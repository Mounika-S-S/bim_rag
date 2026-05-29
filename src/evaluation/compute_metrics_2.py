import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

try:
    from src.core.model_manager import model_manager
except Exception:
    model_manager = None


ROOT = Path(__file__).resolve().parents[2]
CHAT_HISTORY_DIR = ROOT / "data" / "chat_history"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "evaluation"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "both", "by", "can", "do", "for",
    "from", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "let",
    "list", "me", "of", "on", "or", "please", "show", "summarize", "the", "their",
    "them", "this", "to", "up", "use", "what", "which", "with", "you", "your",
}
TOXIC_TERMS = {
    "hate", "idiot", "stupid", "dumb", "moron", "racist", "sexist", "kill",
    "nonsense", "worthless", "trash", "shut up",
}


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else float(default)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_:/\.-]+", (text or "").lower())


def content_tokens(text: str) -> List[str]:
    return [tok for tok in tokenize(text) if tok not in STOPWORDS and len(tok) > 1]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", normalize_text(text))
    return [part.strip() for part in parts if part.strip()]


def count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    syllables = len(groups)
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def flesch_reading_ease(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text or "")
    sentences = split_sentences(text)
    if not words or not sentences:
        return 0.0
    syllables = sum(count_syllables(word) for word in words)
    score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
    return float(score)


def lexical_recall(text: str, reference: str) -> float:
    src = set(content_tokens(text))
    ref = set(content_tokens(reference))
    if not src:
        return 0.0
    return len(src & ref) / len(src)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    a = set(content_tokens(text_a))
    b = set(content_tokens(text_b))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class EmbeddingHelper:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        self.failed = False
        self.cache: Dict[str, np.ndarray] = {}

    def _load_model(self) -> None:
        if self.model is not None or self.failed or model_manager is None:
            return
        try:
            self.model = model_manager.get_model(self.model_name)
        except Exception:
            self.failed = True

    def encode(self, text: str) -> Optional[np.ndarray]:
        text = normalize_text(text)
        if not text:
            return None
        if text in self.cache:
            return self.cache[text]
        self._load_model()
        if self.model is None:
            return None
        embedding = np.asarray(self.model.encode(text))
        self.cache[text] = embedding
        return embedding

    def similarity(self, text_a: str, text_b: str) -> float:
        emb_a = self.encode(text_a)
        emb_b = self.encode(text_b)
        if emb_a is None or emb_b is None:
            return jaccard_similarity(text_a, text_b)
        denom = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        if denom == 0:
            return 0.0
        return float(np.dot(emb_a, emb_b) / denom)


def flatten_json_text(value) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                yield from flatten_json_text(item)
            elif item is not None:
                yield f"{key}: {item}"
    elif isinstance(value, list):
        for item in value:
            yield from flatten_json_text(item)
    elif value is not None:
        yield str(value)


def load_project_corpus(processed_dir: Path, project_id: str) -> List[str]:
    project_dir = processed_dir / project_id
    if not project_dir.exists():
        return []

    texts_path = project_dir / "unified.index.texts"
    if texts_path.exists():
        try:
            corpus = json.loads(texts_path.read_text(encoding="utf-8"))
            return [normalize_text(item) for item in corpus if normalize_text(item)]
        except json.JSONDecodeError:
            pass

    corpus: List[str] = []
    for json_path in sorted(project_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        corpus.extend(normalize_text(text) for text in flatten_json_text(data) if normalize_text(text))
    return corpus


def load_chat_pairs(chat_file: Path) -> List[Tuple[str, str]]:
    try:
        history = json.loads(chat_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    pairs: List[Tuple[str, str]] = []
    for idx in range(len(history) - 1):
        current = history[idx]
        following = history[idx + 1]
        if current.get("role") == "user" and following.get("role") == "assistant":
            pairs.append((normalize_text(current.get("text", "")), normalize_text(following.get("text", ""))))
    return pairs


def pick_top_evidence(query: str, answer: str, corpus: Sequence[str], helper: EmbeddingHelper, top_k: int = 5) -> List[str]:
    if not corpus:
        return []
    scored: List[Tuple[float, str]] = []
    for chunk in corpus:
        lexical = 0.7 * jaccard_similarity(query, chunk) + 0.3 * lexical_recall(answer, chunk)
        semantic = helper.similarity(query, chunk)
        score = 0.55 * semantic + 0.45 * lexical
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def layer_completeness(query: str, answer: str) -> float:
    requested_layers = re.findall(r"\bl\d+\b", query.lower())
    if not requested_layers:
        return 0.0
    answer_lower = answer.lower()
    covered = sum(1 for layer in requested_layers if layer in answer_lower)
    return covered / len(requested_layers)


def aspect_completeness(query: str, answer: str) -> float:
    aspects = []
    separators = [" and ", ",", "/", " vs ", " versus "]
    fragments = [query]
    for separator in separators:
        expanded = []
        for fragment in fragments:
            expanded.extend(fragment.split(separator))
        fragments = expanded
    for fragment in fragments:
        tokens = [tok for tok in content_tokens(fragment) if not re.fullmatch(r"l\d+", tok)]
        if tokens:
            aspects.append(set(tokens))
    if not aspects:
        return 0.0
    answer_tokens = set(content_tokens(answer))
    scores = []
    for aspect in aspects:
        scores.append(len(aspect & answer_tokens) / len(aspect))
    return safe_mean(scores)


def numeric_support(answer: str, evidence: Sequence[str]) -> float:
    answer_numbers = re.findall(r"\d+(?:\.\d+)?", answer)
    if not answer_numbers:
        return 1.0
    evidence_blob = " ".join(evidence)
    supported = sum(1 for number in answer_numbers if re.search(rf"\b{re.escape(number)}\b", evidence_blob))
    return supported / len(answer_numbers)


def sentence_groundedness(answer: str, evidence: Sequence[str], helper: EmbeddingHelper) -> float:
    sentences = split_sentences(answer)
    if not sentences or not evidence:
        return 0.0
    sentence_scores = []
    for sentence in sentences:
        best = 0.0
        for chunk in evidence:
            lexical = lexical_recall(sentence, chunk)
            semantic = helper.similarity(sentence, chunk)
            best = max(best, 0.5 * lexical + 0.5 * semantic)
        sentence_scores.append(best)
    return safe_mean(sentence_scores)


def coherence_and_fluency(answer: str) -> float:
    sentences = split_sentences(answer)
    words = re.findall(r"[A-Za-z]+", answer)
    if not sentences or not words:
        return 0.0

    avg_sentence_length = len(words) / len(sentences)
    length_score = 1.0 - min(abs(avg_sentence_length - 18) / 30, 1.0)

    readability = flesch_reading_ease(answer)
    readability_score = clamp((readability + 20) / 100)

    lower_tokens = content_tokens(answer)
    repetition_penalty = 0.0
    if len(lower_tokens) >= 6:
        trigrams = list(zip(lower_tokens, lower_tokens[1:], lower_tokens[2:]))
        counts = Counter(trigrams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        repetition_penalty = min(repeated / max(len(trigrams), 1), 0.5)

    punctuation_bonus = 1.0 if re.search(r"[.!?]", answer) else 0.7
    score = 0.4 * length_score + 0.35 * readability_score + 0.25 * punctuation_bonus - repetition_penalty
    return clamp(score)


def toxicity_score(answer: str) -> float:
    answer_lower = answer.lower()
    hits = sum(1 for term in TOXIC_TERMS if term in answer_lower)
    if hits == 0:
        return 0.0
    return clamp(hits / 4)


def estimate_user_satisfaction(metrics: Dict[str, float]) -> float:
    contributors = [
        metrics["relevance"],
        metrics["completeness"],
        metrics["coherence_fluency"],
        metrics["faithfulness_groundedness"],
        metrics["factual_accuracy"],
    ]
    penalty = 0.5 * metrics["toxicity_bias_risk"]
    return clamp(safe_mean(contributors) - penalty)


def compute_reference_overlap(answer: str, reference_text: str) -> Dict[str, float]:
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = rouge.score(reference_text, answer)
    bleu = sentence_bleu(
        [reference_text.split()],
        answer.split(),
        smoothing_function=SmoothingFunction().method1,
    )
    return {
        "rouge1_f1": float(rouge_scores["rouge1"].fmeasure),
        "rougeL_f1": float(rouge_scores["rougeL"].fmeasure),
        "bleu": float(bleu),
    }


def compute_turn_metrics(query: str, answer: str, corpus: Sequence[str], helper: EmbeddingHelper) -> Dict[str, object]:
    evidence = pick_top_evidence(query, answer, corpus, helper, top_k=5)
    evidence_blob = " ".join(evidence)

    relevance = helper.similarity(query, answer)
    groundedness = sentence_groundedness(answer, evidence, helper)
    factual_accuracy = clamp(0.7 * groundedness + 0.3 * numeric_support(answer, evidence))
    completeness = clamp(0.5 * aspect_completeness(query, answer) + 0.5 * max(layer_completeness(query, answer), aspect_completeness(query, answer)))
    coherence = coherence_and_fluency(answer)
    faithfulness = clamp(0.6 * groundedness + 0.4 * lexical_recall(answer, evidence_blob))
    toxicity = toxicity_score(answer)
    overlap = compute_reference_overlap(answer, evidence_blob or query)

    metrics = {
        "factual_accuracy": factual_accuracy,
        "relevance": relevance,
        "coherence_fluency": coherence,
        "completeness": completeness,
        "faithfulness_groundedness": faithfulness,
        "toxicity_bias_risk": toxicity,
        "response_time_latency": None,
        "user_satisfaction_proxy": 0.0,
        "robustness": None,
        "rouge1_f1_vs_evidence": overlap["rouge1_f1"],
        "rougeL_f1_vs_evidence": overlap["rougeL_f1"],
        "bleu_vs_evidence": overlap["bleu"],
    }
    metrics["user_satisfaction_proxy"] = estimate_user_satisfaction(metrics)

    return {
        "query": query,
        "answer": answer,
        "top_evidence": evidence,
        "metrics": metrics,
    }


def compute_robustness(turns: List[Dict[str, object]], helper: EmbeddingHelper) -> Optional[float]:
    if len(turns) < 2:
        return None

    pair_scores = []
    for idx, left in enumerate(turns):
        for right in turns[idx + 1:]:
            q1 = left["query"]
            q2 = right["query"]
            query_similarity = helper.similarity(q1, q2)
            if query_similarity < 0.60:
                continue
            a1 = left["answer"]
            a2 = right["answer"]
            answer_similarity = helper.similarity(a1, a2)
            lexical = jaccard_similarity(a1, a2)
            pair_scores.append(clamp(0.7 * answer_similarity + 0.3 * lexical))

    return safe_mean(pair_scores, default=0.0) if pair_scores else None


def aggregate_metrics(turns: List[Dict[str, object]]) -> Dict[str, Optional[float]]:
    keys = [
        "factual_accuracy",
        "relevance",
        "coherence_fluency",
        "completeness",
        "faithfulness_groundedness",
        "toxicity_bias_risk",
        "user_satisfaction_proxy",
        "rouge1_f1_vs_evidence",
        "rougeL_f1_vs_evidence",
        "bleu_vs_evidence",
    ]
    result: Dict[str, Optional[float]] = {}
    for key in keys:
        values = [turn["metrics"][key] for turn in turns if turn["metrics"][key] is not None]
        result[key] = safe_mean(values) if values else None

    response_times = [turn["metrics"]["response_time_latency"] for turn in turns if turn["metrics"]["response_time_latency"] is not None]
    result["response_time_latency"] = safe_mean(response_times) if response_times else None

    robustness_values = [turn["metrics"]["robustness"] for turn in turns if turn["metrics"]["robustness"] is not None]
    result["robustness"] = safe_mean(robustness_values) if robustness_values else None
    return result


def evaluate_chat_history(chat_dir: Path, processed_dir: Path) -> Dict[str, object]:
    helper = EmbeddingHelper()

    session_reports = []
    project_index: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    corpus_cache: Dict[str, List[str]] = {}

    for chat_file in sorted(chat_dir.glob("*.json")):
        if "__" not in chat_file.stem:
            continue
        project_id, session_id = chat_file.stem.split("__", 1)
        corpus = corpus_cache.setdefault(project_id, load_project_corpus(processed_dir, project_id))
        pairs = load_chat_pairs(chat_file)
        turns = [compute_turn_metrics(query, answer, corpus, helper) for query, answer in pairs]
        robustness = compute_robustness(turns, helper)
        if robustness is not None:
            for turn in turns:
                turn["metrics"]["robustness"] = robustness

        session_report = {
            "project_id": project_id,
            "session_id": session_id,
            "turn_count": len(turns),
            "available_corpus_chunks": len(corpus),
            "aggregate_metrics": aggregate_metrics(turns),
            "turns": turns,
        }
        session_reports.append(session_report)
        project_index[project_id].extend(turns)

    project_reports = []
    for project_id, turns in sorted(project_index.items()):
        robustness = compute_robustness(turns, helper)
        if robustness is not None:
            for turn in turns:
                turn["metrics"]["robustness"] = robustness
        project_reports.append(
            {
                "project_id": project_id,
                "turn_count": len(turns),
                "aggregate_metrics": aggregate_metrics(turns),
            }
        )

    all_turns = [turn for turns in project_index.values() for turn in turns]
    overall_robustness = compute_robustness(all_turns, helper)
    if overall_robustness is not None:
        for turn in all_turns:
            turn["metrics"]["robustness"] = overall_robustness

    return {
        "summary": {
            "chat_history_dir": str(chat_dir),
            "processed_dir": str(processed_dir),
            "session_count": len(session_reports),
            "project_count": len(project_reports),
            "turn_count": len(all_turns),
            "notes": [
                "Response-time metrics require timestamps or tracing data; chat_history JSON currently does not store them.",
                "User satisfaction is reported as a proxy score derived from relevance, completeness, groundedness, and fluency.",
                "Toxicity and bias are heuristic lexical checks here; replace with a classifier such as Detoxify or Perspective for stronger moderation evaluation.",
            ],
            "aggregate_metrics": aggregate_metrics(all_turns),
        },
        "projects": project_reports,
        "sessions": session_reports,
    }


def print_summary(report: Dict[str, object]) -> None:
    summary = report["summary"]
    metrics = summary["aggregate_metrics"]
    print("LLM Evaluation Report")
    print("=" * 80)
    print(f"Sessions evaluated : {summary['session_count']}")
    print(f"Projects evaluated : {summary['project_count']}")
    print(f"Turns evaluated    : {summary['turn_count']}")
    print()
    print("Aggregate Metrics")
    print("-" * 80)
    ordered_keys = [
        "factual_accuracy",
        "relevance",
        "coherence_fluency",
        "completeness",
        "faithfulness_groundedness",
        "toxicity_bias_risk",
        "user_satisfaction_proxy",
        "robustness",
        "response_time_latency",
        "rouge1_f1_vs_evidence",
        "rougeL_f1_vs_evidence",
        "bleu_vs_evidence",
    ]
    for key in ordered_keys:
        value = metrics.get(key)
        pretty = "N/A" if value is None else f"{value:.3f}"
        print(f"{key:28} {pretty}")
    print()
    print("Notes")
    print("-" * 80)
    for note in summary["notes"]:
        print(f"- {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BIM-RAG chat history against project knowledge.")
    parser.add_argument("--chat-dir", default=str(CHAT_HISTORY_DIR), help="Directory containing chat history JSON files.")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR), help="Directory containing processed project data.")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "compute_metrics_2_report.json"), help="Path to write JSON report.")
    args = parser.parse_args()

    chat_dir = Path(args.chat_dir)
    processed_dir = Path(args.processed_dir)
    report = evaluate_chat_history(chat_dir, processed_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(report)
    print()
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
