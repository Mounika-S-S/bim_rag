"""
BIM-RAG Evaluation Framework
=============================
Comprehensive evaluation of RAG retrieval quality and LLM response quality.

RAG Metrics: hit_rate, mrr, ndcg, precision, recall, context_relevance, answer_relevance, context_recall
LLM Metrics: faithfulness, answer_correctness, hallucination_rate, format_compliance, completeness, latency
"""
import os, sys, json, time, math, re
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.rag.unified_vector_store import UnifiedVectorStore
from src.reasoning.structured_responder import StructuredResponder
from src.reasoning.llm_client import LLMClient

PROJECT_ID = "project-final"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(PROJECT_ROOT, f"data/processed/{PROJECT_ID}/unified.index")

# ═══════════════════════════════════════════════════════════════════════
# GROUND TRUTH TEST CASES
# ═══════════════════════════════════════════════════════════════════════
# Each test case has:
#   query: the user question
#   relevant_layers: expected layers in top-k results (for hit_rate)
#   relevant_keywords: keywords that MUST appear in retrieved chunks (for precision/recall)
#   expected_element_type: the element type the query is about
#   expected_answer_keywords: keywords that MUST appear in the final LLM answer
#   expected_status: expected compliance status (COMPLIANT / NON_COMPLIANT / MISSING_PROPERTY / None)

TEST_CASES = [
    {
        "id": "Q1",
        "query": "What is the material of the wall?",
        "relevant_layers": ["L1"],
        "relevant_keywords": ["wall", "material", "YC_Test_Brick"],
        "expected_element_type": "wall",
        "expected_answer_keywords": ["YC_Test_Brick", "material"],
        "expected_status": None,
        "category": "property_lookup",
    },
    {
        "id": "Q2",
        "query": "What is the length of beam YC-ST-SF-BIP?",
        "relevant_layers": ["L1"],
        "relevant_keywords": ["beam", "YC-ST-SF-BIP", "Length"],
        "expected_element_type": "beam",
        "expected_answer_keywords": ["length", "YC-ST-SF-BIP"],
        "expected_status": None,
        "category": "property_lookup",
    },
    {
        "id": "Q3",
        "query": "What material is used for columns?",
        "relevant_layers": ["L1"],
        "relevant_keywords": ["column", "material", "Concrete"],
        "expected_element_type": "column",
        "expected_answer_keywords": ["Concrete", "column"],
        "expected_status": None,
        "category": "property_lookup",
    },
    {
        "id": "Q4",
        "query": "Show all non-compliant elements",
        "relevant_layers": ["compliance"],
        "relevant_keywords": ["NON_COMPLIANT", "MISSING"],
        "expected_element_type": None,
        "expected_answer_keywords": ["non-compliant", "missing"],
        "expected_status": "NON_COMPLIANT",
        "category": "compliance_query",
    },
    {
        "id": "Q5",
        "query": "Is the wall width compliant?",
        "relevant_layers": ["compliance", "L1"],
        "relevant_keywords": ["wall", "Width_mm", "complian"],
        "expected_element_type": "wall",
        "expected_answer_keywords": ["width", "wall"],
        "expected_status": None,
        "category": "compliance_query",
    },
    {
        "id": "Q6",
        "query": "What are the regulations for building height?",
        "relevant_layers": ["L4"],
        "relevant_keywords": ["height", "building", "regulation"],
        "expected_element_type": None,
        "expected_answer_keywords": ["height"],
        "expected_status": None,
        "category": "regulation_query",
    },
    {
        "id": "Q7",
        "query": "List all doors in the project",
        "relevant_layers": ["L1"],
        "relevant_keywords": ["door", "IfcDoor"],
        "expected_element_type": "door",
        "expected_answer_keywords": ["door"],
        "expected_status": None,
        "category": "entity_query",
    },
    {
        "id": "Q8",
        "query": "What is the fire rating requirement for walls?",
        "relevant_layers": ["L4", "compliance"],
        "relevant_keywords": ["fire", "wall"],
        "expected_element_type": "wall",
        "expected_answer_keywords": ["fire", "rating"],
        "expected_status": None,
        "category": "regulation_query",
    },
    {
        "id": "Q9",
        "query": "How many beams are in the project?",
        "relevant_layers": ["L1"],
        "relevant_keywords": ["beam"],
        "expected_element_type": "beam",
        "expected_answer_keywords": ["beam"],
        "expected_status": None,
        "category": "entity_query",
    },
    {
        "id": "Q10",
        "query": "What is the width of walls in this building?",
        "relevant_layers": ["L1"],
        "relevant_keywords": ["wall", "Width_mm"],
        "expected_element_type": "wall",
        "expected_answer_keywords": ["width", "wall"],
        "expected_status": None,
        "category": "property_lookup",
    },
    {
        "id": "Q11",
        "query": "Which walls have a length greater than 5000mm and are they compliant with fire safety?",
        "relevant_layers": ["L1", "L4", "compliance"],
        "relevant_keywords": ["wall", "length", "5000", "fire", "compliant"],
        "expected_element_type": "wall",
        "expected_answer_keywords": ["wall", "compliant", "fire"],
        "expected_status": None,
        "category": "complex_query",
    },
    {
        "id": "Q12",
        "query": "Compare the material of beams and columns and identify any non-compliant materials.",
        "relevant_layers": ["L1", "compliance"],
        "relevant_keywords": ["beam", "column", "material", "NON_COMPLIANT"],
        "expected_element_type": None,
        "expected_answer_keywords": ["beam", "column", "material"],
        "expected_status": "NON_COMPLIANT",
        "category": "complex_query",
    },
]


# ═══════════════════════════════════════════════════════════════════════
# RAG EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_hit_rate(results, relevant_keywords):
    """1 if any relevant keyword found in top-k, else 0."""
    for r in results:
        text = r["text"].lower() if isinstance(r, dict) else r.lower()
        for kw in relevant_keywords:
            if kw.lower() in text:
                return 1.0
    return 0.0


def compute_mrr(results, relevant_keywords):
    """Reciprocal rank of first relevant result."""
    for i, r in enumerate(results):
        text = r["text"].lower() if isinstance(r, dict) else r.lower()
        for kw in relevant_keywords:
            if kw.lower() in text:
                return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(results, relevant_keywords, k=10):
    """Normalized Discounted Cumulative Gain."""
    relevance = []
    for r in results[:k]:
        text = r["text"].lower() if isinstance(r, dict) else r.lower()
        score = sum(1 for kw in relevant_keywords if kw.lower() in text)
        relevance.append(score)

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_precision(results, relevant_keywords, k=5):
    """Fraction of retrieved results that are relevant."""
    relevant_count = 0
    for r in results[:k]:
        text = r["text"].lower() if isinstance(r, dict) else r.lower()
        if any(kw.lower() in text for kw in relevant_keywords):
            relevant_count += 1
    return relevant_count / k if k > 0 else 0.0


def compute_recall(results, relevant_layers, metadata_list):
    """Fraction of relevant chunks (by layer) that were retrieved."""
    total_relevant = sum(1 for m in metadata_list
                         if m.get("layer") in relevant_layers)
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = 0
    for r in results:
        meta = r.get("meta", {}) if isinstance(r, dict) else {}
        if meta.get("layer") in relevant_layers:
            retrieved_relevant += 1
    return min(retrieved_relevant / total_relevant, 1.0)


def compute_context_relevance(results, query):
    """Fraction of retrieved chunks that share semantic overlap with query words."""
    query_words = set(query.lower().split())
    relevant = 0
    for r in results:
        text = r["text"].lower() if isinstance(r, dict) else r.lower()
        text_words = set(text.split())
        overlap = len(query_words & text_words)
        if overlap >= 2:  # At least 2 overlapping words
            relevant += 1
    return relevant / len(results) if results else 0.0


def compute_answer_relevance(answer, query):
    """Fraction of query keywords found in the answer."""
    query_words = set(re.findall(r'\w+', query.lower()))
    # remove stop words
    stops = {"is", "the", "a", "an", "of", "in", "for", "what", "how", "are", "this", "all", "show", "list", "tell", "me", "about", "any"}
    query_words -= stops
    if not query_words:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for w in query_words if w in answer_lower)
    return found / len(query_words)


def compute_context_recall(results, expected_answer_keywords):
    """Fraction of expected answer keywords found in retrieved context."""
    context_text = " ".join(
        r["text"].lower() if isinstance(r, dict) else r.lower() for r in results
    )
    if not expected_answer_keywords:
        return 1.0
    found = sum(1 for kw in expected_answer_keywords if kw.lower() in context_text)
    return found / len(expected_answer_keywords)


# ═══════════════════════════════════════════════════════════════════════
# LLM EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_faithfulness(answer, context_chunks):
    """Fraction of answer sentences that are grounded in the context."""
    context_text = " ".join(
        c["text"].lower() if isinstance(c, dict) else c.lower() for c in context_chunks
    )
    sentences = [s.strip() for s in re.split(r'[.!?\n]', answer) if len(s.strip()) > 10]
    if not sentences:
        return 1.0
    grounded = 0
    for sent in sentences:
        words = set(re.findall(r'\w+', sent.lower())) - {"the", "is", "a", "an", "of", "in", "for", "and", "to", "are", "this", "that", "it", "not", "with"}
        if not words:
            grounded += 1
            continue
        overlap = sum(1 for w in words if w in context_text)
        if overlap / len(words) >= 0.3:  # 30% grounding threshold
            grounded += 1
    return grounded / len(sentences)


def compute_bleu_score(answer, expected_keywords):
    """Compute BLEU score against expected keywords as a pseudo-reference."""
    if not expected_keywords:
        return 1.0
    reference = " ".join(expected_keywords).lower().split()
    hypothesis = answer.lower().split()
    try:
        return sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method1)
    except:
        return 0.0

def compute_rouge_score(answer, expected_keywords):
    """Compute ROUGE-L f1 score against expected keywords."""
    if not expected_keywords:
        return 1.0
    reference = " ".join(expected_keywords).lower()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, answer.lower())
    return scores['rougeL'].fmeasure


def compute_hallucination_rate(answer, context_chunks):
    """Fraction of specific claims (numbers, names) not found in context."""
    context_text = " ".join(
        c["text"].lower() if isinstance(c, dict) else c.lower() for c in context_chunks
    )
    # Extract specific values from answer
    numbers = re.findall(r'\d+\.?\d*', answer)
    proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', answer)
    ifc_ids = re.findall(r'YC[-\w]+', answer)
    claims = numbers + proper_nouns + ifc_ids
    if not claims:
        return 0.0
    hallucinated = sum(1 for c in claims if c.lower() not in context_text)
    return hallucinated / len(claims)


def compute_format_compliance(answer):
    """Score for proper formatting: markdown headers, bullet points, structure."""
    score = 0.0
    checks = 0
    # Has some structure (bullets, headers, bold)
    checks += 1
    if any(c in answer for c in ["•", "-", "*", "**", "##", "✅", "❌", "⚠️", "📋", "💡"]):
        score += 1
    # No raw JSON or code dumps
    checks += 1
    if "```" not in answer and "{" not in answer[:50]:
        score += 1
    # Reasonable length (not too short, not too long)
    checks += 1
    if 50 < len(answer) < 3000:
        score += 1
    # Has clear answer (not empty or generic)
    checks += 1
    if len(answer) > 20:
        score += 1
    return score / checks if checks > 0 else 0.0


def compute_completeness(answer, test_case):
    """Does the answer address the query comprehensively?"""
    score = 0.0
    checks = 0

    # Contains element type if applicable
    if test_case.get("expected_element_type"):
        checks += 1
        if test_case["expected_element_type"].lower() in answer.lower():
            score += 1

    # Contains answer keywords
    for kw in test_case.get("expected_answer_keywords", []):
        checks += 1
        if kw.lower() in answer.lower():
            score += 1

    # Has actionable content (not just "not found")
    checks += 1
    not_found_phrases = ["not found", "not present", "no data", "not available"]
    if not any(p in answer.lower() for p in not_found_phrases):
        score += 1

    return score / checks if checks > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def run_full_evaluation():
    print("=" * 70)
    print("   BIM-RAG EVALUATION FRAMEWORK")
    print("=" * 70)

    # Load vector store
    store = UnifiedVectorStore()
    store.load(INDEX_PATH)
    print(f"Loaded vector store: {len(store.text_chunks)} chunks, {len(store.metadata)} metadata entries\n")

    # Initialize responder and LLM
    responder = StructuredResponder(store)

    try:
        llm = LLMClient()
        llm_available = True
        print("LLM client connected.\n")
    except Exception as e:
        llm_available = False
        print(f"LLM not available: {e}. Skipping LLM evaluation.\n")

    # ── Store results ──
    rag_results = []
    llm_results = []
    k_values = [3, 5, 10]

    # ═══════════════════════════════════════════════════════════════
    # PART 1: QUERY-WISE RAG EVALUATION
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print(" PART 1: QUERY-WISE RAG EVALUATION")
    print("=" * 70)

    for tc in TEST_CASES:
        print(f"\n{'-'*60}")
        print(f"  [{tc['id']}] {tc['query']}")
        print(f"  Category: {tc['category']}")
        print(f"{'-'*60}")

        query_rag = {}
        for k in k_values:
            chunks = store.search_with_metadata(
                tc["query"], k=k,
                filter_element_type=tc.get("expected_element_type")
            )

            hit = compute_hit_rate(chunks, tc["relevant_keywords"])
            mrr = compute_mrr(chunks, tc["relevant_keywords"])
            ndcg = compute_ndcg(chunks, tc["relevant_keywords"], k)
            prec = compute_precision(chunks, tc["relevant_keywords"], k)
            rec = compute_recall(chunks, tc["relevant_layers"], store.metadata)
            ctx_rel = compute_context_relevance(chunks, tc["query"])
            ans_rel = compute_answer_relevance(
                " ".join(c["text"] for c in chunks), tc["query"]
            )
            ctx_rec = compute_context_recall(chunks, tc["expected_answer_keywords"])

            metrics = {
                "hit_rate": round(hit, 4),
                "mrr": round(mrr, 4),
                "ndcg": round(ndcg, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "context_relevance": round(ctx_rel, 4),
                "answer_relevance": round(ans_rel, 4),
                "context_recall": round(ctx_rec, 4),
            }
            query_rag[f"k={k}"] = metrics

            if k == 5:
                print(f"\n  k={k} Results:")
                for metric_name, val in metrics.items():
                    bar = "#" * int(val * 20) + "." * (20 - int(val * 20))
                    print(f"    {metric_name:20s} {bar} {val:.4f}")

        rag_results.append({"test_case": tc, "metrics_by_k": query_rag})

    # ═══════════════════════════════════════════════════════════════
    # PART 2: AGGREGATE RAG EVALUATION (across all queries by k)
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print(" PART 2: AGGREGATE RAG METRICS (averaged over all queries)")
    print("=" * 70)

    metric_names = ["hit_rate", "mrr", "ndcg", "precision", "recall",
                     "context_relevance", "answer_relevance", "context_recall"]

    for k in k_values:
        print(f"\n  +--- Top-K = {k} {'-'*48}+")
        for mn in metric_names:
            vals = [r["metrics_by_k"][f"k={k}"][mn] for r in rag_results]
            avg = np.mean(vals)
            std = np.std(vals)
            bar = "#" * int(avg * 20) + "." * (20 - int(avg * 20))
            print(f"  | {mn:20s} {bar} {avg:.4f} (+/-{std:.4f}) |")
        print(f"  +{'-'*60}+")

    # ═══════════════════════════════════════════════════════════════
    # PART 3: QUERY-WISE LLM EVALUATION
    # ═══════════════════════════════════════════════════════════════
    if llm_available:
        print("\n\n" + "=" * 70)
        print(" PART 3: QUERY-WISE LLM EVALUATION")
        print("=" * 70)

        for tc in TEST_CASES:
            print(f"\n{'-'*60}")
            print(f"  [{tc['id']}] {tc['query']}")
            print(f"{'-'*60}")

            # Get context
            chunks = store.search_with_metadata(
                tc["query"], k=5,
                filter_element_type=tc.get("expected_element_type")
            )

            # Time the response
            start = time.time()
            try:
                answer = responder.respond(tc["query"], llm)
            except Exception as e:
                answer = f"ERROR: {e}"
            latency = time.time() - start

            # Compute LLM metrics
            faith = compute_faithfulness(answer, chunks)
            bleu = compute_bleu_score(answer, tc["expected_answer_keywords"])
            rouge = compute_rouge_score(answer, tc["expected_answer_keywords"])
            halluc = compute_hallucination_rate(answer, chunks)
            fmt = compute_format_compliance(answer)
            compl = compute_completeness(answer, tc)
            latency_score = max(0, 1.0 - (latency / 30))  # Normalize: 30s = 0, instant = 1

            metrics = {
                "faithfulness": round(faith, 4),
                "bleu_score": round(bleu, 4),
                "rouge_score": round(rouge, 4),
                "hallucination_rate": round(halluc, 4),
                "format_compliance": round(fmt, 4),
                "completeness": round(compl, 4),
                "latency_seconds": round(latency, 2),
                "latency_score": round(latency_score, 4),
            }

            print(f"  Answer preview: {answer[:120]}...")
            print()
            for mn, val in metrics.items():
                if mn == "latency_seconds":
                    print(f"    {mn:22s}  {val:.2f}s")
                elif mn == "hallucination_rate":
                    bar = "." * int(val * 20) + "#" * (20 - int(val * 20))  # inverted
                    print(f"    {mn:22s}  {bar} {val:.4f} (lower is better)")
                else:
                    bar = "#" * int(val * 20) + "." * (20 - int(val * 20))
                    print(f"    {mn:22s}  {bar} {val:.4f}")

            llm_results.append({"test_case": tc, "metrics": metrics, "answer": answer})

        # ═══════════════════════════════════════════════════════════════
        # PART 4: AGGREGATE LLM EVALUATION
        # ═══════════════════════════════════════════════════════════════
        print("\n\n" + "=" * 70)
        print(" PART 4: AGGREGATE LLM METRICS (averaged over all queries)")
        print("=" * 70)

        llm_metric_names = ["faithfulness", "bleu_score", "rouge_score", "hallucination_rate",
                            "format_compliance", "completeness", "latency_score"]

        print(f"\n  +{'-'*60}+")
        for mn in llm_metric_names:
            vals = [r["metrics"][mn] for r in llm_results]
            avg = np.mean(vals)
            std = np.std(vals)
            if mn == "hallucination_rate":
                bar = "." * int(avg * 20) + "#" * (20 - int(avg * 20))
                print(f"  | {mn:22s} {bar} {avg:.4f} (+/-{std:.4f}) v |")
            else:
                bar = "#" * int(avg * 20) + "." * (20 - int(avg * 20))
                print(f"  | {mn:22s} {bar} {avg:.4f} (+/-{std:.4f}) ^ |")
        avg_lat = np.mean([r["metrics"]["latency_seconds"] for r in llm_results])
        print(f"  | {'avg_latency_seconds':22s}                      {avg_lat:.2f}s     |")
        print(f"  +{'-'*60}+")

    # ═══════════════════════════════════════════════════════════════
    # SAVE RESULTS TO JSON
    # ═══════════════════════════════════════════════════════════════
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_id": PROJECT_ID,
        "num_chunks_in_store": len(store.text_chunks),
        "num_test_cases": len(TEST_CASES),
        "rag_evaluation": {
            "query_wise": [{
                "id": r["test_case"]["id"],
                "query": r["test_case"]["query"],
                "category": r["test_case"]["category"],
                "metrics_by_k": r["metrics_by_k"]
            } for r in rag_results],
            "aggregate": {}
        },
        "llm_evaluation": {
            "query_wise": [{
                "id": r["test_case"]["id"],
                "query": r["test_case"]["query"],
                "metrics": r["metrics"],
                "answer_preview": r["answer"][:300]
            } for r in llm_results] if llm_results else [],
            "aggregate": {}
        }
    }

    # Compute aggregate metrics for JSON
    for k in k_values:
        k_label = f"k={k}"
        output["rag_evaluation"]["aggregate"][k_label] = {}
        for mn in metric_names:
            vals = [r["metrics_by_k"][k_label][mn] for r in rag_results]
            output["rag_evaluation"]["aggregate"][k_label][mn] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

    if llm_results:
        for mn in llm_metric_names:
            vals = [r["metrics"][mn] for r in llm_results]
            output["llm_evaluation"]["aggregate"][mn] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\n✅ Full results saved to: evaluation/evaluation_results.json")
    print("=" * 70)

    return output


if __name__ == "__main__":
    run_full_evaluation()
