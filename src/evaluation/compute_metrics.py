# src/evaluation/compute_metrics.py

import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer


# ===============================
# 🔹 Sample Test Data (REPLACE WITH YOUR REAL DATA)
# ===============================
test_data = [
    {
        "query": "Identify beam with fire rating violation",
        "retrieved_docs": [
            "Fire rating for beams must be 4 hours per L4.",
            "Beam B1 has 2 hours fire rating."
        ],
        "ground_truth_docs": [
            "Fire rating for beams must be 4 hours per L4."
        ],
        "generated_answer": "Beam B1 violates fire rating requirement. Provided 2 hours, required 4 hours.",
        "ground_truth_answer": "Beam B1 violates fire rating as per L4 regulation."
    },
    {
        "query": "Show L124 inference results",
        "retrieved_docs": [
            "L124 infers compliance using L1, L2 and L4.",
            "Beam B1 is non-compliant."
        ],
        "ground_truth_docs": [
            "L124 infers compliance using L1, L2 and L4."
        ],
        "generated_answer": "Beam B1 is non-compliant based on L124 inference.",
        "ground_truth_answer": "L124 identifies non-compliance based on L4 rules."
    }
]


# ===============================
# 🔹 RAG METRICS (MANUAL)
# ===============================
def compute_rag_metrics(data):

    precision_scores = []
    recall_scores = []

    for item in data:

        retrieved = set(item["retrieved_docs"])
        ground_truth = set(item["ground_truth_docs"])

        # Precision
        if len(retrieved) > 0:
            precision = len(retrieved & ground_truth) / len(retrieved)
        else:
            precision = 0

        # Recall
        if len(ground_truth) > 0:
            recall = len(retrieved & ground_truth) / len(ground_truth)
        else:
            recall = 0

        precision_scores.append(precision)
        recall_scores.append(recall)

    return {
        "Context Precision": float(np.mean(precision_scores)),
        "Context Recall": float(np.mean(recall_scores))
    }


# ===============================
# 🔹 LLM METRICS
# ===============================
def compute_llm_metrics(data):

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    rouge_scores = []
    bleu_scores = []
    semantic_scores = []

    for item in data:

        gen = item["generated_answer"]
        gt = item["ground_truth_answer"]

        # ROUGE
        scores = rouge.score(gt, gen)
        rouge_scores.append(scores['rouge1'].fmeasure)

        # BLEU
        bleu_scores.append(sentence_bleu([gt.split()], gen.split()))

        # Semantic similarity
        emb_gen = model.encode(gen)
        emb_gt = model.encode(gt)

        sim = np.dot(emb_gen, emb_gt) / (
            np.linalg.norm(emb_gen) * np.linalg.norm(emb_gt)
        )

        semantic_scores.append(sim)

    return {
        "ROUGE-1 F1": float(np.mean(rouge_scores)),
        "BLEU Score": float(np.mean(bleu_scores)),
        "Semantic Similarity": float(np.mean(semantic_scores))
    }


# ===============================
# 🔹 ANSWER RELEVANCE (QUERY ↔ ANSWER)
# ===============================
def compute_answer_relevance(data):

    model = SentenceTransformer('all-MiniLM-L6-v2')
    scores = []

    for item in data:

        q_emb = model.encode(item["query"])
        a_emb = model.encode(item["generated_answer"])

        sim = np.dot(q_emb, a_emb) / (
            np.linalg.norm(q_emb) * np.linalg.norm(a_emb)
        )

        scores.append(sim)

    return float(np.mean(scores))


# ===============================
# 🔹 MAIN
# ===============================
if __name__ == "__main__":

    print("Running Evaluation...\n")

    rag_scores = compute_rag_metrics(test_data)
    llm_scores = compute_llm_metrics(test_data)
    relevance_score = compute_answer_relevance(test_data)

    print("🔹 RAG Metrics:")
    for k, v in rag_scores.items():
        print(f"{k}: {v:.3f}")

    print(f"Answer Relevance: {relevance_score:.3f}")

    print("\n🔹 LLM Metrics:")
    for k, v in llm_scores.items():
        print(f"{k}: {v:.3f}")