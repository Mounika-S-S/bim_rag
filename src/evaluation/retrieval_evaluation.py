from src.rag.unified_vector_store import UnifiedVectorStore


def evaluate_retrieval(project_id):

    # Define stronger ground truth
    test_cases = [
        {
            "query": "What is minimum road width?",
            "relevant_phrase": "7.2m"
        },
        {
            "query": "Fire rating requirement?",
            "relevant_phrase": "fire rating"
        }
    ]

    store = UnifiedVectorStore()
    path = f"data/processed/{project_id}/unified.index"
    store.load(path)

    k = 5
    top1_correct = 0
    reciprocal_ranks = []

    for case in test_cases:

        results = store.search(case["query"], k=k)

        found_rank = None

        for idx, r in enumerate(results):
            if case["relevant_phrase"].lower() in r.lower():
                found_rank = idx + 1
                break

        print(f"\nQuery: {case['query']}")
        print("Found Rank:", found_rank)

        # Top-1 Accuracy
        if found_rank == 1:
            top1_correct += 1

        # MRR calculation
        if found_rank:
            reciprocal_ranks.append(1 / found_rank)
        else:
            reciprocal_ranks.append(0)

    top1_accuracy = top1_correct / len(test_cases)
    mrr = sum(reciprocal_ranks) / len(test_cases)

    print("\nTop-1 Accuracy:", top1_accuracy)
    print("MRR:", mrr)


if __name__ == "__main__":
    evaluate_retrieval("new")