# src/evaluation/routing_evaluation.py

from src.app import classify_query


def evaluate_routing():

    test_queries = [
        ("Why is YC-ST-SF-BIP beam non compliant?", "COMPLIANCE_EXPLANATION"),
        ("Which beams are non compliant?", "LIST_NON_COMPLIANT"),
        ("What is cost of YC-ST-SF-BIP beam?", "COST_QUERY"),
        ("What is minimum road width?", "REGULATION_QUERY"),
    ]

    correct = 0

    for query, expected in test_queries:

        predicted = classify_query(query)

        if predicted == expected:
            correct += 1

        print(f"Query: {query}")
        print(f"Expected: {expected}, Predicted: {predicted}\n")

    accuracy = correct / len(test_queries)

    print("Routing Accuracy:", accuracy)


if __name__ == "__main__":
    evaluate_routing()