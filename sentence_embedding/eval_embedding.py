import json
from tasks_embedding import interpret_command

# ------------------------------
# Mapping (VERY IMPORTANT)
# ------------------------------
ACTION_MAP = {
    "top burner": "top_burner",
    "bottom burner": "bottom_burner",
    "microwave": "microwave",
    "slide cabinet": "sliding_door",
    "hinge cabinet": "hinge_cabinet",
    "light switch": "light_switch",
    "kettle": "kettle"
}

# ------------------------------
# Load dataset
# ------------------------------
with open("dataset.json", "r") as f:
    data = json.load(f)

# ------------------------------
# Evaluation
# ------------------------------
def evaluate(data):
    total = len(data)
    exact_match = 0

    total_precision = 0
    total_recall = 0

    for sample in data:
        query = sample["query"]
        gt = set(sample["actions"])

        pred_raw = interpret_command(query)

        # map predictions to dataset format
        pred = set([ACTION_MAP.get(p, p) for p in pred_raw])

        # Exact match
        if pred == gt:
            exact_match += 1

        # Precision / Recall
        tp = len(pred & gt)

        precision = tp / len(pred) if pred else 0
        recall = tp / len(gt) if gt else 0

        total_precision += precision
        total_recall += recall

        # Print errors
        if pred != gt:
            print("\n❌ Error")
            print("Query:", query)
            print("GT   :", gt)
            print("Pred :", pred)

    # Final scores
    accuracy = exact_match / total
    avg_precision = total_precision / total
    avg_recall = total_recall / total

    f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)
          if (avg_precision + avg_recall) > 0 else 0)

    print("\n==============================")
    print(f"Total samples: {total}")
    print(f"Exact Match Accuracy: {accuracy:.3f}")
    print(f"Avg Precision: {avg_precision:.3f}")
    print(f"Avg Recall: {avg_recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("==============================")

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    evaluate(data)