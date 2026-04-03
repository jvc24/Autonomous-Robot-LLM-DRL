import json
from Sentence_embedding.train_embedding import interpret_command

# Mapping to match your dataset naming
ACTION_MAP = {
    "top burner": "top_burner",
    "bottom burner": "bottom_burner",
    "microwave": "microwave",
    "slide cabinet": "sliding_door",
    "hinge cabinet": "hinge_cabinet",
    "light switch": "light_switch",
    "kettle": "kettle"
}

# Load dataset (your JSON file)
with open("dataset.json", "r") as f:
    data = json.load(f)

# Helper: compute metrics
def compute_metrics(pred_set, gt_set):
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gt_set) if gt_set else 0
    return precision, recall

# Evaluation function
def evaluate_by_task_count(data):
    categories = {1: [], 2: [], 3: []}  # single, double, triple
    overall_metrics = []

    for sample in data:
        query = sample["query"]
        gt = set(sample["actions"])

        pred_raw = interpret_command(query)
        pred = set([ACTION_MAP.get(p, p) for p in pred_raw])

        precision, recall = compute_metrics(pred, gt)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        exact = int(pred == gt)

        # Categorize by number of ground-truth tasks
        n_tasks = len(gt)
        if n_tasks in categories:
            categories[n_tasks].append((exact, precision, recall, f1))

        overall_metrics.append((exact, precision, recall, f1))

    # Print results per category
    print("\n===== Evaluation by Task Count =====")
    for n, results in categories.items():
        if results:
            total = len(results)
            exact_acc = sum(r[0] for r in results) / total
            avg_precision = sum(r[1] for r in results) / total
            avg_recall = sum(r[2] for r in results) / total
            avg_f1 = sum(r[3] for r in results) / total

            print(f"\nTasks = {n}")
            print(f"  Samples: {total}")
            print(f"  Exact Match Accuracy: {exact_acc:.3f}")
            print(f"  Avg Precision: {avg_precision:.3f}")
            print(f"  Avg Recall: {avg_recall:.3f}")
            print(f"  F1 Score: {avg_f1:.3f}")

    # Overall metrics
    total = len(overall_metrics)
    exact_acc = sum(r[0] for r in overall_metrics) / total
    avg_precision = sum(r[1] for r in overall_metrics) / total
    avg_recall = sum(r[2] for r in overall_metrics) / total
    avg_f1 = sum(r[3] for r in overall_metrics) / total

    print("\n===== Overall Metrics =====")
    print(f"  Total Samples: {total}")
    print(f"  Exact Match Accuracy: {exact_acc:.3f}")
    print(f"  Avg Precision: {avg_precision:.3f}")
    print(f"  Avg Recall: {avg_recall:.3f}")
    print(f"  F1 Score: {avg_f1:.3f}")

# Run evaluation
if __name__ == "__main__":
    evaluate_by_task_count(data)