from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter
import os
import matplotlib.pyplot as plt
import numpy as np
# Create folder for plots
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    hamming_loss,
    multilabel_confusion_matrix,
    classification_report
)

os.makedirs("plots", exist_ok=True)

# -------------------------------
# Configuration
# -------------------------------
BASE_MODEL = "meta-llama/Llama-3.2-1B"
FINETUNED_DIR = "llama31b_lora_final"
TEST_DATA_FILE = "test_robot_dataset.json"
MAX_NEW_TOKENS = 50
ERROR_LOG_FILE = "llm_error_analysis.json"

# Define known tasks
KNOWN_TASKS = [
    "microwave",
    "sliding_door",
    "top_burner",
    "hinge_cabinet",
    "kettle",
    "light_switch"
]

# -------------------------------
# Load Model & Tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, FINETUNED_DIR)
model.eval()

# -------------------------------
# Inference function
# -------------------------------
def infer_tasks(prompt, max_new_tokens=MAX_NEW_TOKENS):
    input_text = f"Instruction: Generate sequence of robot actions.\nInput: {prompt}\nOutput:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract generated actions from result
    predicted = [t for t in KNOWN_TASKS if t.lower() in result.lower()]

    # Fallback to keyword mapping if none predicted
    if not predicted:
        lower_prompt = prompt.lower()
        if "microwave" in lower_prompt or "food" in lower_prompt:
            predicted.append("microwave")
        if "door" in lower_prompt:
            predicted.append("sliding_door")
        if "burner" in lower_prompt or "stove" in lower_prompt:
            predicted.append("top_burner")
        if "cabinet" in lower_prompt or "store" in lower_prompt:
            predicted.append("hinge_cabinet")
        if "kettle" in lower_prompt or "tea" in lower_prompt or "coffee" in lower_prompt:
            predicted.append("kettle")
        if "light" in lower_prompt or "dark" in lower_prompt:
            predicted.append("light_switch")

    return list(set(predicted)), result  # return raw text too


# -------------------------------
# Helper Functions
# -------------------------------
def exact_match(pred, gt):
    return set(pred) == set(gt)

def to_multihot(labels, known_tasks):
    return [1 if task in labels else 0 for task in known_tasks]


# -------------------------------
# Load Test Dataset
# -------------------------------
with open(TEST_DATA_FILE, "r") as f:
    test_data = json.load(f)

# -------------------------------
# Evaluation Containers
# -------------------------------
metrics_by_task = {
    1: {"samples": 0, "exact": 0, "precision": 0, "recall": 0, "f1": 0},
    2: {"samples": 0, "exact": 0, "precision": 0, "recall": 0, "f1": 0},
    3: {"samples": 0, "exact": 0, "precision": 0, "recall": 0, "f1": 0},
}

all_preds = []
all_gts = []

all_preds_bin = []
all_gts_bin = []

error_cases = []

# -------------------------------
# Evaluate
# -------------------------------
for example in tqdm(test_data, desc="Evaluating"):
    prompt = example["input"]
    gt = example["output"]["actions"]
    pred, raw_output = infer_tasks(prompt)

    all_preds.append(pred)
    all_gts.append(gt)

    gt_bin = to_multihot(gt, KNOWN_TASKS)
    pred_bin = to_multihot(pred, KNOWN_TASKS)

    all_gts_bin.append(gt_bin)
    all_preds_bin.append(pred_bin)

    n_tasks = len(gt)
    if n_tasks > 3:
        n_tasks = 3

    # Exact match
    metrics_by_task[n_tasks]["samples"] += 1
    is_exact = exact_match(pred, gt)
    metrics_by_task[n_tasks]["exact"] += int(is_exact)

    # Per-sample precision / recall / f1
    if len(pred) == 0:
        p = 0.0
    else:
        p = len(set(pred) & set(gt)) / len(set(pred))

    r = len(set(pred) & set(gt)) / len(set(gt))
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    metrics_by_task[n_tasks]["precision"] += p
    metrics_by_task[n_tasks]["recall"] += r
    metrics_by_task[n_tasks]["f1"] += f1

    # Save errors for analysis
    if not is_exact:
        missed = list(set(gt) - set(pred))
        extra = list(set(pred) - set(gt))

        error_cases.append({
            "input": prompt,
            "ground_truth": gt,
            "prediction": pred,
            "raw_model_output": raw_output,
            "missed_tasks": missed,
            "extra_tasks": extra,
            "num_gt_tasks": len(gt)
        })

# Convert to numpy
all_gts_bin = np.array(all_gts_bin)
all_preds_bin = np.array(all_preds_bin)

# -------------------------------
# Print Results by Task Count
# -------------------------------
print("\n===== Evaluation by Task Count =====")
for k in [1, 2, 3]:
    m = metrics_by_task[k]
    if m["samples"] == 0:
        continue
    print(f"Tasks = {k}")
    print(f"  Samples: {m['samples']}")
    print(f"  Exact Match Accuracy: {m['exact']/m['samples']:.3f}")
    print(f"  Avg Precision: {m['precision']/m['samples']:.3f}")
    print(f"  Avg Recall: {m['recall']/m['samples']:.3f}")
    print(f"  Avg F1 Score: {m['f1']/m['samples']:.3f}\n")

# -------------------------------
# Overall Metrics
# -------------------------------
subset_acc = accuracy_score(all_gts_bin, all_preds_bin)  # exact match on label vector
micro_p = precision_score(all_gts_bin, all_preds_bin, average="micro", zero_division=0)
micro_r = recall_score(all_gts_bin, all_preds_bin, average="micro", zero_division=0)
micro_f1 = f1_score(all_gts_bin, all_preds_bin, average="micro", zero_division=0)

macro_p = precision_score(all_gts_bin, all_preds_bin, average="macro", zero_division=0)
macro_r = recall_score(all_gts_bin, all_preds_bin, average="macro", zero_division=0)
macro_f1 = f1_score(all_gts_bin, all_preds_bin, average="macro", zero_division=0)

ham_loss = hamming_loss(all_gts_bin, all_preds_bin)
ham_acc = 1 - ham_loss

print("===== Overall Metrics =====")
print(f"Subset Accuracy (Exact Match): {subset_acc:.3f}")
print(f"Micro Precision: {micro_p:.3f}")
print(f"Micro Recall:    {micro_r:.3f}")
print(f"Micro F1 Score:  {micro_f1:.3f}")
print(f"Macro Precision: {macro_p:.3f}")
print(f"Macro Recall:    {macro_r:.3f}")
print(f"Macro F1 Score:  {macro_f1:.3f}")
print(f"Hamming Accuracy:{ham_acc:.3f}")
print(f"Hamming Loss:    {ham_loss:.3f}")

# -------------------------------
# Per-task Classification Report
# -------------------------------
print("\n===== Per-Task Classification Report =====")
print(classification_report(
    all_gts_bin,
    all_preds_bin,
    target_names=KNOWN_TASKS,
    zero_division=0
))

# -------------------------------
# Confusion Matrices
# -------------------------------
print("\n===== Confusion Matrices Per Task =====")
mcm = multilabel_confusion_matrix(all_gts_bin, all_preds_bin)


# Compute multilabel confusion matrices
mcm = multilabel_confusion_matrix(all_gts_bin, all_preds_bin)

# Plot one confusion matrix per task
for i, task in enumerate(KNOWN_TASKS):
    cm = mcm[i]  # 2x2 matrix: [[TN, FP], [FN, TP]]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.set_title(f"Confusion Matrix - {task}")
    plt.colorbar(im, ax=ax)

    class_names = ["Negative", "Positive"]
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    # Add values inside cells
    thresh = cm.max() / 2.
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col, row, format(cm[row, col], 'd'),
                ha="center", va="center",
                color="white" if cm[row, col] > thresh else "black",
                fontsize=12
            )

    plt.tight_layout()

    # Save figure
    plt.savefig(f"plots/confusion_matrix_{task}.png", dpi=300, bbox_inches="tight")
    plt.close()

print("Saved confusion matrix plots in /plots/")

for i, task in enumerate(KNOWN_TASKS):
    tn, fp, fn, tp = mcm[i].ravel()
    print(f"\nTask: {task}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# -------------------------------
# Save Error Analysis
# -------------------------------
with open(ERROR_LOG_FILE, "w") as f:
    json.dump(error_cases, f, indent=4)

print(f"\nSaved {len(error_cases)} error cases to {ERROR_LOG_FILE}")



# ----------------------------------
# Task-level confusion analysis
# ----------------------------------

extended_cols = KNOWN_TASKS + ["MISSED"]
task_conf_matrix = pd.DataFrame(
    0, index=KNOWN_TASKS, columns=extended_cols
)

extra_pred_counter = Counter()

for e in error_cases:
    gt_tasks = set(e["ground_truth"])
    pred_tasks = set(e["prediction"])

    # For each GT task:
    for gt_task in gt_tasks:
        if gt_task in pred_tasks:
            # correctly predicted task
            task_conf_matrix.loc[gt_task, gt_task] += 1
        else:
            # missed completely
            task_conf_matrix.loc[gt_task, "MISSED"] += 1

            # Optional: if there are wrong predictions, count "what it predicted instead"
            wrong_preds = pred_tasks - gt_tasks
            for wp in wrong_preds:
                task_conf_matrix.loc[gt_task, wp] += 1

    # Count extra predicted tasks
    extra_tasks = pred_tasks - gt_tasks
    for extra in extra_tasks:
        extra_pred_counter[extra] += 1

# Also add exact-match correct samples (since error_cases only contains wrong ones)
for gt, pred in zip(all_gts, all_preds):
    gt_tasks = set(gt)
    pred_tasks = set(pred)

    for gt_task in gt_tasks:
        if gt_task in pred_tasks:
            task_conf_matrix.loc[gt_task, gt_task] += 1

print("\n===== Task-Level Confusion Matrix =====")
print(task_conf_matrix)

task_conf_matrix.to_csv("plots/task_confusion_matrix.csv")
print("Saved task confusion matrix to plots/task_confusion_matrix.csv")

# ----------------------------------
# Plot task confusion heatmap
# ----------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

im = ax.imshow(task_conf_matrix.values, cmap="Blues")

ax.set_xticks(np.arange(len(task_conf_matrix.columns)))
ax.set_yticks(np.arange(len(task_conf_matrix.index)))

ax.set_xticklabels(task_conf_matrix.columns, rotation=45, ha="right")
ax.set_yticklabels(task_conf_matrix.index)

ax.set_xlabel("Predicted Task / Outcome")
ax.set_ylabel("Ground Truth Task")
ax.set_title("Task-Level Confusion Analysis")

# Add numbers inside cells
thresh = task_conf_matrix.values.max() / 2.0
for i in range(task_conf_matrix.shape[0]):
    for j in range(task_conf_matrix.shape[1]):
        value = task_conf_matrix.iloc[i, j]
        ax.text(
            j, i, str(value),
            ha="center", va="center",
            color="white" if value > thresh else "black",
            fontsize=10
        )

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("plots/task_confusion_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved task confusion heatmap to plots/task_confusion_heatmap.png")