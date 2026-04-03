from sentence_transformers import SentenceTransformer, util
import re

# ------------------------------
# Load model
# ------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------------------
# Define tasks (better descriptions)
# ------------------------------
TASKS = [
    ("bottom burner", "activate the lower stove burner knob"),
    ("top burner", "activate the upper stove burner knob"),
    ("light switch", "turn on the light switch"),
    ("slide cabinet", "open a sliding cabinet"),
    ("hinge cabinet", "open a cabinet with hinges"),
    ("microwave", "open the microwave door"),
    ("kettle", "place or move the kettle onto a burner")
]

task_names = [t[0] for t in TASKS]
task_desc = [t[1] for t in TASKS]

# Precompute embeddings
task_embeddings = model.encode(
    task_desc,
    convert_to_tensor=True,
    normalize_embeddings=True
)

# ------------------------------
# Split multi-command
# ------------------------------
def split_command(command):
    parts = re.split(r'\band\b|\bthen\b|,', command.lower())
    return [p.strip() for p in parts if p.strip()]

# ------------------------------
# Interpret command → RETURN LIST
# ------------------------------
def interpret_command(command, threshold=0.35):
    parts = split_command(command)

    final_tasks = []

    for part in parts:
        emb = model.encode(
            part,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        similarities = util.cos_sim(emb, task_embeddings)[0]

        # ✅ take ONLY best match
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()

        if best_score > threshold:
            task = task_names[best_idx]

            if task not in final_tasks:
                final_tasks.append(task)

    return final_tasks

# ------------------------------
# INTERACTIVE TEST
# ------------------------------
if __name__ == "__main__":
    print(" Kitchen Command Interpreter")
    print("Type a command (or 'exit')\n")

    while True:
        cmd = input(">> ")

        if cmd.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        tasks = interpret_command(cmd)

        print("Predicted tasks:", tasks)
        print()