# infer_llama_robot.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# -----------------------------
# Model paths
# -----------------------------
BASE_MODEL = "meta-llama/Llama-3.2-1B"
#FINETUNED_DIR = "/LLM/llama31b_lora_final"
FINETUNED_DIR = "./llama31b_lora_final1/checkpoint-2500"

# -----------------------------
# Load tokenizer and model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, FINETUNED_DIR)
model.eval()

# -----------------------------
# Inference function
# -----------------------------
def infer(prompt, max_new_tokens=50):
    # Format the instruction prompt
    input_text = f"Instruction: Generate sequence of robot actions.\nInput: {prompt}\nOutput:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode generated text
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Define known tasks
    KNOWN_TASKS = ["microwave", "sliding_door", "top_burner", "hinge_cabinet", "kettle", "bottom_burner", "light_switch"]

    # Extract tasks mentioned in output
    predicted = [t for t in KNOWN_TASKS if t.lower() in result.lower()]

    # Fallback based on keywords in input
    if not predicted:
        lower_prompt = prompt.lower()
        if "microwave" in lower_prompt:
            predicted = ["microwave"]
        elif "door" in lower_prompt or "slide" in lower_prompt:
            predicted = ["sliding_door"]
        elif "burner" in lower_prompt or "stove" in lower_prompt:
            if "top" in lower_prompt:
                predicted = ["top_burner"]
            else:
                predicted = ["bottom_burner"]
        elif "cabinet" in lower_prompt or "hinge" in lower_prompt:
            predicted = ["hinge_cabinet"]
        elif "kettle" in lower_prompt:
            predicted = ["kettle"]
        elif "light" in lower_prompt:
            predicted = ["light_switch"]

    return predicted, result

# -----------------------------
# Interactive CLI
# -----------------------------
if __name__ == "__main__":
    print("🤖 Robot Command Inference (type 'exit' to quit)")
    while True:
        cmd = input("\n>> Enter command: ").strip()
        if cmd.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        tasks, text_out = infer(cmd)
        print(f"\n Predicted Tasks: {tasks}")
        #print(f"Generated Output: {text_out}")