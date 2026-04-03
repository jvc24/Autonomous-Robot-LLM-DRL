# train_llama1b_lora_final.py

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
from accelerate import infer_auto_device_map
import torch
import json
#from utils import preprocess_example, compute_metrics, LossLoggerCallback
from utils_new import preprocess_example_masked, compute_metrics, LossLoggerCallback

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DATASET_FILE = "robot_dataset1.json"
OUTPUT_DIR = "./llama31b_lora_final1"
MAX_LENGTH = 100

# Load Dataset
dataset = load_dataset("json", data_files=DATASET_FILE)
dataset = dataset["train"]

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load model on CPU first
model_cpu = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype=torch.float16,
)

# Infer device map automatically
device_map = infer_auto_device_map(
    model_cpu,
    dtype=torch.float16,
    max_memory={0: "6GB", "cpu": "10GB"}  # Adjust for GPu available 
)

# BitsAndBytes 8-bit config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# Load quantized model with device map
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map=device_map,
)

#model.gradient_checkpointing_enable() #Not compatible with LoRA 

# LoRA config
lora_config = LoraConfig(
    r=8,  # can reduce to 4 if still OOM
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Tokenize dataset
tokenized = dataset.map(lambda x: preprocess_example_masked(x, tokenizer, MAX_LENGTH))

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # reduced for not running out of GPU memory 
    learning_rate=1e-4,
    num_train_epochs=15,
    logging_steps=10,
    evaluation_strategy="no",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    load_best_model_at_end=False,
    metric_for_best_model="loss",
    greater_is_better=False,
)

# Loss logger callback
loss_logger = LossLoggerCallback()

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    eval_dataset=tokenized,  # replace with validation set if available
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[loss_logger],
)

# Start training
trainer.train()

# Save trainer logs for plotting later
with open("trainer_log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)
print("Trainer log history saved to trainer_log_history.json")

# Save model
model.save_pretrained(OUTPUT_DIR)
print(f"Training finished! Model saved to {OUTPUT_DIR}")

# Save step-wise losses for plotting
with open("train_losses.json", "w") as f:
    json.dump(loss_logger.losses, f)
print("Step-wise losses saved to train_losses.json")