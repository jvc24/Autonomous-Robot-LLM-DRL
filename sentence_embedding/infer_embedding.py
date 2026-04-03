# interactive_inference.py
from tasks_embedding import interpret_command

# Mapping to match your dataset/task naming
ACTION_MAP = {
    "top burner": "top_burner",
    "bottom burner": "bottom_burner",
    "microwave": "microwave",
    "slide cabinet": "sliding_door",
    "hinge cabinet": "hinge_cabinet",
    "light switch": "light_switch",
    "kettle": "kettle"
}

def infer_command_loop():
    """
    Continuously takes a natural language command from the user
    and prints the predicted robot task(s) until 'exit' is typed.
    """
    print("=== Robot Task Command Inference ===")
    print("Type 'exit' to quit.\n")
    
    while True:
        # Get user input
        user_input = input("Enter command: ").strip()
        if user_input.lower() == "exit":
            print("Exiting inference loop.")
            break
        
        # Predict tasks using trained model
        pred_raw = interpret_command(user_input)
        pred_tasks = [ACTION_MAP.get(task, task) for task in pred_raw]
        
        # Print results
        if pred_tasks:
            print(f"Predicted Task(s): {pred_tasks}\n")
        else:
            print("No task recognized. Try a different command.\n")


if __name__ == "__main__":
    infer_command_loop()