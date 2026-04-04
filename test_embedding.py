import sys
import time
import gymnasium as gym
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from buffer import ReplayBuffer
from model import *
from agent import *
from multi_agent import *
from Sentence_embedding.tasks_embedding import interpret_command


# Allowed tasks with trained policies
TRAINED_TASKS = ["microwave", "hinge cabinet", "top burner"]


def normalize_tasks(tasks):
    """
    Normalize task names and apply synonyms mapping.
    Removes duplicates while preserving order.
    """
    if not tasks:
        return []

    mapping = {
        "cabinet": "hinge cabinet",
        "burner": "top burner",
        "microwave": "microwave",
        "sliding_door": "sliding_door",
        "top_burner": "top burner"
    }

    normalized = []
    for t in tasks:
        key = t.strip().lower()
        mapped = mapping.get(key, t)
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized


def main():
    print("Franka Kitchen Robot:")
    print("Type a command for the robot to perform (or 'quit'/'exit'):\n")

    user_input = input(">> ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting...")
        sys.exit(0)

    # Predict tasks from the command
    tasks = interpret_command(user_input)
    if not tasks:
        print("No known task detected. Exiting.")
        sys.exit(1)

    tasks = normalize_tasks(tasks)
    print(f"Predicted task(s) from sentence model: {tasks}\n")

    # Filter to only trained tasks and skip untrained ones
    executable_tasks = [t for t in tasks if t in TRAINED_TASKS]
    skipped_tasks = [t for t in tasks if t not in TRAINED_TASKS]

    if skipped_tasks:
        print(f"⚠️  Skipping untrained tasks: {skipped_tasks}")
    if not executable_tasks:
        print("No executable tasks with trained policy. Exiting.")
        sys.exit(1)

    print(f"✅ Executing trained tasks: {executable_tasks}\n")

    # Environment and hyperparameters
    env_name = "FrankaKitchen-v1"
    max_episode_steps = 1200

    # Initialize environment
    env = gym.make(
        env_name,
        max_episode_steps=max_episode_steps,
        tasks_to_complete=executable_tasks,
        render_mode="human"
    )
    env = RoboGymObservationWrapper(env)

    # Initialize MetaAgent with only the executable tasks
    meta_agent = MetaAgent(env, executable_tasks, max_episode_steps=max_episode_steps)
    meta_agent.initialize_agents()

    # Run test on executable tasks
    try:
        result = meta_agent.test()
        print(f"\n🎯 Execution result: {result}")
    except Exception as e:
        print(f"\n⚠️  Test run finished with an exception: {e}")

    env.close()


if __name__ == "__main__":
    main()