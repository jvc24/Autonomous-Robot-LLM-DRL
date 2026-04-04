import sys
import time
import gymnasium as gym
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from buffer import ReplayBuffer
from model import *
from agent import *
from multi_agent import *
from infer_llama1b_final import infer

def normalize_tasks(tasks):
    """
    Normalize task names and apply synonyms mapping.
    Removes duplicates while preserving order.
    """
    if not tasks:
        return []

    mapping = {
        "hinge_cabinet": "hinge cabinet",
        "top_burner": "top burner",
        "microwave": "microwave",
        "sliding_door": "sliding_cabinet" #Dumb way to change it again find diff sol
    }

    normalized = []
    for t in tasks:
        key = t.strip().lower()
        mapped = mapping.get(key, t)  
        # preserve order and avoid duplicates
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized


def main():
    print("🤖 Franka Kitchen Robot")

    print("Type a command for the robot to perform (or 'quit'/'exit'):\n")

    user_input = input(">> ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting...")
        sys.exit(0)

    tasks, _ = infer(user_input)
    tasks = normalize_tasks(tasks)
    if not tasks:
        print("No known task detected. Exiting.")
        sys.exit(1)

    # Normalize & deduplicate task names (e.g. "burner" -> "top burner")
    tasks = normalize_tasks(tasks)
    print("🤖✨ LLaMA Predicted Task(s):")
    print(f"➡️  {tasks}\n")

    # Environment and training/test hyperparameters same for everything 
    env_name = "FrankaKitchen-v1"
    max_episode_steps = 1200
    replay_buffer_size = 1000000
    gamma = 0.99
    tau = 0.005
    alpha = 0.1
    target_update_interval = 1
    updates_per_step = 4
    hidden_size = 512
    learning_rate = 0.0001
    batch_size = 64
    live_test = True
    generate_score = True

    try:
        if live_test:
            # run a human-rendered live test
            env = gym.make(
                env_name,
                max_episode_steps=max_episode_steps,
                tasks_to_complete=tasks,
                render_mode="human",
            )
            env = RoboGymObservationWrapper(env)

            meta_agent = MetaAgent(env, tasks, max_episode_steps=max_episode_steps)
            meta_agent.initialize_agents()

            # MetaAgent.test() may return a score or just run the test; handle both
            try:
                result = meta_agent.test()
                print(f"Live test result: {result}")
            except Exception as e:
                print(f"Live test finished (MetaAgent.test() raised): {e}")

            env.close()

        if generate_score:
            print("Generating performance score...")
            env = gym.make(
                env_name, max_episode_steps=max_episode_steps, tasks_to_complete=tasks
            )
            env = RoboGymObservationWrapper(env)

            observation, info = env.reset()
            observation_size = observation.shape[0] if hasattr(observation, "shape") else None

            meta_agent = MetaAgent(env, tasks, max_episode_steps=max_episode_steps)
            meta_agent.initialize_agents()

            perf_score_epochs = 10
            total_score = 0.0
            total_time_steps = 0 

            for epoch in range(perf_score_epochs):
                score, num_eps, = meta_agent.test()
                # if meta_agent.test returns None, treat as 0
                score = 0.0 if score is None else float(score)
                total_score += score
                total_time_steps += num_eps
                print(f"Epoch {epoch+1}/{perf_score_epochs} score: {score}")

            # success_ratio: average successes per task normalized by number of tasks
            if len(tasks) > 0:
                success_ratio = ((total_score / len(tasks)) / perf_score_epochs) * 100
                avg_time_steps = total_time_steps / perf_score_epochs
            else:
                success_ratio = 0.0

            

            print("\n" + "="*60)
            print("📊  Performance Evaluation Summary")
            print("="*60)
            print(f"✅ Success Ratio     : {success_ratio:.2f}%")
            print(f"⏱️  Avg. Time Steps   : {avg_time_steps:.2f}")
            print("-"*60)
            print(f"🧠  Tested over {perf_score_epochs} epochs on {len(tasks)} tasks.")
            print("="*60 + "\n")

            env.close()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully.")
        try:
            env.close()
        except Exception:
            pass
        sys.exit(0)
    except Exception as e:
        # Catch-all to help debug runtime errors
        print(f"An error occurred: {e}")
        try:
            env.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
