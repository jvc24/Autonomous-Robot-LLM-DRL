import time
import gymnasium as gym
import numpy as np
import itertools
import random

from gym_robotics_custom import RoboGymObservationWrapper
from buffer import ReplayBuffer
from model import *
from agent import *
from multi_agent import *


if __name__ == '__main__':

    env_name = "FrankaKitchen-v1"
    max_episode_steps = 1200
    tasks = ['top burner', 'hinge cabinet', 'microwave']

    perf_score_epochs = 100  # take one record with a very large number

    print(f"Generating performance score with task combinations")

    # Generate all possible random combinations 
    task_combinations = []
    for r in [1, 2, 3]:
        task_combinations.extend(list(itertools.combinations(tasks, r)))

    # dict to group the taskss
    metrics = {
        1: {"score": 0, "steps": 0, "episodes": 0},
        2: {"score": 0, "steps": 0, "episodes": 0},
        3: {"score": 0, "steps": 0, "episodes": 0},
    }

    for i in range(perf_score_epochs):

        # Sample a random combination
        current_tasks = list(random.choice(task_combinations))
        num_tasks = len(current_tasks)

        print(f"\nEpisode {i+1}: Tasks -> {current_tasks}")

        # Create env with selected tasks
        env = gym.make(
            env_name,
            max_episode_steps=max_episode_steps,
            tasks_to_complete=current_tasks
        )
        env = RoboGymObservationWrapper(env)

        env.set_goal(current_tasks[0])  # initial goal
        observation, info = env.reset()

        
        meta_agent = MetaAgent(env, current_tasks, max_episode_steps=max_episode_steps)
        meta_agent.initialize_agents()

        
        score, num_steps = meta_agent.test()

        
        metrics[num_tasks]["score"] += score
        metrics[num_tasks]["steps"] += num_steps
        metrics[num_tasks]["episodes"] += 1

        env.close()

    # 📊 Final Summary
    print("\n" + "="*60)
    print("📊 Multitask Performance Evaluation ")
    print("="*60)

    for k in [1, 2, 3]:
        if metrics[k]["episodes"] == 0:
            continue

        total_score = metrics[k]["score"]
        total_steps = metrics[k]["steps"]
        episodes = metrics[k]["episodes"]

        success_ratio = (total_score / (episodes * k)) * 100
        avg_steps = total_steps / episodes
        avg_success_steps = total_steps / total_score if total_score > 0 else float('inf')

        print(f"\n🔹 {k}-Task Episodes")
        print("-"*40)
        print(f"Episodes             : {episodes}")
        print(f"✅ Success Ratio     : {success_ratio:.2f}%")
        print(f"⏱️ Avg Time Steps    : {avg_steps:.2f}")
        print(f"⏱️ Success Time Steps: {avg_success_steps:.2f}")

    print("\n" + "="*60 + "\n")