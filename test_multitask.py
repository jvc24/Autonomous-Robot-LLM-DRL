import time
import gymnasium as gym
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from buffer import ReplayBuffer
from model import *
from agent import *
from multi_agent import *

if __name__ == '__main__':

    env_name="FrankaKitchen-v1"
    max_episode_steps=1200
    replay_buffer_size = 1000000
    tasks = ['top burner','hinge cabinet']
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

    if live_test:
        env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=tasks, render_mode='human')
        
        env = RoboGymObservationWrapper(env)

        meta_agent = MetaAgent(env, tasks, max_episode_steps=max_episode_steps)

        meta_agent.initialize_agents()

        meta_agent.test()

        env.close()

    if generate_score:
        print(f"Generating performance score")
        env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=tasks)
        env = RoboGymObservationWrapper(env)
        env.set_goal(tasks[0])   # or any valid initial task
        observation, info = env.reset()

        observation_size = observation.shape[0]

        meta_agent = MetaAgent(env, tasks, max_episode_steps=max_episode_steps)

        meta_agent.initialize_agents()

        perf_score_epochs = 10

        total_score = 0
        total_time_steps = 0 

        """
        for i in range(perf_score_epochs):
            score, num_steps = meta_agent.test()
            total_score += score
            total_time_steps +=  num_steps
        
        success_ratio = ((total_score / len(tasks)) / perf_score_epochs) * 100
        avg_time_steps = total_time_steps / perf_score_epochs
        print(f"Success ratio {success_ratio:.2f}%")
        
        """

        for i in range(perf_score_epochs):
            score, num_steps = meta_agent.test()
            total_score += score
            total_time_steps += num_steps

        success_ratio = ((total_score / len(tasks)) / perf_score_epochs) * 100
        avg_time_steps = total_time_steps / perf_score_epochs
        avg_success_time_steps = total_time_steps / total_score

        print("\n" + "="*60)
        print("📊  Performance Evaluation Summary")
        print("="*60)
        print(f"✅ Success Ratio     : {success_ratio:.2f}%")
        print(f"⏱️  Avg time Steps  : {avg_time_steps:.2f}")
        print(f"⏱️  Avg time for successful episodes   : {avg_success_time_steps:.2f}")
        print("-"*60)
        print(f"🧠  Tested over {perf_score_epochs} epochs on {len(tasks)} tasks.")
        print("="*60 + "\n")


    