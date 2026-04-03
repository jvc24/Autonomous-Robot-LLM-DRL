import os
import time
import datetime

import gymnasium as gym
import numpy as np
import imageio

from gymnasium.wrappers import RecordVideo
from gym_robotics_custom import RoboGymObservationWrapper
from buffer import ReplayBuffer
from model import *
from agent import *


if __name__ == '__main__':
    """
    Main evaluation and recording script for the Franka Kitchen hinge cabinet task.
    This script:
    - loads a trained SAC policy,
    - evaluates it on the hinge cabinet manipulation task,
    - records evaluation videos,
    - optionally saves rendered frames as image files.

    A timestamped run directory is created at each execution so that
    previously recorded videos and frames are preserved.
    """

    # ----------------------------- #
    # Environment / Task Settings   #
    # ----------------------------- #
    env_name = "FrankaKitchen-v1"
    max_episode_steps = 500
    replay_buffer_size = 1000000

    # Select task to evaluate
    task = 'hinge cabinet'
    #task = 'microwave'
    #task = 'top burner'
    #task = 'bottom burner'

    # Replace spaces for cleaner file/folder names
    task_no_spaces = task.replace(" ", "_")

    # ----------------------------- #
    # SAC Hyperparameters           #
    # ----------------------------- #
    gamma = 0.99
    tau = 0.005
    alpha = 0.1
    target_update_interval = 1
    updates_per_step = 4
    hidden_size = 512
    learning_rate = 0.0001
    batch_size = 64

    # ----------------------------- #
    # Recording / Saving Settings   #
    # ----------------------------- #
    SAVE_FRAMES = True
    FRAME_FREQ = 2   # Save one frame every N environment steps
    episodes = 4

    # Create a unique timestamp so files are never overwritten
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Root folders for this evaluation run
    video_output_dir = f"./videos/{task_no_spaces}/{run_timestamp}"
    frames_output_dir = f"./frames/{task_no_spaces}/{run_timestamp}"

    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(frames_output_dir, exist_ok=True)

    # ----------------------------- #
    # Environment Initialization    #
    # ----------------------------- #
    # Use rgb_array rendering so frames and videos can be saved
    env = gym.make(
        env_name,
        max_episode_steps=max_episode_steps,
        tasks_to_complete=[task],
        render_mode="rgb_array"
    )

    # Record every episode to a unique video folder
    env = RecordVideo(
        env,
        video_folder=video_output_dir,
        episode_trigger=lambda episode_id: True,
        name_prefix=f"{task_no_spaces}_{run_timestamp}"
    )

    # Wrap environment to obtain task-specific observations
    env = RoboGymObservationWrapper(env, goal=task)

    # Reset environment and determine observation dimension
    observation, info = env.reset()
    observation_size = observation.shape[0]

    # ----------------------------- #
    # Agent Initialization          #
    # ----------------------------- #
    agent = Agent(
        observation_size,
        env.action_space,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        target_update_interval=target_update_interval,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        goal=task_no_spaces
    )

    # ----------------------------- #
    # Replay Buffer Initialization  #
    # ----------------------------- #
    # Buffer is retained for compatibility with the project structure,
    # although it is not actively used during pure evaluation
    memory = ReplayBuffer(
        replay_buffer_size,
        input_size=observation_size,
        n_actions=env.action_space.shape[0],
        augment_rewards=True,
        augment_data=True
    )

    # ----------------------------- #
    # Load Trained Policy           #
    # ----------------------------- #
    agent.load_checkpoint(evaluate=True)

    print(f"Starting evaluation for task: {task}")
    print(f"Videos will be saved in: {video_output_dir}")
    print(f"Frames will be saved in: {frames_output_dir}")

    # ----------------------------- #
    # Evaluation Loop               #
    # ----------------------------- #
    for episode in range(episodes):

        state, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        # Create a dedicated folder for frames of this episode
        if SAVE_FRAMES:
            episode_frames_dir = f"{frames_output_dir}/episode_{episode}"
            os.makedirs(episode_frames_dir, exist_ok=True)

        while not done and step_count < max_episode_steps:

            # Select deterministic action from trained policy
            action = agent.select_action(state, evaluate=True)

            # Execute action in environment
            next_state, reward, done, _, _ = env.step(action)

            episode_reward += reward
            state = next_state

            # -------- Frame Saving -------- #
            # Save frames at a fixed frequency for later inspection
            if SAVE_FRAMES and step_count % FRAME_FREQ == 0:
                frame = env.render()
                imageio.imwrite(
                    f"{episode_frames_dir}/frame_{step_count:04d}.png",
                    frame
                )

            step_count += 1

        print(f"Episode {episode} | Steps: {step_count} | Reward: {episode_reward}")

    # ----------------------------- #
    # Cleanup                       #
    # ----------------------------- #
    env.close()