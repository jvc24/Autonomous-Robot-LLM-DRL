import time
import gymnasium as gym
import numpy as np

from gym_robotics_custom import RoboGymObservationWrapper
from buffer import ReplayBuffer
from model import *
from agent import *


if __name__ == '__main__':
    """
    Main evaluation script for the Franka Kitchen microwave task.

    This script:
    - creates the Franka Kitchen environment,
    - wraps observations using a custom observation wrapper,
    - initializes the SAC agent and replay buffer,
    - loads trained checkpoints,
    - evaluates the learned policy on the microwave task.
    """

    # ----------------------------- #
    # Environment / Task Selection  #
    # ----------------------------- #
    env_name = "FrankaKitchen-v1"
    max_episode_steps = 500
    replay_buffer_size = 1000000

    # Select task to evaluate
    #task = 'hinge cabinet'
    task = 'microwave'
    #task = 'top burner'
    #task = 'bottom burner'

    # Replace spaces for cleaner checkpoint naming
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
    # Environment Initialization    #
    # ----------------------------- #
    # Create Franka Kitchen environment for the selected task
    env = gym.make(
        env_name,
        max_episode_steps=max_episode_steps,
        tasks_to_complete=[task],
        render_mode='human'
    )

    # Wrap environment to obtain task-specific observations
    env = RoboGymObservationWrapper(env, goal=task)

    # Reset environment and get initial observation
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
    # Buffer is required by the agent structure, even though here
    # the script is only used for evaluation
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

    # ----------------------------- #
    # Policy Evaluation             #
    # ----------------------------- #
    # Run the trained microwave policy for 3 evaluation episodes
    agent.test(env=env, episodes=3, max_episode_steps=max_episode_steps)

    # Close environment after evaluation
    env.close()