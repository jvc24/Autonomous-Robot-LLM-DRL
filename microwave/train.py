import time
import gymnasium as gym
import numpy as np
import logging

from gym_robotics_custom import RoboGymObservationWrapper
from buffer import ReplayBuffer
from model import *
from agent import *

from gymnasium.wrappers import RecordVideo


def setup_logger():
    """
    Configure a simple logger to display training progress and hyperparameters.
    """
    logging.basicConfig(
        format="%(levelname)-8s | %(message)s",
        level=logging.INFO
    )
    logging.info("========== Training Session Started ==========\n")


if __name__ == '__main__':

    # ----------------------------- #
    # Logger Setup                  #
    # ----------------------------- #
    setup_logger()

    # ----------------------------- #
    # Environment Setup             #
    # ----------------------------- #
    env_name = "FrankaKitchen-v1"
    max_episode_steps = 1200  # Maximum allowed steps per episode
    task = 'microwave'
    task_no_spaces = task.replace(" ", "_")  # for folder/video naming

    logging.info(f"Environment: {env_name}")
    logging.info(f"Task: {task}")
    logging.info(f"Max episode steps: {max_episode_steps}\n")

    # ----------------------------- #
    # SAC / Training Hyperparameters
    # ----------------------------- #
    gamma = 0.99
    tau = 0.005
    alpha = 0.1
    target_update_interval = 1
    updates_per_step = 4
    hidden_size = 512
    learning_rate = 0.0001
    batch_size = 64
    replay_buffer_size = 1_000_000

    logging.info("Hyperparameters:")
    logging.info(f"  γ = {gamma}, τ = {tau}, α = {alpha}")
    logging.info(f"  Hidden Size = {hidden_size}, LR = {learning_rate}")
    logging.info(f"  Replay Buffer Size = {replay_buffer_size}")
    logging.info(f"  Batch Size = {batch_size}\n")

    # ----------------------------- #
    # Environment Initialization    #
    # ----------------------------- #
    logging.info("Initializing environment and observation wrapper...")

    # Create Gym environment with RGB array rendering (needed for video recording)
    env = gym.make(
        env_name,
        max_episode_steps=max_episode_steps,
        tasks_to_complete=[task],
        render_mode="rgb_array"
    )

    # Record training videos every 100 episodes
    env = RecordVideo(
        env,
        video_folder=f"./logs/videos/{task_no_spaces}/train",
        episode_trigger=lambda episode_id: episode_id % 100 == 0,
        name_prefix=f"{task_no_spaces}_train"
    )

    # Apply observation wrapper for task-specific goal encoding
    env = RoboGymObservationWrapper(env, goal=task)

    # Get initial observation to determine input size
    observation, info = env.reset()
    observation_size = observation.shape[0]
    logging.info(f"Observation size: {observation_size}\n")

    # ----------------------------- #
    # Agent and Replay Buffer Setup #
    # ----------------------------- #
    logging.info("Initializing agent and replay buffer...")

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

    # Load existing model weights if available
    agent.load_checkpoint(evaluate=False)

    # Replay buffer for storing agent and expert transitions
    memory = ReplayBuffer(
        replay_buffer_size,
        input_size=observation_size,
        n_actions=env.action_space.shape[0],
        augment_rewards=True,
        augment_data=True
    )

    # Load expert demonstration data for imitation guidance
    logging.info("Loading expert data from file...")
    memory.load_from_csv(filename=f'checkpoints/human_memory_{task_no_spaces}.npz')
    logging.info("Expert data successfully loaded.\n")

    # Small delay to ensure environment is stable
    time.sleep(2)

    # ----------------------------- #
    # Training Phases               #
    # ----------------------------- #
    logging.info("========== Beginning Training Phases ==========\n")

    # ----------------------------- #
    # Phase 1: High expert data usage
    # ----------------------------- #
    memory.expert_data_ratio = 0.5  # 50% expert data per batch
    logging.info(f"[Phase 1] Expert Data Ratio: {memory.expert_data_ratio}")
    logging.info("Starting Phase 1 training...\n")
    agent.train(
        env=env,
        memory=memory,
        episodes=150,
        batch_size=batch_size,
        updates_per_step=updates_per_step,
        summary_writer_name=f"live_train_phase_1_{task_no_spaces}",
        max_episode_steps=max_episode_steps,
        phase=1
    )
    logging.info("Phase 1 training completed.\n")

    # ----------------------------- #
    # Phase 2: Reduced expert data
    # ----------------------------- #
    memory.expert_data_ratio = 0.25  # 25% expert data
    logging.info(f"[Phase 2] Expert Data Ratio: {memory.expert_data_ratio}")
    logging.info("Starting Phase 2 training...\n")
    agent.train(
        env=env,
        memory=memory,
        episodes=150,
        batch_size=batch_size,
        updates_per_step=updates_per_step,
        summary_writer_name=f"live_train_phase_2_{task_no_spaces}",
        max_episode_steps=max_episode_steps,
        phase=2
    )
    logging.info("Phase 2 training completed.\n")

    # ----------------------------- #
    # Phase 3: Fully autonomous RL
    # ----------------------------- #
    memory.expert_data_ratio = 0.0  # Only agent-generated transitions
    logging.info(f"[Phase 3] Expert Data Ratio: {memory.expert_data_ratio}")
    logging.info("Starting Phase 3 training...\n")
    agent.train(
        env=env,
        memory=memory,
        episodes=1000,
        batch_size=batch_size,
        updates_per_step=updates_per_step,
        summary_writer_name=f"live_train_phase_3_{task_no_spaces}",
        max_episode_steps=max_episode_steps,
        phase=3
    )
    logging.info("Phase 3 training completed.\n")

    logging.info("========== All Training Phases Completed Successfully ==========")