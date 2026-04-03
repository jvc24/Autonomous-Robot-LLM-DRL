import time
import gymnasium as gym
import numpy as np
import logging
import random
from gym_robotics_custom import RoboGymObservationWrapper
from buffer import ReplayBuffer
from model import *
from agent import *
from multi_agent import MetaAgent


def setup_logger():
    """Configure a clean logger"""
    logging.basicConfig(
        format="%(levelname)-8s | %(message)s",
        level=logging.INFO
    )
    logging.info("========== Training Session Started ==========\n")


if __name__ == '__main__':

    setup_logger()

    # ------------------------- Environment Setup ------------------------- #
    env_name = "FrankaKitchen-v1"
    max_episode_steps = 1200  # Maybe reduce if required
    all_tasks = ["microwave", "hinge cabinet", "top burner"]

    logging.info(f"Environment: {env_name}")
    logging.info(f"Available tasks: {all_tasks}")
    logging.info(f"Max episode steps: {max_episode_steps}\n")

    # ------------------------- Hyperparameters ---------------------------- #
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

    # ------------------------- Environment Initialization ---------------- #
    logging.info("Initializing environment and observation wrapper...")
    env = gym.make(
        env_name,
        max_episode_steps=max_episode_steps,
        tasks_to_complete=all_tasks
    )
    env = RoboGymObservationWrapper(env, goal=all_tasks[1])

    observation, info = env.reset()
    observation_size = observation.shape[0]
    logging.info(f"Observation size: {observation_size}\n")

    # ------------------------- Agent and Memory --------------------------- #
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
        goal=None
    )

    # ✅ Load existing model weights if available
    agent.load_checkpoint(evaluate=False)

    memory = ReplayBuffer(
        replay_buffer_size,
        input_size=observation_size,
        n_actions=env.action_space.shape[0],
        augment_rewards=True,
        augment_data=True
    )

    time.sleep(2)

    # ------------------------- Multi-Task Sequential Training ---------------- #
    logging.info("========== Beginning Sequential Multi-Task Training ==========\n")

    num_episodes = 1000
    memory.expert_data_ratio = 0

    all_tasks_no_spaces = [task.replace(" ", "_") for task in all_tasks]


    for episode in range(1, num_episodes + 1):
        #  Randomly choose how many tasks to train this episode (1, 2, or 3)
        num_tasks = random.randint(1, 3)
        # Randomly choose that no of tasks
        selected_tasks = random.sample(all_tasks, num_tasks)

        logging.info(f"Episode {episode}/{num_episodes} | Selected tasks: {selected_tasks}")

        # Sequentially train on each selected task
        for task in selected_tasks:
            logging.info(f"--- Training on task: {task} ---")

            # Reconfigure environment for this specific task
            env.close()  # ensure clean reset
            env = gym.make(
                env_name,
                max_episode_steps=max_episode_steps,
                tasks_to_complete=[task]
            )
            env = RoboGymObservationWrapper(env)

            observation, info = env.reset()

            # ✅ Train agent using same replay buffer (shared knowledge)
            MetaAgent.train(
                env=env,
                memory=memory,
                episodes=1,  # one episode per task per outer episode
                batch_size=batch_size,
                updates_per_step=updates_per_step,
                summary_writer_name=f"live_train_phase_1_multitask",
                max_episode_steps=max_episode_steps,
                phase=1
            )

            # ✅ Save checkpoint after each task (ensures updated robustness)
            agent.save_checkpoint()

        logging.info(f"✅ Completed Episode {episode} | Tasks trained: {selected_tasks}\n")

    logging.info("========== Multi-Task Sequential Training Completed ==========\n")
