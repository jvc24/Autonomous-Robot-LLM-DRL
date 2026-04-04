import os
import time
import imageio
import gymnasium as gym
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from buffer import ReplayBuffer
from model import *
from agent import *
from multi_agent import *
from gymnasium.wrappers import RecordVideo

if __name__ == '__main__':
    # ------------------------- Settings ------------------------- #
    env_name = "FrankaKitchen-v1"
    max_episode_steps = 1200
    replay_buffer_size = 1_000_000
    tasks = ['top burner','microwave', 'hinge cabinet']

    # RL hyperparameters (not used in test, but needed for agent init)
    gamma = 0.99
    tau = 0.005
    alpha = 0.1
    target_update_interval = 1
    updates_per_step = 4
    hidden_size = 512
    learning_rate = 0.0001
    batch_size = 64

    # Test settings
    live_test = True
    SAVE_FRAMES = True
    FRAME_FREQ = 2  # save frame every N steps
    VIDEO_FOLDER = "./videos"
    FRAMES_FOLDER = "./frames"
    TEST_EPISODES = 1

    tasks_no_spaces = "_".join([task.replace(" ", "_") for task in tasks])

    # ------------------------- Live Test with Video + Frames ------------------------- #
    if live_test:
        # Create environment
        env = gym.make(
            env_name,
            max_episode_steps=max_episode_steps,
            tasks_to_complete=tasks,
            render_mode="rgb_array"
        )

        # Record one video for all tasks
        env = RecordVideo(
            env,
            video_folder=f"{VIDEO_FOLDER}/{tasks_no_spaces}",
            episode_trigger=lambda episode_id: True,
            name_prefix=f"{tasks_no_spaces}"
        )

        # Wrap environment for observations
        env = RoboGymObservationWrapper(env)

        # Initialize multi-task agent
        meta_agent = MetaAgent(env, tasks, max_episode_steps=max_episode_steps)
        meta_agent.initialize_agents()

        for episode in range(TEST_EPISODES):
            print(f"\n[Live Test] Episode {episode + 1}/{TEST_EPISODES}")

            # Create folder for saving frames
            if SAVE_FRAMES:
                episode_frames_dir = f"{FRAMES_FOLDER}/{tasks_no_spaces}/episode_{episode}"
                os.makedirs(episode_frames_dir, exist_ok=True)

            # Reset environment
            state, info = env.reset()
            done = False
            step_count = 0

            # Run multi-task test (record frames inside test)
            score, num_steps = meta_agent.test()

            # Save final frame
            if SAVE_FRAMES:
                try:
                    frame = env.render()
                    if frame is not None:
                        imageio.imwrite(
                            os.path.join(episode_frames_dir, "frame_final.png"),
                            frame
                        )
                except Exception as e:
                    print(f"Final frame save skipped: {e}")

            print(f"Episode {episode} completed | Score: {score} | Steps: {num_steps}")

        env.close()