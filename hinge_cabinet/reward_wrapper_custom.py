import gymnasium as gym
import numpy as np


class DenseRewardWrapper(gym.Wrapper):
    """
    Gym wrapper to provide dense rewards for the 'bottom burner' task.

    Dense rewards are based on the distance between achieved and desired goal positions,
    with an optional bonus when the goal is reached (distance < 0.05).
    """

    def __init__(self, env):
        super().__init__(env)
        self.goal = "bottom burner"  # Task key in the environment observation

    def reset(self, **kwargs):
        """
        Reset environment and return initial observation and info.
        """
        observation, info = self.env.reset(**kwargs)
        return observation, info

    def step(self, action):
        """
        Step environment with given action and compute dense reward.

        Args:
            action: Action to take in the environment

        Returns:
            observation: Next observation
            dense_reward: Shaped reward based on distance to goal
            done: Episode termination flag
            truncated: Episode truncation flag
            info: Extra environment info
        """
        observation, reward, done, truncated, info = self.env.step(action)

        # Extract achieved and desired goal positions for bottom burner
        achieved = observation["achieved_goal"][self.goal]
        desired = observation["desired_goal"][self.goal]

        # Compute distance to goal
        dist = np.linalg.norm(achieved - desired)

        # Dense reward: negative distance
        dense_reward = -dist

        # Bonus for being very close to the goal
        if dist < 0.05:
            dense_reward += 5.0

        return observation, dense_reward, done, truncated, info