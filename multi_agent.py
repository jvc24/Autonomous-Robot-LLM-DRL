import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import *
import datetime
from buffer import ReplayBuffer
import time
from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent
import random
import imageio


class MetaAgent(object):
    """
    Meta-agent for multi-task and sequential task learning.
    Extends the base Agent to handle multiple goals and sequences of tasks per episode.
    Maintains separate agents and replay buffers for each goal.
    """
    
    def __init__(self, env, goal_list=['microwave','hinge cabinet'], replay_buffer_size=1000000, max_episode_steps=500):
        """
        Initialize the MetaAgent.

        Args:
            env: Gym environment.
            goal_list: List of task names to handle.
            replay_buffer_size: Max size of replay buffers.
            max_episode_steps: Maximum steps per episode.
        """
        self.agent_dict = {}  # Holds separate Agent objects for each goal
        self.mem_dict: dict[str, ReplayBuffer] = {}  # Holds ReplayBuffer objects for each goal
        goal_list_no_spaces = [a.replace(" ", "_") for a in goal_list]
        self.goal_dict: dict[str, str] = dict(zip(goal_list_no_spaces, goal_list))  # Mapping: sanitized -> original goal
        self.env = env
        self.agent: Agent = None
        self.replay_buffer_size = replay_buffer_size
        self.max_episode_steps = max_episode_steps

    def initialize_memory(self, augment_rewards=True, augment_data=True, augment_noise_ratio=0.1):
        """
        Initialize separate replay buffers for each goal.
        Supports reward augmentation, data augmentation, and noise augmentation.
        """
        for goal in self.goal_dict:
            self.env.set_goal(self.goal_dict[goal])
            observation, info = self.env.reset()
            observation_size = observation.shape[0]

            memory = ReplayBuffer(self.replay_buffer_size, input_size=observation_size,
                                  n_actions=self.env.action_space.shape[0], augment_rewards=augment_rewards,
                                  augment_data=augment_data, expert_data_ratio=0, augment_noise_ratio=augment_noise_ratio)
            
            self.mem_dict[goal] = memory  # Store buffer per goal
        
    def load_memory(self):
        """
        Load expert/demo data into each goal's replay buffer.
        Assumes pre-saved human demonstration data in CSV/NPZ format.
        """
        for buffer in self.mem_dict:
            self.mem_dict[buffer].load_from_csv(filename=f"checkpoints/human_memory_{buffer}.npz")

    def initialize_agents(self, gamma=0.99, tau=0.005, alpha=0.1,
                          target_update_interval=2, hidden_size=512,
                          learning_rate=0.0001):
        """
        Initialize separate Agent objects for each goal and load their checkpoints.
        Each agent is responsible for learning a specific task.
        """
        for goal in self.goal_dict:
            self.env.set_goal(self.goal_dict[goal])
            observation, info = self.env.reset()
            observation_size = observation.shape[0]

            agent = Agent(observation_size, self.env.action_space, gamma=gamma, tau=tau, alpha=alpha,
                          target_update_interval=target_update_interval, hidden_size=hidden_size,
                          learning_rate=learning_rate, goal=goal)
            
            print(f"Loading checkpoint for {goal}")
            agent.load_checkpoint(evaluate=True)  # Load pretrained weights if available
            self.agent_dict[goal] = agent

    def save_models(self):
        """
        Save all goal-specific agents' checkpoints.
        """
        for agent in self.agent_dict:
            self.agent_dict[agent].save_checkpoint()

    def test(self):
        """
        Sequentially test all tasks in the goal dictionary using their respective agents.
        Returns total reward and steps across all tasks.
        """
        action = None
        episode_reward = 0
        total_steps = 0

        for goal in self.goal_dict:
            print(f"Attempting goal {goal}...")

            self.env.set_goal(self.goal_dict[goal])
            self.agent = self.agent_dict[goal]

            action, reward, steps = self.agent.test(
                env=self.env,
                episodes=1,
                max_episode_steps=self.max_episode_steps,
                prev_action=action
            )

            episode_reward += reward
            total_steps += steps

        return episode_reward, total_steps

    def train(self, episodes, batch_size=64):
        """
        Train all tasks sequentially within each episode.
        Supports multi-task episodes: selects random subset of tasks per episode.
        Updates each agent using its corresponding replay buffer.
        """
        updates = 0

        for episode in range(episodes):
            print("\n" + "=" * 80)
            print(f"Starting Episode {episode + 1}/{episodes}")
            print("=" * 80)

            last_action = None
            action = None
            episode_reward = 0
            episode_steps = 0

            # Randomly select 1-3 tasks to execute sequentially
            num_samples = random.choice([1, 2, 3])
            selected_goals = random.sample(list(self.goal_dict.keys()), num_samples)
            print(f"Selected tasks for this episode: {selected_goals}")
            print("-" * 80)

            # Loop over each task in the selected sequence
            for goal in selected_goals:
                print(f"\n>>> Starting task: {goal}")

                done = False
                self.env.set_goal(self.goal_dict[goal])
                state, _ = self.env.reset()
                task_reward = 0
                task_steps = 0

                while not done and episode_steps < self.max_episode_steps:
                    # Use last_action if task completed in previous iteration
                    if last_action is not None:
                        action = last_action
                        last_action = None
                    else:
                        action = self.agent_dict[goal].select_action(state)

                    # Update agent if enough samples are available
                    if self.mem_dict[goal].can_sample(batch_size=batch_size):
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
                            self.agent_dict[goal].update_parameters(
                                self.mem_dict[goal],
                                batch_size,
                                updates
                            )
                        updates += 1

                        if updates % 50 == 0:
                            print(
                                f"[Update {updates}] "
                                f"C1 Loss: {critic_1_loss:.4f}, "
                                f"C2 Loss: {critic_2_loss:.4f}, "
                                f"Policy Loss: {policy_loss:.4f}, "
                                f"Entropy Loss: {ent_loss:.4f}, "
                                f"Alpha: {alpha:.4f}"
                            )

                    next_state, reward, done, _, _ = self.env.step(action)

                    # Mark task as completed if reward indicates success
                    if reward == 1:
                        done = True
                        last_action = action
                        print(f"Task '{goal}' completed successfully.")

                    episode_steps += 1
                    task_steps += 1
                    episode_reward += reward
                    task_reward += reward

                    # Store transition in the appropriate replay buffer
                    mask = 1 if episode_steps == self.max_episode_steps else float(not done)
                    self.mem_dict[goal].store_transition(state, action, reward, next_state, mask)
                    state = next_state

                print(
                    f"Task finished: {goal} | "
                    f"Task steps: {task_steps} | "
                    f"Task reward: {task_reward}"
                )
                print("-" * 80)

            # Average reward across tasks in this episode
            episode_reward = episode_reward / num_samples

            print(f"Episode {episode + 1} completed")
            print(f"Total episode steps: {episode_steps}")
            print(f"Average episode reward: {episode_reward:.4f}")
            print(f"Total parameter updates so far: {updates}")

            # Periodically save all agents
            if episode % 10 == 0:
                print("Saving models...")
                self.save_models()

        print("\n" + "=" * 80)
        print("Training completed.")
        print("=" * 80)