import numpy as np


class ReplayBuffer:
    """
    Replay buffer that supports expert data mixing and optional data augmentation.

    Attributes:
        expert_data_ratio: ratio of expert (human) transitions used during sampling
        augment_data: flag to add Gaussian noise to states/actions
        augment_rewards: flag to scale rewards
    """

    def __init__(
        self,
        max_size,
        input_size,
        n_actions,
        augment_data=False,
        augment_rewards=False,
        expert_data_ratio=0.1,
        augment_noise_ratio=0.1
    ):
        # ------------------------- Buffer Setup ------------------------- #
        self.mem_size = max_size
        self.mem_ctr = 0

        # Replay memory arrays
        self.state_memory = np.zeros((self.mem_size, input_size))
        self.next_state_memory = np.zeros((self.mem_size, input_size))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        # Data augmentation and expert data settings
        self.augment_data = augment_data
        self.augment_rewards = augment_rewards
        self.augment_noise_ratio = augment_noise_ratio
        self.expert_data_ratio = expert_data_ratio
        self.expert_data_cutoff = 0

    # ------------------------------------------------------------------ #
    def __len__(self):
        """Return current number of stored transitions."""
        return self.mem_ctr

    def can_sample(self, batch_size):
        """
        Check if buffer has enough samples for training.
        Requires at least 1000×batch_size transitions.
        """
        return self.mem_ctr > (batch_size * 1000)

    # ------------------------------------------------------------------ #
    def store_transition(self, state, action, reward, next_state, done):
        """Store one transition in the replay buffer."""
        index = self.mem_ctr % self.mem_size

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    # ------------------------------------------------------------------ #
    def sample_buffer(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        The batch may contain both:
          - expert transitions (controlled by expert_data_ratio)
          - agent-generated transitions
        Optional Gaussian noise is added if augment_data is True.
        Rewards can be scaled if augment_rewards is True.
        """
        max_mem = min(self.mem_ctr, self.mem_size)

        # Mix expert and agent data
        if self.expert_data_ratio > 0:
            expert_data_quantity = int(batch_size * self.expert_data_ratio)
            random_batch = np.random.choice(max_mem, batch_size - expert_data_quantity)
            expert_batch = np.random.choice(self.expert_data_cutoff, expert_data_quantity)
            batch = np.concatenate((random_batch, expert_batch))
        else:
            batch = np.random.choice(max_mem, batch_size)

        # Retrieve batch samples
        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        # Optional data augmentation
        if self.augment_data:
            state_noise_std = self.augment_noise_ratio * np.mean(np.abs(states))
            action_noise_std = self.augment_noise_ratio * np.mean(np.abs(actions))
            states += np.random.normal(0, state_noise_std, states.shape)
            actions += np.random.normal(0, action_noise_std, actions.shape)

        # Optional reward scaling
        if self.augment_rewards:
            rewards = rewards * 100

        return states, actions, rewards, next_states, dones

    # ------------------------------------------------------------------ #
    def save_to_csv(self, filename):
        """Save buffer contents to a .npz file for reuse."""
        np.savez(
            filename,
            state=self.state_memory[:self.mem_ctr],
            action=self.action_memory[:self.mem_ctr],
            reward=self.reward_memory[:self.mem_ctr],
            next_state=self.next_state_memory[:self.mem_ctr],
            done=self.terminal_memory[:self.mem_ctr]
        )
        print(f"Saved replay buffer to {filename}")

    # ------------------------------------------------------------------ #
    def load_from_csv(self, filename, expert_data=True):
        """Load expert or replay data from a .npz file."""
        try:
            data = np.load(filename)
            self.mem_ctr = len(data['state'])

            self.state_memory[:self.mem_ctr] = data['state']
            self.action_memory[:self.mem_ctr] = data['action']
            self.reward_memory[:self.mem_ctr] = data['reward']
            self.next_state_memory[:self.mem_ctr] = data['next_state']
            self.terminal_memory[:self.mem_ctr] = data['done']

            print(f"Successfully loaded {filename} into memory")
            print(f"{self.mem_ctr} transitions loaded")

            if expert_data:
                self.expert_data_cutoff = self.mem_ctr

        except Exception as e:
            print(f"Unable to load memory from {filename}: {e}")

