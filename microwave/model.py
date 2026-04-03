import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os

# Constants for log standard deviation clamping
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6  # Small value for numerical stability


def weights_init_(m):
    """
    Initialize weights of Linear layers using Xavier uniform initialization
    and biases to zero. Applied to all layers of the network.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# ============================= #
# Policy Network for SAC         #
# ============================= #
class Policy(nn.Module):
    """
    Stochastic policy network for SAC.
    Outputs mean and log_std for Gaussian sampling, supports action rescaling.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, checkpoint_dir='checkpoints', name='policy_network'):
        super(Policy, self).__init__()

        # Hidden layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        # Checkpointing info
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # Initialize weights
        self.apply(weights_init_)

        # Action rescaling to environment bounds
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2
            )

    def forward(self, state):
        """
        Forward pass to compute mean and log_std for a given state.
        Log_std is clamped to ensure numerical stability.
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    
    def sample(self, state):
        """
        Sample an action using the reparameterization trick:
        - Computes Gaussian with mean and std from forward()
        - Applies Tanh to constrain action
        - Computes log probability with correction for Tanh squashing
        Returns:
            action: sampled action (scaled to env)
            log_prob: log probability of the action
            mean: deterministic action (for evaluation)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def to(self, device):
        """
        Move network and action rescaling tensors to the specified device (CPU/GPU)
        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Policy, self).to(device)
    
    def save_checkpoint(self):
        """Save model parameters to checkpoint file"""
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        """Load model parameters from checkpoint file"""
        self.load_state_dict(torch.load(self.checkpoint_file))


# ============================= #
# Critic Network for SAC         #
# ============================= #
class Critic(nn.Module):
    """
    Twin Q-network (Double Q-learning) for SAC.
    Returns two Q-values for a given state-action pair.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, checkpoint_dir='checkpoints', name='q_network'):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.output1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        self.output2 = nn.Linear(hidden_dim, 1)

        # Checkpointing info
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # Initialize weights
        self.apply(weights_init_)

    def forward(self, state, action):
        """
        Forward pass for both Q-networks.
        Inputs:
            state: environment state
            action: action taken by policy
        Returns:
            x1: Q1 value
            x2: Q2 value
        """
        xu = torch.cat([state, action], 1)  # Concatenate state and action

        # Q1 forward pass
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        # x1 = F.relu(self.linear3(x1))  # optional 3rd layer
        x1 = self.output1(x1)

        # Q2 forward pass
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        # x2 = F.relu(self.linear6(x2))  # optional 3rd layer
        x2 = self.output2(x2)

        return x1, x2

    def save_checkpoint(self):
        """Save critic parameters to checkpoint file"""
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Load critic parameters from checkpoint file"""
        self.load_state_dict(torch.load(self.checkpoint_file))