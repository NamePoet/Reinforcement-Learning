import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random
import gym
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Hyperparameters
gamma = 0.99  # discount factor
tau = 1e-3  # for soft update of target parameters
lr_actor = 1e-4  # learning rate of the actor
lr_critic = 1e-3  # learning rate of the critic
weight_decay = 0.0  # L2 weight decay
n_episodes = 200  # number of episodes
max_steps = 1000  # max steps per episode
batch_size = 64  # size of the batch
buffer_size = 100000  # size of the replay buffer
start_train = 1000  # number of experiences stored in the replay buffer before training begins


# DDPG Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, state, action):
        q = torch.cat([state, action], 1)
        q = F.relu(self.l1(q))
        q = F.relu(self.l2(q))
        return self.l3(q)


# Environment
env = gym.make('Hopper-v3')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = Actor(state_dim, action_dim, max_action).to(device)
actor_target = Actor(state_dim, action_dim, max_action).to(device)
actor_target.load_state_dict(actor.state_dict())

critic = Critic(state_dim, action_dim).to(device)
critic_target = Critic(state_dim, action_dim).to(device)
critic_target.load_state_dict(critic.state_dict())

# Optimizers
optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor, weight_decay=weight_decay)
optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

# Replay buffer
replay_buffer = deque(maxlen=buffer_size)

# Noise
noise = Normal(0, 0.1)

# Training loop
for episode in range(n_episodes):
    state = env.reset()
    episode_reward = 0
    for step in range(max_steps):
        # Select action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor(state_tensor).detach().cpu().numpy()[0]
        action += noise.sample(action.shape).numpy()
        action = np.clip(action, -max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)

        # Store experience
        replay_buffer.append((state, action, next_state, reward, done))

        # Learn
        if len(replay_buffer) > start_train:
            # Sample batch
            batch = random.sample(replay_buffer, batch_size)
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*batch)

            # Calculate Q values
            with torch.no_grad():
                next_state_array = np.array(next_state_batch)
                next_action = actor_target(torch.FloatTensor(next_state_array).to(device))
                next_q = critic_target(torch.FloatTensor(next_state_array).to(device), next_action)
                done_batch_tensor = torch.FloatTensor(done_batch).to(device)
                reward_batch_tensor = torch.FloatTensor(reward_batch).to(device)
                target_q = reward_batch_tensor + (1 - done_batch_tensor) * gamma * next_q.view(-1)

            # Update Critic
            current_q = critic(torch.FloatTensor(state_batch).to(device), torch.FloatTensor(action_batch).to(device))
            loss_critic = F.mse_loss(current_q, target_q.detach())
            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()

            # Update Actor
            actor_loss = -critic(torch.FloatTensor(state_batch).to(device),
                                 actor(torch.FloatTensor(state_batch).to(device))).mean()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            # Soft update target networks
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode} | Reward: {episode_reward}")

    # Test code with a specific range for starting and ending points
    start_range = [-10.0, -3.0]  # Example starting point range
    end_range = [3.0, 10.0]  # Example ending point range

    # Randomly sample starting and ending points within the specified ranges
    start_point_test = np.random.uniform(start_range[0], start_range[1])
    end_point_test = np.random.uniform(end_range[0], end_range[1])

    # Render environment only for every 10 episodes during testing
    if episode % 10 == 0:
        start_point_test = np.random.uniform(start_range[0], start_range[1])
        end_point_test = np.random.uniform(end_range[0], end_range[1])

        state = env.reset()
        done = False
        success = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor(state_tensor).detach().cpu().numpy()[0]
            next_state, _, done, success = env.step(action)

            # Manually set the starting and ending points for testing
            next_state[0] = start_point_test
            next_state[1] = end_point_test

            # Render environment only during testing
            if episode % 10 == 0:
                env.render()
        # Print success or failure message
        if success:
            print("Testing Successful! Agent reached the goal.")
        else:
            print("Testing Failed! Agent did not reach the goal.")

# Reset state and done for any subsequent code
# state = env.reset()
# done = False

env.close()

