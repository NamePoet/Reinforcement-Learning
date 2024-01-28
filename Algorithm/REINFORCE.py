# 定义REINFORCE算法
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        x = self.fc(state)
        action_probs = self.softmax(x)
        return action_probs


class REINFORCE:
    def __init__(self, input_size, output_size, learning_rate):
        self.policy_network = PolicyNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.policy_network(state)
        action = np.random.choice(range(len(action_probs[0])), p=action_probs.detach().numpy()[0])
        return action

    def update_policy(self, log_probs, rewards):
        discounted_rewards = self.calculate_discounted_rewards(rewards)
        policy_loss = torch.mean(
            torch.stack([-log_prob * reward for log_prob, reward in zip(log_probs, discounted_rewards)]))

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def calculate_discounted_rewards(self, rewards, gamma=0.99):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards