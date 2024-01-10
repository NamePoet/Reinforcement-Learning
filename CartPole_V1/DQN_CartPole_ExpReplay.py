import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt

# 定义深度 Q 网络模型
class DQNNetwork(nn.Module):
    def __init__(self, state_size, num_actions):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 定义经验回放缓冲区
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# 定义深度 Q 网络代理
class DQNAgent:
    def __init__(self, state_size, num_actions, replay_buffer_capacity=10000):
        self.num_actions = num_actions
        self.model = DQNNetwork(state_size, num_actions)
        self.target_model = DQNNetwork(state_size, num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q_values = self.model(torch.FloatTensor(state))
        return int(torch.argmax(q_values))

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        # non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        non_final_next_states = torch.stack([torch.from_numpy(s) for s in batch.next_state if s is not None])

        # state_batch = torch.stack(batch.state)
        state_batch = torch.stack([torch.from_numpy(s) for s in batch.state])

        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward)
        next_state_values = torch.zeros(batch_size)

        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1))

        loss = F.mse_loss(q_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# 定义训练函数
def train_cartpole():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    agent = DQNAgent(state_size, num_actions)

    # 设置渲染帧率
    env.metadata['video.frames_per_second'] = 165

    episodes = 1000
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        max_steps = 500

        for step in range(max_steps):
            # 选择动作并执行
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # 将经验存储到经验回放缓冲区
            agent.replay_buffer.push(state, action, next_state, reward, done)

            # 训练代理并更新状态
            agent.train(batch_size=32)  # 使用经验回放训练

            total_reward += reward
            state = next_state

            if done:
                agent.update_target_model()
                break

            # env.render()  # 使用env.render()展示CartPole状态

        total_rewards.append(total_reward)

        # 每50个episodes输出平均奖励
        if episode % 50 == 0:
            avg_reward = np.mean(total_rewards[-50:])
            print(f"Episode: {episode + 1}, Average Reward (Last 50 episodes): {avg_reward}")

    env.close()

    # 绘制奖励优化曲线
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('CartPole-v1 Training')
    plt.show()

# 开始训练
train_cartpole()
