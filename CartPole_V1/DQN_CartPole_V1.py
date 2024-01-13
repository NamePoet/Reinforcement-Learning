import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import time
import matplotlib.pyplot as plt


total_rewards = []

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


# 定义深度 Q 网络代理
class DQNAgent:
    def __init__(self, state_size, num_actions):
        self.num_actions = num_actions
        self.model = DQNNetwork(state_size, num_actions)
        self.target_model = DQNNetwork(state_size, num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q_values = self.model(torch.FloatTensor(state))
        return int(torch.argmax(q_values))

    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        q_values = self.model(state)
        next_q_values = self.target_model(next_state)

        target = q_values.clone()
        target[action] = reward if done else reward + self.gamma * torch.max(next_q_values)

        loss = F.mse_loss(q_values, target)
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

    episodes = 500
    # total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        max_steps = 500    # 500

        for step in range(max_steps):
            # 选择动作并执行
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # 训练代理并更新状态
            agent.train(state, action, reward, next_state, done)
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
    return agent, state_size, num_actions   # 返回训练后的agent以及state_size和num_actions


# 开始训练
# train_cartpole()


def test_cartpole(agent, state_size, num_actions, max_steps=500):
    env = gym.make('CartPole-v1')

    # 设置渲染帧率
    env.metadata['video.frames_per_second'] = 165

    episodes = 800 # 进行10个测试
    durations = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # 选择动作并执行
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            state = next_state

            if done:
                break

            # env.render()  # 使用env.render()展示CartPole状态

        durations.append(step + 1)  # 记录每个episode的持续时间

    env.close()

    return durations


# 在训练后调用此函数进行测试
trained_agent, state_size, num_actions = train_cartpole()
# 绘制奖励优化曲线
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('CartPole-v1 Training')
plt.show()
durations = test_cartpole(trained_agent, state_size, num_actions)
# 打印测试结果
avg_duration = np.mean(durations)
print(f"Average Duration over 800 episodes: {avg_duration} steps")

