import gym
from gym import spaces
from gym.utils import seeding
import mujoco_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


class CustomEnv(gym.Env):
    def __init__(self, start_point, end_point):
        """
        自定义环境类，用于加载自定义的Mujoco XML模型并在其中进行强化学习训练。

        参数:
            start_point (array_like): 起点的三维坐标，形如 [x, y, z]。
            end_point (array_like): 终点的三维坐标，形如 [x, y, z]。
        """
        self.model = mujoco_py.load_model_from_path("two_link_arm.xml")  # 加载Mujoco XML模型
        self.sim = mujoco_py.MjSim(self.model)  # 创建Mujoco仿真环境
        self.viewer = None

        high = np.array([np.inf] * self.sim.data.qpos.shape[0])
        self.action_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.array([np.inf] * self.sim.data.qpos.shape[0])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()  # 设定随机数种子
        self.viewer = None
        self.state = None
        self.start_point = start_point  # 起点
        self.end_point = end_point  # 终点

    def seed(self, seed=None):
        """
        设定随机数种子。

        参数:
            seed (int): 随机数种子。

        返回:
            list: 返回种子值。
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        与环境进行一步交互。

        参数:
            action (array_like): 代表动作的数组。

        返回:
            tuple: 返回观察、奖励、终止标志和其他信息。
        """
        self.sim.data.ctrl[:] = action
        self.sim.step()
        obs = self._get_obs()
        reward = 0  # 暂时设定为0
        done = False  # 暂时设定为False
        return obs, reward, done, {}

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

    def reset(self):
        """
        重置环境，初始化起点。

        返回:
            array_like: 返回初始观察。
        """
        qpos = np.array([0, 0])  # 设定初始位置，两个关节的角度
        qvel = np.zeros_like(qpos)  # 初始速度为零
        self.set_state(qpos, qvel)  # 假设你有一个set_state方法来设置模型的状态
        self.sim.forward()  # 更新模拟器的状态
        return self._get_obs()  # 返回观察结果

    def _get_obs(self):
        """
        获取当前观察。

        返回:
            array_like: 当前观察。
        """
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])

    def render(self, mode='human'):
        """
        渲染环境。

        参数:
            mode (str): 渲染模式。
        """
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()

    def close(self):
        """
        关闭渲染器。
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        """
        定义DQN网络。

        参数:
            input_size (int): 输入大小。
            output_size (int): 输出大小。
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (tensor): 输入张量。

        返回:
            tensor: 输出张量。
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done


# 定义DQN算法
class DQNAgent:
    def __init__(self, state_size, action_size, capacity=10000, batch_size=64, gamma=0.99, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity)
        self.criterion = nn.MSELoss()

    def select_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.uniform(-1, 1, size=self.action_size)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state).cpu().numpy()[0]

    def update_model(self):
        if len(self.memory.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.long().unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# 训练DQN
def train_dqn(env, num_episodes=1000, max_steps=1000):
    """
    训练DQN模型。

    参数:
        env (CustomEnv): 自定义环境对象。
        num_episodes (int): 训练的回合数。
        max_steps (int): 每个回合的最大步数。
    """
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0])

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            agent.update_model()
            if done:
                break
        if episode % 10 == 0:
            agent.update_target_network()
            print(f"Episode {episode}, Reward: {episode_reward}")


# 在这里定义起点和终点的坐标
start_point = [0, 0, 0]
end_point = [1, 1, 1]

# 创建自定义环境对象
env = CustomEnv(start_point, end_point)

# 运行训练函数
train_dqn(env)
