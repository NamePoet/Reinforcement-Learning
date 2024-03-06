import numpy as np
import mujoco_py
import os

# 加载模型
model_path = './double_pendulum.xml'  # 请将此路径替换为你的XML文件的实际路径
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)

# 设置初始状态
sim.data.qpos[0] = 1.0  # joint1的角度
sim.data.qpos[1] = -0.5  # joint2的角度
sim.data.qvel[:] = 0  # 所有关节的速度都设置为0

# 运行仿真
for i in range(1000):
    # 在此处添加控制逻辑，例如设置电机的目标位置或力矩等。
    # sim.data.ctrl[:] = ...  # 控制信号数组的大小应与你在XML中定义的执行器数量相匹配。

    # 执行一步仿真
    sim.step()

    # 打印当前状态（可选）
    print(sim.data.qpos)  # 打印当前关节位置
    print(sim.data.qvel)  # 打印当前关节速度