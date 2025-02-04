import numpy as np
import matplotlib.pyplot as plt

# 创建一个简单的网格
n_x = 10  # x 方向的点数
n_y = 10  # y 方向的点数
X, Y = np.meshgrid(np.arange(0, n_x), np.arange(0, n_y))  # 网格坐标

# 假设的速度场 u 和 v 分量
u = np.sin(X)  # 速度在 x 方向上的分量
v = np.cos(Y)  # 速度在 y 方向上的分量

# 绘制速度矢量场
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, u, v, scale=1, scale_units='xy', angles='xy', color='blue')

# 设置图形标题和标签
plt.title("Simple Velocity Vector Field")
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal', adjustable='box')  # 确保 x 和 y 轴比例一致

plt.show()
