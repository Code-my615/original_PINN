import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import torch
import sys
from dataset import Dataset
from pinn_model import *
from parameters import get_params


parser = argparse.ArgumentParser(description='take params')
parser.add_argument('--experiment', type=str, default='NS', help='steady-state')
parser.add_argument('--problem', type=str, default='forward')
parser.add_argument('--col_points', type=int, default=2000)
parser.add_argument('--bc_points', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--x_num', type=int, default=256, help='Number of x-coordinates for ground truth grid')  # 测试x轴取的点数
parser.add_argument('--y_num', type=int, default=100, help='Number of y-coordinates for ground truth grid')  # 测试y轴取的点数
parser.add_argument('--range_x', type=float, nargs=2, default=[0, 2 * np.pi])
parser.add_argument('--range_y', type=float, nargs=2, default=[0, 1])
parser.add_argument('--lambda_ic', type=int, default=1.0)
parser.add_argument('--lambda_bc', type=int, default=1.0)
parser.add_argument('--lambda_pde', type=int, default=1.0)
parser.add_argument('--seed', type=int, default=12)
parser.add_argument('--L', type=float, default=0.2)  #管道长度
parser.add_argument("--R", type=float, default=0.1)  #管道宽度半径，暂时设定管道宽度恒定
parser.add_argument("--rho", type=float, default=1e3) #流体密度（假设为水）
parser.add_argument("--nu", type=float, default=1.85e-6) #运动粘度
parser.add_argument("--u_max", type=float, default=0.00925) # 管道入口最大水平速度
parser.add_argument('--layer_sizes', type=int, nargs='+', default=[2, 100, 100, 100, 100, 100, 3])
parser.add_argument('--u0_str', type=str, default='sin(x)')  # 对流方程中的初始条件

args = parser.parse_args()
# 在解析命令行参数之后，再计算 lambda_1 和 lambda_2
args.lambda_1 = args.rho
args.lambda_2 = args.rho * args.nu
# CUDA support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 创建目录
save_dir = f"Seed_{args.seed}_{args.experiment}"
# 替换路径中的括号
u0_str = args.u0_str
safe_u0_str = u0_str.replace('(', '_').replace(')', '_').replace('*', '-')
# 创建子目录
if args.problem == 'forward':
    sub_dir = f"{args.problem}/lbd1={args.lambda_1}_lbd2={args.lambda_2}/col_points_{args.col_points},bc_points_{args.bc_points}/lr={args.lr}/epochs_{args.epochs}"
elif args.problem == 'inverse':
    sub_dir = f"{args.problem}/lbd1={args.lambda_1}_lbd2={args.lambda_2}/col_points_{args.col_points}_usol_num_{args.usol_num}/epochs_{args.epochs}"
else:
    print("输入有误！")
    sys.exit()

# sub_sub_dir = sub_dir + '/predict_picture'
base_dir = os.path.join(save_dir, sub_dir)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Generate training dataset
dataset = Dataset(args.L, args.R, args.u_max, args.col_points, args.bc_points, args.problem, args.seed)

# # Processing the test data
x_star = np.linspace(0, args.L, args.x_num)
y_star = np.linspace(-args.R, args.R, args.y_num)
XX, YY = np.meshgrid(x_star, y_star)  # all the X grid points T times, all the T grid points X times
X_test = np.column_stack((XX.reshape(-1, 1), YY.reshape(-1, 1)))
n_x = x_star.shape[0]
n_y = y_star.shape[0]


params = get_params(dataset, args, x_star, y_star, device)
if args.problem == 'forward':
    model = NS_forward(params)
else:
    model = Convection_inverse(params)

model.train(args.epochs)
# model.plot_losses(args.epochs, base_dir)
# model.text_save(base_dir)

# 预测解
u_pred, v_pred, p_pred = model.predict(X_test)
u_pred_picture = u_pred.reshape(n_y, n_x)
v_pred_picture = v_pred.reshape(n_y, n_x)
p_pred_picture = p_pred.reshape(n_y, n_x)
# l2_error = np.linalg.norm(u_pred - usol) / np.linalg.norm(usol)
# print(l2_error)
# predict solution
plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(XX, YY, u_pred_picture, cmap='jet')
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig(os.path.join(base_dir, "u_predict_solution.png"))
plt.close()

plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(XX, YY, v_pred_picture, cmap='jet')
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig(os.path.join(base_dir, "v_predict_solution.png"))
plt.close()

# 计算速度的模，即 sqrt(u^2 + v^2)
speed_picture = np.sqrt(u_pred_picture**2 + v_pred_picture**2)
plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(XX, YY, speed_picture, cmap='jet')
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig(os.path.join(base_dir, "speed_predict_solution.png"))
plt.close()

plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(XX, YY, p_pred_picture, cmap='jet')
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig(os.path.join(base_dir, "p_predict_solution.png"))
plt.close()
