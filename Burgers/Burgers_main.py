import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import torch
import sys
import pandas as pd
from dataset import Dataset
from pinn_model import *
from parameters import get_params


parser = argparse.ArgumentParser(description='take params')
parser.add_argument('--experiment', type=str, default='burgers')
parser.add_argument('--problem', type=str, default='inverse')
parser.add_argument('--col_points', type=int, default=2000)
parser.add_argument('--ic_points', type=int, default=256)
parser.add_argument('--bc_points', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--x_num', type=int, default=256, help='Number of x-coordinates for ground truth grid')  # 测试x轴取的点数
parser.add_argument('--t_num', type=int, default=128, help='Number of t-coordinates for ground truth grid')  # 测试t轴取的点数
parser.add_argument('--range_x', type=float, nargs=2, default=[-1, 1])
parser.add_argument('--range_t', type=float, nargs=2, default=[0, 1])
parser.add_argument('--nu', type=float, default=0.005)
parser.add_argument('--lambda_ic', type=int, default=1.0)
parser.add_argument('--lambda_bc', type=int, default=1.0)
parser.add_argument('--lambda_pde', type=int, default=1.0)
parser.add_argument('--usol_num', type=int, default=500, help='The number of true solutions used for inverse problems') # 用于逆问题的真实解的个数
parser.add_argument('--seed', type=int, default=12)
parser.add_argument('--layer_sizes', type=int, nargs='+', default=[2, 100, 100, 100, 100, 100, 1])  # 对流方程中的初始条件
parser.add_argument('--u0_str', type=str, default='sin(x)')  # 对流方程中的初始条件

args = parser.parse_args()
# CUDA support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建目录
save_dir = f"Seed_{args.seed}_{args.experiment}"
# os.makedirs(save_dir, exist_ok=True)
# 创建子目录
if args.problem == 'forward':
    sub_dir = f"{args.problem}/nu={args.nu}/col_points_{args.col_points},ic_points_{args.ic_points},bc_points_{args.bc_points}/lr={args.lr}/epochs_{args.epochs}"
elif args.problem == 'inverse':
    sub_dir = f"{args.problem}/nu={args.nu}/col_points_{args.col_points}_usol_num_{args.usol_num}/epochs_{args.epochs}"
else:
    print("输入有误！")
    sys.exit()

# sub_sub_dir = sub_dir + '/predict_picture'
base_dir = os.path.join(save_dir, sub_dir)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Generate training dataset
dataset = Dataset(args.range_x[0], args.range_t[0], args.range_x[1], args.range_t[1], args.col_points, args.bc_points, args.ic_points, args.seed, args.problem)

# Processing the test data
x_star = np.linspace(args.range_x[0], args.range_x[1], args.x_num)
t_star = np.linspace(args.range_t[0], args.range_t[1], args.t_num)
TT, XX = np.meshgrid(t_star, x_star)  # all the X grid points T times, all the T grid points X times
X_test = np.column_stack((XX.reshape(-1, 1), TT.reshape(-1, 1)))

# AC真实解的路径
usol_path = f'./realSolutionBurgers/realSolution/nu={args.nu}_real_solution.csv'
# 读取CSV文件，排除第一列
usol_df = pd.read_csv(usol_path, usecols=range(1, 129), header=None)
usol = usol_df.to_numpy()
usol = usol.reshape(-1, 1)
n_x = x_star.shape[0]
n_t = t_star.shape[0]


usol_picture = usol.reshape(args.x_num, args.t_num)
# reference solution
plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(TT, XX, usol_picture, cmap='jet')
plt.colorbar()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.savefig(os.path.join(base_dir, "reference_solution.pdf"))
plt.close()


params = get_params(dataset, args, base_dir, x_star, t_star, usol, device)
if args.problem == 'forward':
    model = Burgers_forward(params)
else:
    model = Burgers_inverse(params)

model.train(args.epochs)
# model.plot_losses(args.epochs, base_dir)
model.text_save(base_dir)

# 预测解
u_pred = model.predict(X_test)
u_pred_picture = u_pred.reshape(n_x, n_t)
l2_error = np.linalg.norm(u_pred - usol) / np.linalg.norm(usol)
print(l2_error)
# predict solution
plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(TT, XX, u_pred_picture, cmap='jet')
plt.colorbar()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.savefig(os.path.join(base_dir, "predict_solution.png"))
plt.close()