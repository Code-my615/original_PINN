import sys
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import argparse
from dataset import Dataset
from pinn_model import *
from parameters import get_params
from gen_exact_poisson import gen_poisson_exact_solution, poisson_eq_exact_source


parser = argparse.ArgumentParser(description='take params')
parser.add_argument('--experiment', type=str, default='poisson')
parser.add_argument('--problem', type=str, default='inverse')
parser.add_argument('--col_points', type=int, default=2000)
parser.add_argument('--bc_points', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=8000)
parser.add_argument('--x_num', type=int, default=100, help='Number of x-coordinates for ground truth grid')  # 测试x轴取的点数
parser.add_argument('--y_num', type=int, default=100, help='Number of y-coordinates for ground truth grid')  # 测试t轴取的点数
parser.add_argument('--range_x', type=float, nargs=2, default=[-1, 1])
parser.add_argument('--range_y', type=float, nargs=2, default=[-1, 1])
parser.add_argument('--coefficient', type=int, default=1)  # 默认a=b
parser.add_argument('--lambda_bc', type=int, default=1.0)
parser.add_argument('--lambda_pde', type=int, default=1.0)
parser.add_argument('--usol_num', type=int, default=1000, help='The number of true solutions used for inverse problems') # 用于逆问题的真实解的个数
parser.add_argument('--layer_sizes', type=int, nargs='+', default=[2, 100, 100, 100, 100, 100, 1])  # 对流方程中的初始条件
parser.add_argument('--seed', type=int, default=12)

args = parser.parse_args()
# CUDA support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建目录
save_dir = f"Seed_{args.seed}_{args.experiment}"

# 创建子目录
if args.problem == 'forward':
    sub_dir = f"{args.problem}/coefficient={args.coefficient}/col_points_{args.col_points},bc_points_{args.bc_points}/lr={args.lr}/epochs_{args.epochs}"
elif args.problem == 'inverse':
    sub_dir = f"{args.problem}/coefficient={args.coefficient}/col_points_{args.col_points}_usol_num_{args.usol_num}/epochs_{args.epochs}"
else:
    print("输入有误！")
    sys.exit()

base_dir = os.path.join(save_dir, sub_dir)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Generate training dataset
dataset = Dataset(args.coefficient, args.range_x[0], args.range_y[0], args.range_x[1], args.range_y[1], args.col_points, args.bc_points, args.seed, args.problem)

# Processing the test data
x_star = np.linspace(args.range_x[0], args.range_x[1], args.x_num)
y_star = np.linspace(args.range_y[0], args.range_y[1], args.y_num)
XX, YY = np.meshgrid(x_star, y_star)
X_test = np.column_stack((XX.reshape(-1, 1), YY.reshape(-1, 1)))
usol = gen_poisson_exact_solution(args.x_num, args.y_num, args.range_x[0], args.range_x[1], args.range_y[0], args.range_y[1], args.coefficient)
usol = usol.reshape(-1, 1)
real_source = poisson_eq_exact_source(X_test, args.coefficient)

n_x = x_star.shape[0]
n_y = y_star.shape[0]
usol_picture = usol.reshape(n_y, n_x)
# reference solution
plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(YY, XX, usol_picture, cmap='jet')
plt.colorbar()
plt.xlabel('$y$')
plt.ylabel('$x$')
plt.savefig(os.path.join(base_dir, "reference_solution.png"))
plt.close()

real_source_picture = real_source.reshape(n_y, n_x)
plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(YY, XX, real_source_picture, cmap='jet')
plt.colorbar()
plt.xlabel('$y$')
plt.ylabel('$x$')
plt.savefig(os.path.join(base_dir, "true_source.png"))
plt.close()


params = get_params(dataset, args, x_star, y_star, usol, real_source, device)
if args.problem == 'forward':
    model = Poisson_forward(params)
else:
    model = Poisson_inverse(params)

model.train(args.epochs)
# model.plot_losses(args.epochs, base_dir)
model.text_save(base_dir)

# 预测解
u_pred = model.predict(X_test)
u_pred_picture = u_pred.reshape(n_y, n_x)
l2_error = np.linalg.norm(u_pred - usol) / np.linalg.norm(usol)
print(l2_error)

# u_pred_picture = u_pred.reshape(n_y, n_x)
# predict solution
plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(YY, XX, u_pred_picture, cmap='jet')
plt.colorbar()
plt.xlabel('$y$')
plt.ylabel('$x$')
plt.savefig(os.path.join(base_dir, "predict_solution.png"))
plt.close()