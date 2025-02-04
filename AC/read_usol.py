import csv
import numpy as np
import matplotlib.pyplot as plt
import re
import os

lbd = 0.00055
epsilon = 3

# 从第10行开始读取 CSV 文件并提取第三列数据
def extract_third_column_from_line_10(csv_file):
    third_column = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i < 9:
                continue  # 跳过前9行
            if len(row) >= 3:  # 确保行中至少有三列
                third_column.append(float(row[2]))  # 提取第三列数据并转换为浮点数
    return third_column

# 使用示例
csv_file_path = f'AC(csv)/AC_sin_lbd={lbd},epsilon={epsilon}.csv'  # 替换为你的 CSV 文件路径
third_column_data = extract_third_column_from_line_10(csv_file_path)
# print(third_column_data)

# # 使用正则表达式提取 lbd 和 epsilon 的值
# match = re.search(r"lbd=([\d.]+),epsilon=([\d.]+)", csv_file_path)
# if match:
#     lbd = float(match.group(1))     # 提取 lbd 值并转换为浮点数
#     epsilon = float(match.group(2)) # 提取 epsilon 值并转换为浮点数
#     print("lbd:", lbd)
#     print("epsilon:", epsilon)
# else:
#     print("未找到匹配的变量！")

# 目标文件路径
directory = f"./test_dataset/sin_lbd={lbd}_epsilon={epsilon}"
filename = "reference_solution.png"
full_path = os.path.join(directory, filename)
# 如果目录不存在，就创建
os.makedirs(directory, exist_ok=True)


# 测试
x_axis = np.linspace(-1, 1, 256)
t_axis = np.linspace(0, 1, 101)
# 坐标(x, t)
x_x, t_t = np.meshgrid(x_axis, t_axis)
usol = np.array(third_column_data).reshape(101, 256)
# 保存为 NumPy 的 .npy 文件
usol_npy_path = f"./test_dataset/sin_lbd={lbd}_epsilon={epsilon}/usol.npy"
np.save(usol_npy_path, usol)


# Reference solution
plt.figure(figsize=(5, 4), dpi=1000)
plt.pcolor(t_t, x_x, usol, cmap='jet')
plt.colorbar()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.savefig(f"./test_dataset/sin_lbd={lbd}_epsilon={epsilon}/reference_solution.png")
plt.close()
