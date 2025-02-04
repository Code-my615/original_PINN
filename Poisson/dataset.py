import matplotlib.pyplot as plt
import numpy as np
import random
import torch


class Dataset:
    def __init__(self, coefficient, x_min=0, y_min=0, x_max=2 * np.pi, y_max=1,
                 X_f_train_num=30000, boundary_num=1500, seed=0, problem='forward'):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.X_f_train_num = X_f_train_num
        self.boundary_num = boundary_num
        self.seed = seed
        self.coefficient = coefficient

        self._set_seed(seed)
        self.x = None
        self.y = None
        self.X_f_train = None
        self.top_boundary = None
        self.bottom_boundary = None
        self.left_boundary = None
        self.right_boundary = None
        self.boundaries = None
        self.boundaries_u_true = None
        self.problem = problem

        self._generate_points()

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _sample_random(X_all, N, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        idx = np.random.choice(X_all.shape[0], N, replace=False)
        return X_all[idx, :]

    def _generate_points(self):
        # Generate x and y grids
        self.x = np.linspace(self.x_min, self.x_max, self.boundary_num)
        self.y = np.linspace(self.y_min, self.y_max, self.boundary_num)

        if self.problem == 'inverse':
            xx, yy = np.meshgrid(self.x, self.y)
            X_star = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))
            self.X_f_train = Dataset._sample_random(X_star, self.X_f_train_num, seed=self.seed)
            return

        # Generate domain points
        y_noboundary = self.y[1:-1]
        x_noboundary = self.x[1:-1]
        X_noboundary, Y_noboundary = np.meshgrid(x_noboundary, y_noboundary)
        X_star_noboundary = np.hstack(
            (X_noboundary.flatten()[:, None], Y_noboundary.flatten()[:, None])
        )
        self.X_f_train = self._sample_random(X_star_noboundary, self.X_f_train_num, seed=self.seed)

        # Generate boundary points
        # 上边界 (y = y_max)
        self.top_boundary = np.hstack((
            np.random.uniform(self.x_min, self.x_max, (self.boundary_num, 1)),  # x 随机分布在 [x_min, x_max]
            np.full((self.boundary_num, 1), self.y_max)  # y 固定为 y_max
        ))
        self.u_top = np.sin(self.coefficient * self.top_boundary[:, 0:1]) * np.cos(self.coefficient * self.y_max)  # 真实解

        # 下边界 (y = y_min)
        self.bottom_boundary = np.hstack((
            np.random.uniform(self.x_min, self.x_max, (self.boundary_num, 1)),  # x 随机分布在 [x_min, x_max]
            np.full((self.boundary_num, 1), self.y_min)  # y 固定为 y_min
        ))
        self.u_bottom = np.sin(self.coefficient * self.bottom_boundary[:, 0:1]) * np.cos(self.coefficient * self.y_min)  # 真实解

        # 左边界 (x = x_min)
        self.left_boundary = np.hstack((
            np.full((self.boundary_num, 1), self.x_min),  # x 固定为 x_min
            np.random.uniform(self.y_min, self.y_max, (self.boundary_num, 1))  # y 随机分布在 [y_min, y_max]
        ))
        self.u_left = np.sin(self.coefficient * self.x_min) * np.cos(self.coefficient * self.left_boundary[:, 1:])  # 真实解

        # 右边界 (x = x_max)
        self.right_boundary = np.hstack((
            np.full((self.boundary_num, 1), self.x_max),  # x 固定为 x_max
            np.random.uniform(self.y_min, self.y_max, (self.boundary_num, 1))  # y 随机分布在 [y_min, y_max]
        ))
        self.u_right = np.sin(self.coefficient * self.x_max) * np.cos(self.coefficient * self.right_boundary[:, 1:])  # 真实解

        self.boundaries = np.vstack((self.top_boundary, self.bottom_boundary, self.left_boundary, self.right_boundary))
        self.boundaries_u_true = np.vstack((self.u_top, self.u_bottom, self.u_left,  self.u_right))

    def get_collocation_points(self):
        return self.X_f_train

    def get_boundary_points(self):
        return self.boundaries

    def summary(self):
        print(f"Domain points (collocation): {self.X_f_train.shape}")
        print(f"Boundary points: {self.boundaries.shape}")

    def plot(self):
        """Visualize the distribution of collocation, boundary, and initial points."""
        plt.figure(figsize=(10, 6))

        # Collocation points
        plt.scatter(self.X_f_train[:, 0], self.X_f_train[:, 1],
                    s=1, color='blue', label='Collocation Points')

        # Boundary points
        plt.scatter(self.boundaries[:, 0], self.boundaries[:, 1],
                    s=10, color='red', label='Boundary Points')

        # Plot settings
        plt.xlabel('$x$', fontsize=12)
        plt.ylabel('$y$', fontsize=12)
        plt.title('Distribution of Collocation and Boundary Points', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # 使用示例
    dataset = Dataset()
    dataset.summary()
    dataset.plot()
