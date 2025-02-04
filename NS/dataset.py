import matplotlib.pyplot as plt
import numpy as np
import random
import torch


class Dataset:
    def __init__(self, L=1, R=0.5, u_max=0.00925, X_f_train_num=30000, boundary_num=1500, problem='forward', seed=0):
        self.x_min = 0
        self.y_min = -R
        self.x_max = L
        self.y_max = R
        self.X_f_train_num = X_f_train_num
        self.boundary_num = boundary_num
        self.seed = seed
        self.u_max = u_max

        self._set_seed(seed)
        self.x = None
        self.y = None
        self.X_f_train = None
        self.X_pipe_wall = None
        self.u_pipe_wall = None
        self.v_pipe_wall = None

        self.left_inlet = None
        self.u_inlet = None
        self.v_inlet = None
        self.right_outlet = None
        self.p_outlet = None

        self.boundaries = None
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
        x_noboundary = self.x[1:-1]
        y_noboundary = self.y[1:-1]
        X_noboundary, Y_noboundary = np.meshgrid(x_noboundary, y_noboundary)
        X_star_noboundary = np.hstack(
            (X_noboundary.flatten()[:, None], Y_noboundary.flatten()[:, None])
        )
        self.X_f_train = self._sample_random(X_star_noboundary, self.X_f_train_num, seed=self.seed)

        # Generate boundary points
        # 上管道壁 (y = y_max)
        self.top_wall = np.hstack((
            np.random.uniform(self.x_min, self.x_max, (self.boundary_num, 1)),  # x 随机分布在 [x_min, x_max]
            np.full((self.boundary_num, 1), self.y_max)  # y 固定为 y_max
        ))
        self.u_top_wall = np.full((self.boundary_num, 1), 0)
        self.v_top_wall = np.full((self.boundary_num, 1), 0)

        # 下管道壁 (y = y_min)
        self.bottom_wall = np.hstack((
            np.random.uniform(self.x_min, self.x_max, (self.boundary_num, 1)),  # x 随机分布在 [x_min, x_max]
            np.full((self.boundary_num, 1), self.y_min)  # y 固定为 y_min
        ))
        self.u_bottom_wall = np.full((self.boundary_num, 1), 0)
        self.v_bottom_wall = np.full((self.boundary_num, 1), 0)

        self.X_pipe_wall = np.vstack((self.top_wall, self.bottom_wall))
        self.u_pipe_wall = np.vstack((self.u_top_wall, self.u_bottom_wall))
        self.v_pipe_wall = np.vstack((self.v_top_wall, self.v_bottom_wall))

        # 左边界/管道入口 (x = x_min)
        self.left_inlet = np.hstack((
            np.full((self.boundary_num, 1), self.x_min),  # x 固定为 x_min
            np.random.uniform(self.y_min, self.y_max, (self.boundary_num, 1))  # y 随机分布在 [y_min, y_max]
        ))
        self.u_inlet = self.u_max * (1 - (self.left_inlet[:, 1].reshape(-1, 1))**2/self.y_max**2)
        self.v_inlet = np.full((self.boundary_num, 1), 0)

        # 右边界/管道出口 (x = x_max)
        self.right_outlet = np.hstack((
            np.full((self.boundary_num, 1), self.x_max),  # x 固定为 x_max
            np.random.uniform(self.y_min, self.y_max, (self.boundary_num, 1))  # y 随机分布在 [y_min, y_max]
        ))
        self.p_outlet = np.full((self.boundary_num, 1), 0)

        self.boundaries = np.vstack([
            self.top_wall, self.bottom_wall, self.left_inlet, self.right_outlet
        ])

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
