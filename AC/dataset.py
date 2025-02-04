import matplotlib.pyplot as plt
import numpy as np
import random
import torch


class Dataset:
    def __init__(self, x_min=0, t_min=0, x_max=2 * np.pi, t_max=1,
                 X_f_train_num=30000, boundaries_num=1500, ic_num=1500, seed=0, problem='forward'):
        self.x_min = x_min
        self.t_min = t_min
        self.x_max = x_max
        self.t_max = t_max
        self.X_f_train_num = X_f_train_num
        self.boundaries_num = boundaries_num
        self.ic_num = ic_num
        self.seed = seed
        self.problem = problem

        self._set_seed(seed)
        self.x = None
        self.t = None
        self.X_f_train = None
        self.left_boundary = None
        self.right_boundary = None
        self.boundaries = None
        self.initial = None
        self.ic_y_true = None

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
        # Generate x and t grids
        self.x = np.linspace(self.x_min, self.x_max, self.ic_num).reshape(-1, 1)
        self.t = np.linspace(self.t_min, self.t_max, self.boundaries_num).reshape(-1, 1)

        if(self.problem == 'inverse'):
            xx, tt = np.meshgrid(self.x, self.t)
            X_star = np.hstack((xx.flatten()[:, None], tt.flatten()[:, None]))
            self.X_f_train = Dataset._sample_random(X_star, self.X_f_train_num, seed=self.seed)
            return

        # Generate domain points
        t_noinitial = self.t[1:]
        x_noboundary = self.x[1:-1]
        X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
        X_star_noinitial_noboundary = np.hstack(
            (X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None])
        )
        self.X_f_train = Dataset._sample_random(X_star_noinitial_noboundary, self.X_f_train_num, seed=self.seed)

        # Generate boundary points
        # Periodic boundary (left)
        t_values = np.random.uniform(self.t_min, self.t_max, self.boundaries_num)
        X_left = np.full((self.boundaries_num, 1), self.x_min)
        self.left_boundary = np.column_stack((X_left, t_values))

        # Periodic boundary (right)
        right_boundary = np.copy(self.left_boundary)
        right_boundary[:, 0] = self.x_max
        self.right_boundary = right_boundary

        self.boundaries = np.vstack((self.left_boundary, self.right_boundary))

        # Generate initial condition points
        x_values = np.random.uniform(self.x_min, self.x_max, self.ic_num)
        self.initial = np.column_stack((x_values, np.zeros_like(x_values)))

        # Generate initial condition true values
        self.ic_y_true = self.initial[:, 0]**2 * np.cos(np.pi * self.initial[:, 0]).reshape(-1, 1)

    def get_collocation_points(self):
        return self.X_f_train

    def get_boundary_points(self):
        return self.boundaries

    def get_initial_points(self):
        return self.initial

    def get_initial_true_values(self):
        return self.ic_y_true

    def summary(self):
        print(f"Domain points (collocation): {self.X_f_train.shape}")
        print(f"Boundary points: {self.boundaries.shape}")
        print(f"Initial condition points: {self.initial.shape}")
        print(f"Initial true values: {self.ic_y_true.shape}")

    def plot(self):
        """Visualize the distribution of collocation, boundary, and initial points."""
        plt.figure(figsize=(10, 6))

        # Collocation points
        plt.scatter(self.X_f_train[:, 0], self.X_f_train[:, 1],
                    s=1, color='blue', label='Collocation Points')

        # Boundary points
        plt.scatter(self.boundaries[:, 0], self.boundaries[:, 1],
                    s=10, color='red', label='Boundary Points')

        # Initial condition points
        plt.scatter(self.initial[:, 0], self.initial[:, 1],
                    s=10, color='green', label='Initial Condition Points')

        # Plot settings
        plt.xlabel('$x$', fontsize=12)
        plt.ylabel('$t$', fontsize=12)
        plt.title('Distribution of Collocation, Boundary, and Initial Points', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # 使用示例
    dataset = Dataset()
    dataset.summary()
    dataset.plot()
