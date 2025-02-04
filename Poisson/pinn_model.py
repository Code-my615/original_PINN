from abc import abstractmethod
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import os
import time
from network import DNN
from gen_exact_poisson import poisson_eq_exact_source


class BasePINN:
    def __init__(self, params):
        self.device = params["device"]
        # 初始化训练数据
        self.X_f_train = params["train_data"]["X_f_train"]

        # Initalize the network
        self.dnn = DNN(params['train_data']['layer_sizes']).to(self.device)

       # Use optimizers to set optimizer initialization and update functions
        self.lr = params["train_params"]["lr"]
        self.optimizer_fn = params["train_params"]["optimizer"]

        # 各项损失函数前的系数
        self.lambda_bc = params["train_params"]["lambda_bc"]
        self.lambda_pde = params["train_params"]["lambda_pde"]

        # 初始化pde参数
        self.pde_coefficient = params["pde_params"]["coefficient"]


        # 将pde数据分成x,t. 目的为了之后自动微分
        self.X_f_x = torch.tensor(self.X_f_train[:, 0].reshape(-1, 1), dtype=torch.float, device=self.device, requires_grad=True)
        self.X_f_y = torch.tensor(self.X_f_train[:, 1].reshape(-1, 1), dtype=torch.float, device=self.device, requires_grad=True)


        # Creating logs
        self.loss_log = []
        self.loss_ic_log = []
        self.loss_bc_log = []
        self.loss_pde_log = []

        self.abs_err_log = []
        self.rel_l2_err_log = []
        self.beta_log = []

        # Regular Grid for visualization
        x_star = params['viz_params']['x_star']
        y_star = params['viz_params']['y_star']
        self.XX, self.YY = np.meshgrid(x_star, y_star)  # all the X grid points T times, all the T grid points X times
        self.X_star, self.Y_star = self.XX.flatten(), self.YY.flatten()
        self.X_test = np.column_stack((self.X_star, self.Y_star))
        self.usol = params['viz_params']['usol']
        self.n_x = x_star.shape[0]
        self.n_y = y_star.shape[0]

        self.training_time_seconds = None

        # 误差
        self.MAE = None
        self.MSE = None
        self.l1_error = None
        self.l2_error = None

    def neural_net(self, X):
        u = self.dnn(X)
        return u

    @abstractmethod
    def residual_net(self, x, y):
        """ Autograd for calculating the residual for different systems."""
        pass

    def loss_pde(self):
        pde = self.residual_net(self.X_f_x, self.X_f_y)
        loss_pde = torch.mean(pde**2)
        return loss_pde

    @abstractmethod
    def train(self, epochs):
        pass

    def predict(self, X):
        self.dnn.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        u = self.neural_net(X)
        u = u.detach().cpu().numpy()
        return u

    def plot_losses(self, epochs, target_dir, labels=["total Loss", "BC Loss", "PDE Loss"]):
        x = np.linspace(0, epochs, epochs, False)
        loss_lists = []
        loss_lists.append(self.loss_log)
        loss_lists.append(self.loss_bc_log)
        # loss_lists.append(self.loss_ic_log)
        loss_lists.append(self.loss_pde_log)
        for i, loss_list in enumerate(loss_lists):
            plt.plot(x, loss_list, label=labels[i] if labels else f'Loss {i + 1}')
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(f"{labels[i]}_epoch-Loss{i + 1}")
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{target_dir}/plot_{labels[i]}_epoch_{epochs}.png")
            # plt.show()

    def text_save(self, target_dir):  # filename为写入CSV文件的路径，data为要写入数据列表.
        # 文件路径（将文件命名为 'res'）
        file_path = os.path.join(target_dir, "res")

        # 构建数据字典
        data_dict = {
            "MAE": self.MAE,
            "MSE": self.MSE,
            "L1RE": self.l1_error,
            "L2RE": self.l2_error,
            "Training Time(s)": self.training_time_seconds,
        }

        # 打开文件并写入数据
        with open(file_path, 'a') as file:
            for key, value in data_dict.items():
                file.write(f"{key}: {value:.10f} ")
                if key == "L2RE":
                    file.write('\n')
        file.close()

        with open(f"{target_dir}/pinn_l2_error.txt", 'w') as file:
            for epoch, error in enumerate(self.rel_l2_err_log, 1):
                file.write(f"{epoch},{error}\n")
        file.close()


class Poisson_forward(BasePINN):
    def __init__(self, params):
        super().__init__(params)  # 调用父类构造函数
        #  取出边界条件点
        self.boundaries = torch.tensor(params["train_data"]["boundaries"], dtype=torch.float, device=self.device)
        self.boundaries_u_true = torch.tensor(params["train_data"]["boundaries_u_true"], dtype=torch.float, device=self.device)
        self.optimizer = self.optimizer_fn(self.dnn.parameters(), self.lr)

    def residual_net(self, x, y):
        """ Autograd for calculating the residual for different systems."""
        X = torch.cat((x, y), dim=1)
        u = self.neural_net(X)

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]

        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_xx + u_yy + (self.pde_coefficient**2 + self.pde_coefficient**2)*torch.sin(self.pde_coefficient*x)*torch.cos(self.pde_coefficient*y)
        return f

    def train(self, epochs):
        self.dnn.train()
        pbar = trange(epochs)
        start_time = time.time()
        for epoch in pbar:
            self.optimizer.zero_grad()
            u_pre_bc = self.neural_net(self.boundaries)
            loss_bc = torch.mean((u_pre_bc - self.boundaries_u_true)**2)

            loss = self.loss_pde() + loss_bc

            self.loss_log.append(loss.item())
            self.loss_bc_log.append(loss_bc.item())
            self.loss_pde_log.append(self.loss_pde().item())

            loss.backward()
            self.optimizer.step()

            if (epoch+1) % 1000 == 0:
                print("[coefficient: %d], Epoch: %d ,loss: %.16f" % (self.pde_coefficient, (epoch+1), loss))

        end_time = time.time()
        # 计算训练时间
        self.training_time_seconds = end_time - start_time

        u_pred = self.predict(self.X_test)
        self.MAE = np.abs(u_pred - self.usol).mean()
        self.MSE = ((u_pred - self.usol) ** 2).mean()
        self.l1_error = self.MAE / np.abs(self.usol).mean()
        self.l2_error = np.linalg.norm(u_pred - self.usol) / np.linalg.norm(self.usol)


class Poisson_inverse(BasePINN):
    def __init__(self, params):
        super().__init__(params)  # 调用父类构造函数
        # inverse 源未知

        self.real_source = torch.tensor(poisson_eq_exact_source(self.X_f_train, self.pde_coefficient).reshape(-1, 1),dtype=torch.float, device=self.device)
        train_source = torch.tensor(self.real_source + 0.1, dtype=torch.float)
        # train_source = torch.full_like(self.X_f_x, fill_value=-1)  # 随机初始化可能会陷入到局部极小值，也有可能多个源项对应相同的解
        self.train_source = torch.nn.Parameter(train_source.float().to(self.device))
        self.optimizer = self.optimizer_fn(list(self.dnn.parameters()) + [self.train_source], self.lr)

        # 用于逆问题的一些真实解
        x_star = params['viz_params']['x_star']
        y_star = params['viz_params']['y_star']
        usol = params['viz_params']['usol']
        self.XX, self.YY = np.meshgrid(x_star, y_star)  # all the X grid points T times, all the T grid points X times
        X, Y = self.XX.flatten(), self.YY.flatten()
        X_test = np.column_stack((X, Y))
        usol = usol.reshape(-1, 1)
        total_points = X_test.shape[0]
        idx = np.random.choice(total_points, params['train_data']['usol_num'], replace=False)
        self.X_train = torch.tensor(X_test[idx, :], dtype=torch.float, device=self.device)
        self.u_Train = torch.tensor(usol[idx, :], dtype=torch.float, device=self.device)
        # self.real_source = torch.tensor(params['viz_params']['real_source'].reshape(-1,1), dtype=torch.float, device=self.device)


    def residual_net(self, x, y):
        """ Autograd for calculating the residual for different systems."""
        X = torch.cat((x, y), dim=1)
        u = self.neural_net(X)

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]

        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_xx + u_yy
        return f

    def train(self, epochs):
        self.dnn.train()
        pbar = trange(epochs)
        start_time = time.time()
        # X = torch.tensor(self.X_test[:, 0:1], requires_grad=True, dtype=torch.float32, device=self.device)
        # Y = torch.tensor(self.X_test[:, 1:2], requires_grad=True, dtype=torch.float32, device=self.device)
        for epoch in pbar:
            self.optimizer.zero_grad()
            u_pred = self.neural_net(self.X_train)
            fitting_loss = torch.mean((u_pred - self.u_Train) ** 2)

            source_pred = self.residual_net(self.X_f_x, self.X_f_y)
            loss_source = torch.mean((source_pred - self.train_source) ** 2)

            loss = fitting_loss + loss_source

            self.loss_log.append(loss.item())

            loss.backward()
            self.optimizer.step()

            if (epoch+1) % 100 == 0:
                print("[coefficient: %d], Epoch: %d ,loss: %.16f, loss_source: %.16f" % (self.pde_coefficient, (epoch+1), loss, loss_source))

                source_pred = self.residual_net(self.X_f_x, self.X_f_y)
                # source_pred = self.residual_net(X, Y) # 梯度传播可能受限，因为 train_source 是固定维参数，不一定能对未采样的点进行外推。
                # cpu很慢，故将计算放到GPU上。
                self.MAE = torch.abs(source_pred - self.real_source).mean()
                self.MSE = ((source_pred - self.real_source) ** 2).mean()
                self.l1_error = self.MAE / torch.abs(self.real_source).mean()
                self.l2_error = torch.norm(source_pred - self.real_source) / torch.norm(self.real_source)
                self.rel_l2_err_log.append(self.l2_error.cpu().item())

                # plt.figure(figsize=(5, 4), dpi=150)
                # plt.pcolor(self.YY, self.XX, source_pred.detach().cpu().numpy().reshape(100, 256), cmap='jet')
                # plt.colorbar()
                # plt.xlabel('$y$')
                # plt.ylabel('$x$')
                # plt.savefig(f"pre_source_epoch_{epoch}.png")
                # plt.close()


        end_time = time.time()
        # 计算训练时间
        self.training_time_seconds = end_time - start_time

        source_pred = self.residual_net(self.X_f_x, self.X_f_y)
        # source_pred = self.residual_net(X, Y)
        self.MAE = torch.abs(source_pred - self.real_source).mean()
        self.MSE = ((source_pred - self.real_source) ** 2).mean()
        self.l1_error = self.MAE / torch.abs(self.real_source).mean()
        self.l2_error = torch.norm(source_pred - self.real_source) / torch.norm(self.real_source)
        self.rel_l2_err_log.append(self.l2_error.cpu().item())