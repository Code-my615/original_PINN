from abc import abstractmethod

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import os
import time
from network import DNN

class BasePINN:
    def __init__(self, params):
        self.device = params["device"]

        # Initalize the network
        self.dnn = DNN(params['layer_sizes']).to(self.device)

       # Use optimizers to set optimizer initialization and update functions
        self.lr = params["train_params"]["lr"]
        self.optimizer_fn = params["train_params"]["optimizer"]

        # 用于正问题的pde参数/用于逆问题计算L2 error的真实pde参数
        self.pde_beta = params["pde_params"]["beta"]

        # 初始化训练数据
        self.X_f_train = params["train_data"]["X_f_train"]

        # 各项损失函数前的系数
        self.lambda_ic = params["train_params"]["lambda_ic"]
        self.lambda_bc = params["train_params"]["lambda_bc"]
        self.lambda_pde = params["train_params"]["lambda_pde"]


        # 将pde数据分成x,t. 目的为了之后自动微分
        self.X_f_x = torch.tensor(self.X_f_train[:, 0].reshape(-1, 1), dtype=torch.float, device=self.device, requires_grad=True)
        self.X_f_t = torch.tensor(self.X_f_train[:, 1].reshape(-1, 1), dtype=torch.float, device=self.device, requires_grad=True)


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
        t_star = params['viz_params']['t_star']
        self.XX, self.TT = np.meshgrid(x_star, t_star)  # all the X grid points T times, all the T grid points X times
        self.X_star, self.T_star = self.XX.flatten(), self.TT.flatten()
        self.X_test = np.column_stack((self.X_star, self.T_star))
        self.usol = params['viz_params']['usol']
        self.n_x = x_star.shape[0]
        self.n_t = t_star.shape[0]

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
    def residual_net(self, x, t):
        """ Autograd for calculating the residual for different systems."""
        pass

    def loss_pde(self):
        pde = self.residual_net(self.X_f_x, self.X_f_t)
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

    def plot_losses(self, epochs, target_dir, labels=["total Loss", "BC Loss", "IC Loss", "PDE Loss"]):
        x = np.linspace(0, epochs, epochs, False)
        loss_lists = []
        loss_lists.append(self.loss_log)
        loss_lists.append(self.loss_bc_log)
        loss_lists.append(self.loss_ic_log)
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
            # plt.close()

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


class Convection_forward(BasePINN):

    def __init__(self, params):
        super().__init__(params)  # 调用父类构造函数
        self.optimizer = self.optimizer_fn(self.dnn.parameters(), self.lr)
        # 取出初始条件和边界条件点
        self.left_boundary = torch.tensor(params["train_data"]["left_boundary"], dtype=torch.float, device=self.device)
        self.right_boundary = torch.tensor(params["train_data"]["right_boundary"], dtype=torch.float, device=self.device)
        self.initial = torch.tensor(params["train_data"]["initial"], dtype=torch.float, device=self.device)
        self.ic_y_true = torch.tensor(params["train_data"]["ic_y_true"], dtype=torch.float, device=self.device)

    def residual_net(self, x, t):
        """ Autograd for calculating the residual for different systems."""
        X = torch.cat((x, t), dim=1)
        u = self.neural_net(X)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        f = u_t + self.pde_beta * u_x
        return f

    def train(self, epochs):
        self.dnn.train()
        pbar = trange(epochs)
        start_time = time.time()
        for epoch in pbar:

            self.optimizer.zero_grad()

            u_pre_ic = self.neural_net(self.initial)
            loss_ic = torch.mean((u_pre_ic - self.ic_y_true) ** 2)

            u_pre_lbc = self.neural_net(self.left_boundary)
            u_pre_rbc = self.neural_net(self.right_boundary)
            loss_bc = torch.mean((u_pre_lbc - u_pre_rbc) ** 2)

            loss = self.lambda_pde * self.loss_pde() + self.lambda_ic * loss_ic + self.lambda_bc * loss_bc

            self.loss_bc_log.append(loss_bc.item())
            self.loss_ic_log.append(loss_ic.item())
            self.loss_pde_log.append(self.loss_pde().item())
            self.loss_log.append(loss.item())

            loss.backward()
            self.optimizer.step()

            # 如果想看每轮在测试集上的表现，取消注释。注意会增加一些时间
            # u_pred = self.predict(self.X_test)
            # self.MAE = np.abs(u_pred - self.usol).mean()
            # self.MSE = ((u_pred - self.usol) ** 2).mean()
            # self.l1_error = self.MAE / np.abs(self.usol).mean()
            # self.l2_error = np.linalg.norm(u_pred - self.usol) / np.linalg.norm(self.usol)
            #
            # self.abs_err_log.append(self.l1_error)
            # self.rel_l2_err_log.append(self.l2_error)

            if (epoch+1) % 1000 == 0:
                # print("[beta: %d], Epoch: %d ,loss: %.16f, l2_error: %.16f" % (self.pde_beta, (epoch+1), loss, self.l2_error))
                print("[beta: %d], Epoch: %d ,loss: %.16f" % (self.pde_beta, (epoch + 1), loss))

        end_time = time.time()
        # 计算训练时间
        self.training_time_seconds = end_time - start_time

        # 为记录结果做准备
        u_pred = self.predict(self.X_test)
        self.MAE = np.abs(u_pred - self.usol).mean()
        self.MSE = ((u_pred - self.usol) ** 2).mean()
        self.l1_error = self.MAE / np.abs(self.usol).mean()
        self.l2_error = np.linalg.norm(u_pred - self.usol) / np.linalg.norm(self.usol)




class Convection_inverse(BasePINN):

    def __init__(self, params):
        super().__init__(params)
        # inverse
        # 随机选择 0 或 1，来从 self.pde_beta + 2 和 self.pde_beta-2 中选择
        random_index = torch.randint(0, 2, (1,)).item()
        train_beta = (self.pde_beta + 2) if random_index == 0 else (self.pde_beta - 2)
        # 初始化pde参数
        self.train_beta = torch.nn.Parameter(torch.tensor(train_beta).float().to(self.device))
        self.optimizer = self.optimizer_fn(list(self.dnn.parameters()) + [self.train_beta], self.lr)

        # 用于逆问题的一些真实解
        x_star = params['viz_params']['x_star']
        t_star = params['viz_params']['t_star']
        usol = params['viz_params']['usol']
        XX, TT = np.meshgrid(x_star, t_star)  # all the X grid points T times, all the T grid points X times
        X, T = XX.flatten(), TT.flatten()
        X_test = np.column_stack((X, T))
        usol = usol.reshape(-1, 1)
        total_points = X_test.shape[0]
        idx = np.random.choice(total_points, params['train_data']['usol_num'], replace=False)
        self.X_train = torch.tensor(X_test[idx, :], dtype=torch.float, device=self.device)
        self.u_Train = torch.tensor(usol[idx, :], dtype=torch.float, device=self.device)

    def residual_net(self, x, t):
        """ Autograd for calculating the residual for different systems."""
        X = torch.cat((x, t), dim=1)
        u = self.neural_net(X)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        f = u_t + self.train_beta * u_x
        return f

    def train(self, epochs):
        self.dnn.train()
        pbar = trange(epochs)
        start_time = time.time()
        for epoch in pbar:

            self.optimizer.zero_grad()

            u_pred = self.neural_net(self.X_train)
            fitting_loss = torch.mean((u_pred - self.u_Train)**2)
            loss = self.loss_pde() + fitting_loss

            loss.backward()
            self.optimizer.step()

            self.loss_pde_log.append(self.loss_pde().item())
            self.loss_log.append(loss.item())

            self.beta_log.append(self.train_beta.detach().item())
            self.MAE = np.abs(self.train_beta.cpu().detach().numpy() - self.pde_beta).mean()
            self.MSE = ((self.train_beta.cpu().detach().numpy() - self.pde_beta) ** 2).mean()
            self.l1_error = self.MAE / np.abs(self.pde_beta).mean()
            self.l2_error = np.linalg.norm(self.train_beta.cpu().detach().numpy() - self.pde_beta) / np.linalg.norm(self.pde_beta)

            self.abs_err_log.append(self.l1_error)
            self.rel_l2_err_log.append(self.l2_error)

            if (epoch + 1) % 1000 == 0:
                print("[real_beta: %d, Predicted beta: %.6f], Epoch: %d ,loss: %.16f, l2_error: %.16f" % (self.train_beta, self.pde_beta, (epoch+1), loss, self.l2_error))


        end_time = time.time()
        # 计算训练时间
        self.training_time_seconds = end_time - start_time
