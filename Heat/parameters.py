import numpy as np
import os
import sys
import torch


def get_params(dataset, args, x_star, t_star, usol, device):

    params = {
        "problem": args.problem,
        "layer_sizes": args.layer_sizes,
        "train_data":{
            "X_f_train": dataset.X_f_train,
            "left_boundary": dataset.left_boundary,
            "right_boundary": dataset.right_boundary,
            "initial": dataset.initial,
            "ic_y_true":dataset.ic_y_true,
            "usol_num": args.usol_num  #用于逆问题的真实解的个数

        },
        "train_params": {
            "optimizer": torch.optim.Adam,
            "lr": args.lr,
            "lambda_ic": args.lambda_ic,
            "lambda_bc": args.lambda_bc,
            "lambda_pde": args.lambda_pde,
        },
        "viz_params": {
            "x_star": x_star,
            "t_star": t_star,
            "usol": usol,
        },
        "pde_params": {
            "alpha": args.alpha,
        },
        "device": device
    }
    return params
