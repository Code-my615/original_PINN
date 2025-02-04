import numpy as np
import os
import sys
import torch


def get_params(dataset, args, x_star, y_star, device):

    params = {
        "problem": args.problem,
        "layer_sizes": args.layer_sizes,
        "train_data":{
            "X_f_train": dataset.X_f_train,
            "pipe_wall": {"X_pipe_wall": dataset.X_pipe_wall,
                         "u_pipe_wall": dataset.u_pipe_wall,
                         "v_pipe_wall": dataset.v_pipe_wall
                         },
            "inlet":{"X_left_inlet": dataset.left_inlet,
                     "u_inlet": dataset.u_inlet,
                     "v_inlet": dataset.v_inlet
                     },
            "outlet":{"X_right_outlet": dataset.right_outlet,
                      "p_outlet": dataset.p_outlet
            }
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
            "y_star": y_star,
        },
        "pde_params": {
            "lambda_1": args.lambda_1,
            "lambda_2": args.lambda_2
        },
        "device": device
    }
    return params
