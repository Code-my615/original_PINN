import torch


def get_params(dataset, args, x_star, y_star, usol, real_source, device):

    params = {
        "train_data":{
            "X_f_train": dataset.X_f_train,
            "layer_sizes": args.layer_sizes,
            "boundaries": dataset.boundaries,
            "boundaries_u_true": dataset.boundaries_u_true,
            "usol_num": args.usol_num  # 用于逆问题的真实解的个数
        },
        "train_params": {
            "optimizer": torch.optim.Adam,
            "lr": args.lr,
            "lambda_bc": args.lambda_bc,
            "lambda_pde": args.lambda_pde,
        },
        "viz_params": {
            "x_star": x_star,
            "y_star": y_star,
            "usol": usol,
            "real_source": real_source
        },
        "pde_params": {
            "coefficient": args.coefficient,
        },
        "device": device
    }
    return params
