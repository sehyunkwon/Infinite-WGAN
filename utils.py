import json
import numpy as np
import torch


def get_W_D_and_b_D(Nd, scatter_lim):
    W_list = []
    b_list = []
    for i in range(Nd):
        const = np.random.uniform(1, 10)
        W = np.array([const/np.random.uniform(-scatter_lim, scatter_lim),
                      const/np.random.uniform(-scatter_lim, scatter_lim)])
        W_list.append(W)
        b_list.append(-const)
    return torch.Tensor(W_list), torch.Tensor(b_list)


def get_device():
    """ Get available device

    Returns:
        torch.device(device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    return torch.device(device)


def get_latent_vector(batch_size, dim, device):
    return torch.randn(dim, batch_size, dtype=torch.float32, device=device)


def write_fixed_param(fix_dict, fname):
    for key in fix_dict:
        if isinstance(fix_dict[key], torch.Tensor):
            fix_dict[key] = str(fix_dict[key])
    with open(fname, "w") as f:
        json.dump(fix_dict,  f)
