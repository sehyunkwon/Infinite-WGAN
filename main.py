import argparse
import numpy as np
import os
import torch

from args import get_default_opt
from utils import get_W_D_and_b_D
from wrapper import wrapper


if __name__ == "__main__":

    # Fix ForwardX11 related issues (ssh with tmux)
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    parser = argparse.ArgumentParser()
    opt = get_default_opt(parser)

    # Use list for parameter sweep, it will automatically run all possible combinations
    # If you are okay with default values of specific parameters, you can omit those from the following list

    Nd = 1000

    # ################### for 8 modes ###################
    num_locs = 8
    rt2 = np.sqrt(2)
    means = torch.Tensor(np.array([[1., 1.], [1., -1.], [-1., -1.], [-1., 1.], [rt2, 0.], [-rt2, 0.],
                                   [0., -rt2], [0., rt2]]))
    covs = torch.Tensor(0.01 * np.array(num_locs*[np.eye(2)]))
    scatter_lim = 4  # it will be used to generate psi' (a_j and b_j)
    plot_lim = 4  # decide plotting area
    W_D, b_D = get_W_D_and_b_D(Nd, scatter_lim)
    Ng = 5000
    G_lr = 1e-5
    scale_theta = 5e-3

    # ################### for 9 modes ###################
    # num_locs = 9
    # means = torch.Tensor(np.array([[1., 1.], [1., 0.], [1., -1.], [0., 1.], [0., 0.], [0., -1.],
    #                                [-1., 1.], [-1., 0.], [-1., -1.]]))
    # covs = torch.Tensor(0.01 * np.array(num_locs*[np.eye(2)]))
    # scatter_lim = 4  # it will be used to generate psi' (a_j and b_j)
    # plot_lim = 4  # decide plotting area
    # W_D, b_D = get_W_D_and_b_D(Nd, scatter_lim)
    # Ng = 10000
    # G_lr = 5e-6
    # scale_theta = 3e-3

    # ################### for spiral ###################
    # num_locs = 20
    # temp = []
    # for r in range(20):
    #     temp.append(np.array([r/20*np.cos((2*r*np.pi)/20), r/20*np.sin((2*r*np.pi)/20)]))
    # means = torch.Tensor(np.array(temp))
    # covs = torch.Tensor(0.01 * np.array(num_locs*[np.eye(2)]))
    # scatter_lim = 2  # it will be used to generate psi' (a_j and b_j)
    # plot_lim = 4  # decide plotting area
    # W_D, b_D = get_W_D_and_b_D(Nd, scatter_lim)
    # Ng = 10000
    # G_lr = 1e-6
    # scale_theta = 3e-3

    input_dict = {"seed": 0,  # seed
                  "data_dim": 2,  # dimension of data
                  "Nd": Nd,  # width of the discriminator
                  "Ng": Ng,  # width of the generator
                  "latent_dim": 2,  # dimension of latent variable (should be larger than data_dim)
                  "epochs": 25,  # number of epochs
                  "batch_size": 5000,  # size of batch
                  "G_lr": G_lr,  # learning rate for the Generator
                  "D_lr": 1,  # learning rate for the Discriminator
                  "scale_G_W": 10,  # scaling factor of of initial weights (Generator)
                  "scale_G_b": 10,  # scaling factor of of initial bias (Generator)
                  "scale_D_weights": 1,  # scaling factor of of initial weights (Discriminator)
                  "scale_theta": scale_theta,  # scaling factor of of initial theta
                  "scale_gamma": 1,  # scaling factor of of initial gamma
                  "num_locs": num_locs,  # number of mixture components
                  "num_datasets": 100000,  # total number of datasets = num_locs * num_datasets
                  "scatter_lim": scatter_lim,  # xlim and ylim of scatter plot
                  "plot_lim": plot_lim,  # decide plotting area
                  "G_activation": "tanh",  # activation function of G
                  "D_activation": "tanh",  # activation function of D
                  "rate": 0.9,  # decay learning rate of G
                  "num_gen": 5000,  # the number of generated samples
                  "means": means,  # means of true distribution
                  "covs": covs,  # covariance matrices of true distribution
                  "W_D": W_D,  # Weight of the first layer of D (data_dim by Nd Tensor)
                  "b_D": b_D,  # bias of the first layer of D (data_dim by Nd Tensor)
                  }

    wrapper(opt, input_dict)
