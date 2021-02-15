import argparse
import torch


def get_default_opt(parser):
    ''' Get default arguments
    '''
    parser.add_argument("--data_dim", type=int, default=2, help="dimension of data")
    parser.add_argument("--Ng", type=int, default=10000, help="width of the generator")
    parser.add_argument("--Nd", type=int, default=10, help="width of the discriminator")
    parser.add_argument("--latent_dim", type=int, default=2,
                        help="dimension of latent variable (should be larger than data_dim")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="size of batch")
    parser.add_argument("--G_lr", type=float, default=1e-5, help="learning rate for the Generator")
    parser.add_argument("--D_lr", type=float, default=1, help="learning rate for the Discriminator")
    parser.add_argument("--scale_G_W", type=float, default=0.1, help="scaling factor of initial weights (Generator)")
    parser.add_argument("--scale_G_b", type=float, default=0.1, help="scaling factor of initial bias (Generator)")
    parser.add_argument("--scale_D_weights", type=float, default=1, help="scaling factor of initial weights (Discriminator)")
    parser.add_argument("--scale_theta", type=float, default=1, help="scaling factor of initial theta")
    parser.add_argument("--scale_gamma", type=float, default=1, help="scaling factor of initial gamma")
    parser.add_argument("--num_locs", type=int, default=3, help="number of mixture components")
    parser.add_argument("--num_datasets", type=int, default=3000,
                        help="number of data points for each mixture model, "
                             "total number of datasets = num_locs * num_datasets")
    parser.add_argument("--scatter_lim", type=float, default=4,
                        help="xlim and ylim of scatter plot"),
    parser.add_argument("--plot_lim", type=float, default=4,
                        help="decide plotting area")
    parser.add_argument("--G_activation", type=str, default="sigmoid", help="activation function of G")
    parser.add_argument("--D_activation", type=str, default="sigmoid", help="activation function of D")
    parser.add_argument("--rate", type=float, default=0.96,
                        help="decay learning rate of G")
    parser.add_argument("--num_gen", type=int, default=3000,
                        help="the number of generated samples")
    parser.add_argument("--means", type=torch.Tensor, default=None,
                        help="means of true distribution")
    parser.add_argument("--covs", type=torch.Tensor, default=None,
                        help="covs of true distribution")
    parser.add_argument("--W_D", type=torch.Tensor, default=None,
                        help="weights of the first layer of the Discriminator, "
                             "torch.Size([data_dim, Nd])")
    parser.add_argument("--b_D", type=torch.Tensor, default=None,
                        help="bias of the first layer of the Discriminator, "
                             "torch.Size([Nd])")

    opt = parser.parse_args()
    return opt


def update_opt(opt, update_dict):
    ''' Update argparse (opt)
    '''
    opt_dict = vars(opt)
    keys = opt_dict.keys()
    for key in update_dict:
        if key not in keys:
            raise ValueError(f"{key} is not a valid parameter")
        else:
            opt_dict[key] = update_dict[key]

    opt = argparse.Namespace(**opt_dict)
    return opt
