import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    """ Docstring
    """
    def __init__(self, opt, device):
        super(Generator, self).__init__()
        self.Ng = opt.Ng
        self.n = opt.data_dim
        self.d = opt.latent_dim

        # initialize theta
        self.theta = nn.Parameter(opt.scale_theta * torch.randn(1, self.Ng, dtype=torch.float32, requires_grad=True,
                                                                device=device))

        # generate phi's
        self.W = opt.scale_G_W * torch.randn((self.n, self.Ng, self.d),
                                             dtype=torch.float32, device=device)
        self.b = opt.scale_G_b * torch.randn((self.n, self.Ng, 1),
                                             dtype=torch.float32, device=device)

        # add two constant hidden units
        self.W[:, -1, :] = torch.Tensor(np.zeros((self.n, self.d)))
        self.b[:, -1, -1] = torch.Tensor([50]+[0]*(self.n-1))
        self.W[:, -2, :] = torch.Tensor(np.zeros((self.n, self.d)))
        self.b[:, -2, -1] = torch.Tensor([0, 50]+[0]*(self.n-2))

        if opt.G_activation == "sigmoid":
            self.activation = torch.sigmoid
        elif opt.G_activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError("Activation should be either sigmoid or tanh")

    def phi(self, z):
        """ Activation functions for Generator
        Computes $[phi_1(z), phi_2(z), ..., phi_{N_g}(z)]$

        Args:
            torch.Tensor of size [d, batch_size]

        Returns:
            torch.Tensor of size [n, Ng, batch_size]
        """
        hidden = self.activation(self.b + torch.matmul(self.W, z))
        return hidden

    def forward(self, z):
        """ Forward path of the Generator
        Computes $sum_{i=1}^{N_g} theta_i phi_i(z)$

        Args:
            torch.Tensor of size [d, batch_size]

        Returns:
            torch.Tensor of size [batch_size, n]
        """
        hidden = self.phi(z)
        out = torch.matmul(self.theta, hidden).squeeze(dim=1)
        out = torch.transpose(out, 0, 1)
        return out


class Discriminator(nn.Module):
    """ Docstring
    """
    def __init__(self, opt, device):
        super(Discriminator, self).__init__()
        self.Nd = opt.Nd
        self.n = opt.data_dim

        # initialize gamma
        self.gamma = nn.Parameter(opt.scale_gamma * torch.randn(self.Nd, 1, dtype=torch.float32, requires_grad=True,
                                                                device=device))

        # generate psi's
        self.linear = nn.Linear(self.n, self.Nd)
        for param in self.linear.parameters():
            param.requires_grad = False

        self.linear.weight.data = opt.W_D.clone()
        self.linear.bias.data = opt.b_D.clone()

        if opt.D_activation == "sigmoid":
            self.activation = torch.sigmoid
        elif opt.D_activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError("Activation should be either sigmoid or tanh")

    def psi(self, x):
        """ Activation functions for Discriminator
        Computes $[psi_1(x), psi_2(x), ..., psi_{N_d}(x)]$

        Args:
            torch.Tensor of size [batch_size, n]

        Returns:
            torch.Tensor of size [batch_size, Nd]
        """
        hidden = self.activation(self.linear(x))
        return hidden

    def get_param(self):
        for name, param in self.linear.named_parameters():
            if name == "weight":
                W = param.cpu()
            elif name == "bias":
                b = param.cpu()
            else:
                raise ValueError("parameter name should be either weight or bias")
        return W, b

    def forward(self, x):
        """ Forward path of the Discriminator
        Computes $sum_{j=1}^{N_d} gamma psi_j(x)$

        Args:
            torch.Tensor of size [batch_size, n]

        Returns:
            torch.Tensor of size [batch_size, 1]
        """
        hidden = self.psi(x)
        out = torch.matmul(hidden, self.gamma)
        return out
