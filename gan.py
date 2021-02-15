import numpy as np
import pandas as pd
import time
import torch
import tqdm

from data_gen import GMMDataset
from models import Generator, Discriminator
from utils import get_device, get_latent_vector
from visualization import plot_samples, plot_kde_samples


class InfiniteGAN():
    def __init__(self, opt, fname):
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        self.seed = opt.seed

        self.fname = fname
        self.batch_size = opt.batch_size
        self.latent_dim = opt.latent_dim
        self.epochs = opt.epochs
        self.rate = opt.rate
        self.num_gen = opt.num_gen
        self.n = opt.data_dim
        self.Nd = opt.Nd
        self.scatter_lim = opt.scatter_lim
        self.plot_lim = opt.plot_lim

        self.device = get_device()
        self.dataset = GMMDataset(opt.data_dim, opt.num_datasets, opt.num_locs, means=opt.means,
                                  covs=opt.covs)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=True)

        # get generator and discriminator
        self.G = Generator(opt, self.device).to(self.device)
        self.D = Discriminator(opt, self.device).to(self.device)

        self.optimizer_D = torch.optim.SGD(self.D.parameters(), lr=opt.D_lr)
        self.optimizer_G = torch.optim.SGD(self.G.parameters(), lr=opt.G_lr)
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_G,
                                                             lr_lambda=lambda epoch: self.rate ** epoch)

        self.Jtheta_list = []
        self.loss_D_list = []
        self.theta_list = []
        self.gamma_list = []

    def get_device(self):
        return self.device

    def get_Jtheta(self, x, z):
        # compute J(theta)
        with torch.no_grad():
            psi_x = self.D.psi(x).mean(axis=0)
            psi_phi_z = self.D.psi(self.G(z)).mean(axis=0)
            Jtheta = torch.sum(0.5 * torch.square(psi_x-psi_phi_z))
        return Jtheta

    def train(self):
        print('seed number is: ', self.seed)
        start = time.time()
        pbar = tqdm.tqdm(range(self.epochs))

        W, b = self.D.get_param()
        true_samples = next(iter(self.loader))
        # plot true samples (also kde plot)
        plot_samples(true_samples, self.fname+'_true', W, b,
                     scatter_lim=self.scatter_lim, plot_lim=self.plot_lim)
        plot_kde_samples(true_samples, self.fname+'_true', plot_lim=self.plot_lim)

        for epoch_idx in pbar:
            # plot generated samples at the beginning of every epoch
            W, b = self.D.get_param()
            samples = self.get_generated_samples()
            plot_samples(samples, self.fname+f'_epoch{epoch_idx:02d}', W, b,
                         scatter_lim=self.scatter_lim, plot_lim=self.plot_lim)

            for batch_idx, x in enumerate(self.loader):
                x = x.to(self.device)
                z = get_latent_vector(self.batch_size, self.latent_dim, self.device)

                self.optimizer_D.zero_grad()

                psi_x = torch.mean(self.D(x))  # average over batch
                psi_phi_z = torch.mean(self.D(self.G(z)))

                # gradient ascent: update gamma (minimizing the negative objective function)
                loss_D = -psi_x + psi_phi_z + 0.5*torch.norm(self.D.gamma)**2
                loss_D.backward()
                self.optimizer_D.step()

                # gradient descent: update theta
                new_z = get_latent_vector(self.batch_size, self.latent_dim, self.device)

                self.optimizer_G.zero_grad()

                generated_samples = self.G(new_z)
                new_psi_phi_z = torch.mean(self.D(generated_samples).to(self.device))
                loss_G = - new_psi_phi_z
                loss_G.backward()
                self.optimizer_G.step()

                Jtheta = self.get_Jtheta(x, new_z)

                self.loss_D_list.append(-loss_D.item())
                self.Jtheta_list.append(Jtheta.item())

                pbar.set_description(f"[Epoch {epoch_idx:3d}] Processing batch #{batch_idx:3d} Jtheta: {Jtheta.item():.6f}")

            # update learning rate every epoch
            self.scheduler_G.step()

            # save theta and gamma every epoch
            self.theta_list.append(self.G.theta.clone())
            self.gamma_list.append(self.D.gamma.clone())

        end = time.time()
        print("Time ellapsed in training is: {}".format(end - start))

    def get_generated_samples(self):
        with torch.no_grad():
            generated_list = []
            num_repeat = int(self.num_gen/self.batch_size)+1
            for _ in range(num_repeat):
                z = get_latent_vector(self.batch_size, self.latent_dim, self.device)
                generated_samples = self.G(z).cpu().detach().numpy()
                generated_list.append(generated_samples)
        return np.vstack(generated_list)

    def write_csv(self):
        csv_fname = f'{self.fname}.csv'
        df = pd.DataFrame({'Jtheta': self.Jtheta_list,
                           'loss_D': self.loss_D_list,
                           })
        df.to_csv(csv_fname)
