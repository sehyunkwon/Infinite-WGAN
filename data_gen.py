import numpy as np
import torch

from torch.distributions.multivariate_normal import MultivariateNormal


class GMMDataset(torch.utils.data.Dataset):
    def __init__(self, n, num_datasets, num_locs, bound=3, means=None, covs=None):
        ''' Generate Gaussian Mixture Dataset
        Args:
            n (int): dimension of the data (x)
            num_datasets (int): number of data points per mixture component
            num_locs (int): number of mixture components
            bound (float): upper and lower bound of mean value
            means (torch.Tensor): mean vectors of mixture components
                it should be num_locs by n
                if means=None, generate mean vectors uniformly randomly from [-bound, bound]
            covs (torch.Tensor): covariance matrices of mixture components
                it should be num_locs by n by n
                use identity matrices if cov=None
        '''
        self.num_datasets = num_datasets
        if means is None:
            means = torch.Tensor(np.random.uniform(low=-bound, high=bound, size=(num_locs, n)))

        if covs is None:
            covs = torch.Tensor(np.array([np.eye(n)] * num_locs))

        self.num_locs = len(means)

        if len(covs) != self.num_locs or len(means) != self.num_locs:
            raise ValueError("length of means and covs should be equal to num_locs")

        gm_list = []
        for idx, _ in enumerate(means):
            mean = means[idx]
            cov = covs[idx]
            gmm = MultivariateNormal(mean, cov)
            gm_list.append(gmm.sample((num_datasets, )))
        self.gm = torch.cat(gm_list)
        p = torch.randperm(self.num_locs * self.num_datasets)
        self.gm = self.gm[p]

    def __len__(self):
        return self.num_datasets*self.num_locs

    def __getitem__(self, index):
        x = self.gm[index]
        return x


if __name__ == "__main__":
    # set gaussian ceters and covariances in 2D
    n = 2
    num_datasets = 3000
    num_locs = 3

    means = torch.Tensor(np.array([[10.0, 5.0],
                                   [10.0, 5.0],
                                   [0.0, -10.0]]))

    covs = torch.Tensor(np.array([np.diag([2.0, 2.0]),
                                  np.diag([2.0, 2.0]),
                                  np.diag([2.0, 2.0])]))

    dset = GMMDataset(n, num_datasets, means=means, covs=covs, num_locs=num_locs)
    print(len(dset))
