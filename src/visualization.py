import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_samples(samples, fname, W, b, scatter_lim=4, plot_lim=4):
    """
        samples: numpy array (batch_size by n)
    """
    plt.xlim((-plot_lim, plot_lim))
    plt.ylim((-plot_lim, plot_lim))
    plt.scatter(x=samples[:, 0], y=samples[:, 1], s=1, alpha=0.01)
    np.savetxt(f'{fname}_samples.csv', samples, delimiter=',')
    plt.savefig(f'{fname}_scatter.jpg')
    plt.close('all')

    # if input x is 2D, we plot discriminators (psi's)
    if samples.shape[1] == 2:
        x_range = np.linspace(-scatter_lim, scatter_lim, 1000)
        plt.xlim((-plot_lim, plot_lim))
        plt.ylim((-plot_lim, plot_lim))
        plt.scatter(x=samples[:, 0], y=samples[:, 1], s=1, alpha=0.01)
        Nd = W.shape[0]
        for idx in range(Nd):
            w0 = W[idx, 0].item()
            w1 = W[idx, 1].item()
            b0 = b[idx].item()
            if w1 == 0:
                if w0 == 0:
                    continue
                plt.axvline(x=-b0/w0, color='y', linewidth=0.1)
            else:
                y_range = -b0/w1 - w0/w1*x_range
                plt.plot(x_range, y_range, 'y', linewidth=0.1)
        plt.savefig(f'{fname}_scatter_with_D.jpg')
        plt.close('all')


def plot_kde_samples(samples, fname, plot_lim=4):
    """
        kernel density estimation
    """
    sns.set(rc={'axes.facecolor': 'honeydew', 'figure.figsize': (5.0, 5.0)})
    plt.xlim((-plot_lim, plot_lim))
    plt.ylim((-plot_lim, plot_lim))
    g = sns.kdeplot(x=samples[:, 0], y=samples[:, 1], s=0.1, alpha=1, shade=True, n_levels=1000, cmap='Greens')
    g.set(yticklabels=[])
    g.set(xticklabels=[])
    g.grid(False)
    plt.margins(0, 0)
    plt.savefig(f'{fname}_kde_scatter.jpg')
    plt.close('all')


def plot_statistics(fname):
    fig, ax1 = plt.subplots()
    df = pd.read_csv(f"{fname}_loss.csv")

    ax1.set(xlabel='', xticks=[], ylabel='Jtheta')

    color = 'tab:red'
    ymax = df['Jtheta'].max()
    df.plot(y='Jtheta', ax=ax1, ylim=[0, ymax], color=color)

    ax2 = ax1.twinx()
    ax2.set(ylabel='D_loss')
    color = 'tab:blue'
    ymax = df['loss_D'].max()
    ymin = df['loss_D'].min()
    df.plot(y='loss_D', ax=ax2, ylim=[ymin, ymax], color=color)

    plt.savefig(f'{fname}.pdf')
    plt.close('all')
