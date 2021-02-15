import itertools
import os
import pathlib
import time

from args import update_opt
from gan import InfiniteGAN
from utils import write_fixed_param
from visualization import plot_samples, plot_statistics, plot_kde_samples


def wrapper(opt, input_dict):
    iter_dict = {}
    fix_dict = {}
    for key in input_dict:
        if isinstance(input_dict[key], list):
            iter_dict[key] = input_dict[key]
        else:
            fix_dict[key] = input_dict[key]

    fixed_opt = update_opt(opt, fix_dict)

    iter_set = itertools.product(*iter_dict.values())
    iter_keys = list(iter_dict.keys())

    # path to store the results
    result_path = 'results'
    pathlib.Path(result_path).mkdir(exist_ok=True)
    new_time = time.strftime("%m%d-%H%M%S", time.localtime())
    sub_path = f"sweep_{new_time}"
    path = os.path.join(result_path, sub_path)
    pathlib.Path(path).mkdir(exist_ok=True)

    # write fixed_dict
    fixed_param_fname = os.path.join(path, 'fixed_param.json')
    write_fixed_param(fix_dict, fixed_param_fname)

    for vals in iter_set:
        # Update opt_dict for iterating values
        update_dict = {}
        fname = 'run'
        for idx, key in enumerate(iter_keys):
            update_dict[key] = vals[idx]
            # Define log file name with iterating values
            fname += f'_{key}_{vals[idx]}'

        opt_iter = update_opt(fixed_opt, update_dict)

        print(fname)
        print(opt_iter)
        fname_with_path = os.path.join(path, fname)
        gan = InfiniteGAN(opt_iter, fname_with_path)
        gan.train()

        gan.write_csv()
        samples = gan.get_generated_samples()
        W, b = gan.D.get_param()
        plot_samples(samples, fname_with_path, W, b, scatter_lim=gan.plot_lim)
        plot_kde_samples(samples, fname_with_path, plot_lim=gan.plot_lim)
        plot_statistics(fname_with_path)
