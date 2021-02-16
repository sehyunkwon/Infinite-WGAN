<h1 align="center"><b>Infinite WGAN</b></h1>
<h3 align="center"><b>WGAN with an Infinitely Wide Generator has No Spurious Stationary Points</b></h1>
<p align="center">
</p> 
 
--------------

<br>

This is an repository for [WGAN with an Infinitely Wide Generator Has No Spurious Stationary Points](https://arxiv.org/abs/2102.07541). WGAN with a 2-layer generator and a 2-layer discriminator both with random features and sigmoidal activation functions and with the width of the generator (but not the discriminator) being large or infinite has **no spurious stationary points** when trained with stochastic gradient ascent-descent. This repo provides the code for 8-modes, 9-modes and spiral-like GMMs experiments in large width generator case.

## Requirements
Please check `requirements.txt`.

## Instructions
Currently, the user can update parameters directly by modifying `input_dict` in `main.py`.

```
python main.py
```

We set `Nd` to 1000 and generate discrminator feature functions as described in the Appendix.
True distribution is a Gaussian mixture, where `means` and `covs` are given in the `main.py`.
We conduct our experiments on a 12GB Nvidia 1080Ti GPU,
and we set `--Ng` to 10000, and `--batch_size` to 10000 due to the limited GPU memory.
In a GPU with larger memory, you can try larger `--Ng` as well as larger `--batch_size`.

After the training, generated samples, true samples and loss values are stored in results folder in csv format.
One can visualize the KDE plot of true and generated samples, as well as the loss plot using `plotting.ipynb`.

