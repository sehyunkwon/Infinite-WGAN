<h1 align="center"><b>Infinite WGAN</b></h1>
<h3 align="center"><b>WGAN with an Infinitely Wide Width Generator has No Spurious Stationary Points</b></h1>
<p align="center">
  <i>~ in Pytorch ~</i>
</p> 
 
--------------

<br>

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

