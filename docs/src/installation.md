## Setup

- Python 3.11 or higher is required. 
- Git must be installed.

For GPU acceleration, you can install CUDA on systems with an NVIDIA GPU. This is available on both [WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) and [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). While GPU use is optional, it significantly speeds up training the VAE models and other compute-intensive tasks.

## Install with Windows

Currently, the software is supported under Linux, including Windows via WSL2:

- Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) on your Windows system.
- Create a `.wslconfig` file in your Windows home directory to adjust memory, processors, and swap space as needed. Example configuration:
```
[wsl2]
memory=20GB
processors=8
swap=8GB
```

**Restart WSL2 or your machine to apply these configuration changes.**  
For more details on WSL2 configuration, see [Microsoft's documentation](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).

Proceed with the Linux installation instructions once WSL2 is configured and running.

## Install with Linux

### Setting up the environment
We suggest using [Poetry](https://python-poetry.org/docs/main/) for creating and managing the project environment.


### Install mcaqueretina

Navigate to your local macaqueretina git repository root and run:

```
poetry install
```

### Download variational autoencoder (VAE) models separately
You can work with difference of Gaussian (DOG) models with the vanilla repo. 
To use also the variational autoencoder (VAE) models, you need to run 
`./get_vae_models.sh` in repo root and download the about 800 MB of model data.

### Optional: Install Pytorch separately

If you encounter issues with the Pytorch installation through Poetry and have an NVIDIA GPU with CUDA, [install Pytorch using a system-specific command](https://pytorch.org/) into your environment before proceeding. 