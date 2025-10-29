# MacaqueRetina

Python software to build a model of the macaque retina and convert various visual stimuli to ganglion cell action potentials.

This project is under development.

## Setup

- Python 3.11 or higher is required.
- Git must be installed.


For GPU acceleration, you can install CUDA on systems with an NVIDIA GPU. This is available on both [WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) and [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). While GPU use is optional, it significantly speeds up training the VAE models. The standard VAE model takes about an hour to train on a fast CPU.

### Install with Windows

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

### Setting up the Environment
We suggest using [Poetry](https://python-poetry.org/docs/main/) for creating and managing the project environment.


### Install MacaqueRetina

Navigate to your local MacaqueRetina git repository root and run:

```bash
poetry install
```

### Optional: Install Pytorch Separately

If you encounter issues with the Pytorch installation through Poetry and have an NVIDIA GPU with CUDA, [install Pytorch using a system-specific command](https://pytorch.org/) before proceeding with the rest of the setup. 

### How to run

Navigate to your local MacaqueRetina git repository root.

Activate your Poetry-managed environment:

```bash
poetry shell
```

Run the project:

```python
python macaqueretina
```

### How to cite this project

### Contributing

Simo Vanni, 
Henri Hokkanen