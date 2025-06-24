# Built-in
import os
import subprocess
import time
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from sys import exit

# Third-party
import matplotlib.pyplot as plt  # plotting library
import numpy as np  # this module is useful to work with numerical arrays
import pandas as pd
import psutil
import ray
import torch
import torch.optim.lr_scheduler as lr_scheduler
from optuna.samplers import TPESampler
from ray import air, tune
from ray.tune import Callback, CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from scipy.ndimage import fourier_shift, rotate
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics import MeanSquaredError, StructuralSimilarityIndexMeasure
from torchmetrics.image.kid import KernelInceptionDistance
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

# Local
from macaqueretina.retina.apricot_data_module import ApricotData
from macaqueretina.retina.retina_math_module import RetinaMath


class HardDiskWatchDog(Callback):
    def __init__(self, output_path, disk_usage_threshold=90):
        self.disk_usage_threshold = disk_usage_threshold
        self.output_path = output_path

    def on_trial_result(self, iteration, trials, trial, result, **info):
        if result["training_iteration"] % 100 == 0:
            disk_usage = psutil.disk_usage(str(self.output_path))
            if disk_usage.percent > self.disk_usage_threshold:
                print(
                    f"""
                    WARNING: disk_usage_threshold exceeded ({disk_usage.percent:.2f}%
                    Shutting down ray and exiting.
                    """
                )
                ray.shutdown()


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Apricot dataset class for Pytorch.

    The constructor reads the data from the ApricotData class and stores it as
    tensors of shape (n_cells, channels, height, width). While the constructor
    is called with particular gc_type and response_type, all data is retrieved
    and thus the __getitem__ method can be called with any index. This enables
    teaching the network with all data. The gc_type and response_type are, however,
    logged into the ApricotDataset instance object.
    """

    def __init__(self, data, labels, resolution_hw, augmentation_dict=None):
        if augmentation_dict is not None:
            # Multiply the amount of images by the data_multiplier. Take random samples from the data
            len_data = data.shape[0]
            data_multiplier = augmentation_dict["data_multiplier"]

            # Get the number of images to be added
            n_images_to_add = int(data_multiplier * len_data) - len_data

            # Get the indices of the images to be added
            indices_to_add = np.random.choice(len_data, n_images_to_add, replace=True)

            # Get the images to be added
            data_to_add = data[indices_to_add]

            # Get the labels to be added
            labels_to_add = labels[indices_to_add]

            # Concatenate the data and labels
            data = np.concatenate((data, data_to_add), axis=0)
            labels = np.concatenate((labels, labels_to_add), axis=0)

        self.data = data
        self.labels = self._to_tensor(labels)

        self.augmentation_dict = augmentation_dict

        # Calculate mean and std of data
        data_mean = np.mean(self.data)
        data_std = np.std(self.data)

        # Define transforms
        if self.augmentation_dict is None:
            self.transform = transforms.Compose(
                [
                    transforms.Lambda(self._feature_scaling),
                    transforms.Lambda(self._to_tensor),
                    transforms.Resize((resolution_hw, resolution_hw), antialias=True),
                ]
            )

        else:
            self.transform = transforms.Compose(
                [transforms.Lambda(self._feature_scaling)]
            )

            if self.augmentation_dict["noise"] > 0:
                self.transform.transforms.append(transforms.Lambda(self._add_noise))

            if self.augmentation_dict["rotation"] > 0:
                self.transform.transforms.append(
                    transforms.Lambda(self._random_rotate_image)
                )

            if np.sum(self.augmentation_dict["translation"]) > 0:
                self.transform.transforms.append(
                    transforms.Lambda(self._random_shift_image)
                )

            self.transform.transforms.append(transforms.Lambda(self._to_tensor))

            if self.augmentation_dict["flip"] > 0:
                self.transform.transforms.append(
                    transforms.RandomHorizontalFlip(self.augmentation_dict["flip"])
                )
                self.transform.transforms.append(
                    transforms.RandomVerticalFlip(self.augmentation_dict["flip"])
                )

            self.transform.transforms.append(
                transforms.Resize((resolution_hw, resolution_hw), antialias=True)
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Ideally, no data pre-processing steps should be coded anywhere in the whole model training pipeline but for this method.
        """

        image = self.data[idx]
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def _feature_scaling(self, data):
        """
        Scale data to range [0, 1]]. Before scaling the data, the abs median value is close to 0.0,
        consistently =< 0.02 for both parasol and midget data.

        Parameters
        ----------
        data : np.ndarray
            Data to be scaled

        Returns
        -------
        data_scaled : np.ndarray
            Scaled data
        """

        feature_range = (0, 1)
        feat_min, feat_max = feature_range
        data_normalized = (data - data.min()) / (data.max() - data.min())
        data_scaled = data_normalized * (feat_max - feat_min) + feat_min

        return data_scaled

    def _add_noise(self, image):
        """
        Add noise to the input images.

        Parameters
        ----------
        image : np.ndarray
            Input image
        noise_factor : float
            Noise factor

        Returns
        -------
        image_noise : np.ndarray

        """
        noise_factor = self.augmentation_dict["noise"]
        noise = np.random.normal(loc=0, scale=noise_factor, size=image.shape)
        image_noise = np.clip(image + noise, -3.0, 3.0)

        return image_noise

    def _add_noise_t(self, image):
        """
        Add noise to the input images.

        Parameters
        ----------
        image : torch.Tensor
            Input image
        noise_factor : float
            Noise factor

        Returns
        -------
        image_noise : torch.Tensor
        """
        noise_factor = self.augmentation_dict["noise"]
        noise = torch.randn_like(image) * noise_factor
        image_noise = torch.clamp(image + noise, -3.0, 3.0)

        return image_noise

    def _random_rotate_image(self, image):
        """
        Rotate image by a random angle.

        Parameters
        ----------
        image : np.ndarray
            Input image

        Returns
        -------
        image_rot : np.ndarray
            Rotated image
        """
        rot = self.augmentation_dict["rotation"]
        # Take random rot as float
        rot = np.random.uniform(-rot, rot)
        image_rot = rotate(image, rot, axes=(2, 1), reshape=False, mode="reflect")
        return image_rot

    def _random_shift_image(self, image):
        """
        Shift image by a random amount.

        Parameters
        ----------
        image : np.ndarray
            Input image

        Returns
        -------
        image_shift : np.ndarray
            Shifted image
        """
        shift_proportions = self.augmentation_dict["translation"]

        if isinstance(shift_proportions, float):
            shift_proportions = (shift_proportions, shift_proportions)

        shift_max = (
            int(image.shape[1] * shift_proportions[0]),
            int(image.shape[2] * shift_proportions[1]),
        )  # shift in pixels, tuple of (y, x) shift

        # Take random shift as float
        shift = (
            np.random.uniform(-shift_max[0], shift_max[0]),
            np.random.uniform(-shift_max[1], shift_max[1]),
        )

        input_ = np.fft.fft2(np.squeeze(image))
        result = fourier_shift(
            input_, shift=shift
        )  # shift in pixels, tuple of (y, x) shift

        result = np.fft.ifft2(result)
        image_shifted = result.real
        # Expand 0:th dimension
        image_shifted = np.expand_dims(image_shifted, axis=0)

        return image_shifted

    def _to_tensor(self, image):
        image_t = torch.from_numpy(deepcopy(image)).float()  # to float32
        return image_t


class VariationalEncoder(nn.Module):
    """
    Original implementation from Eugenia Anello (https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b)
    """

    def __init__(
        self,
        latent_dims,
        final_side_length,
        kernel_stride=None,
        padding=0,
        channels=8,
        conv_layers=3,
        batch_norm=True,
        latent_distribution="normal",
        device=None,
    ):
        # super(VariationalEncoder, self).__init__()
        super().__init__()
        if kernel_stride is None:
            kernel_stride = {
                "kernel": 3,
                "stride": 1,
            }

        self.device = device
        self.latent_distribution = latent_distribution

        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(
                1,
                channels,
                kernel_size=kernel_stride["kernel"],
                stride=kernel_stride["stride"],
                padding=padding,
            ),
            nn.ReLU(True),
        )

        # Make an OrderedDict to feed into nn.Sequential containing the convolutional layers
        conv_layers_2toN = OrderedDict()
        for i in range(conv_layers - 1):
            n_channels = channels * 2**i
            conv_layers_2toN["conv" + str(i + 2)] = nn.Conv2d(
                n_channels,
                n_channels * 2,
                kernel_size=kernel_stride["kernel"],
                stride=kernel_stride["stride"],
                padding=padding,
            )
            # Add one batch norm layer after second convolutional layer
            # parametrize 0 if need to put b-layer after other conv layers
            if batch_norm and i == 0:  # batch_norm is np.bool_ type, "is True" fails
                conv_layers_2toN["batch" + str(i + 2)] = nn.BatchNorm2d(n_channels * 2)

            conv_layers_2toN["relu" + str(i + 2)] = nn.ReLU(True)

        # OrderedDict works when it is the only argument to nn.Sequential
        self.encoder_conv2toN = nn.Sequential(conv_layers_2toN)

        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Linear(
            int(
                final_side_length
                * final_side_length
                * channels
                * 2 ** (conv_layers - 1)
            ),
            128,
        )

        self.linear2 = nn.Linear(128, latent_dims)  # mu

        match self.latent_distribution:
            case "normal":
                self.linear3 = nn.Linear(128, latent_dims)  # sigma
                self.D = torch.distributions.Normal(0, 1)
                if device is not None and device == "cpu":
                    self.D.loc = self.D.loc.cpu()
                    self.D.scale = self.D.scale.cpu()
                elif device is not None and device == "cuda":
                    self.D.loc = self.D.loc.cuda()  # hack to get sampling on the GPU
                    self.D.scale = self.D.scale.cuda()
            case "uniform":
                self.linear3 = nn.Linear(128, latent_dims)  # sigma
                self.sigmoid = nn.Sigmoid()  # Provides [0, 1] range
                self.D = torch.distributions.uniform.Uniform(0, 1)
                if device is not None and device == "cpu":
                    self.D.low = self.D.low.cpu()
                    self.D.high = self.D.high.cpu()
                elif device is not None and device == "cuda":
                    self.D.low = self.D.low.cuda()
                    self.D.high = self.D.high.cuda()

        self.kl = 0

    def forward(self, x):
        if self.device is not None:
            x = x.to(self.device)

        x = self.encoder_conv1(x)
        x = self.encoder_conv2toN(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        match self.latent_distribution:
            case "normal":
                mu = self.linear2(x)
                sigma = torch.exp(self.linear3(x))  # Ensure positive sigma
                z = mu + sigma * self.D.rsample(mu.shape)

                self.kl = -0.5 * torch.sum(
                    1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
                )
            case "uniform":
                midpoints = self.linear2(x)
                range_Q = self.sigmoid(self.linear3(x))  # [0, 1] range
                z = midpoints + range_Q * self.D.rsample(midpoints.shape)

                volume_P = 1  # Volume of the prior distribution
                # Compute the KL divergence between Q and P
                kl = torch.sum(torch.log(volume_P / range_Q), dim=1)
                self.kl = torch.sum(kl)  # Sum for each sample in the batch

        return z


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dims,
        final_side_length,
        kernel_stride=None,
        padding=0,
        output_padding=0,
        channels=8,
        conv_layers=3,
        batch_norm=True,
        device=None,
    ):
        super().__init__()

        if kernel_stride is None:
            kernel_stride = {
                "kernel": 3,
                "stride": 1,
            }

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(
                128,
                final_side_length
                * final_side_length
                * channels
                * 2 ** (conv_layers - 1),
            ),
            nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(
                channels * 2 ** (conv_layers - 1),
                final_side_length,
                final_side_length,
            ),
        )

        # Make an OrderedDict to feed into nn.Sequential containing the deconvolutional layers
        deconv_layers_Nto2 = OrderedDict()
        conv_layers_list = list(range(conv_layers - 1))
        for i in conv_layers_list:
            n_channels = channels * 2 ** (conv_layers - i - 1)
            deconv_layers_Nto2["deconv" + str(conv_layers - i)] = nn.ConvTranspose2d(
                n_channels,
                n_channels // 2,
                kernel_size=kernel_stride["kernel"],
                stride=kernel_stride["stride"],
                padding=padding,
                output_padding=int(output_padding[conv_layers - i - 1]),
            )

            # parametrize the -1 if you want to change b-layer
            # batch_norm is np.bool_ type, "is True" fails
            if batch_norm and i == conv_layers_list[-1]:
                deconv_layers_Nto2["batch" + str(conv_layers - i)] = nn.BatchNorm2d(
                    n_channels // 2
                )
            deconv_layers_Nto2["relu" + str(conv_layers - i)] = nn.ReLU(True)

        self.decoder_convNto2 = nn.Sequential(deconv_layers_Nto2)

        if kernel_stride["stride"] == 1:
            opadding = 0
        elif kernel_stride["stride"] == 2:
            opadding = output_padding[0]

        self.decoder_end = nn.ConvTranspose2d(
            channels,
            1,
            kernel_size=kernel_stride["kernel"],
            stride=kernel_stride["stride"],
            padding=padding,
            output_padding=opadding,
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_convNto2(x)
        x = self.decoder_end(x)
        x = torch.sigmoid(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dims,
        resolution_hw,
        ksp_key=None,
        channels=8,
        conv_layers=3,
        batch_norm=True,
        latent_distribution="normal",
        device=None,
    ):
        super().__init__()

        # Assert that conv_layers is an integer between 1 and 5
        assert isinstance(
            conv_layers, int
        ), "conv_layers must be an integer, aborting..."
        assert conv_layers >= 1, "conv_layers must be >= 1, aborting..."
        assert conv_layers <= 5, "conv_layers must be <= 5, aborting..."
        if conv_layers > 3:
            assert (
                "s2" not in ksp_key
            ), "stride 2 only supports conv_layers <= 3, check kernel_stride, aborting..."

        # Define the range of valid values for latent_dims
        min_dim = 2
        max_dim = 128

        # Check if latent_dims is a power of 2 bwteen min_dim and max_dim
        assert (
            (latent_dims & (latent_dims - 1) == 0)
            and (latent_dims >= min_dim)
            and (latent_dims <= max_dim)
        ), "Latent_dims must be a power of 2 between 2 and 128, aborting..."

        self._set_ksp_key()
        kernel_stride = self.kernel_stride_keys[ksp_key]
        padding = kernel_stride["kernel"] // 2

        # Get final encoder dimension and output padding for each layer
        final_side_length, output_padding = self._get_final_side_length(
            conv_layers, resolution_hw, kernel_stride, padding
        )

        self.device = device
        self.encoder = VariationalEncoder(
            latent_dims=latent_dims,
            final_side_length=final_side_length,
            kernel_stride=kernel_stride,
            padding=padding,
            channels=channels,
            conv_layers=conv_layers,
            batch_norm=batch_norm,
            latent_distribution=latent_distribution,
            device=self.device,
        )
        self.decoder = Decoder(
            latent_dims,
            final_side_length,
            kernel_stride,
            padding=padding,
            output_padding=output_padding,
            channels=channels,
            conv_layers=conv_layers,
            batch_norm=batch_norm,
            device=self.device,
        )

        # Consider moving for not to unnecessarily print the kid model
        self.mse = MeanSquaredError()

        # Allowed n_features: 64, 192, 768, 2048
        self.kid = KernelInceptionDistance(
            feature=2048,
            reset_real_features=False,
            normalize=True,
            subset_size=16,
        )
        self.ssim = StructuralSimilarityIndexMeasure()

    def _get_final_side_length(
        self, n_conv_layers, resolution_hw, kernel_stride, padding
    ):
        """
        Get the final side length of the image after encoding and decoding.

        Parameters
        ----------
        resolution_hw : int
            height and width of the image
        kernel_stride : dict
            kernel and stride values
        padding : int
            padding value
        conv_layers : int
            number of convolutional layers

        Returns
        -------
        final_side_length : int
            final side length of the image after encoding
        output_padding : int
            output padding values for the deconvolutional layers
        """

        input_size = resolution_hw
        kernel_size = kernel_stride["kernel"]
        stride = kernel_stride["stride"]
        output_padding = np.zeros(n_conv_layers, dtype=int)
        if kernel_stride["stride"] == 1:
            final_side_length = resolution_hw
            # All zeros output_padding is correct when stride is 1
        elif kernel_stride["stride"] == 2:
            for this_layer in range(n_conv_layers):
                this_output = ((input_size - kernel_size + (2 * padding)) // stride) + 1
                a = int((input_size + 2 * padding - kernel_size) % stride)
                inverted_size = (
                    (this_output - 1) * stride + kernel_size - (2 * padding) + a
                )
                print(
                    f"For layer {this_layer}, {input_size=}, {this_output=}, {a=}, {inverted_size=}"
                )
                output_padding[this_layer] = a
                input_size = this_output
            final_side_length = this_output

        return final_side_length, output_padding

    def forward(self, x):
        if self.device is not None:
            x = x.to(self.device)
        z = self.encoder(x)

        return self.decoder(z)

    def _set_ksp_key(self):
        """
        Preset conv2D kernel, stride and padding values for kernel 3 and 5 and for
        reduction (28*28 => 3*3) and preservation (28*28 => 28*28) of representation size.
        """

        self.kernel_stride_keys = {
            "k3s2": {
                "kernel": 3,
                "stride": 2,
            },
            "k5s2": {
                "kernel": 5,
                "stride": 2,
            },
            "k3s1": {
                "kernel": 3,
                "stride": 1,
            },
            "k5s1": {
                "kernel": 5,
                "stride": 1,
            },
            "k7s1": {
                "kernel": 7,
                "stride": 1,
            },
            "k9s1": {
                "kernel": 9,
                "stride": 1,
            },
        }


class TrainableVAE(tune.Trainable):
    """
    Tune will convert this class into a Ray actor, which runs on a separate process.
    By default, Tune will also change the current working directory of this process to
    its corresponding trial-level log directory self.logdir. This is designed so that
    different trials that run on the same physical node wont accidently write to the same
    location and overstep each other

    Generally you only need to implement setup, step, save_checkpoint, and load_checkpoint when subclassing Trainable.

    Accessing config through Trainable.setup

    Return metrics from Trainable.step

    https://docs.ray.io/en/latest/tune/api_docs/trainable.html#tune-trainable-class-api
    """

    def setup(
        self,
        config,
        data_dict=None,
        device=None,
        methods=None,
        fixed_params=None,
    ):
        # Assert that none of the optional arguments are None
        assert data_dict is not None, "data_dict is None, aborting..."
        assert device is not None, "device is None, aborting..."
        assert methods is not None, "methods is None, aborting..."

        self.train_data = data_dict["train_data"]
        self.train_labels = data_dict["train_labels"]
        self.val_data = data_dict["val_data"]
        self.val_labels = data_dict["val_labels"]
        self.test_data = data_dict["test_data"]
        self.test_labels = data_dict["test_labels"]

        # Augment training and validation data.
        augmentation_dict = {
            "rotation": config.get("rotation"),
            "translation": (
                config.get("translation"),
                config.get("translation"),
            ),
            "noise": config.get("noise"),
            "flip": config.get("flip"),
            "data_multiplier": config.get("data_multiplier"),
        }

        self._augment_and_get_dataloader = methods["_augment_and_get_dataloader"]

        self.train_loader = self._augment_and_get_dataloader(
            data_type="train",
            augmentation_dict=augmentation_dict,
            batch_size=config.get("batch_size"),
            shuffle=True,
        )

        self.val_loader = self._augment_and_get_dataloader(
            data_type="val",
            augmentation_dict=augmentation_dict,
            batch_size=config.get("batch_size"),
            shuffle=True,
        )

        self.device = device
        self._train_epoch = methods["_train_epoch"]
        self._validate_epoch = methods["_validate_epoch"]

        self.model = VariationalAutoencoder(
            latent_dims=config.get("latent_dim"),
            resolution_hw=config.get("resolution_hw"),
            ksp_key=config.get("kernel_stride"),
            channels=config.get("channels"),
            conv_layers=config.get("conv_layers"),
            batch_norm=config.get("batch_norm"),
            latent_distribution=config.get("latent_distribution"),
            device=self.device,
        )

        # Will be saved with checkpoint model
        self.model.augmentation_dict = augmentation_dict

        self.model.to(self.device)

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=config.get("lr"), weight_decay=1e-5
        )
        # Define the scheduler with a step size and gamma factor
        self.scheduler = lr_scheduler.StepLR(
            self.optim,
            step_size=fixed_params["lr_step_size"],
            gamma=fixed_params["lr_gamma"],
        )

    def step(self):
        train_loss = self._train_epoch(
            self.model, self.device, self.train_loader, self.optim, self.scheduler
        )

        (
            val_loss_epoch,
            mse_epoch,
            ssim_epoch,
            kid_mean_epoch,
            kid_std_epoch,
        ) = self._validate_epoch(self.model, self.device, self.val_loader)

        # Convert to float, del & empty cache to free GPU memory
        train_loss_out = float(train_loss)
        val_loss_out = float(val_loss_epoch)
        mse_out = float(mse_epoch)
        ssim_out = float(ssim_epoch)
        kid_mean_out = float(kid_mean_epoch)
        kid_std_out = float(kid_std_epoch)

        del (
            train_loss,
            val_loss_epoch,
            mse_epoch,
            ssim_epoch,
            kid_mean_epoch,
            kid_std_epoch,
        )
        torch.cuda.empty_cache()

        return {
            "iteration": self.iteration
            + 1,  # Do not remove, plus one for 0=>1 indexing
            "train_loss": train_loss_out,
            "val_loss": val_loss_out,
            "mse": mse_out,
            "ssim": ssim_out,
            "kid_mean": kid_mean_out,
            "kid_std": kid_std_out,
        }

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        # torch.save(self.model.state_dict(), checkpoint_path)
        torch.save(self.model, checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        # self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        self.model = torch.load(checkpoint_path, weights_only=False)


class RetinaVAE(RetinaMath):
    """
    Class to apply variational autoencoder to Apricot retina data and run single learning run
    or Ray[Tune] hyperparameter search.

    Refereces for validation metrics:
    FID : Heusel_2017_NIPS
    KID : Binkowski_2018_ICLR
    SSIM : Wang_2009_IEEESignProcMag, Wang_2004_IEEETransImProc
    """

    def __init__(self, context, training_mode) -> None:
        # Dependency injection at ProjectManager construction
        self._context = context
        self.training_mode = training_mode

    @property
    def context(self):
        return self._context

    def build(
        self,
        gc_type,
        response_type,
        save_tuned_models=False,
    ):

        self.apricot_metadata_parameters = self.context.apricot_metadata_parameters
        self.gc_type = gc_type
        self.response_type = response_type

        # Fixed values for both single training and ray tune runs
        self.epochs = 5
        self.lr_step_size = 20  # Learning rate decay step size (in epochs)
        self.lr_gamma = 0.9  # Learning rate decay (multiplier for learning rate)
        # how many times to get the data, applied only if augmentation_dict is not None
        self.resolution_hw = 13  # Both x and y. Images will be sampled to this space.

        # For ray tune only
        # If grid_search is True, time_budget and grace_period are ignored
        self.grid_search = True  # False for tune by Optuna, True for grid search
        self.time_budget = 60 * 60 * 24 * 4  # in seconds
        self.grace_period = 50  # epochs. ASHA stops earliest at grace period.

        #######################
        # Single run parameters
        #######################
        # Set common VAE model parameters
        self.latent_dim = 32  # 32  # 2**1 - 2**6, use powers of 2 btw 2 and 128
        self.channels = 16
        # lr will be reduced by scheduler down to lr * gamma ** (epochs/step_size)
        self.lr = 0.0005
        # self._show_lr_decay(self.lr, self.lr_gamma, self.lr_step_size, self.epochs)

        self.batch_size = 256  # None will take the batch size from test_split size.
        self.test_split = 0.2  # Split data for validation and testing (both will take this fraction of data)

        self.kernel_stride = "k7s1"  # "k3s1", "k3s2" # "k5s2" # "k5s1"
        self.conv_layers = 2  # 1 - 5 for s1, 1 - 3 for k3s2 and k5s2
        self.batch_norm = True
        self.latent_distribution = "uniform"  # "normal" or "uniform"

        # Augment training and validation data.
        augmentation_dict = {
            "rotation": 0,  # rotation in degrees
            "translation": (
                0,  # 0.07692307692307693,
                0,  # 0.07692307692307693,
            ),  # fraction of image, (x, y) -directions
            "noise": 0,  # 0.005,  # noise float in [0, 1] (noise is added to the image)
            "flip": 0.5,  # flip probability, both horizontal and vertical
            "data_multiplier": 4,  # how many times to get the data w/ augmentation
        }
        self.augmentation_dict = augmentation_dict  # None

        ####################
        # Utility parameters
        ####################
        self.train_by = [[gc_type], [response_type]]  # Train by these factors

        # Set the random seed for reproducible results for both torch and numpy
        self.random_seed = self.context.numpy_seed
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.latent_space_plot_scale = 15.0  # Scale for plotting latent space

        self.models_folder = self._set_models_folder(self.context)
        self.train_log_folder = self.models_folder / "train_logs"

        self.dependent_variables = [
            "train_loss",
            "val_loss",
            "mse",
            "ssim",
            "kid_std",
            "kid_mean",
        ]

        self.device = self.context.device

        # torch.serialization.add_safe_globals([VariationalAutoencoder])
        # torch.serialization.add_safe_globals([retina.vae_module.VariationalAutoencoder])
        self._get_and_split_apricot_data()

        match self.training_mode:
            case "train_model":

                self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Create datasets and dataloaders

                self.train_loader = self._augment_and_get_dataloader(
                    data_type="train",
                    augmentation_dict=self.augmentation_dict,
                    batch_size=self.batch_size,
                    shuffle=True,
                )
                self.val_loader = self._augment_and_get_dataloader(
                    data_type="val",
                    augmentation_dict=self.augmentation_dict,
                    batch_size=self.batch_size,
                    shuffle=True,
                )
                # Create model and set optimizer and learning rate scheduler
                self._prep_training()

                # # Init tensorboard. This cleans up the folder.
                # self._prep_tensorboard_logging()
                # Init logging.
                self._prep_logging()

                # Train
                self._train()

                self._save_logging()

                # Save model
                model_path = self._save_model()
                summary(
                    self.vae,
                    input_size=(1, self.resolution_hw, self.resolution_hw),
                    batch_size=-1,
                )

            case "tune_model":
                # self._get_and_split_apricot_data()
                self.ray_dir = self._set_ray_folder(self.context)

                # This will be captured at _set_ray_tuner
                # Search space of the tuning job. Both preprocessor and dataset can be tuned here.
                # Use grid search to try out all values for each parameter. values: iterable
                # Note that initial_params under _set_ray_tuner MUST be included in the search space.
                # Grid search: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#ray.tune.grid_search
                # Sampling: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs
                self.search_space = {
                    "lr": [0.0005],
                    "latent_dim": [32],  # 32 best # 4, 8, 16, 32 search space
                    "resolution_hw": [13],  # Both x and y, 13 or 28
                    # k3s2,k3s1,k5s2,k5s1,k7s1, k9s1 Kernel-stride-padding for conv layers. NOTE you cannot use >3 conv layers with stride 2
                    "kernel_stride": ["k7s1"],  # "k3s1", "k5s1", "k7s1", "k9s1"
                    "channels": [16],  # 4, 8, 16, 32
                    "batch_size": [256],
                    "conv_layers": [2],  # 1, 2, 3, 4
                    "batch_norm": [True],  # False, True
                    "latent_distribution": ["uniform"],  # "normal", "uniform"
                    "rotation": [0],  # Augment: max rotation in degrees
                    # Augment: fract of im, max in (x, y)/[xy] dir
                    "translation": [0],  # 1/13 pixels
                    # Augment: noise added, btw [0., 1.]
                    "noise": [0],  # 0, 0.025, 0.05, 0.1
                    "flip": [0.5],  # Augment: flip prob, both horiz and vert
                    "data_multiplier": [4],  # N times to get the data w/ augmentation
                    "num_models": 8,  # repetitions of the same model
                }

                # The first metric is the one that will be used to prioritize the checkpoints and pruning.
                self.multi_objective = {
                    "metric": ["val_loss"],
                    "mode": ["min"],
                }

                # Fraction of GPU per trial. 0.25 for smaller models is enough. Larger may need 0.33 or 0.5.
                # Increase if you get CUDA out of memory errors.
                self.gpu_fraction = 0.5

                self.disk_usage_threshold = 90  # %, stops training if exceeded

                # Save tuned models. > 100 MB / model.
                self.save_tuned_models = save_tuned_models

                tuner = self._set_ray_tuner(grid_search=self.grid_search)
                self.result_grid = tuner.fit()

                results_df = self.result_grid.get_dataframe()
                print(
                    "Shortest training time:",
                    results_df["time_total_s"].min(),
                    "for config:",
                    results_df[
                        results_df["time_total_s"] == results_df["time_total_s"].min()
                    ].index.values,
                )
                print(
                    "Longest training time:",
                    results_df["time_total_s"].max(),
                    "for config:",
                    results_df[
                        results_df["time_total_s"] == results_df["time_total_s"].max()
                    ].index.values,
                )

                self.best_result = self.result_grid.get_best_result(
                    metric="val_loss", mode="min"
                )
                print("Best result:", self.best_result)
                result_df = self.best_result.metrics_dataframe
                result_df[["training_iteration", "val_loss", "time_total_s"]]

                # Load model state dict from checkpoint to new self.vae and return the state dict.
                self.vae = self._load_model(best_result=self.best_result)

                self._update_retinavae_to_ray_result(self.best_result)

                # Give one second to write the checkpoint to disk
                time.sleep(1)

            case "load_model":
                # Load previously calculated model for vizualization
                # Load model to self.vae

                if (
                    self.context.retina_parameters["ray_tune_trial_id"] is not None
                ):  # After tune_model
                    self.ray_dir = self._set_ray_folder(self.context)
                    trial_name = self.context.retina_parameters["ray_tune_trial_id"]
                    self.vae, result_grid, trial_folder = self._load_model(
                        trial_name=trial_name
                    )

                    [this_result] = [
                        result
                        for result in result_grid
                        if trial_name in result.metrics["trial_id"]
                    ]
                    self._update_retinavae_to_ray_result(this_result)

                elif (
                    self.context.retina_parameters["ray_tune_trial_id"] is None
                ):  # After train_model
                    if self.context.retina_parameters["model_file_name"] is None:
                        self.vae = self._load_model(model_path=self.models_folder)
                        self._load_logging()
                    else:
                        model_file_name = self.context.retina_parameters[
                            "model_file_name"
                        ]
                        self._validate_model_file_name(model_file_name)
                        model_path_full = self.models_folder / model_file_name
                        self.vae = self._load_model(model_path=model_path_full)
                        self._load_logging(model_file_name=model_file_name)
                    # Get datasets for RF generation and vizualization
                    # Original augmentation and data multiplication is applied to train and val ds
                    self.train_loader = self._augment_and_get_dataloader(
                        data_type="train", augmentation_dict=self.vae.augmentation_dict
                    )
                    self.val_loader = self._augment_and_get_dataloader(
                        data_type="val", augmentation_dict=self.vae.augmentation_dict
                    )

                else:
                    raise ValueError(
                        "No output path (models_folder) or trial name given, cannot load model, aborting..."
                    )

                summary(
                    self.vae.to(self.device),
                    input_size=(1, self.resolution_hw, self.resolution_hw),
                    batch_size=-1,
                )

        # This attaches test data to the model.
        self.test_loader = self._augment_and_get_dataloader(
            data_type="test", shuffle=False
        )

    def _validate_model_file_name(self, model_file_name):
        assert (
            self.gc_type in model_file_name
        ), "gc_type does not match model_file_name, aborting..."
        assert (
            self.response_type in model_file_name
        ), "response_type not in model_file_name, aborting..."

    def _show_lr_decay(self, lr, gamma, step_size, epochs):
        lrs = np.zeros(epochs)
        for this_epoch in range(epochs):
            lrs[this_epoch] = lr * gamma ** np.floor(this_epoch / step_size)
        plt.plot(lrs)
        plt.show()
        exit()

    def _update_retinavae_to_ray_result(self, this_result):
        """
        Update the VAE to match the one model found by ray tune.
        """

        attributes_to_update = {
            "latent_dim": "latent_dim",
            "channels": "channels",
            "lr": "lr",
            "latent_distribution": "latent_distribution",
            "batch_size": "batch_size",
            "kernel_stride": "kernel_stride",
            "conv_layers": "conv_layers",
            "batch_norm": "batch_norm",
        }

        augmentation_keys = [
            "rotation",
            "translation",
            "noise",
            "flip",
            "data_multiplier",
        ]

        for key in augmentation_keys:
            try:
                self.augmentation_dict[key] = this_result.config[key]
            except KeyError:
                print(
                    f"WARNING: Key '{key}' is missing in augmentation_dict and will not be updated"
                )

        for attr_name, config_key in attributes_to_update.items():
            try:
                setattr(self, attr_name, this_result.config[config_key])
            except KeyError:
                print(f"WARNING: Key '{config_key}' is missing and will not be updated")

        self.train_loader = self._augment_and_get_dataloader(
            data_type="train",
            augmentation_dict=self.augmentation_dict,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_loader = self._augment_and_get_dataloader(
            data_type="val",
            augmentation_dict=self.augmentation_dict,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def _run_tensorboard(self, tb_dir):
        """Run tensorboard in a new subprocess"""
        subprocess.run(
            [
                "tensorboard",
                "--logdir",
                tb_dir,
                "--host",
                "localhost",
                "--port",
                "6006",
            ]
        )

    def _set_ray_tuner(self, grid_search=True):
        """Set ray tuner"""

        # List of strings from the self.search_space dictionary which should be reported.
        # Include only the parameters which have more than one item listed in the search space.
        parameters_to_report = []
        for key, value in self.search_space.items():
            if key == "num_models":
                continue
            if len(value) > 1:
                parameters_to_report.append(key)

        print(f"parameters_to_report: {parameters_to_report}")
        reporter = CLIReporter(
            metric_columns=[
                "time_total_s",
                "iteration",
                "train_loss",
                "val_loss",
                "mse",
                "ssim",
                "kid_mean",
                "kid_std",
            ],
            parameter_columns=parameters_to_report,
        )

        trainable = tune.with_resources(TrainableVAE, {"gpu": self.gpu_fraction})
        trainable_with_parameters = tune.with_parameters(
            trainable,
            data_dict={
                "train_data": self.train_data,
                "train_labels": self.train_labels,
                "val_data": self.val_data,
                "val_labels": self.val_labels,
                "test_data": self.test_data,  # For later evaluation and viz
                "test_labels": self.test_labels,
            },
            device=self.device,
            methods={
                "_train_epoch": self._train_epoch,
                "_validate_epoch": self._validate_epoch,
                "_augment_and_get_dataloader": self._augment_and_get_dataloader,
            },
            fixed_params={
                "lr_step_size": self.lr_step_size,
                "lr_gamma": self.lr_gamma,
            },
        )

        if grid_search:
            param_space = {
                "lr": tune.grid_search(self.search_space["lr"]),
                "latent_dim": tune.grid_search(self.search_space["latent_dim"]),
                "resolution_hw": tune.grid_search(self.search_space["resolution_hw"]),
                "kernel_stride": tune.grid_search(self.search_space["kernel_stride"]),
                "channels": tune.grid_search(self.search_space["channels"]),
                "batch_size": tune.grid_search(self.search_space["batch_size"]),
                "conv_layers": tune.grid_search(self.search_space["conv_layers"]),
                "batch_norm": tune.grid_search(self.search_space["batch_norm"]),
                "latent_distribution": tune.grid_search(
                    self.search_space["latent_distribution"]
                ),
                "rotation": tune.grid_search(self.search_space["rotation"]),
                "translation": tune.grid_search(self.search_space["translation"]),
                "noise": tune.grid_search(self.search_space["noise"]),
                "flip": tune.grid_search(self.search_space["flip"]),
                "data_multiplier": tune.grid_search(
                    self.search_space["data_multiplier"]
                ),
                "model_id": tune.grid_search(
                    [
                        "model_{}".format(i)
                        for i in range(self.search_space["num_models"])
                    ]
                ),
            }

            # Efficient hyperparameter selection. Search Algorithms are wrappers around open-source
            # optimization libraries. Each library has a
            # specific way of defining the search space.
            # https://docs.ray.io/en/latest/ray-air/package-ref.html#ray.tune.tune_config.TuneConfig
            tune_config = tune.TuneConfig(
                search_alg=tune.search.basic_variant.BasicVariantGenerator(
                    constant_grid_search=True,
                ),
            )
        else:
            # Note that the initial parameters must be included in the search space
            initial_params = [
                {
                    "lr": 0.001,
                    "latent_dim": 16,
                    "kernel_stride": "k7s1",
                    "channels": 16,
                    "batch_size": 128,
                    "conv_layers": 2,
                    "batch_norm": False,
                    "rotation": 0,
                    "translation": 0,
                    "noise": 0.0,
                    "flip": 0.5,
                    "data_multiplier": 4,
                    "model_id": "model_0",
                }
            ]

            # tune (log)uniform etc require two positional arguments, so we need to unpack the list
            param_space = {
                "lr": tune.loguniform(
                    self.search_space["lr"][0], self.search_space["lr"][-1]
                ),
                "latent_dim": tune.choice(self.search_space["latent_dim"]),
                "kernel_stride": tune.choice(self.search_space["kernel_stride"]),
                "channels": tune.choice(self.search_space["channels"]),
                "batch_size": tune.choice(self.search_space["batch_size"]),
                "conv_layers": tune.choice(self.search_space["conv_layers"]),
                "batch_norm": tune.choice(self.search_space["batch_norm"]),
                "latent_distribution": tune.choice(
                    self.search_space["latent_distribution"]
                ),
                "rotation": tune.uniform(
                    self.search_space["rotation"][0], self.search_space["rotation"][-1]
                ),
                "translation": tune.uniform(
                    self.search_space["translation"][0],
                    self.search_space["translation"][-1],
                ),
                "noise": tune.uniform(
                    self.search_space["noise"][0], self.search_space["noise"][-1]
                ),
                "flip": tune.uniform(
                    self.search_space["flip"][0], self.search_space["flip"][-1]
                ),
                "data_multiplier": tune.choice(self.search_space["data_multiplier"]),
                "model_id": tune.choice(
                    [
                        "model_{}".format(i)
                        for i in range(self.search_space["num_models"])
                    ]
                ),
            }

            # Efficient hyperparameter selection. Search Algorithms are wrappers around open-source
            # optimization libraries. Each library has a specific way of defining the search space.
            # https://docs.ray.io/en/latest/ray-air/package-ref.html#ray.tune.tune_config.TuneConfig
            tune_config = tune.TuneConfig(
                # Local optuna search will generate study name "optuna" indicating in-memory storage
                search_alg=OptunaSearch(
                    sampler=TPESampler(),
                    metric=self.multi_objective["metric"],
                    mode=self.multi_objective["mode"],
                    points_to_evaluate=initial_params,
                ),
                scheduler=ASHAScheduler(
                    time_attr="training_iteration",
                    metric=self.multi_objective["metric"][
                        0
                    ],  # Only 1st metric used for pruning
                    mode=self.multi_objective["mode"][0],
                    max_t=self.epochs,
                    grace_period=self.grace_period,
                    reduction_factor=2,
                ),
                time_budget_s=self.time_budget,
                num_samples=-1,
            )

        # Runtime configuration that is specific to individual trials. Will overwrite the run config passed to the
        # Trainer. for API, see https://docs.ray.io/en/latest/ray-air/package-ref.html#ray.air.config.RunConfig

        if self.save_tuned_models is True:
            hard_disk_watchdog = [
                HardDiskWatchDog(
                    self.ray_dir, disk_usage_threshold=self.disk_usage_threshold
                )
            ]
        else:
            hard_disk_watchdog = None

        run_config = (
            air.RunConfig(
                stop={"training_iteration": self.epochs},
                progress_reporter=reporter,
                local_dir=self.ray_dir,
                callbacks=hard_disk_watchdog,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_score_attribute=self.multi_objective["metric"][0],
                    checkpoint_score_order=self.multi_objective["mode"][0],
                    num_to_keep=1,
                    checkpoint_at_end=self.save_tuned_models,
                    checkpoint_frequency=0,
                ),
                verbose=1,
            ),
        )

        tuner = tune.Tuner(
            trainable_with_parameters,
            param_space=param_space,
            run_config=run_config[0],
            tune_config=tune_config,
        )

        return tuner

    def _set_models_folder(self, context=None):
        """Set the folder where models are saved"""

        # If output_folder is Path instance or string, use it as models_folder
        if isinstance(self.context.input_folder, Path) or isinstance(
            self.context.input_folder, str
        ):
            models_folder = self.context.input_folder
        else:
            models_folder = self.context.output_folder / "models"
        Path(models_folder).mkdir(parents=True, exist_ok=True)

        return models_folder

    def _set_ray_folder(self, context):
        """Set the folder where ray tune results are saved"""

        if isinstance(self.context.input_folder, Path) or isinstance(
            self.context.input_folder, str
        ):
            ray_dir = self.context.input_folder / "ray_results"
        else:
            ray_dir = self.context.output_folder / "ray_results"
        Path(ray_dir).mkdir(parents=True, exist_ok=True)

        return ray_dir

    def _save_model(self):
        """
        Save model for single trial, a.k.a. 'train_model' training_mode
        """
        filename = f"model_{self.gc_type}_{self.response_type}_{self.device}_{self.timestamp}.pt"
        model_path = f"{self.models_folder}/{filename}"
        # Create models folder if it does not exist using pathlib

        # Get key VAE structural parameters and save them with the full model
        # This is only partial config data, but only latent_dims are needed for loading
        self.vae.config = {
            "latent_dims": self.latent_dim,
        }
        print(f"Saving model to {model_path}")
        # torch.save(self.vae.state_dict(), model_path)
        torch.save(self.vae, model_path)
        # model_scripted = torch.jit.script(self.vae)  # Export to TorchScript
        # model_scripted.save(model_path)  # Save

        return model_path

    def _load_model(self, model_path=None, best_result=None, trial_name=None):
        """Load model if exists. Use either model_path, best_result, or trial_name to load model"""

        if best_result is not None:
            # ref https://medium.com/distributed-computing-with-ray/simple-end-to-end-ml-from-selection-to-serving-with-ray-tune-and-ray-serve-10f5564d33ba
            log_dir = best_result.log_dir
            checkpoint_dir = [
                d for d in os.listdir(log_dir) if d.startswith("checkpoint")
            ][0]
            checkpoint_path = os.path.join(log_dir, checkpoint_dir, "model.pth")

            vae_model = torch.load(checkpoint_path, weights_only=False)

        elif model_path is not None:
            model_path = Path(model_path)
            if Path.exists(model_path) and model_path.is_file():
                print(
                    f"Loading model from {model_path}. \nWARNING: This will replace the current model in-place."
                )
                vae_model = torch.load(model_path, weights_only=False)
            elif Path.exists(model_path) and model_path.is_dir():
                try:
                    prefix = f"model_{self.gc_type}_{self.response_type}_{self.device}_"
                    model_path = max(Path(self.models_folder).glob(f"{prefix}*.pt"))
                    time_stamp = "_".join(model_path.stem.split("_")[-2:])
                    self.timestamp_for_loading = time_stamp
                    if not model_path.is_file():
                        raise FileNotFoundError("VAE model not found, aborting...")
                    try:
                        vae_model = torch.load(model_path, weights_only=False)
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError(
                            "Module path changed, you need to train VAE model. Aborting..."
                        )
                    print(f"Most recent model is {model_path}.")
                except ValueError:
                    raise FileNotFoundError("No model files found. Aborting...")

            else:
                print(f"Model {model_path} does not exist.")

        elif trial_name is not None:
            # trial_name = "TrainableVAE_XXX" from ray.tune results table.
            # Search under self.ray_dir for folder with the trial name. Under that folder,
            # there should be a checkpoint folder which contains the model.pth file.
            try:
                correct_trial_folder = [
                    p for p in Path(self.ray_dir).glob(f"**/") if trial_name in p.stem
                ][0]
            except IndexError:
                raise FileNotFoundError(
                    f"Could not find trial with name {trial_name}. Aborting..."
                )
            # However, we need to first check that the model dimensionality is correct.
            # This will be hard coded for checking and changing only latent_dim.
            # More versatile version is necessary if other dimensions are searched.

            # Get the results as dataframe from the ray directory / correct run
            results_folder = correct_trial_folder.parents[0]
            tuner = tune.Tuner.restore(str(results_folder))
            results = tuner.get_results()

            # Load the model from the checkpoint folder.
            try:
                checkpoint_folder_name = [
                    p for p in Path(correct_trial_folder).glob("checkpoint_*")
                ][0]
            except IndexError:
                raise FileNotFoundError(
                    f"Could not find checkpoint folder in {correct_trial_folder}. Aborting..."
                )
            model_path = Path.joinpath(checkpoint_folder_name, "model.pth")
            vae_model = torch.load(model_path, weights_only=False)

            # Move new model to same device as the input data
            vae_model.to(self.device)
            return vae_model, results, correct_trial_folder

        else:
            # Get the most recent model. Max recognizes the timestamp with the largest value
            try:
                model_path = max(Path(self.models_folder).glob("*.pt"))
                print(f"Most recent model is {model_path}.")
            except ValueError:
                raise FileNotFoundError("No model files found. Aborting...")
            vae_model = torch.load(model_path, weights_only=False)

        vae_model.to(self.device)

        return vae_model

    def _visualize_augmentation_and_exit(self):
        """
        Visualize the augmentation effects and exit
        """

        # Get numpy data
        data_np, labels_np, data_names2labels_dict = self._get_spatial_apricot_data()

        # Split to training, validation and testing
        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            data_np,
            labels_np,
            test_size=self.test_split,
            random_state=self.random_seed,
            stratify=labels_np,
        )

        # Augment training and validation data
        train_val_ds = AugmentedDataset(
            train_val_data,
            train_val_labels,
            self.resolution_hw,
            augmentation_dict=self.augmentation_dict,
        )

        # Do not augment test data
        test_ds = AugmentedDataset(
            test_data, test_labels, self.resolution_hw, augmentation_dict=None
        )

        # Split into train and validation
        train_ds, val_ds = random_split(
            train_val_ds,
            [
                int(np.round(len(train_val_ds) * (1 - self.test_split))),
                int(np.round(len(train_val_ds) * self.test_split)),
            ],
        )

        # Get n items for the three sets
        self.n_train = len(train_ds)
        self.n_val = len(val_ds)
        self.n_test = len(test_ds)

        # Make a figure with 2 rows and 5 columns, with upper row containing 5 original and the lower row 5 augmented images.
        # The original images are in the train_val_data (numpy array with dims (N x C x H x W)), and the augmented images are in the train_val_ds
        # (torch dataset with dims (N x C x H x W)). The labels are in train_val_labels (numpy array with dims (N, 1)).
        fig, axs = plt.subplots(2, 5, figsize=(10, 5))
        for i in range(5):
            axs[0, i].imshow(train_val_data[i, 0, :, :], cmap="gray")
            axs[0, i].axis("off")
            axs[1, i].imshow(train_val_ds[i][0][0, :, :], cmap="gray")
            axs[1, i].axis("off")

        # Set the labels as text upper left inside the images of the upper row
        for i in range(5):
            axs[0, i].text(
                0.05,
                0.85,
                self.apricot_data.data_labels2names_dict[train_val_labels[i][0]],
                fontsize=10,
                color="blue",
                transform=axs[0, i].transAxes,
            )

        # Set subtitle "Original" for the first row
        axs[0, 0].set_title("Original", fontsize=14)
        # Set subtitle "Augmented" for the second row
        axs[1, 0].set_title("Augmented", fontsize=14)

        plt.show()
        exit()

    def _get_and_split_apricot_data(self):
        """
        Load data
        Split into training, validation and testing
        """

        # Get numpy data
        data_np, labels_np, data_names2labels_dict = self._get_spatial_apricot_data()

        # Split to training+validation and testing
        (
            train_val_data_np,
            test_data_np,
            train_val_labels_np,
            test_labels_np,
        ) = train_test_split(
            data_np,
            labels_np,
            test_size=self.test_split,
            random_state=self.random_seed,
            stratify=labels_np,
        )

        # Split into train and validation
        train_data_np, val_data_np, train_labels_np, val_labels_np = train_test_split(
            train_val_data_np,
            train_val_labels_np,
            test_size=self.test_split,
            random_state=self.random_seed,
            stratify=train_val_labels_np,
        )

        # These are all numpy arrays
        self.train_data = train_data_np
        self.train_labels = train_labels_np
        self.val_data = val_data_np
        self.val_labels = val_labels_np
        self.test_data = test_data_np
        self.test_labels = test_labels_np

    def _augment_and_get_dataloader(
        self, data_type="train", augmentation_dict=None, batch_size=32, shuffle=True
    ):
        """
        Augmenting data
        Creating dataloaders

        Parameters:
            data_type (str): "train", "val" or "test"
            augmentation_dict (dict): augmentation dictionary
            batch_size (int): batch size
            shuffle (bool): shuffle data

        Returns:
            dataloader (torch.utils.data.DataLoader): dataloader
        """

        # Assert that data_type is "train", "val" or "test"
        assert data_type in [
            "train",
            "val",
            "test",
        ], "data_type must be 'train', 'val' or 'test', aborting..."

        # Assert that self. has attribute "train_data", "val_data" or "test_data"
        assert hasattr(self, data_type + "_data"), (
            "\nself has no attribute '" + data_type + "_data', aborting...\n"
        )

        data = getattr(self, data_type + "_data")
        labels = getattr(self, data_type + "_labels")

        # Augment training and validation data
        data_ds = AugmentedDataset(
            data,
            labels,
            self.resolution_hw,
            augmentation_dict=augmentation_dict,
        )

        # set self. attribute "n_train", "n_val" or "n_test"
        setattr(self, "n_" + data_type, len(data_ds))

        # set self. attribute "train_ds", "val_ds" or "test_ds"
        setattr(self, data_type + "_ds", data_ds)

        data_loader = DataLoader(data_ds, batch_size=batch_size, shuffle=shuffle)

        return data_loader

    def _get_spatial_apricot_data(self):
        """
        Get spatial ganglion cell data from file using the apricot_data method read_spatial_filter_data().
        All data is returned, the requested data is logged in the class attributes gc_type and response_type.

        Returns
        -------
        collated_gc_spatial_data_np : np.ndarray
            Spatial data with shape (n_gc, 1, ydim, xdim)
        collated_gc_spatial_labels_np : np.ndarray
            Labels with shape (n_gc, 1)
        data_names2labels_dict : dict
            Dictionary with gc names as keys and labels as values
        """

        # Get requested data
        self.apricot_data = ApricotData(
            self.apricot_metadata_parameters, self.gc_type, self.response_type
        )

        # Log requested label
        self.gc_label = self.apricot_data.data_names2labels_dict[
            f"{self.gc_type}_{self.response_type}"
        ]

        # We train by more data, however.
        # Build a list of combinations of gc types and response types from self.train_by
        train_by_combinations = [
            f"{gc}_{response}"
            for (gc, response) in product(self.train_by[0], self.train_by[1])
        ]

        response_labels = [
            self.apricot_data.data_names2labels_dict[key]
            for key in train_by_combinations
        ]

        # Log trained_by labels
        self.train_by_labels = response_labels

        # Get all available gc types and response types
        gc_types = [key[: key.find("_")] for key in train_by_combinations]
        response_types = [key[key.find("_") + 1 :] for key in train_by_combinations]

        # Initialise numpy arrays to store data
        collated_gc_spatial_data_np = np.empty(
            (
                0,
                1,
                self.apricot_data.metadata["data_spatialfilter_height"],
                self.apricot_data.metadata["data_spatialfilter_width"],
            )
        )
        collated_labels_np = np.empty((0, 1), dtype=int)

        # Get data for learning
        for gc_type, response_type, label in zip(
            gc_types, response_types, response_labels
        ):
            print(f"Loading data for {gc_type}_{response_type} (label {label})")
            apricot_data = ApricotData(
                self.apricot_metadata_parameters, gc_type, response_type
            )
            bad_data_idx = apricot_data.manually_picked_bad_data_idx
            (
                gc_spatial_data_np_orig,
                _,
            ) = apricot_data.read_spatial_filter_data()

            # Drop bad data
            gc_spatial_data_np = np.delete(
                gc_spatial_data_np_orig, bad_data_idx, axis=0
            )

            # Invert data arrays with negative sign for fitting and display.
            gc_spatial_data_np = self.flip_negative_spatial_rf(gc_spatial_data_np)

            gc_spatial_data_np = np.expand_dims(gc_spatial_data_np, axis=1)

            # Collate data
            collated_gc_spatial_data_np = np.concatenate(
                (collated_gc_spatial_data_np, gc_spatial_data_np), axis=0
            )
            labels = np.full((gc_spatial_data_np.shape[0], 1), label)
            collated_labels_np = np.concatenate((collated_labels_np, labels), axis=0)

        return (
            collated_gc_spatial_data_np,
            collated_labels_np,
            apricot_data.data_names2labels_dict,
        )

    def _prep_training(self):
        self.vae = VariationalAutoencoder(
            latent_dims=self.latent_dim,
            resolution_hw=self.resolution_hw,
            ksp_key=self.kernel_stride,
            channels=self.channels,
            conv_layers=self.conv_layers,
            batch_norm=self.batch_norm,
            latent_distribution=self.latent_distribution,
            device=self.device,
        )

        # Will be saved with model for later eval and viz
        self.vae.test_data = self.test_data
        self.vae.test_labels = self.test_labels
        self.vae.augmentation_dict = self.augmentation_dict

        self.optim = torch.optim.Adam(
            self.vae.parameters(), lr=self.lr, weight_decay=1e-5
        )

        # Define the scheduler with a step size and gamma factor
        self.scheduler = lr_scheduler.StepLR(
            self.optim, step_size=self.lr_step_size, gamma=self.lr_gamma
        )

        print(f"Selected device: {self.device}")
        self.vae.to(self.device)

    def _save_logging(self):
        """
        Save logging to train_log_folder
        """
        # Save log_df as csv
        self.log_df.to_csv(
            self.train_log_folder / f"train_log_{self.timestamp}.csv", index=False
        )

        # Save log_df as pickle
        self.log_df.to_pickle(self.train_log_folder / f"train_log_{self.timestamp}.pkl")

    def _load_logging(self, model_file_name=None):
        """
        Load logging from train_log_folder
        """

        if model_file_name is None:
            # This is set in _load_model, when there is no model_file_name
            time_stamp = self.timestamp_for_loading
        else:
            # Get the time stamp from the file name
            name_stem = model_file_name.split(".")[0]
            times = name_stem.split("_")[-2:]
            time_stamp = "_".join(times)

        # Get the most recent log file
        try:
            if time_stamp is not None:
                # log file name is of type train_log_[TIME_STAMP].csv
                log_file_name = f"train_log_{time_stamp}.csv"
                log_path = Path(self.train_log_folder) / log_file_name
                print(f"Loading log file from {log_path}.")
            else:
                log_path = max(Path(self.train_log_folder).glob("*.csv"))
                print(f"Most recent log file is {log_path}.")
        except ValueError:
            raise FileNotFoundError("No log files found. Aborting...")

        # Load the log file
        self.log_df = pd.read_csv(log_path)

    def _prep_logging(self):
        """
        Prepare logging
        """
        self.train_log_folder = Path(self.train_log_folder)

        # Create a folder for the experiment tensorboard logs
        Path.mkdir(self.train_log_folder, parents=True, exist_ok=True)

        # Make an empty dataframe for logging
        self.log_df = pd.DataFrame(
            columns=self.dependent_variables, index=range(self.epochs)
        )

    ### Training function
    def _train_epoch(self, vae, device, dataloader, optimizer, scheduler):
        # Set train mode for both the encoder and the decoder
        vae.train()
        train_loss = 0.0
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)

        for x, _ in dataloader:
            x = x.to(device)
            x_hat = vae(x)

            # Evaluate loss
            loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # # Print batch loss
            # print("\t partial train loss (single batch): %f" % (loss.item()))
            train_loss += loss.item()

        # Update the learning rate at the end of each epoch
        scheduler.step()

        train_loss_out = float(train_loss)
        del train_loss, loss, x, x_hat

        return train_loss_out / len(dataloader.dataset)

    ### Testing function
    def _validate_epoch(self, vae, device, dataloader):
        # Set evaluation mode for encoder and decoder
        vae.eval()
        val_loss = 0.0
        vae.mse.reset()
        vae.ssim.reset()
        vae.kid.reset()

        with torch.no_grad():  # No need to track the gradients
            for x, _ in dataloader:
                # Move tensor to the proper device
                x = x.to(device)
                x_hat = vae(x)
                loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl
                val_loss += loss.item()

                vae.mse.update(x_hat, x)  # MSE
                vae.ssim.update(x_hat, x)  # SSIM

                # Expand dim 1 to 3 for x and x_hat
                x_expanded = x.expand(-1, 3, -1, -1)
                x_hat_expanded = x_hat.expand(-1, 3, -1, -1)

                vae.kid.update(x_expanded, real=True)  # KID
                vae.kid.update(x_hat_expanded, real=False)  # KID

        n_samples = len(dataloader.dataset)
        val_loss_epoch = val_loss / n_samples
        mse_epoch = vae.mse.compute()
        ssim_epoch = vae.ssim.compute()
        kid_mean_epoch, kid_std_epoch = vae.kid.compute()

        # Test all output variables for type, and covert to value if needed
        if isinstance(val_loss_epoch, torch.Tensor):
            val_loss_epoch = val_loss_epoch.item()
        if isinstance(mse_epoch, torch.Tensor):
            mse_epoch = mse_epoch.item()
        if isinstance(ssim_epoch, torch.Tensor):
            ssim_epoch = ssim_epoch.item()
        if isinstance(kid_mean_epoch, torch.Tensor):
            kid_mean_epoch = kid_mean_epoch.item()
        if isinstance(kid_std_epoch, torch.Tensor):
            kid_std_epoch = kid_std_epoch.item()

        return (
            val_loss_epoch,
            mse_epoch,
            ssim_epoch,
            kid_mean_epoch,
            kid_std_epoch,
        )

    def _train(self):
        """
        Train for training_mode = train_model
        """
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(
                self.vae, self.device, self.train_loader, self.optim, self.scheduler
            )
            (
                val_loss_epoch,
                mse_epoch,
                ssim_epoch,
                kid_mean_epoch,
                kid_std_epoch,
            ) = self._validate_epoch(self.vae, self.device, self.val_loader)

            # For every 100th epoch, print the outputs of the autoencoder
            # if epoch == 0 or epoch % 100 == 0:
            print(
                f""" 
                EPOCH {epoch + 1}/{self.epochs} \t train_loss {train_loss:.3f} \t val loss {val_loss_epoch:.3f}
                mse {mse_epoch:.3f} \t ssim {ssim_epoch:.3f} \t kid mean {kid_mean_epoch:.3f} \t kid std {kid_std_epoch:.3f}
                Learning rate: {self.optim.param_groups[0]['lr']:.3e}
                """
            )

            # Convert to float, del & empty cache to free GPU memory
            train_loss_out = float(train_loss)
            val_loss_out = float(val_loss_epoch)
            mse_out = float(mse_epoch)
            ssim_out = float(ssim_epoch)
            kid_mean_out = float(kid_mean_epoch)
            kid_std_out = float(kid_std_epoch)

            del (
                train_loss,
                val_loss_epoch,
                mse_epoch,
                ssim_epoch,
                kid_mean_epoch,
                kid_std_epoch,
            )
            torch.cuda.empty_cache()

            # Add each metric to df separately
            self.log_df.loc[epoch, "train_loss"] = train_loss_out
            self.log_df.loc[epoch, "val_loss"] = val_loss_out
            self.log_df.loc[epoch, "mse"] = mse_out
            self.log_df.loc[epoch, "ssim"] = ssim_out
            self.log_df.loc[epoch, "kid_mean"] = kid_mean_out
            self.log_df.loc[epoch, "kid_std"] = kid_std_out

    def _show_image(self, img, latent=None):
        npimg = img.numpy()
        # Enc 0 as x-axis, 1 as y-axis
        npimg_transposed = np.transpose(npimg, (2, 1, 0))
        sidelength = int(npimg_transposed.shape[0] / 10)
        npimg_transposed = np.flip(npimg_transposed, 0)  # flip the image ud
        plt.imshow(npimg_transposed)
        plt.xticks([])
        plt.yticks([])
        if latent is not None:
            # Make x and y ticks from latent space (rows, cols) values
            x_ticks = np.linspace(latent[:, 1].min(), latent[:, 1].max(), 10)
            y_ticks = np.linspace(latent[:, 0].max(), latent[:, 0].min(), 10)
            # Limit both x and y ticks to 2 significant digits
            x_ticks = np.around(x_ticks, 2)
            y_ticks = np.around(y_ticks, 2)
            plt.xticks(
                np.arange(0 + sidelength / 2, sidelength * 10, sidelength), x_ticks
            )
            plt.yticks(
                np.arange(0 + sidelength / 2, sidelength * 10, sidelength), y_ticks
            )
            # X label and Y label
            plt.xlabel("EncVariable 0")
            plt.ylabel("EncVariable 1")

    def get_encoded_samples(self, ds_name=None, dataset=None):
        """Get encoded samples from a dataset.

        Parameters
        ----------
        ds_name : str, optional
            Dataset name

        Returns
        -------
        pd.DataFrame
            Encoded samples
        """

        # Assert that either ds_name or dataset is given
        assert (
            ds_name is not None or dataset is not None
        ), "Either ds_name or dataset must be given, aborting... "

        # After training, train and valid datasets contain augmentation
        if ds_name == "train_ds":
            ds = self.train_ds
        elif ds_name == "valid_ds":
            ds = self.valid_ds
        elif ds_name == "test_ds":
            ds = self.test_ds
        else:
            ds = dataset

        encoded_samples = []
        for sample in tqdm(ds):
            img = sample[0].unsqueeze(0).to(self.device)
            label = self.apricot_data.data_labels2names_dict[sample[1].item()]
            # Encode image
            self.vae.eval()
            with torch.no_grad():
                encoded_img = self.vae.encoder(img)
            # Append to list
            encoded_img = encoded_img.flatten().cpu().numpy()
            encoded_sample = {
                f"EncVariable {i}": enc for i, enc in enumerate(encoded_img)
            }
            encoded_sample["label"] = label
            encoded_samples.append(encoded_sample)

        encoded_samples = pd.DataFrame(encoded_samples)

        return encoded_samples

    def check_kid_and_exit(self):
        """
        Check KernelInceptionDistance between real and fake data and exit.
        """

        def kid_compare(dataloader_real, dataloader_fake, n_features=64):
            # Set evaluation mode for encoder and decoder
            kid = KernelInceptionDistance(
                feature=n_features,
                reset_real_features=True,
                normalize=True,
                subset_size=16,
            )

            kid.reset()
            kid.to(self.device)

            with torch.no_grad():  # No need to track the gradients
                # for x, _ in dataloader_real:
                for real_batch, fake_batch in zip(dataloader_real, dataloader_fake):
                    # Move tensor to the proper device
                    real_img_batch = real_batch[0].to(self.device)
                    fake_img_batch = fake_batch[0].to(self.device)
                    # Expand dim 1 to 3 for x and x_hat
                    real_img_batch_expanded = real_img_batch.expand(-1, 3, -1, -1)
                    fake_img_batch_hat_expanded = fake_img_batch.expand(-1, 3, -1, -1)

                    kid.update(real_img_batch_expanded, real=True)  # KID
                    kid.update(fake_img_batch_hat_expanded, real=False)  # KID

            kid_mean_epoch, kid_std_epoch = kid.compute()

            return kid_mean_epoch, kid_std_epoch

        dataloader_real = self._augment_and_get_dataloader(
            data_type="train",
            augmentation_dict=None,
            batch_size=self.batch_size,
            shuffle=True,
        )

        dataloader_fake = self._augment_and_get_dataloader(
            data_type="train",
            augmentation_dict=self.augmentation_dict,
            # augmentation_dict=None,
            batch_size=self.batch_size,
            shuffle=True,
        )

        kid_mean, kid_std = kid_compare(dataloader_real, dataloader_fake, n_features=64)
        print(f"KID mean: {kid_mean}, KID std: {kid_std} for 64 features")

        kid_mean, kid_std = kid_compare(
            dataloader_real, dataloader_fake, n_features=192
        )
        print(f"KID mean: {kid_mean}, KID std: {kid_std} for 192 features")

        kid_mean, kid_std = kid_compare(
            dataloader_real, dataloader_fake, n_features=768
        )
        print(f"KID mean: {kid_mean}, KID std: {kid_std} for 768 features")

        kid_mean, kid_std = kid_compare(
            dataloader_real, dataloader_fake, n_features="2048"
        )
        print(f"KID mean: {kid_mean}, KID std: {kid_std} for 2048 features")

        exit()
