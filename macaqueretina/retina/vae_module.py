# Built-in
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
import torch
import torch.optim.lr_scheduler as lr_scheduler
from scipy.ndimage import fourier_shift, rotate
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, StructuralSimilarityIndexMeasure
from torchmetrics.image.kid import KernelInceptionDistance
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

# Local
from macaqueretina.retina.experimental_data_module import ExperimentalData
from macaqueretina.retina.retina_math_module import RetinaMath


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Experimental dataset class for Pytorch.

    The constructor reads the data from the ExperimentalData class and stores it as
    tensors of shape (n_cells, channels, height, width). While the constructor
    is called with particular gc_type and response_type, all data is retrieved
    and thus the __getitem__ method can be called with any index. This enables
    teaching the network with all data. The gc_type and response_type are, however,
    logged into the ExperimentalDataset instance object.
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


class RetinaVAE(RetinaMath):
    """
    Class to apply variational autoencoder to experimental retina data and run single learning run.

    Refereces for validation metrics:
    FID : Heusel_2017_NIPS
    KID : Binkowski_2018_ICLR
    SSIM : Wang_2009_IEEESignProcMag, Wang_2004_IEEETransImProc
    """

    def __init__(self, config) -> None:

        self._config = config
        self.vae_run_mode = config.vae_train_parameters["vae_run_mode"]
        self.gc_type = config.retina_parameters["gc_type"]
        self.response_type = config.retina_parameters["response_type"]
        self.gc_response_types = [[self.gc_type], [self.response_type]]

        self.random_seed = self.config.numpy_seed
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

    @property
    def config(self):
        return self._config

    def client(
        self,
    ):

        self.experimental_metadata = self.config.experimental_metadata
        vae_train_parameters = self.config.vae_train_parameters
        self.epochs = vae_train_parameters["epochs"]
        self.lr_step_size = vae_train_parameters["lr_step_size"]
        self.lr_gamma = vae_train_parameters["lr_gamma"]
        self.resolution_hw = vae_train_parameters["resolution_hw"]

        self.latent_dim = vae_train_parameters["latent_dim"]
        self.channels = vae_train_parameters["channels"]
        self.lr = vae_train_parameters["lr"]
        self.batch_size = vae_train_parameters["batch_size"]
        self.test_split = vae_train_parameters["test_split"]
        self.kernel_stride = vae_train_parameters["kernel_stride"]
        self.conv_layers = vae_train_parameters["conv_layers"]
        self.batch_norm = vae_train_parameters["batch_norm"]
        self.latent_distribution = vae_train_parameters["latent_distribution"]
        self.augmentation_dict = vae_train_parameters["augmentation_dict"]

        ####################
        # Utility parameters
        ####################
        self.latent_space_plot_scale = 15.0
        self.models_folder = self._set_models_folder(self.config)
        self.train_log_folder = self.models_folder / "train_logs"
        self.dependent_variables = [
            "train_loss",
            "val_loss",
            "mse",
            "ssim",
            "kid_std",
            "kid_mean",
        ]
        self.device = self.config.device

        match self.vae_run_mode:

            case "load_model":
                model_file_name = self.config.retina_parameters.get(
                    "model_file_name", None
                )
                if model_file_name is None:
                    self.vae = self._load_model(model_path=self.models_folder)
                    self._load_logging()
                    self._load_latent_stats()
                else:
                    # model_file_name = self.config.retina_parameters["model_file_name"]
                    self._validate_model_file_name(model_file_name)
                    model_path_full = self.models_folder / model_file_name
                    self.vae = self._load_model(model_path=model_path_full)
                    self._load_logging(model_file_name=model_file_name)
                    self._load_latent_stats()

                summary(
                    self.vae.to(self.device),
                    input_size=(1, self.resolution_hw, self.resolution_hw),
                    batch_size=-1,
                )

            case "train_model":
                self.get_and_split_experimental_data()

                self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                self.train_loader = self.augment_and_get_dataloader(
                    data_type="train",
                    augmentation_dict=self.augmentation_dict,
                    batch_size=self.batch_size,
                    shuffle=True,
                )
                self.val_loader = self.augment_and_get_dataloader(
                    data_type="val",
                    augmentation_dict=self.augmentation_dict,
                    batch_size=self.batch_size,
                    shuffle=True,
                )

                self._prep_training()
                self._prep_logging()
                self._train()

                self.test_loader = self.augment_and_get_dataloader(
                    data_type="test", shuffle=False
                )
                model_path = self._save_model()
                self._save_logging()
                self._save_latent_stats()

                summary(
                    self.vae,
                    input_size=(1, self.resolution_hw, self.resolution_hw),
                    batch_size=-1,
                )

    def _validate_model_file_name(self, model_file_name):
        assert (
            self.gc_type in model_file_name
        ), "gc_type does not match model_file_name, aborting..."
        assert (
            self.response_type in model_file_name
        ), "response_type not in model_file_name, aborting..."

    def _set_models_folder(self, config=None):
        """Set the folder where models are saved"""

        # If input_folder is Path instance or string, use it as models_folder
        if isinstance(self.config.input_folder, Path) or isinstance(
            self.config.input_folder, str
        ):
            models_folder = self.config.input_folder
        else:
            models_folder = self.config.output_folder / "models"
        Path(models_folder).mkdir(parents=True, exist_ok=True)

        return models_folder

    def _save_model(self):
        """
        Save model for single trial, a.k.a. 'train_model' vae_run_mode
        """
        model_filename = f"model_{self.gc_type}_{self.response_type}_{self.device}_{self.timestamp}.pt"
        model_path = f"{self.models_folder}/{model_filename}"

        self.vae.config = {
            "latent_dims": self.latent_dim,
        }
        print(f"Saving model to {model_path}")
        torch.save(self.vae.state_dict(), model_path)

    def _save_latent_stats(self):
        """
        Save latent data after 'train_model' vae_run_mode
        """
        latent_stats_filename = f"latent_stats_{self.gc_type}_{self.response_type}_{self.device}_{self.timestamp}.npy"
        latent_stats_path = f"{self.models_folder}/{latent_stats_filename}"

        vae_latent_stats = self._get_vae_latent_stats()
        print(f"Saving latent data to {latent_stats_path}")
        np.save(latent_stats_path, vae_latent_stats)

        self.vae_latent_stats = vae_latent_stats  # Save to self for use after training

    def _get_vae_latent_stats(self):
        # Get the latent space data
        train_df = self.get_encoded_samples(dataset=self.train_loader.dataset)
        valid_df = self.get_encoded_samples(dataset=self.val_loader.dataset)
        test_df = self.get_encoded_samples(dataset=self.test_loader.dataset)
        latent_df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)

        # Extract data from latent_df into a numpy array from columns whose title include "EncVariable"
        vae_latent_stats = latent_df.filter(regex="EncVariable").to_numpy()

        return vae_latent_stats

    def _parse_timestamp_from_model_filename(self, model_file_name):
        """
        Parse the timestamp from the model file name.
        The timestamp is expected to be in the format 'YYYYMMDD_HHMMSS'.
        """
        # Get the time stamp from the file name
        name_stem = model_file_name.split(".")[0]
        times = name_stem.split("_")[-2:]
        timestamp = "_".join(times)

        return timestamp

    def _load_latent_stats(self, model_file_name=None):
        # Load latent data after 'load_model' vae_run_mode
        if model_file_name is None:
            latent_stats_file_name = f"latent_stats_{self.gc_type}_{self.response_type}_{self.device}_{self.timestamp_for_loading}.npy"  # timestamp_for_loading is set in _load_model
            timestamp = self.timestamp_for_loading
        else:
            timestamp = self._parse_timestamp_from_model_filename(model_file_name)

        try:
            if timestamp is not None:
                # latent data file name is of type latent_stats_[TIMESTAMP].npy
                latent_stats_file_name = f"latent_stats_{self.gc_type}_{self.response_type}_{self.device}_{timestamp}.npy"
            else:
                # Get the most recent latent data file
                latent_stats_file_name = max(
                    Path(self.models_folder).glob(
                        f"latent_stats_{self.gc_type}_{self.response_type}_{self.device}_*.npy"
                    )
                )
                print(f"Most recent latent data file is {latent_stats_file_name}.")
        except ValueError:
            raise FileNotFoundError("No latent data files found. Aborting...")

        latent_stats_path = self.models_folder / latent_stats_file_name

        print(f"Loading latent data from {latent_stats_path}")
        self.vae_latent_stats = np.load(latent_stats_path)

    def _load_model(self, model_path=None, best_result=None, trial_name=None):
        """Load model if exists. Use either model_path, best_result, or trial_name to load model"""

        vae_model = self._create_empty_model()

        if model_path is not None:
            model_path = Path(model_path)
            if Path.exists(model_path) and model_path.is_file():
                print(
                    f"Loading model from {model_path}. \nWARNING: This will replace the current model in-place."
                )
                vae_model.load_state_dict(torch.load(model_path, weights_only=True))
            elif Path.exists(model_path) and model_path.is_dir():
                try:
                    prefix = f"model_{self.gc_type}_{self.response_type}_{self.device}_"
                    model_path = max(Path(self.models_folder).glob(f"{prefix}*.pt"))
                    timestamp = "_".join(model_path.stem.split("_")[-2:])
                    self.timestamp_for_loading = timestamp
                    if not model_path.is_file():
                        raise FileNotFoundError("VAE model not found, aborting...")
                    try:
                        vae_model.load_state_dict(
                            torch.load(model_path, weights_only=True)
                        )
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError(
                            "Module path changed, you need to train VAE model. Aborting..."
                        )
                    print(f"Most recent model is {model_path}.")
                except ValueError:
                    raise FileNotFoundError("No model files found. Aborting...")

            else:
                print(f"Model {model_path} does not exist.")

        else:
            # Get the most recent model. Max recognizes the timestamp with the largest value
            try:
                model_path = max(Path(self.models_folder).glob("*.pt"))
                print(f"Most recent model is {model_path}.")
            except ValueError:
                raise FileNotFoundError("No model files found. Aborting...")
            vae_model.load_state_dict(torch.load(model_path, weights_only=True))

        vae_model.to(self.device)

        return vae_model

    def _get_spatial_experimental_data(self):
        """
        Get spatial ganglion cell data from file using the experimental_data method read_spatial_filter_data().
        All data is returned, the requested data is logged in the class attributes gc_type and response_type.
        """

        # Get requested data
        self.experimental_data = ExperimentalData(
            self.experimental_metadata, self.gc_type, self.response_type
        )

        # Log requested label
        self.gc_label = self.experimental_data.data_names2labels_dict[
            f"{self.gc_type}_{self.response_type}"
        ]

        # We train by more data, however.
        # Build a list of combinations of gc types and response types from self.gc_response_types
        train_by_combinations = [
            f"{gc}_{response}"
            for (gc, response) in product(
                self.gc_response_types[0], self.gc_response_types[1]
            )
        ]

        response_labels = [
            self.experimental_data.data_names2labels_dict[key]
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
                self.experimental_data.metadata["data_spatialfilter_height"],
                self.experimental_data.metadata["data_spatialfilter_width"],
            )
        )
        collated_labels_np = np.empty((0, 1), dtype=int)

        # Get data for learning
        for gc_type, response_type, label in zip(
            gc_types, response_types, response_labels
        ):
            print(f"Loading data for {gc_type}_{response_type} (label {label})")
            experimental_data = ExperimentalData(
                self.experimental_metadata, gc_type, response_type
            )
            bad_data_idx = experimental_data.known_bad_data_idx
            (
                gc_spatial_data_np_orig,
                _,
            ) = experimental_data.read_spatial_filter_data()

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
            experimental_data.data_names2labels_dict,
        )

    def _create_empty_model(self):
        vae = VariationalAutoencoder(
            latent_dims=self.latent_dim,
            resolution_hw=self.resolution_hw,
            ksp_key=self.kernel_stride,
            channels=self.channels,
            conv_layers=self.conv_layers,
            batch_norm=self.batch_norm,
            latent_distribution=self.latent_distribution,
            device=self.device,
        )
        return vae

    def _prep_training(self):
        self.vae = self._create_empty_model()

        self.vae.test_data = self.test_data
        self.vae.test_labels = self.test_labels
        self.vae.augmentation_dict = self.augmentation_dict

        self.optim = torch.optim.Adam(
            self.vae.parameters(), lr=self.lr, weight_decay=1e-5
        )

        self.scheduler = lr_scheduler.StepLR(
            self.optim, step_size=self.lr_step_size, gamma=self.lr_gamma
        )

        print(f"Selected device: {self.device}")
        self.vae.to(self.device)

    def _save_logging(self):
        """
        Save logging to train_log_folder
        """
        self.log_df.to_csv(
            self.train_log_folder / f"train_log_{self.timestamp}.csv", index=False
        )

        self.log_df.to_pickle(self.train_log_folder / f"train_log_{self.timestamp}.pkl")

    def _load_logging(self, model_file_name=None):
        """
        Load logging from train_log_folder
        """

        if model_file_name is None:
            timestamp = self.timestamp_for_loading
        else:
            timestamp = self._parse_timestamp_from_model_filename(model_file_name)

        # Get the most recent log file
        try:
            if timestamp is not None:
                log_file_name = f"train_log_{timestamp}.csv"
                log_path = Path(self.train_log_folder) / log_file_name
                print(f"Loading log file from {log_path}.")
            else:
                log_path = max(Path(self.train_log_folder).glob("*.csv"))
                print(f"Most recent log file is {log_path}.")
        except ValueError:
            raise FileNotFoundError("No log files found. Aborting...")

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

    def _train_epoch(self, vae, device, dataloader, optimizer, scheduler):
        """Train model"""

        # Set train mode for both the encoder and the decoder
        vae.train()
        train_loss = 0.0

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

    def _validate_epoch(self, vae, device, dataloader):
        """Test model"""

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
        Train for vae_run_mode = train_model
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

    def get_and_split_experimental_data(self):
        """
        Load data
        Split into training, validation and testing
        """

        # Get numpy data
        data_np, labels_np, data_names2labels_dict = (
            self._get_spatial_experimental_data()
        )

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

    def augment_and_get_dataloader(
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

    def get_encoded_samples(self, ds_name=None, dataset=None):
        """Get encoded samples from a dataset.

        Parameters
        ----------
        ds_name : str, optional
            Dataset name
        dataset : torch.utils.data.Dataset, optional
            Dataset to encode. If ds_name is given, this is ignored.

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
            label = self.experimental_data.data_labels2names_dict[sample[1].item()]
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
