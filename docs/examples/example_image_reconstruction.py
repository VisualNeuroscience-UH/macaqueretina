"""
Quantitative evaluation of macaque retina simulator

Simulate responses to natural images, build linear reconstruction models from spike data,
and evaluate model performance by comparing reconstructed images to originals.
This requires van Hateren image dataset.

Sequence of operations:
dataset = "train", 3330 images available
1) operation = "transform_images" # Creates the retina spike data for model construction
2) operation = "simulate" # Creates the retina spike data for model construction
3) operation = "construct_model" # Creates the model from train data

dataset = "test", 834 images available
4) operation = "transform_images" # Creates the retina spike data for testing the model
5) operation = "simulate" # Creates the retina spike data for testing the model
6) operation = "reconstruct" # Reconstructs the images from the test spike data
7) operation = "show_model_comparison" # show_model_comparisons the reconstructed images from the test spike data
8) operation = "fourier_transform" # Computes and compares the Fourier transform of the reconstructed images
"""

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Third-party
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import kruskal, mannwhitneyu
from torchvision.transforms import v2

# Local
import macaqueretina as mr
from macaqueretina.analysis.image_reconstruction_module import (
    ImageReconstruction,
)

start_time = time.time()

mr.load_parameters()

mr.config.device = "cuda" if torch.cuda.is_available() else "cpu"

# Fluid parameters for workstation run.
dataset = "test"  # "train" or "test"
n_images = 834  # 3330 for train, 834 for test
operation = "transform_images"  # "transform_images", "simulate", "construct_model", "reconstruct", "show_model_comparison", "fourier_transform", show_slurm_arrays
spatial_model_type = "DOG"  # "DOG" or "VAE"
temporal_model_type = "fixed"  # "fixed", "dynamic" or "subunit"
array_idx_str = "01"
H = W = 240

image_rootpath = Path(
    f"/opt3/images/vanHateren/imc_images_{dataset}"
)  # Change to your path
# torch.manual_seed(42)

# QA Print captured env vars
print("The following ENV VAR captured")
print(f"{dataset=}")
print(f"{n_images=}")
print(f"{operation=}")
print(f"{spatial_model_type=}")
print(f"{temporal_model_type=}")
print(f"{array_idx_str=}")
print(f"{H=}, {W=}")

mr.config.retina_parameters.spatial_model_type = spatial_model_type
mr.config.retina_parameters.temporal_model_type = temporal_model_type
mr.config.retina_parameters.ecc_limits_deg = (4.5, 5.5)
mr.config.retina_parameters.pol_limits_deg = (-1.5, 1.5)

# mr.config.retina_parameters.ecc_limits_deg = (3.5, 6.5)
# mr.config.retina_parameters.pol_limits_deg = (-10, 10)

mr.config.external_stimulus_parameters.ext_pix_per_deg = 30
mr.config.visual_stimulus_parameters.image_height = H
mr.config.visual_stimulus_parameters.image_width = W
mr.config.visual_stimulus_parameters.stimulus_size = 1.6
mr.config.visual_stimulus_parameters.pix_per_deg = 60
mr.config.visual_stimulus_parameters.stimulus_form = "rectangular"

mr.config.visual_stimulus_parameters.duration_seconds = 0.1
mr.config.visual_stimulus_parameters.baseline_start_seconds = 0.1
mr.config.visual_stimulus_parameters.baseline_end_seconds = 0.3
mr.config.visual_stimulus_parameters.pattern = "natural_image"


# gc_types = ["midget"]
gc_types = ["parasol", "midget"]
response_types = ["on", "off"]

# Update session data.
mr.config.path = Path(mr.config.model_root_path).joinpath(
    Path(mr.config.project), mr.config.experiment
)

session_suffix = f"{H}x{W}_{spatial_model_type}_{temporal_model_type}"

model_filename = f"W_{session_suffix}.npz"

transformed_dir = (
    mr.config.path / f"transformed_{H}x{W}_{dataset}_images_{array_idx_str}"
)


transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Grayscale(),
        v2.RandomCrop((H, W)),
        v2.ToDtype(torch.float16, scale=True),
    ]
)


def save_transformed_images(transformed_dir, data_loader):
    """
    Save transformed images to disk for retina simulator.
    """
    dataset = data_loader.dataset
    filenames = data_loader.filenames

    if len(filenames) != len(dataset):
        raise ValueError(
            f"Length of filenames ({len(filenames)}) does not match dataset length ({len(dataset)})"
        )

    transformed_dir.mkdir(parents=True, exist_ok=True)

    output_names = []

    for idx in range(len(dataset)):
        img_transformed, _ = dataset[idx]
        img_transformed = img_transformed.numpy().squeeze(0) * 255.0
        img_transformed = img_transformed.astype(np.uint8)  # Convert to uint8

        output_name = transformed_dir / f"{Path(filenames[idx]).stem}_{H}x{W}.png"
        output_names.append(output_name)
        mr.data_io.save_data(output_name, img_transformed)

    return output_names


def get_transformed_filenames(transformed_dir):
    """
    Get a list of transformed image filenames in the output directory.
    """
    transformed_filenames = list(transformed_dir.glob("*.png"))
    if not transformed_filenames:
        raise FileNotFoundError(
            f"No transformed images found in {transformed_dir}. Please run 'transform_images' first."
        )
    return transformed_filenames


def update_folders():
    # Remove existing output and stimulus folders which do not yet define the train/test division.
    shutil.rmtree(mr.config.output_folder, ignore_errors=True)
    shutil.rmtree(mr.config.stimulus_folder, ignore_errors=True)

    mr.config.output_folder = mr.config.path / f"resolution_{session_suffix}"
    mr.config.stimulus_folder = (
        mr.config.path / f"stim_resolution_{H}_{W}_{array_idx_str}"
    )
    mr.config.input_folder = None

    # Create appropriate output and stimulus folders for the current dataset.
    mr.config.output_folder = Path(str(mr.config.output_folder) + f"_{dataset}")
    mr.config.output_folder.mkdir(parents=True, exist_ok=True)
    mr.config.stimulus_folder = Path(str(mr.config.stimulus_folder) + f"_{dataset}")
    mr.config.stimulus_folder.mkdir(parents=True, exist_ok=True)


def transfer_retina_to_test_folder():
    """
    Transfer a file from the train to the test output folder.
    """
    files_to_transfer_to_test = [
        "*_metadata.yaml",
        "*_mosaic.csv",
        "*_ret.npz",
        "*_spatial_rfs.npz",
    ]
    train_folder_path = Path(str(mr.config.output_folder)[:-5] + "_train")
    if not train_folder_path.exists():
        raise FileNotFoundError(
            f"Train folder {train_folder_path} does not exist. Please run the 'train' dataset first."
        )

    for pattern in files_to_transfer_to_test:
        for file in train_folder_path.glob(pattern):
            shutil.copy(file, mr.config.output_folder)


def simulate_retina():
    # Save transformed images to disk for simulation
    cone_noise = None

    for gc_type in gc_types:
        for response_type in response_types:
            mr.config.retina_parameters.gc_type = gc_type
            mr.config.retina_parameters.response_type = response_type

            retina, ganglion_cell = mr.retina_constructor.construct(return_objects=True)
            hashstr = mr.config.retina_parameters["retina_parameters_hash"]

            transformed_filenames = get_transformed_filenames(transformed_dir)

            for this_name in transformed_filenames:
                mr.config.external_stimulus_parameters.ext_stimulus_file = str(
                    this_name
                )

                filename_stem = Path(this_name).stem

                mr.config.visual_stimulus_parameters.stimulus_video_name = (
                    f"stim_{filename_stem}.mp4"
                )
                this_video = mr.stimulus_factory.generate()

                mr.retina_simulator.simulate(
                    retina=retina,
                    ganglion_cell=ganglion_cell,
                    stimulus=this_video,
                    filename=f"{gc_type}_{response_type}_{hashstr}_{filename_stem}_results.gz",
                    cone_noise=cone_noise,
                )

                # Get cone noise from the first simulation and use it for all subsequent simulations
                if cone_noise is None:
                    cone_noise = mr.retina_simulator.get_cone_noise()

            # mr.viz.show_stimulus_with_gcs(frame_number=31)


reco = ImageReconstruction(
    mr.config,
    mr.data_io,
    n_images=n_images,
    image_rootpath=image_rootpath,
    H=H,
    W=W,
    transform=transform,
)

data_loader = reco.get_vanhateren_dataloader(batch_size=32, shuffle=True, num_workers=4)

update_folders()

# Checks
if dataset == "test" and operation == "construct_model":
    raise ValueError("Do not construct model with test dataset!")

if dataset == "test" and operation == "simulate":
    transfer_retina_to_test_folder()

match operation:
    case "transform_images":
        if transformed_dir.is_dir():
            raise FileExistsError(
                f"""
                Output directory {transformed_dir} already exists.
                Currently, we want the same images with different model combinations, 
                separately for train and test datasets."""
            )
        transformed_filenames = save_transformed_images(transformed_dir, data_loader)

    case "simulate":
        simulate_retina()

    case "construct_model":
        R, S, retina_mask = reco.get_spikes_and_images(
            gc_types=gc_types, response_types=response_types
        )

        W, S_mean = reco.create_model(R, S, ridge_lambda=0.0)

        # If any value in W or S_mean is NaN, raise an error
        if np.isnan(W).any() or np.isnan(S_mean).any():
            raise ValueError("Model contains NaN values.")

        # Pack W and S_mean into npz
        model_data = {"W": W, "S_mean": S_mean}

        mr.data_io.save_data(
            filename=model_filename, data=model_data, path=mr.config.path
        )

    case "reconstruct":
        if Path(mr.config.path / model_filename).is_file():
            model_data = mr.data_io.load_data(filename=model_filename)
            W = model_data["W"]
            S_mean = model_data["S_mean"]
        else:
            raise FileNotFoundError(
                f"Model file {model_filename} not found. Please run 'construct_model' first."
            )

        R_test, S_test, retina_mask = reco.get_spikes_and_images(
            gc_types=gc_types, response_types=response_types
        )

        S_estimated = reco.estimate_model(W, R_test, S_mean)
        S_estimated_img = reco.reconstruct_images(S_estimated, retina_mask)

        corr = mr.retina_math.rowwise_correlation(S_estimated, S_test)
        rho = np.nanmean(corr, axis=0)

        print(f"Correlation: {rho:.4f} between reconstructed and original images")

        mr.data_io.save_data(
            filename=f"reconstruction_results_{session_suffix}.npz",
            data={
                "S_test": S_test,
                "S_estimated": S_estimated,
                "retina_mask": retina_mask,
                "rho": rho,
                "R_test": R_test,
            },
            path=mr.config.path,
        )

    case "show_model_comparison":
        spatial_models = ["DOG", "VAE"]
        temporal_models = ["fixed", "dynamic", "subunit"]

        stimulus_sample = [23, 273, 204, 211]
        n_samples = len(stimulus_sample)

        # Create all possible session suffixes based on the model combinations
        model_combinations = [
            f"{spatial_model}_{temporal_model}"
            for spatial_model in spatial_models
            for temporal_model in temporal_models
        ]
        n_models = len(model_combinations)

        #############################
        # Load reconstruction data
        #############################

        # Get all reconstruction result files matching the session suffix
        reconstruction_files = {}
        for this_model in model_combinations:
            session_files = list(
                mr.config.path.glob(f"reconstruction_results*{this_model}*")
            )
            # sorted is critical to get array idx right, glob does not sort
            reconstruction_files[this_model] = sorted(session_files)

        n_files = [len(reconstruction_files[x]) for x in model_combinations]

        ###########################################################
        # Init S_test and S_estimated arrays with the correct shape
        ###########################################################

        this_array = int(array_idx_str) - 1

        # Read one file to get the shape of S_test and S_estimated and S_test_img
        sample_file = reconstruction_files[
            model_combinations[np.where(np.array(n_files) > 0)[0][0]]
        ][this_array]
        sample_data = mr.data_io.load_data(filename=sample_file, hush=True)
        S_test = sample_data["S_test"][stimulus_sample]
        S_test_img = np.zeros((*S_test.shape[:1], H, W))

        S_estimated = sample_data["S_estimated"][stimulus_sample]
        S_estimated_img = np.zeros((*S_estimated.shape[:1], n_models, H, W))
        retina_mask = sample_data["retina_mask"]

        stimulus_sample_idx = range(n_samples)

        # Here we choose the S_test_img (ground truth on the left) from sample_file
        for idx in stimulus_sample_idx:
            S_test_img[idx, ...] = reco.reconstruct_images(
                sample_data["S_test"][stimulus_sample[idx]], retina_mask
            )

        for i in range(n_models):
            this_model = model_combinations[i]
            try:
                file = reconstruction_files[this_model][this_array]
                array_idx = file.stem.rfind("_") + 1
                this_array_idx_str = file.stem[array_idx:]
                if this_array_idx_str != array_idx_str:
                    raise ValueError(
                        f"Array index mismatch: expected {array_idx_str}, got {this_array_idx_str}"
                    )

                data = mr.data_io.load_data(filename=file, hush=True)
                S_estimated_img[:, i, ...] = reco.reconstruct_images(
                    data["S_estimated"][stimulus_sample], retina_mask
                )

            except IndexError:
                print(f"No reconstruction file found for this_model: {this_model}")

        # #########################################
        # Analyze and group the correlation values
        # #########################################
        n_iterations = max(n_files)
        rho_values = (
            np.zeros((n_iterations, len(reconstruction_files.keys()), n_images))
            * np.nan
        )  # Initialize with NaN to handle missing data

        # Get the correlation values for each reconstruction file
        for i, this_model in enumerate(model_combinations):
            for j, file in enumerate(reconstruction_files[this_model]):
                data = mr.data_io.load_data(filename=file, hush=True)
                S_test_all = data["S_test"]
                S_estimated_all = data["S_estimated"]
                # Compute the correlation for each sample in stimulus_sample
                this_n_images = S_test_all.shape[0]

                rho_values[j, i, :this_n_images] = mr.retina_math.rowwise_correlation(
                    S_estimated_all, S_test_all
                )

        # returns array of shape (2, n_models): [lower, upper] for each column
        # Reshape rho_values to become (n_iterations * n_images_in_data, n_models) for bootstrap_ci
        reshaped_rho_values = np.transpose(rho_values, (0, 2, 1)).reshape(-1, 6)
        cis = mr.retina_math.bootstrap_ci(
            reshaped_rho_values, n_bootstraps=1000, stat_func=np.nanmean
        )

        # Prepare data: list of arrays, one per model. After removal of nan values,
        # arrays are no longer the same length.
        rho_tmp = [reshaped_rho_values[:, i] for i in range(n_models)]
        rho_data_nan_free = [
            model_data[~np.isnan(model_data)] for model_data in rho_tmp
        ]

        dog = np.concatenate(
            (
                np.array(rho_data_nan_free[0]),
                np.array(rho_data_nan_free[1]),
                np.array(rho_data_nan_free[2]),
            )
        )
        vae = np.concatenate(
            (
                np.array(rho_data_nan_free[3]),
                np.array(rho_data_nan_free[4]),
                np.array(rho_data_nan_free[5]),
            )
        )
        spatial_stat, spatial_p = mannwhitneyu(dog, vae, alternative="two-sided")

        # Temporal test: fixed (cols 0,3), dynamic (cols 1,4), subunit (cols 2,5)
        fixed = np.concatenate(
            (np.array(rho_data_nan_free[0]), np.array(rho_data_nan_free[3]))
        )
        dynamic = np.concatenate(
            (np.array(rho_data_nan_free[1]), np.array(rho_data_nan_free[4]))
        )
        subunit = np.concatenate(
            (np.array(rho_data_nan_free[2]), np.array(rho_data_nan_free[5]))
        )
        temporal_stat, temporal_p = mannwhitneyu(
            np.array(rho_data_nan_free[2]), np.array(rho_data_nan_free[5])
        )

        ####################
        # Create the figure
        ####################
        fig = plt.figure(figsize=(16, 14))
        outer = gridspec.GridSpec(
            2, 2, height_ratios=[1, 2], width_ratios=[1, 6], wspace=0.1, hspace=0.1
        )

        # Top-level panels
        ax01 = fig.add_subplot(outer[0, 1])
        ax10 = fig.add_subplot(outer[1, 0])
        ax11 = fig.add_subplot(outer[1, 1])

        # Nested 2x6 subgrid in [1]
        test_inner = gridspec.GridSpecFromSubplotSpec(
            n_samples, 1, subplot_spec=outer[1, 0]
        )
        reco_inner = gridspec.GridSpecFromSubplotSpec(
            n_samples, 6, subplot_spec=outer[1, 1]
        )

        test_axes = [fig.add_subplot(test_inner[i]) for i in stimulus_sample_idx]
        reco_axes = [
            fig.add_subplot(reco_inner[i, j])
            for i in stimulus_sample_idx
            for j in range(n_models)
        ]

        mean_rho_values = np.nanmean(reshaped_rho_values, axis=0)

        # Prepare data for Seaborn: long-form DataFrame
        plot_data = []
        for i, model in enumerate(model_combinations):
            for rho_val in rho_data_nan_free[i]:
                plot_data.append({"model": model, "rho": rho_val})
        df = pd.DataFrame(plot_data)

        # Create Seaborn violin plot
        sns.violinplot(
            data=df,
            x="model",
            y="rho",
            ax=ax01,
            color="lightgray",
            inner=None,  # We'll add our own mean/CI markers
            cut=0,  # Extend density to extremes
            scale="width",
        )

        # Add mean markers and 95% CI error bars
        for i, model in enumerate(model_combinations):
            ax01.scatter(
                i,
                mean_rho_values[i],
                color="black",
                s=100,
                zorder=3,
                marker="_",
                label="Mean" if i == 0 else "",
            )

        # Add text annotations for mean and CI values
        for i, (mean, lower, upper) in enumerate(
            zip(mean_rho_values, cis[0, :], cis[1, :])
        ):
            ax01.text(
                i,
                mean + 0.1,
                f"{mean:.3f}\n({lower:.3f}, {upper:.3f})\nn={len(rho_data_nan_free[i])}",
                ha="center",
                va="bottom",
                fontsize=8,
                zorder=4,
            )

        ax01.set_ylim(-0.2, 1)
        ax01.set_xlabel("Model Combination")
        ax01.set_ylabel("Correlation (rho)")
        ax01.set_title(
            "Distribution of Correlation between Reconstructed and Original Images"
        )

        # Rotate x-axis labels for readability
        plt.setp(ax01.get_xticklabels(), rotation=45, ha="right")

        # Add legend for mean marker
        ax01.legend(loc="upper right")

        # Annotate the figure with the p-values and statistics
        ax01.text(
            0.5,
            0.95,
            f"Spatial: U = {spatial_stat:.4f}, p = {spatial_p:.4f}, test: Mann-Whitney U",
            transform=ax01.transAxes,
            ha="center",
        )
        ax01.text(
            0.5,
            0.90,
            f"Temporal: H = {temporal_stat:.4f}, p = {temporal_p:.4f}, test: Kruskal-Wallis",
            transform=ax01.transAxes,
            ha="center",
        )

        # Show test images.
        for i in stimulus_sample_idx:
            test_axes[i].imshow(
                S_test_img[i, ...],
                cmap="gray",
                vmin=0,
                vmax=255,
            )

        # Show estimated images.
        for i in range(len(stimulus_sample_idx)):
            for j in range(n_models):
                this_model = model_combinations[j]
                ax = reco_axes[i * n_models + j]
                this_S_estimated = S_estimated_img[i, j, ...][retina_mask]
                this_S_test = S_test_img[i, ...][retina_mask]

                rho = np.corrcoef(this_S_estimated, this_S_test)[0, 1]
                rho2 = rho_values[this_array, j, stimulus_sample[i]]
                if not np.isclose(rho, rho2, atol=1e-3):
                    raise ValueError(
                        f"Correlation mismatch for sample {stimulus_sample[i]} and model {this_model}: {rho:.4f} vs {rho2:.4f}"
                    )

                ax.imshow(
                    S_estimated_img[i, j, ...],
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )
                ax.text(
                    0.05,
                    0.95,
                    f"rho = {rho:.3f}",
                    color="white",
                    fontsize=8,
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                )
                ax.set_title(this_model)
                ax.axis("off")

        ax10.axis("off")
        ax11.axis("off")

        plt.tight_layout()

        # mr.viz._figsave(
        #     figurename=mr.config.path.joinpath(
        #         f"Fig7_reco_array_{array_idx_str}_samples_{'_'.join(map(str, stimulus_sample))}_260827.eps"
        #     ),
        # )

    case "fourier_transform":
        # 115, 83,  11, 20, 107
        stimulus_sample = [0]

        # Load the reconstruction results
        session_suffix = (
            f"{H}x{W}_{spatial_model_type}_{temporal_model_type}_{array_idx_str}"
        )
        reconstruction_file = (
            mr.config.path / f"reconstruction_results_{session_suffix}.npz"
        )
        if not reconstruction_file.exists():
            raise FileNotFoundError(
                f"Reconstruction results file {reconstruction_file} not found. Please run 'reconstruct' first."
            )

        data = mr.data_io.load_data(filename=reconstruction_file)
        S_test_img = reco.reconstruct_images(data["S_test"], data["retina_mask"])
        S_estimated_img = reco.reconstruct_images(
            data["S_estimated"], data["retina_mask"]
        )
        retina_mask = data["retina_mask"]

        # Get a rectangle inside the retina mask. Without this the retina border dominates the mean fourier transform.
        min_row, min_col = (
            np.where(retina_mask)[0].min(),
            np.where(retina_mask)[1].min(),
        )
        max_row, max_col = (
            np.where(retina_mask)[0].max(),
            np.where(retina_mask)[1].max(),
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(
            S_test_img[stimulus_sample[0], ...], cmap="gray", vmin=0, vmax=255
        )
        axes[0].add_patch(
            plt.Rectangle(
                (min_col, min_row),
                max_col - min_col,
                max_row - min_row,
                edgecolor="red",
                facecolor="none",
                linewidth=2,
            )
        )
        axes[0].set_title("Original Image with Retina Mask")
        axes[1].imshow(
            S_estimated_img[stimulus_sample[0], ...], cmap="gray", vmin=0, vmax=255
        )
        axes[1].add_patch(
            plt.Rectangle(
                (min_col, min_row),
                max_col - min_col,
                max_row - min_row,
                edgecolor="red",
                facecolor="none",
                linewidth=2,
            )
        )
        axes[1].set_title("Reconstructed Image with Retina Mask")
        plt.tight_layout()

        fig2, axes2 = plt.subplots(1, 1, figsize=(12, 6))
        # Compute the 2D Fourier transform inside the rectangle of the original and reconstructed images
        S_test_fft = np.fft.fft2(S_test_img[:, min_row:max_row, min_col:max_col])
        S_estimated_fft = np.fft.fft2(
            S_estimated_img[:, min_row:max_row, min_col:max_col]
        )

        # Compute the magnitude of the Fourier transform
        S_test_magnitude = np.abs(np.fft.fftshift(S_test_fft))
        S_estimated_magnitude = np.abs(np.fft.fftshift(S_estimated_fft))
        # We have 30 pixels per degree. Please calculate and plot the absolute magnitude as a function of spatial frequency.
        pixels_per_degree = 30
        freqs = np.fft.fftshift(
            np.fft.fftfreq(max_col - min_col, d=1 / pixels_per_degree)
        )
        freq_mask = freqs > 0
        test_magnitude_mean = S_test_magnitude.mean(axis=0)[
            S_test_magnitude.shape[1] // 2, :
        ][freq_mask]
        estimated_magnitude_mean = S_estimated_magnitude.mean(axis=0)[
            S_estimated_magnitude.shape[1] // 2, :
        ][freq_mask]

        # Plot the magnitude spectra
        axes2.plot(
            np.log(freqs[freq_mask]),
            test_magnitude_mean,
            label="Original Image",
        )
        axes2.set_xlabel("Spatial Frequency (log cycles/degree)")
        axes2.set_ylabel("Magnitude")
        axes2.legend()

        axes2.plot(
            np.log(freqs[freq_mask]),
            estimated_magnitude_mean,
            label="Reconstructed Image",
        )

        plt.tight_layout()

    case "show_slurm_arrays":
        # Show reconstruction separately for each SLURM array index. Only one set of array indexes
        # are processed at a time, so we can show the results for each array index separately.
        # Get all reconstruction result files matching the session suffix
        reconstruction_files = sorted(
            list(mr.config.path.glob(f"reconstruction_results*{session_suffix}*"))
        )
        n_files = len(reconstruction_files)

        rho_values = (
            np.zeros((n_files, n_images)) * np.nan
        )  # Initialize with NaN to handle missing data
        has_R_test = np.zeros(n_files)
        R_values = np.zeros(n_files)

        for idx, file in enumerate(reconstruction_files):
            data = mr.data_io.load_data(filename=file, hush=True)
            S_test_all = data["S_test"]
            S_estimated_all = data["S_estimated"]
            this_n_images = S_test_all.shape[0]

            rho_values[idx, :this_n_images] = mr.retina_math.rowwise_correlation(
                S_estimated_all, S_test_all
            )
            if "R_test" in data.keys():
                # plt.hist(data["R_test"].flatten(),100);plt.show()
                R_values[idx] = data["R_test"].mean()
                has_R_test[idx] = 0  # temporarily set to 0 to force recomputation of R

        result_paths = sorted(
            list(mr.config.path.parent.glob(f"resolution_{session_suffix}/*_{dataset}"))
        )

        def save_R(R, idx):
            data = mr.data_io.load_data(filename=reconstruction_files[idx], hush=True)
            data = dict(data)
            data[f"R_{dataset}"] = R
            mr.data_io.save_data(
                filename=reconstruction_files[idx],
                data=data,
            )

        # --- Parallel version using ThreadPoolExecutor for async disk I/O ---
        # Use a lock for numpy array access if you want strict thread-safety
        has_R_test_lock = threading.Lock()

        def process_single_path(this_path, idx):
            # Check with lock for strict thread-safety
            with has_R_test_lock:
                if has_R_test[idx]:
                    return None
            print(f"Processing {this_path}...")
            R, _ = reco.get_rates(gc_types, response_types, this_path)
            return R  # Return the full array, not the mean

        with ThreadPoolExecutor(max_workers=min(n_files, 5)) as executor:
            futures = [
                executor.submit(process_single_path, this_path, idx)
                for idx, this_path in enumerate(result_paths)
            ]
            for idx, future in enumerate(futures):
                if not has_R_test[idx]:
                    R = future.result()
                    if R is not None:  # Only process if not skipped
                        R_values[idx] = R.mean()
                        save_R(R, idx)  # Pass the full array R

        mean_rho_values = np.nanmean(rho_values, axis=1)
        fig, ax1 = plt.subplots(figsize=(12, 6))

        sns.violinplot(
            x=np.repeat(
                range(n_files), rho_values.shape[1]
            ),  # Repeat each index for all samples
            y=rho_values.flatten(),  # Flatten all rho_values
            ax=ax1,
            color="lightgray",
            inner=None,  # No inner markings
            cut=0,  # Clip the KDE to the range of the data
            scale="width",
        )

        # Plot mean correlation line
        ax1.plot(range(n_files), mean_rho_values, "b-", label="Correlation (rho)")
        ax1.set_xlabel("SLURM Array Index")
        ax1.set_ylabel("Correlation (rho)", color="b")
        ax1.set_ylim(-0.2, 1)

        # Add gorizontal grid lines for better readability
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # Twin axes for R_values
        # Divide R_values by epoch duration to get firing rates.
        firing_rates = R_values / 0.15  # NOTE hard coded
        ax2 = ax1.twinx()
        ax2.plot(range(n_files), firing_rates, "r-", label="Mean Response (R)")
        # ax2.plot(range(n_files), R_values, "r-", label="Mean Response (R)")
        ax2.set_ylabel("Mean Response (R)", color="r")
        ax2.set_ylim(5, 30)

        # Title and layout
        plt.title("Correlation and Mean Response vs SLURM Array Index")
        plt.tight_layout()

        # Optional: Add legend for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        outpath = Path("/opt3/projects/macaqueretina/experiment_scripts")

        mr.viz._figsave(
            # figurename=None,
            figurename=outpath.joinpath(
                f"Fig7_gain_multiplier_rho_R_{spatial_model_type}_{temporal_model_type}_poisson.eps"
            ),
        )

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")


plt.show()
