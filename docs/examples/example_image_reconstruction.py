import os
import shutil
import time
from pathlib import Path

# Third-party
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kruskal, mannwhitneyu
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2

# Local
import macaqueretina as mr
from macaqueretina.analysis.image_reconstruction_module import ImageReconstruction

mr.load_parameters()

start_time = time.time()

"""
Quantitative evaluation of macaque retina simulator

Simulate responses to natural images, build linear reconstruction models from spike data,
and evaluate model performance by comparing reconstructed images to originals.

Sequence of operations:
dataset = "train", 3330 images available
1) operation = "simulate" # Creates the retina spike data for model construction
2) operation = "construct_model" # Creates the model from train data

dataset = "test", 834 images available
3) operation = "simulate" # Creates the retina spike data for testing the model
4) operation = "reconstruct" # Reconstructs the images from the test spike data
"""
# # Fluid parameters.
# dataset = os.environ["DATASET"]  # "train" or "test"
# n_images = int(os.environ["N_IMAGES"])
# operation = os.environ["OPERATION"]  # "transform_images", "simulate", "construct_model", or "reconstruct"
# spatial_model_type = os.environ["SPATIAL_MODEL_TYPE"]  # "DOG" or "VAE"
# temporal_model_type = os.environ[
#     "TEMPORAL_MODEL_TYPE"
# ]  # "fixed", "dynamic" or "subunit"
# array_idx = int(os.environ["ARRAY_IDX"])
# array_idx_str = f"{array_idx:02d}"
# H = W = int(os.environ["HEIGHT_AND_WIDTH"])

# Fluid parameters.
dataset = "train"  # "train" or "test"
n_images = 2
operation = "construct_model"  # "transform_images", "simulate", "construct_model", "reconstruct", "display"
spatial_model_type = "DOG"  # "DOG" or "VAE"
temporal_model_type = "fixed"  # "fixed", "dynamic" or "subunit"
array_idx_str = "00"
H = W = 240

mr.config.experiment = "image_reconstruction_hpc_240_tmp"
image_rootpath = Path(f"/opt3/images/vanHateren/imc_images_{dataset}")
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

if dataset == "test" and operation == "construct_model":
    raise ValueError("Do not construct model with test dataset!")

# Remove empty experiment folder.
shutil.rmtree(mr.config.path, ignore_errors=True)

# Update session data.
mr.config.path = Path(mr.config.model_root_path).joinpath(
    Path(mr.config.project), mr.config.experiment
)

mr.config.retina_parameters.spatial_model_type = spatial_model_type
mr.config.retina_parameters.temporal_model_type = temporal_model_type
mr.config.retina_parameters.ecc_limits_deg = (3.5, 6.5)
mr.config.retina_parameters.pol_limits_deg = (-10, 10)

session_suffix = f"{H}x{W}_{spatial_model_type}_{temporal_model_type}_{array_idx_str}"

mr.config.output_folder = mr.config.path / f"resolution_{session_suffix}"
mr.config.stimulus_folder = mr.config.path / f"stim_resolution_{H}_{W}_{array_idx_str}"
mr.config.input_folder = None
model_filename = f"W_{session_suffix}.npz"

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Grayscale(),
        v2.RandomCrop((H, W)),
        v2.ToDtype(torch.float16, scale=True),
    ]
)


class TransformWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.get_original_image = getattr(dataset.dataset, "get_original_image", None)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)


class VanHaterenDataset(Dataset):
    """
    Custom Dataset class for the Van Hateren image dataset.

    Note that Van Hateren image names start from 1, so indexes will be one off
    """

    def __init__(self, root_dir):
        self.image_paths = sorted(
            [
                path
                for path in root_dir.iterdir()
                if path.suffix.lower() in (".imc", ".iml")
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.get_original_image(idx)

        # Image trasforms follow Simo's van_hateren_script2
        # Z normalize
        img = (img - img.mean()) / img.std()

        # tanh normalize
        img = np.tanh(img)

        # Scale to 0-1
        img = (img - img.min()) / (img.max() - img.min())

        # Add dummy color channel
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img)

        # Return a dummy label (0) since Van Hateren images don't have labels
        return img, 0

    def get_original_image(self, idx):
        """Return the original image as a numpy array."""
        with open(self.image_paths[idx], "rb") as handle:
            s = handle.read()

        img = np.frombuffer(s, dtype="uint16").byteswap()
        img = img.reshape(1024, 1536).astype(np.float32)

        return img


def create_filtered_imagenet(root, H, W, batch_size=1024, split="train"):
    """
    Create a filtered ImageNet dataset containing only images with resolution >= (H, W).
    Saves the indices of valid images to a .pt file for future use.

    Note: The ImageNet dataset is large, and this function does not load the actual images into memory, only their metadata.
    This functionality needs the imagesize library to check image dimensions without loading the images.
    Nevertheless, this is slow, but needs to be done only once. The indices are saved to a .pt file for future use.
    """
    import imagesize

    root = Path(root)
    full_dataset = datasets.ImageNet(
        root=root,
        split=split,
        transform=None,
    )

    total = len(full_dataset)
    valid_mask = np.zeros(total, dtype=bool)  # Pre-allocate boolean mask

    print(f"Filtering ImageNet-{split} for resolution >= ({H}, {W})...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        for idx in range(start, end):
            path, _ = full_dataset.samples[idx]
            w, h = imagesize.get(path)
            if w >= W and h >= H:
                valid_mask[idx] = True  # O(1) assignment, no resizing

        print(f"Processed {end}/{total} images, found {valid_mask.sum()} valid")

    # Extract valid indices in one vectorized operation
    valid_indices = np.where(valid_mask)[0].tolist()

    # Save and return
    indices_path = root / f"imagenet_{split}_{H}_{W}_indices.pt"
    torch.save(valid_indices, indices_path)
    print(f"Saved indices to {indices_path}")

    subset = Subset(full_dataset, valid_indices)
    print(f"Filtered dataset: {len(subset)} images")


def get_imagenet_dataloader(batch_size=32, shuffle=True, num_workers=4):
    """
    Get a DataLoader for the filtered ImageNet dataset with images of resolution >= (H, W).
    """
    # 1. Load the full dataset (metadata only, no actual image loading yet)
    full_dataset = datasets.ImageNet(
        root=image_rootpath,
        transform=None,
    )

    # 2. Load precalculated indices for images with resolution >= (H, W)
    full_path = image_rootpath / f"imagenet_train_{H}_{W}_indices.pt"
    if full_path.exists():
        indices = torch.load(full_path)
        print(f"Loaded {len(indices)} valid indices from {full_path}")
    else:
        print(
            f"Indices file {full_path} not found. Getting and saving ImageNet indices..."
        )
        create_filtered_imagenet(image_rootpath, H, W)
        indices = torch.load(full_path)

    filtered_dataset = Subset(full_dataset, indices)

    # 3. Randomly select a subset of n_images from the filtered dataset
    # subset_indices = range(2, 3)
    subset_indices = torch.randperm(len(filtered_dataset))[:n_images]
    subset = Subset(filtered_dataset, subset_indices)

    # 4. Apply transformations to the subset
    subset = TransformWrapper(subset, transform=transform)

    # 5. Create a DataLoader for the subset
    data_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # pin_memory=True,
    )

    data_loader.filenames = get_filenames_from_dataloader(data_loader)

    return data_loader


def get_vanhateren_dataloader(batch_size=32, shuffle=True, num_workers=4):
    # 1. Load dataset (metadata only, no transforms)
    full_dataset = VanHaterenDataset(root_dir=image_rootpath)

    # 2. Select a random subset of n_images
    subset_indices = torch.randperm(len(full_dataset))[:n_images]
    subset = Subset(full_dataset, subset_indices)

    # 3. Apply transformations via TransformWrapper
    subset = TransformWrapper(subset, transform=transform)

    # 4. Create DataLoader
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # pin_memory=True,
    )

    dataloader.filenames = [str(full_dataset.image_paths[i]) for i in subset_indices]

    return dataloader


def get_filenames_from_dataloader(data_loader):
    """Extract original filenames from the ImageNet data_loader."""
    wrapper = data_loader.dataset
    subset1 = wrapper.dataset
    subset2 = subset1.dataset
    imagenet = subset2.dataset

    subset_indices = subset1.indices
    filtered_indices = subset2.indices

    return [imagenet.samples[filtered_indices[i]][0] for i in subset_indices]


def save_transformed_images(output_dir, data_loader):
    """
    Save transformed images to disk for retina simulator.
    """
    dataset = data_loader.dataset
    filenames = data_loader.filenames

    if len(filenames) != len(dataset):
        raise ValueError(
            f"Length of filenames ({len(filenames)}) does not match dataset length ({len(dataset)})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_names = []

    for idx in range(len(dataset)):
        img_transformed, _ = dataset[idx]
        img_transformed = img_transformed.numpy().squeeze(0) * 255.0
        img_transformed = img_transformed.astype(np.uint8)  # Convert to uint8

        output_name = output_dir / f"{Path(filenames[idx]).stem}_{H}x{W}.png"
        output_names.append(output_name)
        mr.data_io.save_data(output_name, img_transformed)

    return output_names


def get_filenames(this_name):
    filename_stem = Path(this_name).stem
    gc_type = mr.config.retina_parameters["gc_type"]
    response_type = mr.config.retina_parameters["response_type"]
    hashstr = mr.config.retina_parameters["retina_parameters_hash"]
    simulation_results_filename = (
        f"{gc_type}_{response_type}_{hashstr}_{filename_stem}_results.gz"
    )

    stimulus_video_name = f"stim_{filename_stem}.mp4"

    return simulation_results_filename, stimulus_video_name


def update_folders(dataset):
    # Remove existing output and stimulus folders which do not yet define the train/test division.
    shutil.rmtree(mr.config.output_folder, ignore_errors=True)
    shutil.rmtree(mr.config.stimulus_folder, ignore_errors=True)

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


def bootstrap_ci(data, n_bootstraps=10000, stat_func=np.mean, alpha=0.05):
    n = data.shape[0]
    idx = np.random.randint(0, n, size=(n_bootstraps, n))
    stats = stat_func(data[idx], axis=1)
    cis = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return cis, data[idx]


data_loader = get_vanhateren_dataloader(batch_size=32, shuffle=True, num_workers=4)

update_folders(dataset)

if dataset == "test" and operation == "simulate":
    transfer_retina_to_test_folder()

# Spatial parameters. H = external stimulus height (pix), W = external stimulus width (pix)
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

# Main loop
gc_types = ["parasol", "midget"]
response_types = ["on", "off"]

output_dir = mr.config.path / f"transformed_{H}x{W}_{dataset}_images"


def simulate_retina():
    # Save transformed images to disk for simulation

    for gc_type in gc_types:
        for response_type in response_types:
            mr.config.retina_parameters.gc_type = gc_type
            mr.config.retina_parameters.response_type = response_type

            mr.retina_constructor.construct()  # Reuses existing matching retina
            transformed_filenames = output_dir.glob("*.png")
            # If empty, raise an error
            if not any(transformed_filenames):
                raise FileNotFoundError(
                    f"No transformed images found in {output_dir}. Please run 'transform_images' first."
                )

            for this_name in transformed_filenames:
                mr.config.external_stimulus_parameters.ext_stimulus_file = str(
                    this_name
                )

                simulation_results_filename, stimulus_video_name = get_filenames(
                    this_name
                )

                mr.config.visual_stimulus_parameters.stimulus_video_name = (
                    stimulus_video_name
                )

                mr.stimulus_factory.generate()  # Reuses existing matching stimulus

                mr.retina_simulator.simulate(filename=simulation_results_filename)

            # mr.viz.show_stimulus_with_gcs(frame_number=31)


reco = ImageReconstruction(mr.config, mr.data_io)


match operation:
    case "transform_images":
        if output_dir.is_dir():
            raise FileExistsError(
                f"""
                Output directory {output_dir} already exists.
                Currently, we want the same images with different model combinations, 
                separately for train and test datasets."""
            )
        data_loader = get_vanhateren_dataloader(
            batch_size=32, shuffle=True, num_workers=4
        )
        transformed_filenames = save_transformed_images(output_dir, data_loader)
    case "simulate":
        simulate_retina()

    case "construct_model":
        reco = ImageReconstruction(mr.config, mr.data_io)

        R, S, _ = reco.get_spikes_and_images(
            n_images=n_images, gc_types=gc_types, response_types=response_types
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
            n_images=n_images, gc_types=gc_types, response_types=response_types
        )

        S_estimated = reco.estimate_model(W, R_test, S_mean)
        S_estimated_img = reco.reconstruct_images(S_estimated, retina_mask)

        rho = np.corrcoef(S_estimated.flatten(), S_test.flatten())[0, 1]
        print(f"Correlation: {rho:.4f} between reconstructed and original images")

        mr.data_io.save_data(
            filename=f"reconstruction_results_{session_suffix}.npz",
            data={
                "S_test": S_test,
                "S_estimated": S_estimated,
                "retina_mask": retina_mask,
                "rho": rho,
            },
            path=mr.config.path,
        )

    case "display":
        spatial_models = ["DOG", "VAE"]
        temporal_models = ["fixed", "dynamic", "subunit"]
        stimulus_sample = range(2)

        # Create all possible session suffixes based on the model combinations
        model_combinations = [
            f"{spatial_model}_{temporal_model}"
            # f"{H}x{W}_{spatial_model}_{temporal_model}"
            for spatial_model in spatial_models
            for temporal_model in temporal_models
        ]
        n_cond = len(model_combinations)

        #############################
        # Load the correlation values
        #############################

        # Get all reconstruction result files matching the session suffix
        reconstruction_files = {}
        for session_suffix in model_combinations:
            session_files = list(
                mr.config.path.glob(f"reconstruction_results*{session_suffix}*")
            )

            reconstruction_files[session_suffix] = session_files

        rho_values = np.zeros(
            (
                len(reconstruction_files[model_combinations[0]]),
                len(reconstruction_files.keys()),
            )
        )

        # Get the correlation values for each reconstruction file
        for i, session_suffix in enumerate(model_combinations):
            for j, file in enumerate(reconstruction_files[session_suffix]):
                data = mr.data_io.load_data(filename=file, hush=True)
                rho_values[j, i] = data["rho"]

        ###########################################################
        # Init S_test and S_estimated arrays with the correct shape
        ###########################################################

        # Read one file to get the shape of S_test and S_estimated
        sample_file = reconstruction_files[model_combinations[0]][0]
        sample_data = mr.data_io.load_data(filename=sample_file, hush=True)

        S_test = sample_data["S_test"][stimulus_sample]
        S_test_img = np.zeros((*S_test.shape[:1], H, W))

        S_estimated = sample_data["S_estimated"][stimulus_sample]
        S_estimated_img = np.zeros((*S_estimated.shape[:1], n_cond, H, W))
        retina_mask = sample_data["retina_mask"]
        # for i in range(2):

        for this_sample_idx in stimulus_sample:
            S_test_img[this_sample_idx, ...] = reco.reconstruct_images(
                sample_data["S_test"][stimulus_sample[this_sample_idx]], retina_mask
            )
        for i in range(n_cond):
            session_suffix = model_combinations[i]
            try:
                file = reconstruction_files[session_suffix][0]
                data = mr.data_io.load_data(filename=file, hush=True)

                # breakpoint()
                S_estimated_img[:, i, ...] = reco.reconstruct_images(
                    data["S_estimated"][stimulus_sample], retina_mask
                )

            except IndexError:
                print(
                    f"No reconstruction file found for session_suffix: {session_suffix}"
                )

        ##########################################
        # Analyze and group the correlation values
        ##########################################

        # returns array of shape (2, 6): [lower, upper] for each column
        cis, bootstrap_samples = bootstrap_ci(rho_values)

        # Spatial test: DOG (columns 0-2) vs VAE (columns 3-5)
        dog = rho_values[:, :3].flatten()
        vae = rho_values[:, 3:].flatten()
        spatial_stat, spatial_p = mannwhitneyu(dog, vae, alternative="two-sided")

        # Temporal test: fixed (cols 0,3), dynamic (cols 1,4), subunit (cols 2,5)
        fixed = rho_values[:, [0, 3]].flatten()
        dynamic = rho_values[:, [1, 4]].flatten()
        subunit = rho_values[:, [2, 5]].flatten()
        temporal_stat, temporal_p = kruskal(fixed, dynamic, subunit)

        ####################
        # Create the figure
        ####################
        fig = plt.figure()
        outer = gridspec.GridSpec(
            2, 2, height_ratios=[1, 2], width_ratios=[1, 6], wspace=0.1, hspace=0.1
        )

        # Top-level panels
        ax01 = fig.add_subplot(outer[0, 1])
        ax10 = fig.add_subplot(outer[1, 0])
        ax11 = fig.add_subplot(outer[1, 1])

        # Nested 2x6 subgrid in [1]
        test_inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 0])
        reco_inner = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer[1, 1])
        # axes = [[i, j] for i in range(6) for j in range(2)]
        test_axes = [fig.add_subplot(test_inner[i]) for i in stimulus_sample]
        reco_axes = [
            fig.add_subplot(reco_inner[i, j])
            for i in stimulus_sample
            for j in range(n_cond)
        ]

        # Make a bar graph of the mean of the correlation values for each model combination. Add SEM error bars to the bar graph.
        mean_rho_values = rho_values.mean(axis=0)
        ax01.bar(
            model_combinations,
            mean_rho_values,
            yerr=[mean_rho_values - cis[0, :], cis[1, :] - mean_rho_values],
            capsize=20,
            linewidth=1,
        )
        ax01.set_ylim(0.3, 1)
        ax01.set_xlabel("Model Combination")
        ax01.set_ylabel("Mean Correlation (rho)")
        ax01.set_title("Mean Correlation between Reconstructed and Original Images")

        # Annotate the figure with the p-values and statistics from the statistical tests
        ax01.text(
            0.5,
            0.95,
            f"Spatial: U = {spatial_stat:.4f}, p = {spatial_p:.4f}, test: Mann-Whitney U",
            transform=ax01.transAxes,
        )
        ax01.text(
            0.5,
            0.90,
            f"Temporal: H = {temporal_stat:.4f}, p = {temporal_p:.4f}, test: Kruskal-Wallis",
            transform=ax01.transAxes,
        )

        # Show test images.

        for i in stimulus_sample:
            test_axes[i].imshow(
                S_test_img[i, ...],
                cmap="gray",
                vmin=0,
                vmax=255,
            )

        # Show sample images.
        for i in range(len(stimulus_sample)):
            for j in range(n_cond):
                session_suffix = model_combinations[j]
                ax = reco_axes[i * n_cond + j]

                ax.imshow(
                    S_estimated_img[i, j, ...],
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )
                ax.set_title(session_suffix)
                ax.axis("off")

        # ax01.axis("off")
        ax10.axis("off")
        ax11.axis("off")

        plt.tight_layout()

        mr.viz._figsave(
            figurename=mr.config.path.joinpath(
                f"fig_reconstruction_results_{session_suffix}_summary.eps"
            ),
        )

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")


plt.show()
