import time
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2

# Local
import macaqueretina as mr
from macaqueretina.analysis.image_reconstruction_module import ImageReconstruction

mr.load_parameters()


start_time = time.time()

"""
Quantitative evaluation of Macaque Retina Simulator

Simulate responses to natural images, build linear reconstruction models from spike data,
and evaluate model performance by comparing reconstructed images to originals.

Sequence of operations:
1) dataset = "train", 3333 images available
2) operation = "simulate" # Creates the retina spike data for model construction
3) operation = "construct_model" # Creates the model from train data

4) dataset = "test", 834 images available
5) operation = "simulate" # Creates the retina spike data for testing the model
6) operation = "reconstruct" # Reconstructs the images from the test spike data
"""


dataset = "train"
n_images = 3333
operation = "simulate"
# torch.manual_seed(42)

image_rootpath = Path(f"/opt3/images/vanHateren/imc_images_{dataset}")
H = W = 120

model_filename = f"W_{H}x{W}.npz"

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


def get_imagenet_dataloader(batch_size=32, shuffle=True, num_workers=0):
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
    Save transformed images to disk for later use.
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

        output_name = output_dir / f"{Path(filenames[idx]).stem}_{H}x{W}.jpg"
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


data_loader = get_vanhateren_dataloader(batch_size=32, shuffle=True, num_workers=4)
# data_loader = get_imagenet_dataloader(batch_size=4, shuffle=True, num_workers=0)

# Spatial parameters. H = external stimulus height (pix), W = external stimulus width (pix)
mr.config.external_stimulus_parameters.ext_pix_per_deg = 30

mr.config.visual_stimulus_parameters.image_height = H
mr.config.visual_stimulus_parameters.image_width = W
mr.config.visual_stimulus_parameters.stimulus_size = 0.8
mr.config.visual_stimulus_parameters.pix_per_deg = 60

mr.config.visual_stimulus_parameters.duration_seconds = 0.1
mr.config.visual_stimulus_parameters.baseline_start_seconds = 0.1
mr.config.visual_stimulus_parameters.baseline_end_seconds = 0.3
mr.config.visual_stimulus_parameters.pattern = "natural_image"

# Main loop
gc_types = ["parasol", "midget"]
response_types = ["on", "off"]


def simulate_retina():
    # Save transformed images to disk for simulation
    output_dir = Path(f"{image_rootpath}_transformed")
    transformed_filenames = save_transformed_images(output_dir, data_loader)

    for gc_type in gc_types:
        for response_type in response_types:
            mr.config.retina_parameters.gc_type = gc_type
            mr.config.retina_parameters.response_type = response_type

            mr.retina_constructor.construct()  # Reuses existing matching retina

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


match operation:
    case "simulate":
        simulate_retina()

    case "construct_model":
        reco = ImageReconstruction(mr.config, mr.data_io)

        R, S, _ = reco.get_spikes_and_images(
            n_images=n_images, gc_types=gc_types, response_types=response_types
        )
        W, S_mean = reco.create_model(R, S, ridge_lambda=0.0)

        # Pack W and S_mean into npz
        model_data = {"W": W, "S_mean": S_mean}

        mr.data_io.save_data(
            filename=model_filename, data=model_data, path=mr.config.path
        )

    case "reconstruct":
        reco = ImageReconstruction(mr.config, mr.data_io)

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

        S_test_img = reco.reconstruct_images(S_test, retina_mask)

        S_estimated = reco.estimate_model(W, R_test, S_mean)
        S_estimated_img = reco.reconstruct_images(S_estimated, retina_mask)

        print(
            f"Correlation: {np.corrcoef(S_estimated.flatten(), S_test.flatten())[0, 1]}"
        )


end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")


plt.show()
