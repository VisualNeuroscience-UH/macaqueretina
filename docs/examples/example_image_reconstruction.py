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

mr.load_parameters()


start_time = time.time()

rootpath = Path("/opt3/images/vanHateren/imc_images")
# rootpath = Path("/opt3/images/ImageNet")
H = W = 256
n_images = 1000
# torch.manual_seed(42)

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
        with open(self.image_paths[idx], "rb") as handle:
            s = handle.read()

        img = np.frombuffer(s, dtype="uint16").byteswap()

        img = img.reshape(1024, 1536).astype(np.float32)

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
        root=rootpath,
        transform=None,
    )

    # 2. Load precalculated indices for images with resolution >= (H, W)
    full_path = rootpath / f"imagenet_train_{H}_{W}_indices.pt"
    if full_path.exists():
        indices = torch.load(full_path)
        print(f"Loaded {len(indices)} valid indices from {full_path}")
    else:
        print(
            f"Indices file {full_path} not found. Getting and saving ImageNet indices..."
        )
        create_filtered_imagenet(rootpath, H, W)
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
    full_dataset = VanHaterenDataset(root_dir=rootpath)

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


def show_original_vs_transformed(data_loader, filenames, wait_time=0.5):
    """
    Quality control.
    Display original and transformed images side-by-side, one pair at a time.
    """
    import matplotlib.image as mpimg

    dataset = data_loader.dataset

    if len(filenames) != len(dataset):
        raise ValueError(
            f"Length of filenames ({len(filenames)}) does not match dataset length ({len(dataset)})"
        )

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Original vs Transformed")

    for idx in range(len(dataset)):
        # Get transformed image
        img_transformed, _ = dataset[idx]
        img_transformed = img_transformed.numpy()

        # Handle channel dimension
        if img_transformed.ndim == 3 and img_transformed.shape[0] == 1:
            img_transformed = img_transformed.squeeze(0)

        # Get original image
        img_original = mpimg.imread(filenames[idx])
        if img_original.ndim == 3 and img_original.shape[2] == 3:
            img_original = np.mean(img_original, axis=2)
        if img_original.max() > 1:
            img_original = img_original / 255.0

        # Display
        axes[0].imshow(img_original, cmap="gray", vmin=0, vmax=1)
        axes[0].imshow(img_original, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(img_transformed, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Transformed")
        axes[1].axis("off")

        plt.draw()
        plt.pause(wait_time)

        axes[0].clear()
        axes[1].clear()

    plt.ioff()


def save_transformed_images(output_dir, data_loader, filenames):
    """
    Save transformed images to disk for later use.
    """
    dataset = data_loader.dataset

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


data_loader = get_vanhateren_dataloader(batch_size=4, shuffle=True, num_workers=0)
# data_loader = get_imagenet_dataloader(batch_size=4, shuffle=True, num_workers=0)
filenames = data_loader.filenames


# Quality control
# show_original_vs_transformed(data_loader, filenames, wait_time=0.5)

# Save transformed ImageNet images to disk for later use
output_dir = Path(f"{rootpath}_transformed")
transformed_filenames = save_transformed_images(output_dir, data_loader, filenames)

# # Get one batch from the DataLoader
# images, labels = next(iter(data_loader))

# Spatial parameters. H = external stimulus height (pix), W = external stimulus width (pix)
mr.config.external_stimulus_parameters.ext_pix_per_deg = 30

mr.config.visual_stimulus_parameters.image_height = 120
mr.config.visual_stimulus_parameters.image_width = 120
mr.config.visual_stimulus_parameters.stimulus_size = 0.8
mr.config.visual_stimulus_parameters.pix_per_deg = 60

mr.config.visual_stimulus_parameters.duration_seconds = 0.1
mr.config.visual_stimulus_parameters.baseline_start_seconds = 0.1
mr.config.visual_stimulus_parameters.baseline_end_seconds = 0.3
mr.config.visual_stimulus_parameters.pattern = "natural_image"

# plt.ion()

# Main loop
gc_types = ["parasol", "midget"]
response_types = ["on", "off"]
for gc_type in gc_types:
    for response_type in response_types:
        mr.config.retina_parameters.gc_type = gc_type
        mr.config.retina_parameters.response_type = response_type

        mr.retina_constructor.construct()  # Reuses existing matching retina

        for this_name in transformed_filenames:
            mr.config.external_stimulus_parameters.ext_stimulus_file = str(this_name)

            simulation_results_filename, stimulus_video_name = get_filenames(this_name)

            mr.config.visual_stimulus_parameters.stimulus_video_name = (
                stimulus_video_name
            )

            mr.stimulus_factory.generate()  # Reuses existing matching stimulus

            mr.retina_simulator.simulate(filename=simulation_results_filename)

            # mr.viz.show_all_gc_responses_after_simulate(savefigname=None)
            # mr.viz.show_stimulus_with_gcs(frame_number=31, savefigname=None)
            # plt.show()
            # plt.draw()

# plt.ioff()

print(f"Output folder: {mr.config.output_folder}")

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")


plt.show()
