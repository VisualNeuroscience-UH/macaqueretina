# Built-in
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
import torch


class ImageReconstruction:
    """
    Class for performing linear reconstruction of input images from
    responses of the four macaque RGC types.
    """

    def __init__(self, config, data_io, **kwargs) -> None:
        self._config = config
        self._data_io = data_io

        for attr, value in kwargs.items():
            setattr(self, attr, value)

    @property
    def config(self):
        return self._config

    @property
    def data_io(self):
        return self._data_io

    def linear_regression_fit(
        self, R, S, device="auto", dtype=torch.float32, ridge_lambda=0.0
    ):
        """
        Compute optimal weights W for the linear model S = R @ W using least squares.
        The ridge_lambda parameter controls the bias-variance tradeoff.
        The ridge_lambda = 0 corresponds to ordinary least squares, while
        ridge_lambda > 0 adds L2 regularization.

        Parameters
        ----------
        R : torch.Tensor
            Input tensor of shape (N_images, N_cells)
        S : torch.Tensor
            Target tensor of shape (N_images, N_pixels)
        device : str, optional
            Computation device ('cpu', 'cuda', or 'auto'), by default 'auto'
        dtype : torch.dtype, optional
            Data type for computation, by default torch.float32
        ridge_lambda : float, optional
            Ridge regularization coefficient, by default 0.0

        Returns
        -------
        W : torch.Tensor
            Weight tensor of shape (N_cells, N_pixels)

        Raises
        ------
        ValueError
            If input dimensions are inconsistent
        """

        if R.ndim != 2 or S.ndim != 2:
            raise ValueError("R and S must be 2-dimensional tensors")
        if R.shape[0] != S.shape[0]:
            raise ValueError(
                f"Number of images must match: R has {R.shape[0]}, S has {S.shape[0]}"
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        R = R.to(device=device, dtype=dtype)
        S = S.to(device=device, dtype=dtype)

        if ridge_lambda > 0:
            RtR = R.T @ R
            RtR.diagonal(dim1=-2, dim2=-1).add_(ridge_lambda)
            RtS = R.T @ S
            W = torch.linalg.solve(RtR, RtS)
        else:
            result = torch.linalg.lstsq(R, S)
            W = result.solution

        return W

    def _get_spike_filenames(self, gc_types: list[str], response_types: list[str]):
        """
        Get the filenames of the responses for a specific RGC type and response type.

        """
        output_folder = self.config.output_folder
        response_files = list(output_folder.glob("*results*.gz"))

        # Raise error if no response files are found
        if not response_files:
            raise FileNotFoundError(
                f"No response files found in {output_folder}. "
                "Please ensure that the responses have been generated and saved."
            )

        filenames = {
            gc_type: {response_type: [] for response_type in response_types}
            for gc_type in gc_types
        }
        for response_file in response_files:
            for gc_type in gc_types:
                for response_type in response_types:
                    if f"{gc_type}_{response_type}" in response_file.name:
                        filenames[gc_type][response_type].append(response_file)

        return filenames

    def _load_spikes(self, filenames: dict[str, dict[str, list[Path]]], n_images: int):
        """
        Load the responses of a specific RGC type and response type.

        Parameters
        ----------
        filenames : dict
            Dictionary containing the filenames of the responses for each RGC type and response type.
            The structure is:
            {
                "gc_type": {
                    "response_type": [Path, Path, ...]
                }
            }

        Returns
        -------
        spike_data : dict
            Dictionary containing the loaded spike responses for each RGC type and response type.

        """
        # Load the responses
        spike_data = {}
        for gc_type, response_dict in filenames.items():
            spike_data[gc_type] = {}
            for response_type, file_list in response_dict.items():
                # Sort files by video hash or filename to ensure consistent order
                file_list_sorted = sorted(file_list, key=lambda x: x.name)
                spike_data[gc_type][response_type] = []
                for filename in file_list_sorted[:n_images]:
                    data_dict = self.data_io.load_data(filename, hush=True)
                    spike_data[gc_type][response_type].append(data_dict)
        return spike_data

    def _load_images(self, n_images: int):
        stimulus_folder = self.config.stimulus_folder
        image_files = list(stimulus_folder.glob("stim_*.hdf5"))
        image_files = sorted(image_files, key=lambda x: x.name)  # Sort by filename

        return [self.data_io.load_data(f, hush=True) for f in image_files[:n_images]]

    def _get_response_matrix(
        self,
        spike_data_dicts: dict[str, dict[str, list[dict]]],
        gc_types: list[str],
        response_types: list[str],
        n_images: int,
        epoch: tuple[float, float] = (0, 0.15),
    ):
        """
        Get the response matrix R for the linear model S = R @ W.

        Parameters
        ----------
        spike_data_dicts : dict
            Dictionary containing the loaded spike responses for each RGC type and response type.
        gc_types : list
            List of RGC types to include in the response matrix.
        response_types : list
            List of response types to include in the response matrix.
        n_images : int
            Number of images to include in the response matrix.
        epoch : tuple, optional
            Time window (start, end) in seconds to consider for spike counting,
            by default (0, 0.15)

        Returns
        -------
        R : np.ndarray
            Response matrix of shape (N_images, N_cells)
        R_hash : np.ndarray
            Ordered array of video hashes corresponding to the spike data in R.

        Notes
        -----
        The order of looping gc_types and response_types is important to ensure that
        the response matrix R is constructed consistently. The order of the unit types in R
        corresponds to the order of gc_types and response_types provided as input.
        """
        n_unit_types = len(gc_types) * len(response_types)

        n_gc_units_per_type = [
            spike_data_dicts[gc_type][response_type][0]["n_units"]
            for gc_type in gc_types
            for response_type in response_types
        ]

        if len(set(n_gc_units_per_type)) != n_unit_types:
            raise ValueError("Inconsistent number of cells across unit types. ")

        R = np.zeros((n_images, sum(n_gc_units_per_type)))
        R_hash = np.zeros((n_images, n_unit_types), dtype=object)
        unit_i = 0
        type_i = 0
        for gc_type in gc_types:
            for response_type in response_types:
                spike_data_dict_list = spike_data_dicts[gc_type][response_type]
                n_units = spike_data_dict_list[0]["n_units"]
                bins = np.arange(n_units + 1)  # Bins for digitizing unit IDs

                for image_i, spike_data_dict in enumerate(spike_data_dict_list):
                    # This approach enables multiple n_sweeps which become keys spikes_0, spikes_1 ...
                    spike_keys = [k for k in spike_data_dict.keys() if "spikes" in k]
                    unit_ids = []
                    spike_times = []
                    for this_key in spike_keys:
                        unit_ids.append(spike_data_dict[this_key][0])
                        spike_times.append(spike_data_dict[this_key][1])

                    unit_ids_array = np.array(unit_ids)
                    spike_times_array = np.array(spike_times)

                    # Mask spikes that are outside the epoch
                    mask = (spike_times_array >= epoch[0]) & (
                        spike_times_array <= epoch[1]
                    )
                    unit_ids_array = unit_ids_array[mask]

                    (
                        R[
                            image_i,
                            unit_i : unit_i + n_units,
                        ],
                        _,
                    ) = np.histogram(
                        unit_ids_array,
                        bins=bins,
                    )

                    R_hash[image_i, type_i] = spike_data_dict["video_hash"]

                    if image_i + 1 >= n_images:
                        break

                unit_i += n_units
                type_i += 1

        # Check for hash order across unit types, i.e. that they saw the same video.
        if not (R_hash == R_hash[:, [0]]).all():
            raise ValueError(
                "Inconsistent video hashes across unit types for the same image. "
            )

        return R, R_hash

    def _get_stimulus_matrix(
        self, image_data_dicts: list[dict], n_images: int, retina_mask: np.ndarray
    ):
        """
        Get the stimulus matrix S for the linear model S = R @ W.

        Parameters
        ----------
        image_data_dicts : list
            List of dictionaries containing the loaded image data.
        n_images : int
            Number of images to include in the stimulus matrix.

        Returns
        -------
        S : np.ndarray
            Stimulus matrix of shape (N_images, N_pixels)
        S_hash : np.ndarray
            Ordered array of video hashes corresponding to the image data in S.
        """

        # Assuming all images have the same shape
        n_pixels = retina_mask.sum()

        S = np.zeros((n_images, n_pixels))
        retina_mask_flat = retina_mask.flatten()
        S_hash = np.zeros(n_images, dtype=object)

        baseline_len_tp = image_data_dicts[0]["baseline_len_tp"]
        n_stim_tp = image_data_dicts[0]["n_stim_tp"]
        time_mask = np.arange(baseline_len_tp, baseline_len_tp + n_stim_tp)

        for image_i, image_data_dict in enumerate(image_data_dicts):
            S[image_i, :] = (
                image_data_dict["frames"][time_mask]
                .mean(axis=0)
                .flatten()[retina_mask_flat]
            )
            S_hash[image_i] = image_data_dict["video_hash"]
            if image_i + 1 >= n_images:
                break

        return S, S_hash

    def get_spikes_and_images(
        self, n_images: int, gc_types: list[str], response_types: list[str]
    ):
        """
        Create a linear model for image reconstruction.

        Parameters
        ----------
        n_images : int
            Number of images.
        gc_types : list
            List of RGC types.
        response_types : list
            List of RGC response types ('on', 'off').

        Returns
        -------
        W : torch.Tensor
            Weight tensor of shape (N_cells, N_pixels)
        R : np.ndarray
            Response matrix of shape (N_images, N_cells)
        S : np.ndarray
            Stimulus matrix of shape (N_images, N_pixels)
        retina_mask : np.ndarray
            Boolean mask indicating the pixels corresponding to the retina patch
            inside the stimulus images.

        Notes
        -----
        The order of videos and responses are sorted according to filenames.
        The hash order check ensures that the the stimulus video matches the response.
        """

        filenames_spikes = self._get_spike_filenames(gc_types, response_types)

        spike_data_dicts = self._load_spikes(filenames_spikes, n_images)

        R, R_hash = self._get_response_matrix(
            spike_data_dicts, gc_types, response_types, n_images
        )

        retina_mask = spike_data_dicts["parasol"]["on"][0]["retina_patch_pixel_mask"]

        image_data_dicts = self._load_images(n_images)

        S, S_hash = self._get_stimulus_matrix(image_data_dicts, n_images, retina_mask)

        # Some images may be corrupted. These are removed.
        nan_rows = np.where(np.isnan(S).any(axis=1))[0]

        if len(nan_rows) > 0:
            S = np.delete(S, nan_rows, axis=0)
            S_hash = np.delete(S_hash, nan_rows, axis=0)
            R = np.delete(R, nan_rows, axis=0)
            R_hash = np.delete(R_hash, nan_rows, axis=0)

        # Check for video hash order between stimulus videos and responses.
        if not (R_hash == S_hash[:, np.newaxis]).all():
            raise ValueError(
                "Inconsistent video hashes between stimulus videos and responses."
            )

        return R, S, retina_mask

    def create_model(self, R, S, ridge_lambda=0.0):
        """
        Create a linear model for image reconstruction.

        Parameters
        ----------
        R : np.ndarray
            Response matrix of shape (N_images, N_cells)
        S : np.ndarray
            Stimulus matrix of shape (N_images, N_pixels)
        ridge_lambda : float, optional
            Ridge regularization coefficient, by default 0.0

        Returns
        -------
        W : torch.Tensor
            Weight tensor of shape (N_cells, N_pixels)
        """

        if R.shape[0] < 2:
            raise ValueError(
                "At least two images and responses are required to create a model."
            )

        S_mean = S.mean(axis=0, keepdims=True)

        # Remove mean values from R and S to center the data
        R -= R.mean(axis=0, keepdims=True)
        S -= S_mean
        R_tensor = torch.tensor(R, dtype=torch.float32)
        S_tensor = torch.tensor(S, dtype=torch.float32)

        W = self.linear_regression_fit(R_tensor, S_tensor, ridge_lambda=ridge_lambda)

        return W.cpu().numpy(), S_mean

    def estimate_model(self, W, R, S_mean):
        """
        Estimate the stimulus matrix S from the response matrix R and weight matrix W.

        Parameters
        ----------
        W : np.ndarray
            Weight matrix of shape (N_cells, N_pixels)
        R : np.ndarray
            Response matrix of shape (N_images, N_cells)
        S_mean : np.ndarray
            Mean stimulus values of shape (1, N_pixels)

        Returns
        -------
        S_estimate : np.ndarray
            Estimated stimulus matrix of shape (N_images, N_pixels)
        """

        if R.shape[0] < 2:
            raise ValueError(
                "At least two images and responses are required to estimate a model."
            )

        R -= R.mean(axis=0, keepdims=True)

        S_estimate = R @ W
        S_estimate += S_mean  # Return the mean values to the estimated stimulus matrix

        return S_estimate

    def reconstruct_images(self, S, retina_mask):
        """
        Create an estimate of the input images from the responses and weights.

        Parameters
        ----------
        S : np.ndarray
            Stimulus matrix of shape (N_images, N_pixels)
        retina_mask : np.ndarray
            Boolean mask indicating the pixels corresponding to the retina patch.

        Returns
        -------
        S_img : np.ndarray
            Estimated images of shape (N_images, original_image_size)
        """
        S = np.expand_dims(S, 0) if S.ndim == 1 else S
        S_img = np.zeros((S.shape[0], retina_mask.size))
        S_img[:, retina_mask.flatten()] = S

        n_samples = S_img.shape[0]
        original_size = S_img.shape[1]
        height = int(np.sqrt(original_size))
        width = int(np.sqrt(original_size))

        if height * width != original_size:
            raise ValueError(
                f"Original size {original_size} is not a square, cannot reshape to 2D image."
            )

        S_img = S_img.reshape((n_samples, height, width))

        mean_value = self.config.visual_stimulus_parameters.mean
        mean_background = np.where(retina_mask, 0, mean_value)

        # Broadcast the mean background to all images
        S_img += mean_background[np.newaxis, :, :]

        return S_img
