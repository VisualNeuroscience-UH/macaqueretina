# Built-in
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from cv2 import resize
from scipy.ndimage import rotate
from scipy.signal import resample_poly
from torchvision.transforms.functional import rotate

plt.rcParams["image.cmap"] = "gray"

"""
This module creates the visual stimuli.

Input: stimulus definition, image (formats jpg, png) or video (formats avi, mp4)
Output: video stimulus frames in hdf5 (reloading) and mp4 (viewing)
"""

if TYPE_CHECKING:
    from macaqueretina.data_io.config_io import Configuration
    from macaqueretina.data_io.data_io_module import DataIO
    from macaqueretina.viz.viz_module import Viz


class VideoBaseClass:
    def __init__(self):
        """
        Initialize standard video stimulus
        The base class methods are applied to every stimulus
        """
        options = {}
        options["image_height"] = 720  # Image height in pixels
        options["image_width"] = 1280  # Image width in pixels
        options["container"] = "mp4"  # file format to export
        options["codec"] = "mp4v"  # only mp4v works for my ubuntu 22.04
        options["fps"] = 100.0  # 64.0  # Frames per second
        options["duration_seconds"] = 1.0  # seconds
        # Video luminance range. If none, defined by mean and contrast.
        options["intensity"] = None
        options["mean"] = 128  # intensity mean
        options["contrast"] = 1

        # Dynamic range before scaling, set by each stimulus pattern method
        options["raw_intensity"] = None
        options["pattern"] = "sine_grating"

        # 0 - 2pi, to have grating or temporal oscillation phase shifted
        options["phase_shift"] = 0
        options["stimulus_form"] = "circular"

        # Stimulus center position in degrees inside the video. (0,0) is the center.
        options["stimulus_position"] = (0.0, 0.0)

        # In degrees. Radius for circle and annulus, half-width for rectangle. 0 gives smallest distance from image borders, ie max radius
        options["stimulus_size"] = 0.0

        # Init optional arguments
        options["spatial_frequency"] = None
        options["temporal_frequency"] = None
        options["temporal_frequency_range"] = None
        options["temporal_band_pass"] = None
        options["orientation"] = 0.0  # No rotation or vertical
        options["size_inner"] = None
        options["size_outer"] = None

        # Binary noise options
        # between 0 and 1, proportion of stimulus-on time
        options["on_proportion"] = 0.5
        options["on_time"] = 0.1  # in seconds
        options["direction"] = "increment"  # or 'decrement'

        # Limits, no need to go beyond these
        options["min_spatial_frequency"] = 0.0625  # cycles per degree
        options["max_spatial_frequency"] = 16.0  # cycles per degree
        options["min_temporal_frequency"] = 0.5  # cycles per second, Hz
        options["max_temporal_frequency"] = 32.0  # cycles per second, Hz.

        options["background"] = 128  # Background grey value

        # Get resolution
        options["pix_per_deg"] = 60

        # For external image and video files
        options["ext_pix_per_deg"] = None
        options["ext_stimulus_file"] = None

        options["baseline_start_seconds"] = 0
        options["baseline_end_seconds"] = 0
        options["stimulus_video_name"] = None
        options["dtype_name"] = "uint8"
        options["ND_filter"] = 0.0  # ND filter value at log10 units

        # Run parameter, but necessary to log to experiiment metadata
        options["n_sweeps"] = 1
        options["video_hash"] = "video_hash"

        self.options = options

    def _scale_intensity(self):
        """
        Scale intensity to 8-bit grey scale. Calculating peak-to-peak here allows different
        luminances and contrasts
        """

        raw_min_value = np.min(self.options["raw_intensity"])
        raw_max_value = np.max(self.options["raw_intensity"])
        raw_peak_to_peak = np.ptp(self.options["raw_intensity"])
        frames = self.frames

        # Rotation may exceed min and max values by antialiasing
        frames[frames < raw_min_value] = raw_min_value
        frames[frames > raw_max_value] = raw_max_value

        if self.options["intensity"] is not None:
            Lmax = np.max(self.options["intensity"])
            Lmin = np.min(self.options["intensity"])
        else:
            mean = self.options["mean"]  # This is the mean of final dynamic range
            contrast = self.options["contrast"]
            Lmax = mean * (1 + contrast)
            Lmin = mean * (1 - contrast)

        peak_to_peak = Lmax - Lmin

        frames = frames - raw_min_value
        frames = frames / raw_peak_to_peak
        frames = frames * peak_to_peak
        frames = frames + Lmin

        self.frames = frames

    def _rotate(self, frames: np.ndarray, orientation: float) -> np.ndarray:
        """
        Rotate frames using PyTorch and GPU acceleration.

        Parameters:
        -----------
        frames: np.ndarray
            Input array of shape [time points, height, width].
        orientation: float
            Rotation angle in degrees.

        Returns:
        -------
            Rotated frames as a numpy array.
        """
        # Convert numpy array to PyTorch tensor and move to GPU
        frames_tensor = torch.from_numpy(frames).to(
            self.config.device, dtype=torch.float32
        )

        # Reshape to [time_points, 1, height, width] for batch processing
        frames_tensor = frames_tensor.unsqueeze(1)

        # Rotate each frame in the batch
        rotated_frames_tensor = rotate(frames_tensor, angle=orientation)

        # Remove the channel dimension and convert back to numpy
        rotated_frames = rotated_frames_tensor.squeeze(1).cpu().numpy()

        return rotated_frames

    def _prepare_grating(self, grating_type="sine"):
        """Create temporospatial grating based on specified grating type."""

        # Common setup for both grating types
        spatial_frequency = self.options.get("spatial_frequency", 1)
        temporal_frequency = self.options.get("temporal_frequency", 1)
        fps = self.options["fps"]
        duration_seconds = self.options["duration_seconds"]
        orientation = self.options["orientation"]
        image_height = self.options["image_height"]
        image_width = self.options["image_width"]
        diameter = np.ceil(np.sqrt(image_height**2 + image_width**2)).astype(np.int32)
        image_width_diameter = diameter
        image_height_diameter = diameter
        n_frames = int(fps * duration_seconds)
        # scipy.ndimage.rotate cannot handle 16 bit data => float32
        frames = np.zeros(
            (n_frames, image_height_diameter, image_width_diameter)
        ).astype(np.float16)

        # Specific part for sine grating
        if grating_type == "sine":
            one_cycle = 2 * np.pi
            cycles_per_degree = spatial_frequency
            image_position_vector = np.linspace(
                0,
                one_cycle
                * cycles_per_degree
                * image_width_diameter
                / self.options["pix_per_deg"],
                image_width_diameter,
            )
            large_frames = np.tile(
                image_position_vector, (image_height_diameter, n_frames, 1)
            )
            large_frames = np.moveaxis(large_frames, 1, 0)
            temporal_shift_vector = np.linspace(
                0,
                temporal_frequency * one_cycle * duration_seconds
                - (temporal_frequency * one_cycle) / fps,
                n_frames,
            )
            large_frames += temporal_shift_vector[:, np.newaxis, np.newaxis]
            frames = large_frames

            # Turn to sine values
            frames += self.options["phase_shift"]
            frames = np.sin(frames)

        # Specific part for square grating
        elif grating_type == "square":
            cycle_width_pix = self.options["pix_per_deg"] / spatial_frequency

            phase_shift_in_pixels = cycle_width_pix * (
                self.options["phase_shift"] / (2 * np.pi)
            )

            # X coords refects luminance values, % 2 < 1 is white, % 2 > 1 is black
            bar_coords = np.arange(image_width_diameter) / (cycle_width_pix / 2)
            # Apply the phase shift in the square grating calculation
            for frame in range(n_frames):
                temporal_shift = cycle_width_pix * temporal_frequency * frame / fps
                relative_bar_coords = (
                    bar_coords
                    + ((temporal_shift + phase_shift_in_pixels) / (cycle_width_pix / 2))
                ) % 2
                frames[frame] = np.where(relative_bar_coords < 1, 1, -1)

        # Common post-processing: Rotate and cut to original dimensions
        # rotate(frames, orientation, axes=(2, 1), reshape=False, output=frames, order=3)
        frames = self._rotate(frames, orientation)

        marginal_height = (diameter - image_height) // 2
        marginal_width = (diameter - image_width) // 2
        frames = frames[
            :, marginal_height:-marginal_height, marginal_width:-marginal_width
        ]

        # In case of rounding errors, clip to image_height and image_width
        frames = frames[:, :image_height, :image_width]

        self.frames = frames
        # Set raw_intensity to [-1 1]
        self.options["raw_intensity"] = (-1, 1)

    def _prepare_form(
        self, stimulus_size: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        center_deg = self.options["stimulus_position"]  # in degrees
        radius_deg = stimulus_size  # in degrees
        height = self.options["image_height"]  # in pixels
        width = self.options["image_width"]  # in pixels
        pix_per_deg = self.options["pix_per_deg"]

        # Turn position in degrees to position in mask, shift 0,0 to center of image
        center_pix = np.array([0, 0])
        # NOTE Width goes to x-coordinate
        center_pix[0] = int(width / 2 + pix_per_deg * center_deg[0])
        # NOTE Height goes to y-coordinate. Inverted to get positive up
        center_pix[1] = int(height / 2 + pix_per_deg * -center_deg[1])

        self.options["center_pix"] = center_pix  # x, y

        if (
            radius_deg == 0
        ):  # use the smallest distance between the center and image walls
            radius_pix = min(
                center_pix[0],
                center_pix[1],
                width - center_pix[0],
                height - center_pix[1],
            )
        else:
            radius_pix = pix_per_deg * radius_deg

        Y, X = np.ogrid[:height, :width]

        return X, Y, center_pix, radius_pix

    def _prepare_circular_mask(self, stimulus_size: float) -> np.ndarray:
        X, Y, center_pix, radius_pix = self._prepare_form(stimulus_size)
        dist_from_center = np.sqrt((X - center_pix[0]) ** 2 + (Y - center_pix[1]) ** 2)

        mask = dist_from_center <= radius_pix
        return mask

    def _combine_background(self, mask: np.ndarray) -> None:
        frames = self.frames
        shape = self.frames.shape
        background = self.options["background"]

        frames_background = np.ones(shape) * background
        frames_background[:, mask] = frames[:, mask]

        self.frames = frames_background

    def _prepare_temporal_sine_pattern(self):
        """Prepare temporal sine pattern"""

        temporal_frequency = self.options["temporal_frequency"]
        fps = self.options["fps"]
        duration_seconds = self.options["duration_seconds"]
        phase_shift = self.options["phase_shift"]

        if not temporal_frequency:
            print("Temporal_frequency missing, setting to 1")
            temporal_frequency = 1

        # Create sine wave
        n_frames = self.frames.shape[0]
        image_height = self.options["image_height"]
        image_width = self.options["image_width"]

        # time_vector in radians, temporal modulation via np.sin()
        time_vec_end = 2 * np.pi * temporal_frequency * duration_seconds
        time_vec = np.linspace(
            0 + phase_shift, time_vec_end + phase_shift, int(fps * duration_seconds)
        )
        temporal_modulation = np.sin(time_vec)

        # Set the frames to sin values
        frames = (
            np.ones(self.frames.shape) * temporal_modulation[:, np.newaxis, np.newaxis]
        )

        # Set raw_intensity to [-1 1]
        self.options["raw_intensity"] = (-1, 1)

        assert temporal_modulation.shape[0] == n_frames, "Unequal N frames, aborting..."
        assert (
            image_height != n_frames
        ), "Errors in 3D broadcasting, change image width/height NOT to match n frames "
        assert (
            image_width != n_frames
        ), "Errors in 3D broadcasting, change image width/height NOT to match n frames "

        self.frames = frames

    def _prepare_temporal_chirp_pattern(self):
        """Prepare temporal chirp pattern"""

        temporal_frequency_range = self.options["temporal_frequency_range"]
        fps = self.options["fps"]
        duration_seconds = self.options["duration_seconds"]
        phase_shift = self.options["phase_shift"]

        if not temporal_frequency_range:
            print("Temporal_frequency_range missing, setting to (1, 10)")
            start_frequency, end_frequency = (1, 10)
        else:
            start_frequency, end_frequency = temporal_frequency_range

        # Create chirp signal
        n_frames = self.frames.shape[0]
        image_height = self.options["image_height"]
        image_width = self.options["image_width"]

        # Time vector in seconds
        time_vec = np.linspace(0, duration_seconds, int(fps * duration_seconds))

        # Linearly increasing frequency over time
        k = (end_frequency - start_frequency) / duration_seconds

        # Creating the chirp signal
        phase = (
            2 * np.pi * (start_frequency * time_vec + (k / 2) * time_vec**2)
            + phase_shift
        )
        temporal_modulation = np.sin(phase)

        # Set the frames to chirp values
        frames = (
            np.ones(self.frames.shape) * temporal_modulation[:, np.newaxis, np.newaxis]
        )

        # Set raw_intensity to [-1 1]
        self.options["raw_intensity"] = (-1, 1)

        assert temporal_modulation.shape[0] == n_frames, "Unequal N frames, aborting..."
        assert (
            image_height != n_frames
        ), "Errors in 3D broadcasting, change image width/height NOT to match n frames "
        assert (
            image_width != n_frames
        ), "Errors in 3D broadcasting, change image width/height NOT to match n frames "

        self.frames = frames

    def _raw_intensity_from_data(self):
        self.options["raw_intensity"] = (np.min(self.frames), np.max(self.frames))

    def _create_frames(self, epoch_in_seconds: float) -> np.ndarray:
        # Create frames for the requested duration in sec
        frames = (
            np.ones(
                (
                    int(self.options["fps"] * epoch_in_seconds),
                    self.options["image_height"],
                    self.options["image_width"],
                ),
                dtype=self.options["dtype"],
            )
            * self.options["background"]
        )

        return frames

    def _set_zero_masked_pixels_to_bg(self):
        # Set masked pixels to background value.
        mask_value = 1
        self.frames[self.frames == mask_value] = self.options["background"]

    def _extract_frames(self, cap: cv2.VideoCapture, frames: np.ndarray) -> np.ndarray:
        """
        Loads the frames from a cv2.VideoCapture object s array of gray values between 0-255.

        Parameters
        ----------
        cap : cv2.VideoCapture
            The video capture object from which frames are extracted.
        frames: np.ndarray
            Empty 3D array for loading the frames
        """
        cvtColor = cv2.cvtColor
        COLOR_BGR2GRAY = cv2.COLOR_BGR2GRAY

        # Load each frame as array of gray values between 0-255
        for frame_ix in range(frames.shape[0]):
            _, this_frame = cap.read()

            frames[frame_ix, :, :] = cvtColor(this_frame, COLOR_BGR2GRAY)

        return frames

    def _spatial_resample(
        self,
        frames: np.ndarray,
        req_video_height: int,
        req_video_width: int,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        """
        Spatial resampling of frames

        Parameters
        ----------
        frames : nd.array
            Input video 3D array [time, height, width]
        req_video_height : int
            Input video requested image height (no aspect ration change)
        req_video_width : int
            Input video requested image width (no aspect ration change)
        target_height : int
            Target image height
        target_width : int
            Target image width
        """

        video_n_frames = frames.shape[0]
        video_height = frames.shape[1]
        video_width = frames.shape[2]

        height_cut = int((video_height - req_video_height) / 2)
        width_cut = int((video_width - req_video_width) / 2)
        frames_cut = frames[:, height_cut:-height_cut, width_cut:-width_cut]

        frames_resampled = np.zeros((video_n_frames, target_height, target_width))

        INTER_AREA = cv2.INTER_AREA
        for frame_ix in range(video_n_frames):
            frame_in = frames_cut[frame_ix, ...]
            frames_resampled[frame_ix, ...] = resize(
                frame_in, (target_width, target_height), interpolation=INTER_AREA
            )

        return frames_resampled

    def _temporal_resample(
        self,
        frames: np.ndarray,
        req_video_n_frames: int,
        target_n_frames: int,
        mean_value=128,
    ) -> np.ndarray:
        """
        Temporal resampling of frames

        Parameters
        ----------
        frames : nd.array
            Input video 3D array [time, height, width]
        req_video_n_frames: int
            Requested N frames from input video
        target_n_frames: int
            Target N frames
        mean_value: float
            Stimulus mean value for padding during resampling
        """
        ratio = target_n_frames / req_video_n_frames
        fraction = Fraction(ratio).limit_denominator()
        up = fraction.numerator
        down = fraction.denominator
        resampled_frames = resample_poly(
            frames[:req_video_n_frames, ...],
            up=up,
            down=down,
            axis=0,
            padtype="constant",
            cval=mean_value,
            window=("hann",),
        )

        if resampled_frames.shape[0] != target_n_frames:
            raise ValueError(
                f"Temporal resampling error: got {resampled_frames.shape[0]} frames, expected {target_n_frames} frames"
            )

        return resampled_frames


class StimulusPattern:
    """
    Construct the stimulus image pattern.
    This class is for isolating logic. Self comes from
    the calling function as argument
    """

    def sine_grating(self):
        """
        Create a sine wave grating stimulus pattern.

        This method applies a sine function to the temporospatial grating
        frames, with an optional phase shift. It sets the raw intensity
        range of the stimulus pattern to [-1, 1].
        """

        self._prepare_grating(grating_type="sine")

    def square_grating(self):
        """
        Create a square wave grating stimulus pattern.
        """

        self._prepare_grating(grating_type="square")

    def white_gaussian_noise(self):
        """
        Generate a white Gaussian noise stimulus pattern.

        This method fills the frames with white Gaussian noise, using a normal
        distribution centered at 0.0 with a standard deviation of 1.0. The shape
        of the noise array matches the shape of the existing frames.

        After generating the noise, it updates the raw intensity values based on
        the data in the frames.
        """

        self.frames = np.random.normal(loc=0.0, scale=1.0, size=self.frames.shape)

        self._raw_intensity_from_data()

    def white_uniform_noise(self):
        """
        Generate a white uniform noise stimulus pattern.

        This method fills the frames with white Gaussian noise, using a normal
        distribution centered at 0.0 with a standard deviation of 1.0. The shape
        of the noise array matches the shape of the existing frames.

        After generating the noise, it updates the raw intensity values based on
        the data in the frames.
        """

        self.frames = np.random.uniform(loc=0.0, scale=1.0, size=self.frames.shape)

        self._raw_intensity_from_data()

    def temporal_sine_pattern(self):
        """
        Create a temporal sine wave pattern.

        This method prepares a temporal sine wave pattern by invoking the
        `_prepare_temporal_sine_pattern` method, which handles the detailed
        implementation of this pattern.
        """
        self._prepare_temporal_sine_pattern()

    def temporal_chirp_pattern(self):
        """
        Create a temporal chirp pattern.

        This method prepares a temporal sine wave pattern by invoking the
        `_prepare_temporal_chirp_pattern` method, which handles the detailed
        implementation of this pattern.
        """
        self._prepare_temporal_chirp_pattern()

    def contrast_chirp_pattern(self):
        """
        Create a contrast chirp pattern.

        This method prepares a contrast chirp pattern by invoking the
        '_prepare_temporal_sine_pattern' method, and then creates a linear
        ramp for amplitude.
        """
        self._prepare_temporal_sine_pattern()
        # Create linear ramp for amplitude from 0 to 1
        amplitude_ramp = np.linspace(0, 1, self.n_stim_tp)
        self.frames = self.frames * amplitude_ramp[:, np.newaxis, np.newaxis]

    def temporal_square_pattern(self):
        """
        Create a temporal square wave pattern.

        This method starts by preparing a temporal sine pattern using
        `_prepare_temporal_sine_pattern`. It then converts this pattern
        into a square wave pattern by applying a threshold, defaulting to zero.
        Values equal to or above the threshold are set to 1, and values below
        are set to -1.

        The threshold can be adjusted between [-1, 1] for creating uneven
        grating patterns.
        """
        self._prepare_temporal_sine_pattern()
        # Turn to square grating values, threshold at zero.
        threshold = (
            0  # Change this between [-1 1] if you want uneven grating. Default is 0
        )

        self.frames[self.frames >= threshold] = 1
        self.frames[self.frames < threshold] = -1

    def temporal_digitized_pattern(self):
        """
        Create a temporal wave pattern from digitized temporal sequence.

        """
        filepath = self.config.literature_data_files["temporal_pattern_datafile"]
        data_npz = self.data_io.load_data(filepath)
        tp, amp = self.get_xy_from_npz(data_npz)
        duration = self.options["duration_seconds"]

        # Transform digitized times to stimulus duration
        tp_min = np.min(tp)
        tp_max = np.max(tp)
        tp_zeroed = tp - tp_min
        tp_scaled = tp_zeroed / (tp_max - tp_min)
        tp_scaled = tp_scaled * duration

        # Get closest frame index
        fps = self.options["fps"]
        n_frames = self.frames.shape[0]
        frame_indices = np.round(tp_scaled * fps).astype(int)
        intensity_array = np.zeros(n_frames)

        for idx in range(len(frame_indices) - 1):
            fr_idx = range(frame_indices[idx + 1] - frame_indices[idx])
            d_y = (amp[idx + 1] - amp[idx]) / len(fr_idx)
            y0 = amp[idx]
            fr_y = y0 + d_y * fr_idx
            intensity_array[frame_indices[idx] : frame_indices[idx + 1]] = fr_y

        # Scale intensity to [-1, 1]
        self.options["raw_intensity"] = (-1, 1)
        intensity_array = intensity_array - np.min(intensity_array)
        intensity_array = intensity_array / np.max(intensity_array)
        intensity_array = intensity_array * 2 - 1

        self.frames = np.zeros(self.frames.shape) + intensity_array[:, None, None]

    def spatially_uniform_binary_noise(self):
        """
        Generate a spatially uniform binary noise pattern.

        This method creates a binary noise pattern based on the specified 'on_proportion'
        and applies it to all frames. The noise is either incrementing or decrementing
        based on the 'direction' option. The contrast is adjusted to account for the
        dynamic range of the stimulus.

        The method throws a NotImplementedError if the 'direction' is neither 'increment'
        nor 'decrement'. The raw intensity range is set to [-1, 1].
        """
        on_proportion = self.options["on_proportion"]
        on_time = self.options["on_time"]
        fps = self.options["fps"]
        number_of_frames = self.frames.shape[0]
        direction = self.options["direction"]

        def _rand_bin_array(
            number_of_frames, on_proportion, number_of_successive_on_frames
        ):
            N = number_of_frames // number_of_successive_on_frames
            K = np.uint(N * on_proportion)
            arr = np.zeros(N)
            arr[:K] = 1
            np.random.shuffle(arr)
            arr = np.repeat(arr, number_of_successive_on_frames)

            if len(arr) < number_of_frames:
                arr = np.append(arr, np.zeros(number_of_frames - len(arr)))
            elif len(arr) > number_of_frames:
                arr = arr[:number_of_frames]

            return arr

        number_of_successive_on_frames = int(on_time * fps)
        frame_time_series = _rand_bin_array(
            number_of_frames, on_proportion, number_of_successive_on_frames
        )

        if direction == "decrement":
            frame_time_series = frame_time_series * -1  # flip
        elif direction == "increment":
            frame_time_series = frame_time_series * 1
        else:
            raise NotImplementedError(
                'Unknown option for "direction", should be "increment" or "decrement"'
            )

        # Note the stim has dyn range 1 and thus contrast will be halved by the
        # dyn range -1 1, thus the doubling
        # self.options["contrast"] = self.options["contrast"] * 2
        self.options["raw_intensity"] = (-1, 1)
        self.frames = (
            np.zeros(self.frames.shape) + frame_time_series[:, np.newaxis, np.newaxis]
        )

    def natural_image(self):
        """
        Process natural images for use in stimulus patterns.

        This method handles natural images loading an image file based on the provided
        stimulus metadata. The selected image is then resized to match the frame dimensions.
        The resized image is integrated with the frames by multiplying it, enabling the
        creation of astimulus pattern.

        After this integration, the method updates the raw intensity values based on the new data.
        """
        image_file_name = self.config.external_stimulus_parameters["ext_stimulus_file"]
        self.image = self.data_io.load_data(image_file_name)

        # resize image by specifying custom width and height
        resized_image = resize(self.image, self.frames.shape[1:])

        # add new axis to b to use numpy broadcasting
        resized_image = resized_image[np.newaxis, :, :]

        self.frames = self.frames * resized_image

        self._raw_intensity_from_data()

    def natural_video(self):
        """
        Process natural video for use in stimulus patterns.

        This method processes a natural video file specified in the stimulus metadata.
        It extracts frames from the video, resizes them to match the desired dimensions,
        and integrates them into the stimulus frames. The method also sets the frames
        to a specified frames per second (fps) and pixels per degree (pix_per_deg).

        After processing, it updates the raw intensity values based on the new data.
        """

        target_fps = self.options["fps"]
        target_pix_per_deg = self.options["pix_per_deg"]
        video_pix_per_deg = self.config.external_stimulus_parameters["ext_pix_per_deg"]

        target_height = self.frames.shape[1]
        target_width = self.frames.shape[2]
        target_n_frames = self.frames.shape[0]

        # Get external input video
        video_file_name = self.config.external_stimulus_parameters["ext_stimulus_file"]
        video_cap = self.data_io.load_data(video_file_name)

        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_n_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)

        # Rquired dimensions with no empty background or aspect ratio change or speed change
        req_video_height = int(target_height * (video_pix_per_deg / target_pix_per_deg))
        req_video_width = int(target_width * (video_pix_per_deg / target_pix_per_deg))
        req_video_n_frames = int(target_n_frames * (video_fps / target_fps))

        if video_height < req_video_height:
            raise ImportError("Input video too small for requested height in pixels")
        if video_width < req_video_width:
            raise ImportError("Input video too narrow for requested width in pixels")
        if video_n_frames < req_video_n_frames:
            raise ImportError("Input video shorter than requested n frames")

        video_frames = np.zeros((video_n_frames, video_height, video_width))

        video_frames = self._extract_frames(video_cap, video_frames)
        video_frames_res = self._spatial_resample(
            video_frames, req_video_height, req_video_width, target_height, target_width
        )

        mean_value = self.config.visual_stimulus_parameters.mean
        self.frames = self._temporal_resample(
            video_frames_res, req_video_n_frames, target_n_frames, mean_value=mean_value
        )
        self.fps = target_fps

        print(
            "Original movie dimensions %d height %d width, %d frames at %d fps."
            % (video_height, video_width, video_n_frames, video_fps)
        )
        print(
            "From which we used %d height %d width, %d frames."
            % (req_video_height, req_video_width, req_video_n_frames)
        )
        print(
            "To get movie with dimensions %d height %d width, %d frames at %d fps."
            % (
                self.frames.shape[1],
                self.frames.shape[2],
                self.frames.shape[0],
                self.fps,
            )
        )

        self._raw_intensity_from_data()
        video_cap.release()


class StimulusForm:
    """
    Mask the stimulus images. This class is for isolating logic.
    Self comes from the calling function as argument.
    """

    def circular(self):
        mask = self._prepare_circular_mask(self.options["stimulus_size"])

        self._combine_background(mask)

    def rectangular(self):
        X, Y, center_pix, radius_pix = self._prepare_form(self.options["stimulus_size"])

        # Prepare rectangular distance map in pixels
        x_distance_vector = np.abs(X - center_pix[0])
        X_distance_matrix = np.tile(x_distance_vector, (Y.shape[0], 1))
        y_distance_vector = np.abs(Y - center_pix[1])
        Y_distance_matrix = np.tile(y_distance_vector, (1, X.shape[1]))
        mask = np.logical_and(
            (X_distance_matrix <= radius_pix), (Y_distance_matrix <= radius_pix)
        )

        self._combine_background(mask)

    def annulus(self):
        size_inner = self.options["size_inner"]
        size_outer = self.options["size_outer"]
        if not size_inner:
            print("Size_inner missing, setting to 1")
            size_inner = 1
        if not size_outer:
            print("Size_outer missing, setting to 2")
            size_outer = 2

        mask_inner = self._prepare_circular_mask(size_inner)
        mask_outer = self._prepare_circular_mask(size_outer)

        mask = mask_outer ^ mask_inner
        # self.frames = self.frames * mask[..., np.newaxis]
        self._combine_background(mask)


class VisualStimulus(VideoBaseClass):
    """
    Create stimulus video and save
    """

    def __init__(
        self, config: Configuration, data_io: DataIO, get_xy_from_npz: callable
    ):
        super().__init__()

        self._config = config
        self._data_io = data_io
        self.get_xy_from_npz = get_xy_from_npz

    @property
    def config(self) -> Configuration:
        return self._config

    @property
    def data_io(self) -> DataIO:
        return self._data_io

    def make_stimulus_video(self, options: dict | None = None) -> np.ndarray:
        """
        Valid stimulus_options include

        image_height: in pixels
        image_width: in pixels
        container: file format to export
        codec: compression format
        fps: frames per second
        duration_seconds: stimulus duration
        baseline_start_seconds: midgray at the beginning
        baseline_end_seconds: midgray at the end
        pattern:
            'sine_grating'; 'square_grating'; 'white_gaussian_noise';
            'natural_image'; 'natural_video'; 'temporal_sine_pattern'; 'temporal_square_pattern';
            'spatially_uniform_binary_noise'
        stimulus_form: 'circular'; 'rectangular'; 'annulus'
        stimulus_position: in degrees, (0,0) is the center.
        stimulus_size: In degrees. Radius for circle and annulus, half-width for rectangle.
        contrast: between 0 and 1
        mean: mean stimulus intensity between 0, 256

        Note if mean + ((contrast * max(intensity)) / 2) exceed 255 or if
                mean - ((contrast * max(intensity)) / 2) go below 0
                the stimulus generation fails

        For sine_grating and square_grating, additional arguments are:
        spatial_frequency: in cycles per degree
        temporal_frequency: in Hz
        orientation: in degrees

        For all temporal and spatial gratings, additional argument is
        phase_shift: between 0 and 2pi

        For spatially_uniform_binary_noise, additional argument is
        on_proportion: between 0 and 1, proportion of on-stimulus, default 0.5
        direction: 'increment' or 'decrement'
        stimulus_video_name: name of the stimulus video

        ------------------------
        Output: saves the stimulus video file to output path if stimulus_video_name is not empty str or None

        Returns
        -------
         stimulus_video: VisualStimulus
            the stimulus video object, including attribute frames as numpy array of shape [time points, height, width]
        """

        if options is not None:
            self.config.visual_stimulus_parameters = options

        visual_stimulus_parameters = self.config.visual_stimulus_parameters

        match visual_stimulus_parameters.pattern:
            case "natural_image" | "natural_video":
                self.config.visual_stimulus_parameters.update(
                    self.config.external_stimulus_parameters
                )

        # Load stimulus if it exists, identified by hash of parameters. Otherwise, make new stimulus and save.
        video_hash = self.config.visual_stimulus_parameters.hash()
        video_name_stem = Path(visual_stimulus_parameters.stimulus_video_name).stem
        video_file_name = video_name_stem + "_" + video_hash + ".hdf5"
        video_file_full = self.data_io.parse_path("", substring=video_file_name)
        if video_file_full:
            print(
                "Video stimulus hash exists, loading stimulus from file:",
                video_file_full,
            )
            stimulus_video = self.data_io.load_stimulus_from_videofile(video_file_full)
            # The following two are references to self.config.visual_stimulus_parameters
            visual_stimulus_parameters.stimulus_video_name = video_file_name
            visual_stimulus_parameters.video_hash = video_hash
            return stimulus_video  # This does not return to simulation. Simulation reloads stimulus from file.
        else:
            print(
                "Did not find existing stimulus video hash, making a stimulus with the following properties:"
            )
            visual_stimulus_parameters.stimulus_video_name = video_file_name

        visual_stimulus_parameters["video_hash"] = video_hash
        for this_option in visual_stimulus_parameters:
            print(this_option, ":", visual_stimulus_parameters[this_option])
            if this_option not in self.options.keys():
                raise KeyError(f"The option '{this_option}' was not recognized")

        self.options.update(visual_stimulus_parameters)
        bg = self.options["background"]

        if isinstance(bg, str):
            match bg:
                case "mean":
                    self.options["background"] = self.options["mean"]
                case "intensity_max":
                    self.options["background"] = int(self.options["intensity"][1])
                case "intensity_min":
                    self.options["background"] = int(self.options["intensity"][0])

        self.options["dtype"] = getattr(np, self.options["dtype_name"])

        self.n_stim_tp = int(self.options["duration_seconds"] * self.options["fps"])
        # background for stimulus
        self.frames = self._create_frames(self.options["duration_seconds"])

        # Check that phase shift is in radians
        assert (
            0 <= self.options["phase_shift"] <= 2 * np.pi
        ), "Phase shift should be between 0 and 2 pi"

        # Call StimulusPattern class method to get patterns (numpy array)
        # self.frames updated according to the pattern
        # Direct call to class.method() requires the self as argument
        eval(f'StimulusPattern.{self.options["pattern"]}(self)')

        # Now only the stimulus is scaled. The baseline and bg comes from options
        self._scale_intensity()

        # For natural images, set zero-masked pixels to background value
        if self.options["pattern"] == "natural_image":
            self._set_zero_masked_pixels_to_bg()

        # Call StimulusForm class method to mask frames
        # self.frames updated according to the form
        eval(
            f'StimulusForm.{self.options["stimulus_form"]}(self)'
        )  # Direct call to class.method() requires the self argument

        # background for baseline before stimulus:
        frames_baseline_start = self._create_frames(
            self.options["baseline_start_seconds"]
        )
        # background for baseline after stimulus:
        frames_baseline_end = self._create_frames(self.options["baseline_end_seconds"])
        # Concatenate baselines and stimulus, recycle to self.frames
        self.frames = np.concatenate(
            (frames_baseline_start, self.frames, frames_baseline_end), axis=0
        )

        self.frames = self.frames.astype(self.options["dtype"])
        self.video = self.frames
        self.fps = self.options["fps"]
        self.pix_per_deg = self.options["pix_per_deg"]
        self.baseline_len_tp = frames_baseline_start.shape[0]

        self.video_n_frames = len(self.video)
        self.video_width = self.frames.shape[2]
        self.video_height = self.frames.shape[1]
        self.video_width_deg = self.video_width / self.pix_per_deg
        self.video_height_deg = self.video_height / self.pix_per_deg

        self.video_hash = self.options["video_hash"]

        stimulus_video = self
        self.data_io.save_stimulus_to_videofile(video_file_name, stimulus_video)

        return stimulus_video
