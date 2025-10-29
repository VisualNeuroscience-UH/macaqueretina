# Built-in
from pathlib import Path

# Third-party
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

plt.rcParams["image.cmap"] = "gray"

"""
This module creates the visual stimuli. Stimuli include patches of sinusoidal gratings at different orientations
and spatial frequencies. The duration can be defined in seconds and size (radius), and center location (x,y) 
in degrees.

Input: stimulus definition
Output: video stimulus frames

Formats .avi .mov .mp4 ?


"""


class VideoBaseClass(object):
    def __init__(self):
        """
        Initialize standard video stimulus
        The base class methods are applied to every stimulus
        """
        options = {}
        options["image_width"] = 1280  # Image width in pixels
        options["image_height"] = 720  # Image height in pixels
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

        options["baseline_start_seconds"] = 0
        options["baseline_end_seconds"] = 0
        options["stimulus_video_name"] = None
        options["dtype_name"] = "uint8"
        options["ND_filter"] = 0.0  # ND filter value at log10 units

        # Run parameter, but necessary to log to experiiment metadata
        options["n_sweeps"] = 1

        self.options = options

    def _scale_intensity(self):
        """
        Scale intensity to 8-bit grey scale. Calculating peak-to-peak here allows different
        luminances and contrasts
        """

        raw_min_value = np.min(self.options["raw_intensity"])
        raw_peak_to_peak = np.ptp(self.options["raw_intensity"])
        frames = self.frames

        # Rotation may exceed min and max values by antialiasing
        frames[frames < np.min(self.options["raw_intensity"])] = np.min(
            self.options["raw_intensity"]
        )
        frames[frames > np.max(self.options["raw_intensity"])] = np.max(
            self.options["raw_intensity"]
        )

        if self.options["intensity"] is not None:
            Lmax = np.max(self.options["intensity"])
            Lmin = np.min(self.options["intensity"])
        else:
            mean = self.options["mean"]  # This is the mean of final dynamic range
            contrast = self.options["contrast"]
            Lmax = mean * (1 + contrast)
            Lmin = mean * (1 - contrast)

        peak_to_peak = Lmax - Lmin

        # Scale values
        # Shift to 0
        frames = frames - raw_min_value
        # Scale to 1
        frames = frames / raw_peak_to_peak

        # Final scale
        frames = frames * peak_to_peak

        # Final offset
        frames = frames + Lmin

        # Here was rounding "to avoid unnecessary errors" but the
        # frames = np.round(frames, 1), was problematic with low intensity stimuli.
        # In case of problems, consider rounding to 3 decimal places.

        # Return
        self.frames = frames.astype(self.options["dtype"])

    def _prepare_grating(self, grating_type="sine"):
        """Create temporospatial grating based on specified grating type."""

        # Common setup for both grating types
        spatial_frequency = self.options.get("spatial_frequency", 1)
        temporal_frequency = self.options.get("temporal_frequency", 1)
        fps = self.options["fps"]
        duration_seconds = self.options["duration_seconds"]
        orientation = self.options["orientation"]
        image_width = self.options["image_width"]
        image_height = self.options["image_height"]
        # image_width_in_degrees = image_width / self.options["pix_per_deg"]
        diameter = np.ceil(np.sqrt(image_height**2 + image_width**2)).astype(np.int32)
        image_width_diameter = diameter
        image_height_diameter = diameter
        n_frames = int(fps * duration_seconds)
        self.frames = np.zeros((n_frames, image_height_diameter, image_width_diameter))

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
            self.frames = large_frames
            # Turn to sine values
            self.frames = np.sin(self.frames + self.options["phase_shift"])

        # Specific part for square grating
        elif grating_type == "square":
            # n_cycles = spatial_frequency * image_width_in_degrees
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
                self.frames[frame] = np.where(relative_bar_coords < 1, 1, -1)

        # Common post-processing: Rotate and cut to original dimensions
        for frame in range(n_frames):
            self.frames[frame] = ndimage.rotate(
                self.frames[frame], orientation, reshape=False
            )
        marginal_height = (diameter - image_height) // 2
        marginal_width = (diameter - image_width) // 2
        self.frames = self.frames[
            :, marginal_height:-marginal_height, marginal_width:-marginal_width
        ]

        # In case of rounding errors, clip to image_height and image_width
        self.frames = self.frames[:, :image_height, :image_width]

        # Set raw_intensity to [-1 1]
        self.options["raw_intensity"] = (-1, 1)

    def _prepare_form(self, stimulus_size):
        center_deg = self.options["stimulus_position"]  # in degrees
        radius_deg = stimulus_size  # in degrees
        height = self.options["image_height"]  # in pixels
        width = self.options["image_width"]  # in pixels
        pix_per_deg = self.options["pix_per_deg"]

        # Turn position in degrees to position in mask, shift 0,0 to center of image
        center_pix = np.array([0, 0])
        center_pix[0] = int(
            width / 2 + pix_per_deg * center_deg[0]
        )  # NOTE Width goes to x-coordinate
        center_pix[1] = int(
            height / 2 + pix_per_deg * -center_deg[1]
        )  # NOTE Height goes to y-coordinate. Inverted to get positive up

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

    def _prepare_circular_mask(self, stimulus_size):
        X, Y, center_pix, radius_pix = self._prepare_form(stimulus_size)
        dist_from_center = np.sqrt((X - center_pix[0]) ** 2 + (Y - center_pix[1]) ** 2)

        mask = dist_from_center <= radius_pix
        return mask

    def _combine_background(self, mask):
        self.frames_background = np.ones(self.frames.shape) * self.options["background"]
        self.frames_background[:, mask] = self.frames[:, mask]
        self.frames = self.frames_background

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
        image_width = self.options["image_width"]
        image_height = self.options["image_height"]

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
            image_width != n_frames
        ), "Errors in 3D broadcasting, change image width/height NOT to match n frames "
        assert (
            image_height != n_frames
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
        image_width = self.options["image_width"]
        image_height = self.options["image_height"]

        # Time vector in seconds
        time_vec = np.linspace(0, duration_seconds, int(fps * duration_seconds))

        # Linearly increasing frequency over time
        k = (end_frequency - start_frequency) / duration_seconds
        instantaneous_frequency = start_frequency + k * time_vec

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
            image_width != n_frames
        ), "Errors in 3D broadcasting, change image width/height NOT to match n frames "
        assert (
            image_height != n_frames
        ), "Errors in 3D broadcasting, change image width/height NOT to match n frames "

        self.frames = frames

    def _raw_intensity_from_data(self):
        self.options["raw_intensity"] = (np.min(self.frames), np.max(self.frames))

    def _create_frames(self, epoch_in_seconds):
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

    def _extract_frames(self, cap, n_frames):
        """
        Extracts and resizes the frames from a cv2.VideoCapture object

        Parameters
        ----------
        cap : cv2.VideoCapture
            The video capture object from which frames are extracted.
        n_frames : int
            The number of frames to extract.
        """
        w = self.options["image_width"]
        h = self.options["image_height"]
        # Load each frame as array of gray values between 0-255
        for frame_ix in range(n_frames):
            _, frame = cap.read()

            frame_out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frames[frame_ix, :, :] = cv2.resize(
                frame_out, (w, h), interpolation=cv2.INTER_AREA
            )


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

        # Turn to sine values
        self.frames = np.sin(self.frames + self.options["phase_shift"])

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
        filepath = self.config.literature_data_files["temporal_pattern_path"]
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

    def natural_images(self):
        """
        Process natural images for use in stimulus patterns.

        This method handles natural images loading an image file based on the provided
        stimulus metadata. The selected image is then resized to match the frame dimensions.
        The resized image is integrated with the frames by multiplying it, enabling the
        creation of astimulus pattern.

        After this integration, the method updates the raw intensity values based on the new data.
        """
        image_file_name = self.config.external_stimulus_parameters["stimulus_file"]
        self.image = self.data_io.load_data(image_file_name)

        # resize image by specifying custom width and height
        resized_image = cv2.resize(self.image, self.frames.shape[1:])

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

        video_file_name = self.config.external_stimulus_parameters["stimulus_file"]
        video_cap = self.data_io.load_data(video_file_name)

        self.fps = self.options["fps"]
        self.pix_per_deg = self.options["pix_per_deg"]

        # Cut to desired length at desired fps
        n_frames = self.frames.shape[0]
        self.frames = np.ones(
            n_frames, (self.options["image_height"], self.options["image_width"])
        )

        video_n_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_cap.get(cv2.CAP_PROP_FPS)

        self._extract_frames(video_cap, n_frames)

        print(
            "Original movie dimensions %d x %d px, %d frames at %d fps."
            % (video_width, video_height, video_n_frames, fps)
        )
        print(
            "Resized movie dimensions %d x %d px, %d frames at %d fps."
            % (
                self.options["image_width"],
                self.options["image_height"],
                n_frames,
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

    def __init__(self, config, data_io, get_xy_from_npz):
        super().__init__()

        self._config = config
        self._data_io = data_io

        self.get_xy_from_npz = get_xy_from_npz

    @property
    def config(self):
        return self._config

    @property
    def data_io(self):
        return self._data_io

    def make_stimulus_video(self, options=None):
        """
        Valid stimulus_options include

        image_width: in pixels
        image_height: in pixels
        container: file format to export
        codec: compression format
        fps: frames per second
        duration_seconds: stimulus duration
        baseline_start_seconds: midgray at the beginning
        baseline_end_seconds: midgray at the end
        pattern:
            'sine_grating'; 'square_grating'; 'white_gaussian_noise';
            'natural_images'; 'natural_video'; 'temporal_sine_pattern'; 'temporal_square_pattern';
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
        """

        # Set input arguments to video-object, updates the defaults from VideoBaseClass
        if options is not None:
            self.config.visual_stimulus_parameters = options

        visual_stimulus_parameters = self.config.visual_stimulus_parameters

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
            visual_stimulus_parameters.stimulus_video_name = video_file_name
            return stimulus_video  # This does not return to simulation. Simulation reloads stimulus from file.
        else:
            print(
                "Did not find existing stimulus video hash, making a stimulus with the following properties:"
            )
            visual_stimulus_parameters.stimulus_video_name = video_file_name

        for this_option in visual_stimulus_parameters:
            print(this_option, ":", visual_stimulus_parameters[this_option])
            assert (
                this_option in self.options.keys()
            ), f"The option '{this_option}' was not recognized"

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
        if self.options["pattern"] == "natural_images":
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

        stimulus_video = self

        self.data_io.save_stimulus_to_videofile(video_file_name, stimulus_video)

        return stimulus_video


class AnalogInput:
    """
    Creates analog input in CxSystem compatible video mat file format. Analog stimulus comprises of
    continuous waveforms of types 'quadratic_oscillation', 'noise' or 'step_current'. You get few input
    channels (N_units) of temporal signals. These signals do not pass through the retina, instead
    they are saved as .mat files. Use e.g. as current injection to a unit.

    frameduration assumes milliseconds
    """

    def __init__(self, config, data_io, viz, **kwargs):
        super().__init__()

        self._config = config
        self._data_io = data_io
        self._viz = viz

        for attr, value in kwargs.items():
            setattr(self, attr, value)

    @property
    def config(self):
        return self._config

    @property
    def data_io(self):
        return self._data_io

    @property
    def viz(self):
        return self._viz

    def make_stimulus_video(self, analog_options=None):
        assert analog_options is not None, "analog_options not set, aborting... "

        N_units = analog_options["N_units"]
        N_tp = analog_options["N_tp"]
        filename_out = analog_options["filename_out"]
        input_type = analog_options["input_type"]
        coord_type = analog_options["coord_type"]
        N_cycles = analog_options["N_cycles"]
        frameduration = analog_options["dt"]

        # get Input
        if input_type == "noise":
            Input = self.create_noise_input(Nx=N_units, N_tp=N_tp)
        elif input_type == "quadratic_oscillation":
            if N_units != 2:
                print(
                    f"NOTE: You requested {input_type} input type, setting excessive units to 0 value"
                )
            Input = self.create_quadratic_oscillation_input(
                Nx=N_units, N_tp=N_tp, N_cycles=N_cycles
            )
        elif input_type == "step_current":
            Input = self.create_step_input(Nx=N_units, N_tp=N_tp)

        # get coordinates
        if coord_type == "dummy":
            w_coord, z_coord = self._get_dummy_coordinates(Nx=N_units)
        elif coord_type == "real":
            rf = self.ReceptiveFields(
                self.config.retina_parameters,
                self.config.experimental_metadata,
                self.data_io.load_data,
                self.pol2cart_df,
            )

            w_coord, z_coord = self._get_real_coordinates(rf, Nx=N_units)

        assert (
            "w_coord" in locals()
        ), "coord_type not set correctly, check __init__, aborting"
        w_coord = np.expand_dims(w_coord, 1)
        z_coord = np.expand_dims(z_coord, 1)

        # For potential plotting from conf
        self.Input = Input
        self.frameduration = frameduration

        if analog_options["save_stimulus"] is True:
            self.data_io.save_analog_stimulus(
                filename_out=filename_out,
                Input=Input,
                z_coord=z_coord,
                w_coord=w_coord,
                frameduration=frameduration,
            )

    def _gaussian_filter(self):
        sigma = 30  # was abs(30)
        w = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -1 * np.power(np.arange(1000) - 500, 2) / (2 * np.power(sigma, 2))
        )
        w = w / np.sum(w)
        return w

    def _normalize(self, Input):
        # Scale to interval [0, 1]
        Input = Input - min(np.ravel(Input))
        Input = Input / max(np.ravel(Input))
        return Input

    def create_noise_input(self, Nx=0, N_tp=None, amplitude=15.0):
        """
        Create signal for simulated current injection with noise input in the AnalogInput class.

        Generates a multivariate Gaussian noise input for a specified number of units (Nx)
        and timepoints (N_tp). Applies a Gaussian filter to the noise input and scales it
        to simulate current injection in a neural network model.

        Parameters
        ----------
        Nx : int
            Number of units. Must be non-zero.
        N_tp : int or None
            Number of timepoints. Must be specified.
        amplitude : float
            Amplitude of the noise input. Default is 15.

        Returns
        -------
        ndarray
            The generated noise input signal after filtering and scaling.
        """
        assert Nx != 0, "N units not set, aborting..."
        assert N_tp is not None, "N timepoints not set, aborting..."
        Input = (np.random.multivariate_normal(np.zeros([Nx]), np.eye(Nx), N_tp)).T

        # Get gaussian filter, apply
        w = self._gaussian_filter()
        A = amplitude  # Deneve project was 2000, from their Learning.py file
        for d in np.arange(Nx):
            Input[d, :] = A * np.convolve(Input[d, :], w, "same")

        return Input

    def create_quadratic_oscillation_input(
        self, Nx=0, N_tp=None, N_cycles=0, amplitude=5.0
    ):
        """
        Create analog oscillatory input for a specified number of units, timepoints, and cycles.

        Parameters
        ----------
        Nx : int
            Number of units. Must be non-zero.
        N_tp : int
            Number of time points. Must be specified.
        N_cycles : int, float, or list
            Number of oscillatory cycles. Scalar for quadratic pair, list for distinct frequencies.
        amplitude : float
            Amplitude of the oscillatory input. Default is 5.

        Returns
        -------
        ndarray
            The generated oscillatory input signal.
        """

        assert Nx != 0, "N units not set, aborting..."
        assert N_cycles != 0, "N cycles not set, aborting..."
        assert N_tp is not None, "N timepoints not set, aborting..."

        tp_vector = np.arange(N_tp)
        A = amplitude  # Deneve project was 2000, from their Learning.py file

        if isinstance(N_cycles, int) or isinstance(N_cycles, float):
            # frequency, this gives N_cycles over all time points
            freq = N_cycles * 2 * np.pi * 1 / N_tp
            sine_wave = np.sin(freq * tp_vector)
            cosine_wave = np.cos(freq * tp_vector)
            Input = A * np.array([sine_wave, cosine_wave])
            if Nx > 2:
                unit_zero_input = np.zeros(sine_wave.shape)
                stack_to_add = np.tile(unit_zero_input, (Nx - 2, 1))
                zero_padded_input_stack = np.vstack((Input, stack_to_add))
                Input = zero_padded_input_stack
        elif isinstance(N_cycles, list):
            for index, this_Nx in enumerate(range(Nx)):
                if index > len(N_cycles) - 1:
                    freq = 0
                else:
                    freq = N_cycles[this_Nx] * 2 * np.pi * 1 / N_tp

                if index % 2 == 0:
                    oscillations = np.sin(freq * tp_vector)
                else:
                    oscillations = np.cos(freq * tp_vector)
                    if freq == 0:
                        oscillations = oscillations * 0
                if "Input" not in locals():
                    Input = A * np.array([oscillations])
                else:
                    Input = np.vstack((Input, A * np.array([oscillations])))

        return Input

    def create_step_input(self, Nx=0, N_tp=None, amplitude=5.0):
        """
        Create a step function input for simulated current injection.

        Parameters
        ----------
        Nx : int
            Number of units. Must be non-zero.
        N_tp : int
            Number of time points. Must be specified.

        Returns
        -------
        ndarray
            The generated step function input signal after amplification.
        """

        assert Nx != 0, "N units not set, aborting..."
        assert N_tp is not None, "N timepoints not set, aborting..."

        # Create your input here. Zeros and ones at this point.
        # Create matrix of zeros with shape of Input
        Input = np.concatenate(
            (
                np.zeros((N_tp // 3,), dtype=int),
                np.ones((N_tp // 3), dtype=int),
                np.zeros((N_tp // 3), dtype=int),
            ),
            axis=None,
        )
        Input = np.concatenate(
            (Input, np.zeros((N_tp - np.size((Input), 0),), dtype=int)), axis=None
        )
        Input = np.tile(Input.T, (Nx, 1))

        A = amplitude
        Input = A * Input

        minI = np.min(Input)
        maxI = np.max(Input)
        print(f"minI = {minI}")
        print(f"maxI = {maxI}")
        return Input

    def _get_dummy_coordinates(self, Nx=0):
        # Create dummy coordinates for CxSystem format video input.

        assert Nx != 0, "N units not set, aborting..."

        # N units btw 4 and 6 deg ecc
        z_coord = np.linspace(4.8, 5.2, Nx)
        z_coord = z_coord + 0j  # Add second dimension

        visual2cortical_params = self.config.retina_parameters_options[
            "visual2cortical_params"
        ]
        a = visual2cortical_params["a"]
        k = visual2cortical_params["k"]
        w_coord = k * np.log(z_coord + a)

        return w_coord, z_coord

    def _get_real_coordinates(self, rf, Nx=0):
        # For realistic coordinates, we use Macaque retina module

        assert Nx != 0, "N units not set, aborting..."

        # Initialize SimulateRetina
        w_coord, z_coord = self.get_w_z_coords(rf)

        # Get random sample sized N_units, assert for too small sample

        Nmosaic_units = w_coord.size
        assert (
            Nx <= Nmosaic_units
        ), "Too few units in mosaic, increase ecc and / or sector limits in _get_real_coordinates method"
        idx = np.random.choice(Nmosaic_units, size=Nx, replace=False)
        w_coord, z_coord = w_coord[idx], z_coord[idx]

        return w_coord, z_coord
        return w_coord, z_coord
