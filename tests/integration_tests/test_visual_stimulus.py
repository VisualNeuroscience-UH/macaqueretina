# Built-in
from pathlib import Path

# Third-party
import numpy as np
import pytest

# Local
import macaqueretina as mr
from macaqueretina.stimuli.visual_stimulus_module import VideoBaseClass

test_root = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def stimulus_config(tmp_path_factory):
    """
    Fixture to reset and provide the stimulus configuration.
    """
    temp_dir = tmp_path_factory.mktemp("stimulus_test")
    mr.config.visual_stimulus_parameters = VideoBaseClass().options
    mr.config.stimulus_folder = temp_dir
    return mr.config


PATTERNS = [
    "sine_grating",
    "square_grating",
    "white_gaussian_noise",
    "natural_images",
    "natural_video",
    "temporal_sine_pattern",
    "temporal_square_pattern",
    "spatially_uniform_binary_noise",
]
STIMULUS_FORMS = ["circular", "rectangular", "annulus"]
DTYPES = ["float16", "uint8"]


@pytest.mark.parametrize(
    "pattern,stimulus_form,dtype_name",
    [
        (pattern, form, dtype)
        for pattern in PATTERNS
        for form in STIMULUS_FORMS
        for dtype in DTYPES
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_make_stimulus_video(
    stimulus_config,
    tmp_path,
    pattern,
    stimulus_form,
    dtype_name,
):
    # Set parameters
    stimulus_config.stimulus_folder = Path(tmp_path)
    stimulus_config.visual_stimulus_parameters.pattern = pattern
    stimulus_config.visual_stimulus_parameters.stimulus_form = stimulus_form
    stimulus_config.visual_stimulus_parameters.dtype_name = dtype_name
    stimulus_config.visual_stimulus_parameters.stimulus_video_name = (
        f"test_{pattern}_{stimulus_form}_{dtype_name}"
    )
    stimulus_config.visual_stimulus_parameters.duration_seconds = 0.1
    stimulus_config.visual_stimulus_parameters.fps = 30
    stimulus_config.visual_stimulus_parameters.image_width = 128
    stimulus_config.visual_stimulus_parameters.image_height = 128
    stimulus_config.visual_stimulus_parameters.mean = 128
    stimulus_config.visual_stimulus_parameters.contrast = 0.5
    stimulus_config.visual_stimulus_parameters.stimulus_size = 1.0
    stimulus_config.visual_stimulus_parameters.stimulus_position = (0, 0)
    # For patterns that require additional parameters
    if pattern in ["sine_grating", "square_grating"]:
        stimulus_config.visual_stimulus_parameters.spatial_frequency = 2.0
        stimulus_config.visual_stimulus_parameters.temporal_frequency = 2.0
        stimulus_config.visual_stimulus_parameters.orientation = 0.0
        stimulus_config.visual_stimulus_parameters.phase_shift = 0.0
    if pattern == "spatially_uniform_binary_noise":
        stimulus_config.visual_stimulus_parameters.on_proportion = 0.5
        stimulus_config.visual_stimulus_parameters.direction = "increment"
    if pattern in ["natural_images"]:
        stimulus_config.input_folder = test_root / "mock_data"
        mr.config.external_stimulus_parameters.stimulus_file = "test_image.jpg"
    if pattern in ["natural_video"]:
        pytest.skip("Not fully implemented yet")

    # Generate stimulus
    stimulus_video = mr.make_stimulus()

    # Assertions
    assert hasattr(
        stimulus_video, "video"
    ), "Stimulus video object missing 'video' attribute"
    assert hasattr(
        stimulus_video, "fps"
    ), "Stimulus video object missing 'fps' attribute"
    assert hasattr(
        stimulus_video, "pix_per_deg"
    ), "Stimulus video object missing 'pix_per_deg' attribute"
    assert hasattr(
        stimulus_video, "baseline_len_tp"
    ), "Stimulus video object missing 'baseline_len_tp' attribute"
    assert hasattr(
        stimulus_video, "video_n_frames"
    ), "Stimulus video object missing 'video_n_frames' attribute"
    assert hasattr(
        stimulus_video, "video_width"
    ), "Stimulus video object missing 'video_width' attribute"
    assert hasattr(
        stimulus_video, "video_height"
    ), "Stimulus video object missing 'video_height' attribute"
    assert hasattr(
        stimulus_video, "video_width_deg"
    ), "Stimulus video object missing 'video_width_deg' attribute"
    assert hasattr(
        stimulus_video, "video_height_deg"
    ), "Stimulus video object missing 'video_height_deg' attribute"
    assert stimulus_video.video is not None, "Stimulus video is None"
    assert (
        stimulus_video.video.shape[0] == stimulus_video.video_n_frames
    ), "Video frame count mismatch"
    assert (
        stimulus_video.video.shape[1]
        == stimulus_config.visual_stimulus_parameters.image_height
    ), "Video height mismatch"
    assert (
        stimulus_video.video.shape[2]
        == stimulus_config.visual_stimulus_parameters.image_width
    ), "Video width mismatch"
    assert stimulus_video.video.dtype == getattr(
        np, dtype_name
    ), f"Video dtype is not {dtype_name}"

    # Check if file was saved
    stimulus_folder = Path(stimulus_config.stimulus_folder)
    assert stimulus_folder.exists(), "Stimulus folder does not exist"
    stimulus_video_stem = Path(
        stimulus_config.visual_stimulus_parameters.stimulus_video_name
    ).stem
    video_files = list(stimulus_folder.glob(f"{stimulus_video_stem}*"))
    assert len(video_files) == 2, f"Expected 2 video files, found {len(video_files)}"
