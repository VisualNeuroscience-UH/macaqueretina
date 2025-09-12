"""Type stubs for MacaqueRetina package."""

# Built-in
from typing import Any

# Local
from .data_io.config_io import ConfigManager
from .project.project_manager_module import ProjectManager
from .retina.retina_math_module import RetinaMath
from .parameters.param_validation import RetinaParameters
from .viz.viz_module import Viz

config: ConfigManager
"""Access and modify configuration values.

Example:

>>> mr.config.something = 12 # Set "something" to 12
>>> dir(mr.config) # List all attributes + parameters
"""

class _Config:
    retina_parameters: RetinaParameters
    """Retina parameters"""

config: _Config

PM: ProjectManager

def build_retina() -> None:
    """Build the retina.

    For each call, a new retina is built with the parameters in the mr.config object.

    After being built, the retina is saved to mr.config.output_folder.
    """
    ...

def make_stimulus(options: Any) -> None:
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
        'sine_grating'; 'square_grating'; 'colored_temporal_noise'; 'white_gaussian_noise';
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
    ...

def simulate_retina():
    """Simulate the retina."""
    ...

retina_math: RetinaMath

class _Viz:
    pass

viz: _Viz
"""Visualize"""

__all__: list[str]
