"""
This module is run when macaqueretina is imported. It connects the top-level
macaqueretina namespace to the various sub-modules. It also runs the
ProjectManager to load the configuration parameters.
"""

# Built-in
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable

# Third-party
import tomli

# Local
from .analysis.analysis_module import Analysis
from .project.project_manager_module import ProjectManager as _ProjectManager
from .retina.retina_math_module import RetinaMath
from .stimuli.experiment_module import Experiment
from .viz.viz_module import Viz, VizResponse

if TYPE_CHECKING:
    from data_io.config_io import Configuration

# ProjectManager instance
if os.environ.get("YAML_TMPDIR"):
    yaml_tmpdir = Path(os.environ.get("YAML_TMPDIR"))

PM: _ProjectManager = _ProjectManager(yaml_path=yaml_tmpdir)

# This connects the top-level macaqueretina namespace to the various modules. Look here if you are lost.
config: "Configuration" = PM.config
analysis: Analysis = PM.ana
construct_retina: Callable = PM.construct_retina.build_retina_client
countlines = PM.countlines
DataSampler = PM.data_sampler
experiment: Experiment = PM.experiment
load_data = PM.data_io.load_data
make_stimulus: Callable = PM.stimulate.make_stimulus_video
retina_math: RetinaMath = PM.retina_math
simulate_retina: Callable = PM.simulate_retina.client
viz: Viz = PM.viz
viz_spikes_with_stimulus: VizResponse = PM.viz_spikes_with_stimulus.client


# Define what is imported when doing: from macaqueretina import *. Only the objects in
# __all__ can be imported this way.
__all__ = [
    "analysis",
    "config",
    "construct_retina",
    "countlines",
    "DataSampler",
    "experiment",
    "load_data",
    "make_stimulus",
    "retina_math",
    "simulate_retina",
    "viz",
    "viz_spikes_with_stimulus",
]

del _ProjectManager


def get_version():
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)
        return data["tool"]["poetry"]["version"]


__version__ = get_version()
