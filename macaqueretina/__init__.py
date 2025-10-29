"""
Macaque retina simulator.
"""

# Built-in
import os
from typing import Callable

# Third-party
import tomli

# Local
from .analysis.analysis_module import Analysis
from .project.project_conf_module import load_parameters as _load_parameters
from .project.project_manager_module import ProjectManager as _ProjectManager
from .retina.retina_math_module import RetinaMath
from .stimuli.experiment_module import Experiment
from .viz.viz_module import Viz, VizResponse

config = _load_parameters()
PM: _ProjectManager = _ProjectManager(config)

# This connects the top-level macaqueretina namespace to the various modules. Look here if you are lost.
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

del (_load_parameters, _ProjectManager)


def get_version():
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)
        return data["tool"]["poetry"]["version"]


__version__ = get_version()
