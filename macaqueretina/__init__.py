"""
Macaque retina simulator.
"""

# Built-in
import os
from typing import Callable

# Third-party
import tomli

# Local
from .project.project_conf_module import load_parameters as _lp
from .project.project_manager_module import ProjectManager as _PM
from .retina.retina_math_module import RetinaMath
from .viz.viz_module import Viz, VizResponse
from .stimuli.experiment_module import Experiment
from .analysis.analysis_module import Analysis

config = _lp()

PM: _PM = _PM(config)

construct_retina: Callable = PM.construct_retina.build_retina_client
make_stimulus: Callable = PM.stimulate.make_stimulus_video
simulate_retina: Callable = PM.simulate_retina.client
viz: Viz = PM.viz
viz_spikes_with_stimulus: VizResponse = PM.viz_spikes_with_stimulus.client
retina_math: RetinaMath = PM.retina_math
experiment: Experiment = PM.experiment
analysis: Analysis = PM.ana

countlines = PM.countlines

get_data = PM.data_io.get_data

__all__ = [
    "config",
    "construct_retina",
    "stimulate",
    "simulate_retina",
    "viz",
    "retina_math",
]

del (_lp, _PM)


def get_version():
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)
        return data["tool"]["poetry"]["version"]


__version__ = get_version()
