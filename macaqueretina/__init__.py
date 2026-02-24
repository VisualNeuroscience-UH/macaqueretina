"""
This module runs when macaqueretina is imported. It connects the top-level
macaqueretina namespace to the various sub-modules.
"""

# Local
from .project.project_manager_module import (
    load_parameters as _load_parameters,
    data_sampler,
)
from . import viz
from . import analysis
from . import retina as retina_math
from .stimuli import make_stimulus, run_experiment
from .retina import build_retina, save_retina, simulate_retina
from .project import countlines
from .data_io import load_data

config = None


def load_parameters():
    """Load parameters and store in module namespace."""
    global config
    config = _load_parameters()
    print("Parameters loaded successfully.")


__all__ = [
    "load_parameters",
    "retina_math",
    "data_sampler",
    "construct_retina",
    "load_data",
    "config",
    "make_stimulus",
    "run_experiment",
    "build_retina",
    "save_retina",
    "simulate_retina",
    "countlines",
]


def get_version():
    import os

    import tomli

    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)
        return data["tool"]["poetry"]["version"]


__version__ = get_version()
