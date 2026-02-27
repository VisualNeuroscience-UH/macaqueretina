"""
This module runs when macaqueretina is imported. It connects the top-level
macaqueretina namespace to the various sub-modules. The __init__.py files in
the sub-modules contain wrapper functions that call the appropriate classes
and methods from the project_manager_module. This design allows for a
clean separation of concerns and makes it easier to maintain and extend the codebase in the future.
"""

# Local
from . import analysis, viz
from .data_io import load_data
from .project import countlines
from .project.project_manager_module import data_sampler
from .project.project_manager_module import load_parameters as _load_parameters
from .retina import build_retina, retina_math, save_retina, simulate_retina
from .stimuli import make_stimulus, run_experiment

config = None


def load_parameters():
    """Load parameters and store in module namespace."""
    global config
    config = _load_parameters()
    print("Parameters loaded successfully.")


__all__ = [
    "load_parameters",
    "retina_math",
    "analysis",
    "viz",
    "data_sampler",
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
