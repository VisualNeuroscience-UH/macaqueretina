"""
This module runs when macaqueretina is imported. It connects the top-level
macaqueretina namespace to the various sub-modules. The __init__.py files in
the sub-modules contain wrapper functions that call the appropriate classes
and methods from the project_manager_module. This design allows for a
clean separation of concerns and makes it easier to maintain and extend the codebase in the future.
"""

# Local
from . import analysis, viz
from .data_io import data_io
from .project import data_sampler, project_utilities
from .project.project_manager_module import load_parameters as _load_parameters
from .retina import construct_retina, retina_math, simulate_retina
from .stimuli import experiment, visual_stimulus

config = None


def load_parameters():
    """Load parameters and store in module namespace."""
    global config
    config = _load_parameters()
    print("Parameters loaded successfully.")


__all__ = [
    "config",
    "construct_retina",
    "data_io",
    "data_sampler",
    "experiment",
    "load_parameters",
    "project_utilities",
    "retina_math",
    "simulate_retina",
    "visual_stimulus",
    "analysis",
    "viz",
]


def get_version():
    import os
    import tomli

    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)
        return data["tool"]["poetry"]["version"]


__version__ = get_version()
