"""
This module runs when macaqueretina is imported. It connects the top-level
macaqueretina namespace to the various sub-modules. The __init__.py files in
the sub-modules contain wrapper functions that call the appropriate classes
and methods from the project_manager_module. This design allows for a
clean separation of concerns and makes it easier to maintain and extend the codebase in the future.
"""

# Local
from .analysis import analysis
from .data_io import data_io
from .project import data_sampler, project_utilities
from .project.project_manager_module import load_parameters as _load_parameters
from .retina import retina_constructor, retina_math, retina_simulator
from .stimuli import experiment, stimulus_factory
from .viz import viz, viz_response

config = None


def load_parameters():
    """Load parameters and store in module namespace."""
    global config
    config = _load_parameters()
    print("Parameters loaded successfully.")


# This enables from macaqueretina import something to work.
__all__ = [
    "analysis",
    "config",
    "retina_constructor",
    "data_io",
    "data_sampler",
    "experiment",
    "load_parameters",
    "project_utilities",
    "retina_math",
    "retina_simulator",
    "stimulus_factory",
    "viz",
    "viz_response",
]


def get_version():
    import os

    import tomllib

    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
        return data["tool"]["poetry"]["version"]


__version__ = get_version()
