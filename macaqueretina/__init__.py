"""
This module runs when macaqueretina is imported. It connects the top-level
macaqueretina namespace to the various sub-modules.
"""

# Local
from .project.project_manager_module import (
    build_retina,
    load_parameters as _load_parameters,
)
from . import viz

config = None


def load_parameters():
    """Load parameters and store in module namespace."""
    global config
    config = _load_parameters()
    return config


__all__ = ["load_parameters", "build_retina", "config"]


def get_version():
    import os

    import tomli

    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)
        return data["tool"]["poetry"]["version"]


__version__ = get_version()
