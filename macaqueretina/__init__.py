"""
This module runs when macaqueretina is imported. It connects the top-level
macaqueretina namespace to the various sub-modules.
"""

# Local

from .project.project_manager_module import load_parameters, build_retina


# Define what is imported when doing: from macaqueretina import *. Only the objects in
# __all__ can be imported this way.
__all__ = [
    "load_parameters",
    "build_retina",
]


def get_version():
    import os
    import tomli

    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)
        return data["tool"]["poetry"]["version"]


__version__ = get_version()
