"""
Macaque retina simulator.
"""

# Built-in
from typing import Callable

# Local
from .project.project_conf_module import load_parameters as _lp
from .project.project_manager_module import ProjectManager as _PM
from .retina.retina_math_module import RetinaMath

config = _lp()

PM: _PM = _PM(config)

construct_retina: Callable = PM.construct_retina.build_retina_client
make_stimulus: Callable = PM.stimulate.make_stimulus_video
simulate_retina: Callable = PM.simulate_retina
viz = PM.viz
retina_math: RetinaMath = PM.retina_math

__all__ = ["construct_retina", "stimulate", "simulate_retina", "viz", "retina_math"]

del (_lp, _PM)


# import os
# import tomli

# def get_version():
#     pyproject_path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
#     with open(pyproject_path, 'rb') as f:
#         data = tomli.load(f)
#         return data['tool']['poetry']['version']

# __version__ = get_version()
