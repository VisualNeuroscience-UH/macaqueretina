# Local
from .project.project_conf_module import load_parameters as _lp
from .project.project_manager_module import ProjectManager as _PM

config = _lp()

PM = _PM(config)

construct_retina = PM.construct_retina
stimulate = PM.stimulate
simulate_retina = PM.simulate_retina
viz = PM.viz

__all__ = ["construct_retina", "stimulate", "simulate_retina", "viz"]

del _lp, _PM, PM

# import os
# import tomli

# def get_version():
#     pyproject_path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
#     with open(pyproject_path, 'rb') as f:
#         data = tomli.load(f)
#         return data['tool']['poetry']['version']

# __version__ = get_version()
