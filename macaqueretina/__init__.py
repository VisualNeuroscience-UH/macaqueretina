import os
import tomli

def get_version():
    pyproject_path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
    with open(pyproject_path, 'rb') as f:
        data = tomli.load(f)
        return data['tool']['poetry']['version']

__version__ = get_version()
