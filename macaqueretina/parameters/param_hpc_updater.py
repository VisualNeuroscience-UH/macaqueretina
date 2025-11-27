#!/usr/bin/env python3
import os
import sys

import yaml

"""
This module allows yaml configuration files to be updated based on environment variables.
This is handy for eg clusters using SLURM where parameters can be passed as environment variables.

Usage:
    python param_hpc_updater.py <input_yaml_file> <top_level_key>
"""

if len(sys.argv) < 3:
    print("Usage: python param_hpc_updater.py <input_yaml_file> <top_level_key>")
    sys.exit(1)

yaml_path = sys.argv[1]
top_level_key = sys.argv[2]

with open(yaml_path) as f:
    data = yaml.safe_load(f)

if top_level_key not in data:
    print(f"Error: '{top_level_key}' not found in YAML file.")
    sys.exit(1)

for key, value in data[top_level_key].items():
    env_var = key.upper()
    if env_var in os.environ and isinstance(value, str):
        data[top_level_key][key] = os.environ[env_var]

with open(yaml_path, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)
print(
    f"Updated '{yaml_path}' with environment variables for top-level key '{top_level_key}'."
)
