"""
Example of running the project_manager_module from container in SLURM environment.
Parameters are changed via environment variables using param_hpc_updater.py.
The environment variables are set in SLURM job script.
"""

# Built-in
import os
import sys
from pathlib import Path

# Third-party
import yaml
def update_yaml_with_env_vars(yaml_path, top_level_key):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if top_level_key not in data:
        print(f"Error: '{top_level_key}' not found in YAML file.")
        sys.exit(1)

    for key, value in data[top_level_key].items():
        env_var = key.upper()
        if env_var in os.environ and isinstance(value, str):
            breakpoint()
            data[top_level_key][key] = os.environ[env_var]

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(
        f"Updated '{yaml_path}' with environment variables for top-level key '{top_level_key}'."
)

##################################################
# Repeat for all parameters file & key pairs whose
# parameters should be updated via env vars

yaml_path = Path("macaqueretina/parameters/retina_parameters.yaml")
top_level_key = "retina_parameters"

update_yaml_with_env_vars(yaml_path, top_level_key)
############################################
# Local
import macaqueretina as mr


stimulus_folder = f"stim_{mr.config.experiment}"
mr.config.output_folder = f"{mr.config.experiment}_{mr.config.gc_type}_{mr.config.response_type}_{mr.config.spatial_model_type}_{mr.config.temporal_model_type}"

mr.construct_retina()

###############################
## Build and run experiment ###
###############################

# These are the variables to be changed in the experiment
# See visual_stimulus_parameters, safe up to two variables
exp_variables = ["contrast"]  # ["contrast", "spatial_frequency"]
mr.config.experiment_parameters = {
    "exp_variables": exp_variables,
    # two vals below for each exp_variable, even is it is not changing
    "min_max_values": [[0, 1.0]],  # [[0, 0.6], [0.1, 15.0]]
    "n_steps": [5],  # [10 ,16]
    "logarithmic": [False],  # [True, True]
    "n_sweeps": 1,
    # "distributions": {"gaussian": {"sweeps": 10, "mean": [-30, 30], "sd": [5, 5]}},
    "distributions": {"uniform": None},
}

filename = mr.experiment.build_and_run(build_without_run=True)

########################################
## Analyze and visualize experiment ###
########################################

# my_analysis_options = {
#     "exp_variables": exp_variables,
#     "t_start_ana": 0.5,
#     "t_end_ana": 1.5,
# }
# mr.analysis.analyze_experiment(filename, my_analysis_options)

##############################################


# plt.show()
