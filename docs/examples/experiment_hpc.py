"""
Example of running the project_manager_module from container in SLURM environment.
Parameters are changed via environment variables using __main__.py when invoking python macaqueretina
The environment variables are set in SLURM job script.
"""

import macaqueretina as mr

# import matplotlib.pyplot as plt

# Stimulus folder is the same for the distinct ganglion cell types
mr.config.stimulus_folder = mr.config.path.joinpath(f"stim_{mr.config.experiment}")
mr.config.stimulus_folder.mkdir(parents=True, exist_ok=True)

# Output folders are separate for the distinct ganglion cell types
retina_parameters = mr.config.retina_parameters
output_folder = f"{mr.config.experiment}_{retina_parameters.gc_type}_{retina_parameters.response_type}_{retina_parameters.spatial_model_type}_{retina_parameters.temporal_model_type}"
mr.config.output_folder = mr.config.path.joinpath(output_folder)
mr.config.output_folder.mkdir(parents=True, exist_ok=True)
print(f"\n{output_folder=}\n")

mr.construct_retina()

###############################
## Build and run experiment ###
###############################

exp_variables = ["contrast", "temporal_frequency"]
mr.config.experiment_parameters = {
    "exp_variables": exp_variables,
    "min_max_values": [[0, 1.0], [0.2, 60.0]],
    "n_steps": [10, 16],
    "logarithmic": [True, True],
    "n_sweeps": 1,
    "distributions": {"uniform": None},
}

filename = mr.experiment.build_and_run(build_without_run=True)

# ########################################
# ## Analyze and visualize experiment ###
# ########################################

# my_analysis_options = {
#     "exp_variables": exp_variables,
#     "t_start_ana": 0.5,
#     "t_end_ana": 12.5,
# }
# mr.analysis.analyze_experiment(filename, my_analysis_options)

# #########################################

# # Contrast sensitivity
# mr.viz.contrast_sensitivity(
#     filename,
#     ["contrast", "spatial_frequency"],
#     xlog=True,
#     ylog=True,
#     xlim=[0.1, 100],
#     ylim=[1, 200],
#     savefigname=f"F1_unit_{output_folder}.eps",
# )

# plt.show()
