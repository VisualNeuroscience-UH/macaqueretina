"""
Example of running the project_manager_module from container in SLURM environment.
Parameters are changed via environment variables using __main__.py when invoking python macaqueretina
The environment variables are set in SLURM job script.
"""

import time

start_time = time.time()

import matplotlib.pyplot as plt

import macaqueretina as mr

###############################
## Build and run experiment ###
###############################
rp = mr.config.retina_parameters

sf = int(mr.config.visual_stimulus_parameters["spatial_frequency"])
output_folder = f"{mr.config.experiment}_{rp.gc_type}_{rp.response_type}_{rp.spatial_model_type}_{rp.temporal_model_type}_{sf}cpd"
mr.config.output_folder = mr.config.path.joinpath(output_folder)
mr.config.output_folder.mkdir(parents=True, exist_ok=True)
stimulus_folder = mr.config.path.joinpath(f"stimuli_{sf}cpd")
mr.config.stimulus_folder = stimulus_folder
stimulus_folder.mkdir(parents=True, exist_ok=True)

mr.construct_retina()

# These are the variables to be changed in the experiment
# See visual_stimulus_parameters, safe up to two variables
exp_variables = ["contrast", "temporal_frequency"]
mr.config.experiment_parameters = {
    "exp_variables": exp_variables,
    # two vals below for each exp_variable, even is it is not changing
    "min_max_values": [[0, 1.0], [0.2, 60]],  # [[0, 0.6], [0.1, 15.0]]
    "n_steps": [10, 16],  # [10 ,16]
    "logarithmic": [True, True],  # [True, True]
    "n_sweeps": 1,
    # "distributions": {"gaussian": {"sweeps": 10, "mean": [-30, 30], "sd": [5, 5]}},
    "distributions": {"uniform": None},
}

filename = mr.experiment.build_and_run(build_without_run=False)

########################################
## Analyze and visualize experiment ###
########################################

my_analysis_options = {
    "exp_variables": exp_variables,
    "t_start_ana": 0.5,
    "t_end_ana": 12.5,
}
mr.analysis.analyze_experiment(filename, my_analysis_options)

# #########################################
# # filename = "exp_metadata_contrast_temporal_frequency_0da761a886.csv"

# # mr.viz.show_fr4c_response(
# #     filename,
# #     exp_variables,
# #     xlog=False,
# #     ylog=False,
# #     xlim=None,
# #     ylim=None,
# #     savefigname=f"{mr.config.experiment}_{rp.gc_type}_{rp.response_type}_{rp.spatial_model_type}_{rp.temporal_model_type}.eps",
# # )

# Contrast sensitivity
# filename = "exp_metadata_contrast_spatial_frequency_0f60a89182e4.csv"
mr.viz.contrast_sensitivity(
    filename,
    ["contrast", "temporal_frequency"],
    xlog=True,
    ylog=True,
    xlim=[0.1, 100],
    ylim=[1, 500],
    savefigname=f"{mr.config.experiment}_{rp.gc_type}_{rp.response_type}_{rp.spatial_model_type}_{rp.temporal_model_type}_{sf}cpd.eps",
)

end_time = time.time()
print(f"Elapsed time: {end_time - start_time} seconds")
plt.show()
