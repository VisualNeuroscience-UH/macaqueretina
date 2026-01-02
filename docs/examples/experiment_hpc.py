"""
Example of running the project_manager_module from container in SLURM environment.
Parameters are changed via environment variables using __main__.py when invoking python macaqueretina
The environment variables are set in SLURM job script.
"""

# import matplotlib.pyplot as plt

import macaqueretina as mr

# Stimulus folder
mr.config.stimulus_folder = mr.config.path.joinpath(f"stim_{mr.config.experiment}")
mr.config.stimulus_folder.mkdir(parents=True, exist_ok=True)

# Output folder
retina_parameters = mr.config.retina_parameters

for calibrated_gain in range(1, 11):
    ###############################
    ## Build and run experiment ###
    ###############################

    mr.config.retina_parameters.calibrated_gain = float(calibrated_gain)
    contrast = str(mr.config.visual_stimulus_parameters.contrast).replace(".", "p")
    sf = str(mr.config.visual_stimulus_parameters.spatial_frequency).replace(".", "p")
    gain = str(calibrated_gain)
    output_folder = f"{retina_parameters.gc_type}_{retina_parameters.response_type}_{retina_parameters.spatial_model_type}_{retina_parameters.temporal_model_type}_c{contrast}_g{gain}_sf{sf}"
    print(f"\n{output_folder=}\n")
    mr.config.output_folder = mr.config.path.joinpath(output_folder)
    mr.config.output_folder.mkdir(parents=True, exist_ok=True)
    mr.construct_retina()

    # These are the variables to be changed in the experiment
    # See visual_stimulus_parameters, safe up to two variables
    exp_variables = ["temporal_frequency"]
    mr.config.experiment_parameters = {
        "exp_variables": exp_variables,
        # two vals below for each exp_variable, even is it is not changing
        "min_max_values": [[1.0, 32.0]],  # [[0, 0.6], [0.1, 15.0]]
        # "n_steps": [2],  # [10 ,16]
        "n_steps": [16],  # [10 ,16]
        "logarithmic": [True],  # [True, True]
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

    ############################################

# threshold = 10
# # folder_pattern = "parasol_on_DOG_fixed_contrast0p035_gain*_temporal_frequency"
# folder_pattern = "gain_calibration_sf_parasol_on_VAE_dynamic_c0p035_g*_sf2"
# # folder_pattern = "parasol_on_DOG_subunit_c0p035_g*_sf2p0"
# # folder_pattern = "parasol_on_DOG_subunit_c0p035"
# gain_multiplier = 1.0
# mr.viz.show_gain_calibration(
#     threshold, folder_pattern, gain_multiplier=gain_multiplier, savefigname=None
# )

# plt.show()

# To replicate old Fig 6.
# core_parameters.yaml:
#   experiment: "gain_calibration_flicker"
# threshold = 10
# folder_pattern = "parasol_on_DOG_fixed_contrast0p035_gain*_temporal_frequency"
# # folder_pattern = "gain_calibration_sf_parasol_on_DOG_fixed_c0p035_g*_sf2"
# # folder_pattern = "parasol_on_DOG_subunit_c0p035_g*_sf2p0"
# # folder_pattern = "parasol_on_DOG_subunit_c0p035"
# gain_multiplier = 0.5
# mr.viz.show_gain_calibration(
#     threshold, folder_pattern, gain_multiplier=gain_multiplier, savefigname=None
# )

# plt.show()
