"""
Example of running the project_manager_module from container in SLURM environment.
Parameters are changed via environment variables ??using param_hpc_updater.py.??
The environment variables are set in SLURM job script.
"""

if __name__ == "__main__":
    import macaqueretina as mr

    # Stimulus folder
    mr.config.stimulus_folder = mr.config.path.joinpath(f"stim_{mr.config.experiment}")
    mr.config.stimulus_folder.mkdir(parents=True, exist_ok=True)

    # Output folder
    retina_parameters = mr.config.retina_parameters
    output_folder = f"{retina_parameters.gc_type}_{retina_parameters.response_type}_{retina_parameters.spatial_model_type}_{retina_parameters.temporal_model_type}"
    mr.config.output_folder = mr.config.path.joinpath(output_folder)
    mr.config.output_folder.mkdir(parents=True, exist_ok=True)

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

    filename = mr.experiment.build_and_run(build_without_run=False)

    ########################################
    ## Analyze and visualize experiment ###
    ########################################

    my_analysis_options = {
        "exp_variables": exp_variables,
        "t_start_ana": 0.5,
        "t_end_ana": 1.5,
    }
    mr.analysis.analyze_experiment(filename, my_analysis_options)

    #############################################

    # plt.show()
