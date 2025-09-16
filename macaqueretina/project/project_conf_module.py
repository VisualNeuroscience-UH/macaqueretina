# Built-in
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable

# Third-party
import matplotlib.pyplot as plt

# Local
from macaqueretina.data_io.config_io import load_yaml
from macaqueretina.parameters.param_reorganizer import ParamReorganizer
from macaqueretina.project.project_manager_module import ProjectManager

if TYPE_CHECKING:
    from macaqueretina.data_io.config_io import ConfigManager

start_time = time.time()
warnings.simplefilter("ignore")


def _validation_switch(base: Path) -> Callable | None:
    """
    Perform parameter validation if a .py file with 'validation' in its name
    is found in the parameters/ subfolder.

    Returns:
        Callable or None: validation function (validate_params) if found, None otherwise
    """
    validation_files = list(base.glob("*validation*.py"))
    match len(validation_files):
        case 0:
            print(
                f"No validation file provided in {base}. "
                f"Proceeding without parameter validation."
            )
            return None
        case 1:
            try:
                from macaqueretina.parameters.param_validation import validate_params

                return validate_params
            except ImportError as e:
                print(f"Could not import validation file: {e}")
                return None
        case n:
            raise ValueError(
                f"Expected at most 1 validation file in {base}, but found {n} files"
                f" with 'validation' in their name:"
                f"{[file.name for file in validation_files]}"
            )


def load_parameters() -> "ConfigManager":
    """Load configuration parameters."""
    project_conf_module_file_path = Path(__file__).resolve()
    git_repo_root_path = project_conf_module_file_path.parent.parent

    reorganizer = ParamReorganizer()

    base: Path = git_repo_root_path.joinpath("parameters/")
    yaml_files = list(base.glob("*.yaml"))

    validate_params: Callable | None = _validation_switch(base)

    config: ConfigManager = load_yaml(yaml_files)

    if validate_params:
        validated_config = validate_params(
            config.as_dict(), project_conf_module_file_path, git_repo_root_path
        )
        validated_config = validated_config.model_dump()
        reorganized_config = reorganizer.reorganize(validated_config)

        config._config = reorganized_config

    return config


def dispatcher(PM: "ProjectManager", config: "ConfigManager"):
    run = config.run
    if run.build_retina:
        PM.construct_retina.build_retina_client()
    if run.make_stimulus:
        PM.stimulate.make_stimulus_video()
    if run.simulate_retina:
        PM.simulate_retina.client()
    if run.visualize_DoG_img_grid.show:
        options = run.visualize_DoG_img_grid
        PM.viz.show_DoG_img_grid(
            gc_list=options.gc_list,
            n_samples=options.n_samples,
            savefigname=options.savefigname,
        )
    if run.visualize_all_gc_responses:
        options = run.visualize_all_gc_responses
        PM.viz.show_all_gc_responses()


def main():
    config = load_parameters()

    if config.profile is True:
        # Built-in
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        end_time = time.time()

    PM = ProjectManager(config)

    dispatcher(PM, config)

    end_time = time.time()
    print(
        "Total time taken: ",
        time.strftime(
            "%H hours %M minutes %S seconds", time.gmtime(end_time - start_time)
        ),
    )

    plt.show()

    if config.profile is True:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("tottime")
        stats.print_stats(20)


if __name__ == "__main__":
    main()


#     ###################################
#     ###################################
#     ###         Single Trial        ###
#     ###################################
#     ###################################

#     ########################
#     ### Create stimulus ###
#     ########################

#     # Based on visual_stimulus_parameters above
#     PM.stimulate.make_stimulus_video()

#     ####################################
#     ### Run multiple trials or units ###
#     ####################################

#     PM.simulate_retina.client()

#     ##########################################
#     ### Show single ganglion cell features ###
#     ##########################################
#     # PM.viz.show_spatiotemporal_filter_summary(savefigname=None)

#     # PM.viz.show_spatiotemporal_filter(unit_index=1, savefigname=None)
#     # PM.viz.show_temporal_kernel_frequency_response(unit_index=2, savefigname=None)
#     # PM.viz.plot_midpoint_contrast(unit_index=2, savefigname=None)
#     # PM.viz.plot_local_rms_contrast(unit_index=2, savefigname=None)
#     # PM.viz.plot_local_michelson_contrast(unit_index=2, savefigname=None)
#     # PM.viz.show_single_gc_view(unit_index=2, frame_number=300, savefigname=None)

#     # ################################################################################
#     # #####     Interactive plot of spike frequency on stimulus video     ############
#     # ################################################################################

#     # video_file_name = visual_stimulus_parameters["stimulus_video_name"]
#     # response_file_name = run_parameters["gc_response_filenames"][0] + ".gz"
#     # window_length = 0.1  # seconds
#     # rate_scale = 20  # Hz, Colorscale max amplitude
#     # PM.viz_spikes_with_stimulus.client(
#     #     video_file_name, response_file_name, window_length, rate_scale
#     # )

#     ################################################
#     ###   Show multiple trials for single unit,  ###
#     ###   or multiple units for single trial     ###
#     ################################################

#     # Based on run_parameters above
#     PM.viz.show_all_gc_responses(savefigname=None)
#     # PM.viz.show_all_generator_potentials(savefigname=None)
#     # PM.viz.show_generator_potential_histogram(savefigname=None)
#     # PM.viz.show_cone_responses(time_range=[0.4, 1.1], savefigname=None)
#     # PM.viz.show_gc_noise(savefigname=None)

#     # PM.viz.show_stimulus_with_gcs(
#     #     example_gc=None,  # [int,], None for all
#     #     frame_number=31,  # 31 depends on fps, and video and baseline lengths
#     #     show_rf_id=True,
#     #     savefigname=f"stimulus_with_gcs_{gc_type}_{response_type}.png",
#     # )

#     ##########################################
#     ###       Show impulse response        ###
#     ##########################################

#     # # Contrast applies only for parasol units with dynamic model, use [1.0] for others
#     # PM.context.run_parameters["contrasts_for_impulse"] = [1.0]
#     # PM.simulate_retina.client(impulse=True)
#     # PM.viz.show_impulse_response(savefigname="sununit_impulse_response.eps")

#     #########################################
#     ##          Show unity data           ###
#     #########################################

#     # Get uniformity data and exit
#     # See Gauthier_2009_PLoSBiol for details
#     # TODO: MAKE UNITY MAX EXPERIMENT, SET CENTER MASK TH TO UNITY MAX
#     # PM.simulate_retina.client(unity=True)
#     # PM.viz.show_unity(savefigname=None)

#     #################################################################
#     #################################################################
#     ###   Experiment with multiple units, conditions and trials   ###
#     #################################################################
#     #################################################################

#     ################################
#     ### Build and run Experiment ###
#     ################################

#     # # Retina needs to be built for this to work.
#     # # visual_stimulus_parameters above defines the stimulus. From that dictionary,
#     # # defined keys' values are dynamically changed in the experiment.
#     # # Note that tuple values from visual_stimulus_parameters are captured for varying each tuple value separately.

#     # # from visual_stimulus_parameters, safe up to two variables
#     # exp_variables = ["temporal_frequency"]
#     # experiment_parameters = {
#     #     "exp_variables": exp_variables,
#     #     # two vals below for each exp_variable, even is it is not changing
#     #     # "min_max_values": [[5, 5]],
#     #     "min_max_values": [[1, 32]],
#     #     # "min_max_values": [[1, 32], [0.015, 0.5]],
#     #     "n_steps": [2],  # [6 ,10]
#     #     # "n_steps": [16],  # [6 ,10]
#     #     "logarithmic": [True],
#     #     # "logarithmic": [True, True],
#     #     "n_sweeps": 1,
#     #     # "distributions": {"gaussian": {"sweeps": 10, "mean": [-30, 30], "sd": [5, 5]}},
#     #     "distributions": {"uniform": None},
#     # }

#     # # # # N trials or N units must be 1, and the other > 1. This is set above in run_parameters.
#     # filename = PM.experiment.build_and_run(
#     #     experiment_parameters, build_without_run=False, show_histogram=False
#     # )
#     # filename = "exp_metadata_orientation_spatial_frequency_edd770296d0e.csv"

#     # # #########################
#     # # ## Analyze Experiment ###
#     # # #########################

#     # my_analysis_options = {
#     #     "exp_variables": exp_variables,
#     #     "t_start_ana": 0.5,
#     #     "t_end_ana": 6.5,
#     # }

#     # # PM.ana.analyze_experiment(filename, my_analysis_options)
#     # # PM.ana.unit_correlation(
#     # #     filename, my_analysis_options, gc_type, response_type, gc_units=None
#     # # )
#     # PM.ana.relative_gain(filename, my_analysis_options)
#     # # PM.ana.response_vs_background(filename, my_analysis_options)

#     # # ############################
#     # # ### Visualize Experiment ###
#     # # ############################
#     # PM.viz.spike_raster_response(filename, sweeps_to_show=[0], savefigname="exp.png")
#     # # # PM.viz.show_relative_gain(filename, exp_variables, savefigname=None)
#     # # # PM.viz.show_response_vs_background_experiment(unit="cd/m2", savefigname=None)

#     # # # PM.viz.show_unit_correlation(
#     # # #     filename, exp_variables, time_window=[-0.2, 0.2], savefigname=None
#     # # # )

#     # PM.viz.F1F2_popul_response(
#     #     filename,
#     #     exp_variables,
#     #     xlog=True,
#     #     F1_only=False,
#     #     savefigname=None,
#     # )
#     # # # PM.viz.F1F2_unit_response(filename, exp_variables, xlog=True, savefigname=None)
#     # # # PM.viz.F1F2_unit_response(
#     # # #     filename, exp_variables, xlog=True, hue="temporal_frequency", savefigname=None
#     # # # )
#     # # ### PM.viz.ptp_response(filename, exp_variables, x_of_interest=None, savefigname=None)
#     # # ### PM.viz.fr_response(filename, exp_variables, xlog=True, savefigname=None)
#     # # # PM.viz.tf_vs_fr_cg(
#     # # #     filename,
#     # # #     exp_variables,
#     # # #     n_contrasts=10,
#     # # #     xlog=True,
#     # # #     ylog=False,
#     # # #     savefigname=None,
#     # # # )
