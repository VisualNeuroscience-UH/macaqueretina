"""Example usage of MacaqueRetina."""

# Built-in
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# Local
import macaqueretina as mr

###################################
###################################
###         Single Trial        ###
###################################
###################################
mr.PM.construct_retina._get_parameters_for_build()  # Hack to get the retina_parameters updated


# mr.construct_retina()
# mr.make_stimulus()
# mr.simulate_retina()


################################################
###   Show multiple units for single trial   ###
################################################
# # mr.viz.show_all_gc_responses(savefigname=None)
# mr.viz.show_all_generator_potentials(savefigname=None)
# mr.viz.show_generator_potential_histogram(savefigname=None)


##########################################
###  After active mr.simulate_retina() ###
##########################################

# mr.viz.show_stimulus_with_gcs(
#     example_gc=None,  # [int,], None for all
#     frame_number=31,  # 31 depends on fps, and video and baseline lengths
#     show_rf_id=True,
#     savefigname=None,
# )

## for fixed temporal response model only ##
# mr.viz.show_spatiotemporal_filter_sums(savefigname=None)
# mr.viz.show_spatiotemporal_filter(unit_index=1, savefigname=None)
# mr.viz.show_temporal_kernel_frequency_response(unit_index=2, savefigname=None)

## for subunit temporal response model only ##
# mr.viz.show_cone_responses(time_range=[0.4, 1.1], savefigname=None)

## for all temporal response models ##
# mr.viz.show_single_gc_view(unit_index=2, frame_number=31, savefigname=None)
# mr.viz.show_gc_noise_hist_cov_mtx(savefigname=None)

# ################################################################################
# #####     Interactive plot of spike frequency on stimulus video     ############
# ################################################################################

# video_file_name = mr.config.visual_stimulus_parameters["stimulus_video_name"]
# # Zero index points to first file
# file_idx = 0
# response_file_name = mr.config.run_parameters["gc_response_filenames"][file_idx] + ".gz"

# window_length = 0.1  # seconds
# rate_scale = 20  # Hz, Colorscale max amplitude
# mr.PM.viz_spikes_with_stimulus.client(
#     video_file_name, response_file_name, window_length, rate_scale
# )


################################################################
################################################################
##   Experiment with multiple units, conditions and trials   ###
################################################################
################################################################

###############################
## Build and run experiment ###
###############################

# # See visual_stimulus_parameters, safe up to two variables
# exp_variables = ["temporal_frequency"]
# experiment_parameters = {
#     "exp_variables": exp_variables,
#     # two vals below for each exp_variable, even is it is not changing
#     # "min_max_values": [[5, 5]],
#     "min_max_values": [[1, 32]],
#     # "min_max_values": [[1, 32], [0.015, 0.5]],
#     "n_steps": [3],  # [6 ,10]
#     # "n_steps": [16],  # [6 ,10]
#     "logarithmic": [True],
#     # "logarithmic": [True, True],
#     "n_sweeps": 1,
#     # "distributions": {"gaussian": {"sweeps": 10, "mean": [-30, 30], "sd": [5, 5]}},
#     "distributions": {"uniform": None},
# }

# filename = mr.PM.experiment.build_and_run(
#     experiment_parameters, build_without_run=False, show_histogram=False
# )
# filename = "exp_metadata_orientation_spatial_frequency_edd770296d0e.csv"

# ########################################
# ## Analyze and visualize experiment ###
# ########################################

# my_analysis_options = {
#     "exp_variables": exp_variables,
#     "t_start_ana": 0.5,
#     "t_end_ana": 6.5,
# }
# mr.PM.ana.analyze_experiment(filename, my_analysis_options)
# mr.PM.ana.unit_correlation(
#     filename, my_analysis_options, "parasol", "on", gc_units=None
# )

############################
### Visualize experiment ###
############################
# mr.PM.viz.spike_raster_response(filename, sweeps_to_show=[0], savefigname="exp.png")
# mr.PM.viz.F1F2_popul_response(
#     filename,
#     exp_variables,
#     xlog=True,
#     savefigname=None,
# )
# mr.PM.viz.F1F2_unit_response(filename, exp_variables, xlog=True, savefigname=None)

# # Contrast gain
# mr.PM.viz.tf_vs_fr_cg(
#     filename,
#     exp_variables,
#     n_contrasts=2,
#     xlog=True,
#     ylog=False,
#     savefigname=None,
# )

# # Unit correlation vs distance
# mr.PM.viz.show_unit_correlation(
#     filename, exp_variables, time_window=[-0.2, 0.2], savefigname=None
# )

# mr.PM.viz.fr_response(filename, exp_variables, xlog=True, savefigname=None)


#################################
#################################
###   Utility functions       ###
#################################
#################################

# mr.PM.countlines(Path("macaqueretina"))

# # Load arbitrary data to workspace
# filename_parents = mr.config.output_folder
# filename_offspring = f"gc_response_00.gz"
# filename = Path(filename_parents).joinpath(filename_offspring)
# xx = mr.PM.data_io.get_data(filename)
# print(type(xx))


# ###################################
# ###  Show spikes from gz files  ###
# ###################################
# filename_parents = mr.config.output_folder
# filename_offspring = f"gc_response_00.gz"
# filename = Path(filename_parents).joinpath(filename_offspring)
# mr.viz.show_spikes_from_gz_file(
#     filename=filename,
#     sweepname="spikes_0",  # "spikes_0", "spikes_1", ...
#     savefigname=None,  # None for no save, or string with type suffix
# )

#########################################
##          Show unity data           ###
#########################################
# # See Gauthier_2009_PLoSBiol for details
# mr.PM.simulate_retina.client(unity=True)
# mr.viz.show_unity(savefigname=None)

##########################################
###       Show impulse response        ###
##########################################
# mr.PM.context.run_parameters["contrasts_for_impulse"] = [1.0]
# mr.PM.simulate_retina.client(impulse=True)
# mr.viz.show_impulse_response(savefigname=None)


# #####################################################
# ##   Luminance and Photoisomerization calculator   ##
# #####################################################

# I_cone = 4000  # photoisomerizations per second per cone
# A_pupil = 9.0

# luminance = mr.retina_math.get_luminance_from_photoisomerizations(
#     I_cone, A_pupil=A_pupil
# )
# print(f"{luminance:.2f} cd/m2")
# print(f"{luminance * A_pupil} trolands")

# luminance = 128  # Luminance in cd/m2
# A_pupil = 9.0

# I_cone = mr.retina_math.get_photoisomerizations_from_luminance(
#     luminance, A_pupil=A_pupil
# )
# print(f"{I_cone:.2f} photoisomerizations per second per cone")
# print(f"{luminance * A_pupil} trolands")


##########################################
#   Sample figure data from literature  ##
##########################################

# # TODO. make the git_repo_root_path available from the package
# project_conf_module_file_path = Path(__file__).resolve()
# git_repo_root_path = project_conf_module_file_path.parent.parent.parent
# # breakpoint()

# # Example validation file
# filename = "Derrington_1984b_Fig10B_magno_spatial.jpg"
# filename_full = git_repo_root_path.joinpath(
#     r"macaqueretina/retina/validation_data", filename
# )

# # Local
# from macaqueretina.project.project_utilities_module import DataSampler

# # Plot lowest and highest tick values in the image, use these as calibration points
# min_X, max_X, min_Y, max_Y = (0.1, 10, 1, 100)  # Needs to be set for each figure
# ds = DataSampler(filename_full, min_X, max_X, min_Y, max_Y, logX=True, logY=True)

# # ds.collect_and_save_points() # Will overwrite existing data file
# ds.quality_control() # Show image with calibration and data points
# # x_data, y_data = ds.get_data_arrays() # Get data arrays for further processing


#########################################
### NOT FUNCTIONAL YET ###
#########################################
# # Relative gain
# mr.PM.ana.relative_gain(filename, my_analysis_options)
# mr.PM.viz.show_relative_gain(filename, exp_variables, savefigname=None)

# # Subunit response vs background
# mr.PM.ana.response_vs_background(filename, my_analysis_options)
# mr.PM.viz.show_response_vs_background_experiment(unit="cd/m2", savefigname=None)


############################
###  Measure model gain  ###
############################

# mr.viz.show_gain_calibration(
#     signal_gain["threshold"],
#     f"parasol_on_DOG_subunit_c0p035_g*_temporal_frequency",
#     # f"{gc_type}_{response_type}_{spatial_model_type}_{temporal_model_type}_c{contrast_str}_g*_temporal_frequency",
#     signal_gain=10,
#     savefigname=None,
# )
##############################################
### NOT FUNCTIONAL YET ENDS ###
##############################################


plt.show()
