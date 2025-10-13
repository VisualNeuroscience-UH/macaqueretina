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

mr.construct_retina()
mr.make_stimulus()
mr.simulate_retina()  # Requires mr.construct_retina() and mr.make_stimulus() to be run first

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
# mr.config.experiment_parameters = {
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

# filename = mr.experiment.build_and_run(build_without_run=False, show_histogram=False)
# filename = "exp_metadata_orientation_spatial_frequency_edd770296d0e.csv"

# ########################################
# ## Analyze and visualize experiment ###
# ########################################

# my_analysis_options = {
#     "exp_variables": exp_variables,
#     "t_start_ana": 0.5,
#     "t_end_ana": 6.5,
# }
# mr.analysis.analyze_experiment(filename, my_analysis_options)
# mr.analysis.unit_correlation(
#     filename, my_analysis_options, "parasol", "on", gc_units=None
# )

############################
### Visualize experiment ###
############################
# mr.viz.spike_raster_response(filename, sweeps_to_show=[0], savefigname="exp.png")
# mr.viz.F1F2_popul_response(
#     filename,
#     exp_variables,
#     xlog=True,
#     savefigname=None,
# )
# mr.viz.F1F2_unit_response(filename, exp_variables, xlog=True, savefigname=None)

# # Contrast gain
# mr.viz.tf_vs_fr_cg(
#     filename,
#     exp_variables,
#     n_contrasts=2,
#     xlog=True,
#     ylog=False,
#     savefigname=None,
# )

# # Unit correlation vs distance
# mr.viz.show_unit_correlation(
#     filename, exp_variables, time_window=[-0.2, 0.2], savefigname=None
# )

# mr.viz.fr_response(filename, exp_variables, xlog=True, savefigname=None)


#################################
#################################
###   Utility functions       ###
#################################
#################################

# mr.countlines(Path("macaqueretina"))

# # Load arbitrary data to workspace
# filename_parents = mr.config.output_folder
# filename_offspring = f"gc_response_00.gz"
# filename = Path(filename_parents).joinpath(filename_offspring)
# xx = mr.get_data(filename)
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
# See Gauthier_2009_PLoSBiol for details
# mr.simulate_retina(unity=True)
# mr.viz.show_unity(savefigname=None)

##########################################
###       Show impulse response        ###
##########################################
# mr.config.run_parameters["contrasts_for_impulse"] = [1.0]
# mr.simulate_retina(impulse=True)
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

# # Example validation file
# filename = "Derrington_1984b_Fig10B_magno_spatial.jpg"
# filename_full = mr.config.git_repo_root_path.joinpath(
#     r"macaqueretina/retina/validation_data", filename
# )

# # Plot lowest and highest tick values in the image, use these as calibration points
# min_X, max_X, min_Y, max_Y = (0.1, 10, 1, 100)  # Needs to be set for each figure
# ds = mr.DataSampler(filename_full, min_X, max_X, min_Y, max_Y, logX=True, logY=True)

# # ds.collect_and_save_points() # Will overwrite existing data file
# ds.quality_control()  # Show image with calibration and data points
# # x_data, y_data = ds.get_data_arrays() # Get data arrays for further processing


#########################################
### NOT FUNCTIONAL YET ###
#########################################
# # Relative gain
# mr.analysis.relative_gain(filename, my_analysis_options)
# mr.viz.show_relative_gain(filename, exp_variables, savefigname=None)

# # Subunit response vs background
# mr.ana.response_vs_background(filename, my_analysis_options)
# mr.viz.show_response_vs_background_experiment(unit="cd/m2", savefigname=None)


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
