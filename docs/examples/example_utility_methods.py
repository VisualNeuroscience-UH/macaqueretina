"""Example usage of MacaqueRetina."""

# Built-in
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt

# Local
import macaqueretina as mr

mr.load_parameters()

#########################
##  Print parameters  ###
#########################

print("\nRetina parameters:")
print(mr.config.retina_parameters)

print("\nVisual stimulus parameters:")
print(mr.config.visual_stimulus_parameters)

print("\nSimulation parameters:")
print(mr.config.simulation_parameters)

################################################################
###  Count lines in codebase, relative to working directory  ###
################################################################

mr.project_utilities.countlines(Path("macaqueretina"))


# #####################################################
# ##   Luminance and Photoisomerization calculator   ##
# #####################################################

# I_cone = 4000  # photoisomerizations per second per cone
# A_pupil = 9.0

# luminance = mr.retina_math.get_luminance_from_photoisomerizations(
#     I_cone, A_pupil=A_pupil
# )
# print(f"{luminance:.2f} cd/m2")
# print(f"{luminance * A_pupil:.2f} trolands")

# luminance = 128  # Luminance in cd/m2
# A_pupil = 9.0

# I_cone = mr.retina_math.get_photoisomerizations_from_luminance(
#     luminance, A_pupil=A_pupil
# )
# print(f"{I_cone:.2f} photoisomerizations per second per cone")
# print(f"{luminance * A_pupil:.2f} trolands")


# #####################################################
# ###  Sample and view figure data from literature  ###
# #####################################################

# # Example validation file, at
# filename = "Derrington_1984b_Fig10B_magno_spatial.jpg"
# filename_full = mr.config.git_repo_root_path.joinpath(
#     r"retina/validation_data", filename
# )

# # Plot lowest and highest tick values in the image, use these as calibration points
# min_X, max_X, min_Y, max_Y = (0.1, 10, 1, 100)  # Needs to be set for each figure
# ds = mr.data_sampler(filename_full, min_X, max_X, min_Y, max_Y, logX=True, logY=True)

# # ds.collect_and_save_points() # Will overwrite existing data file -- change filename or move existing file first
# ds.quality_control()  # Show image with calibration and data points
# # x_data, y_data = ds.get_data_arrays() # Get data arrays for further processing


# #######################################################################
# ###  For the rest, you need to run these once to create data files  ###
# #######################################################################

# mr.retina_constructor.construct()
# mr.stimulus_factory.generate()
# mr.retina_simulator.simulate()

# #########################################
# ##  Load arbitrary data to workspace  ###
# #########################################
# output_folder = mr.config.output_folder
# # Copy the filename from output folder after running build_retina() once.
# filename_map = output_folder.glob("*_mosaic.csv")
# filename = list(filename_map)[0]
# data = mr.data_io.load_data(filename)
# print(type(data))
# print(data.shape)


# ##################################
# ### Show spikes from gz files  ###
# ##################################

# output_folder = mr.config.output_folder
# ret_file_map = output_folder.glob("*.gz")
# filename = list(ret_file_map)[0]
# mr.viz.show_spikes_from_gz_file(
#     filename=filename,
#     sweepname="spikes_0",  # "spikes_0", "spikes_1", ...
#     savefigname=None,
# )


# #########################################
# ##       Show impulse response        ###
# #########################################
# # unity and impulse make a dummy video which uses visual stimulus parameters.

# mr.config.retina_parameters.gc_type = "parasol"  # "parasol", "midget"
# mr.config.retina_parameters.response_type = "on"  # "on", "off"
# mr.config.retina_parameters.spatial_model_type = "DOG"  # "DOG", "VAE"
# mr.config.retina_parameters.temporal_model_type = (
#     "fixed"  # "fixed", "dynamic", "subunit"
# )
# mr.retina_constructor.construct()

# mr.config.simulation_parameters["contrasts_for_impulse"] = (1.0,)
# mr.retina_simulator.simulate(impulse=True)
# mr.viz.show_impulse_response_after_simulate(savefigname=None)

# ######################################
# ###        Show unity data         ###
# ######################################
# mr.config.retina_parameters.ecc_limits_deg = [3.5, 6.5]  # eccentricity in degrees
# mr.config.retina_parameters.pol_limits_deg = [-15, 15]  # polar angle in degrees
# mr.retina_constructor.construct()

# mr.retina_simulator.simulate(unity=True)
# mr.viz.show_unity(savefigname=None)


# plt.show()
