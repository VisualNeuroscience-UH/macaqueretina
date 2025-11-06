"""Example usage of MacaqueRetina."""

# Built-in
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# Local
import macaqueretina as mr

# ##########################
# ###  Print parameters  ###
# ##########################

print("\nRetina parameters:")
print(mr.config.retina_parameters)

print("\nVisual stimulus parameters:")
print(mr.config.visual_stimulus_parameters)

print("\nSimulation parameters:")
print(mr.config.simulation_parameters)

# ################################################################
# ###  Count lines in codebase, relative to working directory  ###
# ################################################################

# mr.countlines(Path("macaqueretina"))


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
# ds = mr.DataSampler(filename_full, min_X, max_X, min_Y, max_Y, logX=True, logY=True)

# # ds.collect_and_save_points() # Will overwrite existing data file -- change filename or move existing file first
# ds.quality_control()  # Show image with calibration and data points
# # x_data, y_data = ds.get_data_arrays() # Get data arrays for further processing


# #######################################################################
# ###  For the rest, you need to run these once to create data files  ###
# #######################################################################

# mr.construct_retina()
# mr.make_stimulus()
# mr.simulate_retina()

##########################################
###  Load arbitrary data to workspace  ###
##########################################
# filename_parents = mr.config.output_folder
# filename = Path(filename_parents).joinpath(mr.config.retina_parameters.mosaic_file)

# data = mr.load_data(filename)
# print(type(data))
# print(data.shape)


# ##################################
# ### Show spikes from gz files  ###
# ##################################

# filename_parents = mr.config.output_folder
# # Build response filename from scratch
# gc_type = mr.config.retina_parameters["gc_type"]
# response_type = mr.config.retina_parameters["response_type"]
# hashstr = mr.config.retina_parameters["retina_parameters_hash"]
# filename_offspring = f"{gc_type}_{response_type}_{hashstr}_response_00.gz"
# filename = Path(filename_parents).joinpath(filename_offspring)
# mr.viz.show_spikes_from_gz_file(
#     filename=filename,
#     sweepname="spikes_0",  # "spikes_0", "spikes_1", ...
#     savefigname=None,
# )


# #########################################
# ##       Show impulse response        ###
# #########################################

# mr.config.retina_parameters.gc_type = "parasol"  # "parasol", "midget"
# mr.config.retina_parameters.response_type = "on"  # "on", "off"
# mr.config.retina_parameters.spatial_model_type = "DOG"  # "DOG", "VAE"
# mr.config.retina_parameters.temporal_model_type = (
#     "fixed"  # "fixed", "dynamic", "subunit"
# )
# mr.construct_retina()

# mr.config.simulation_parameters["contrasts_for_impulse"] = [1.0]
# mr.simulate_retina(impulse=True)
# mr.viz.show_impulse_response(savefigname=None)

# ######################################
# ###        Show unity data         ###
# ######################################
# mr.config.retina_parameters.ecc_limits_deg = [3.5, 6.5]  # eccentricity in degrees
# mr.config.retina_parameters.pol_limits_deg = [-15, 15]  # polar angle in degrees
# mr.construct_retina()

# mr.simulate_retina(unity=True)
# mr.viz.show_unity(savefigname=None)


plt.show()
