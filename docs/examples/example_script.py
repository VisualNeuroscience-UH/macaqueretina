"""Example usage of MacaqueRetina."""

# Local
import macaqueretina as mr

#################################
#################################
###   Utility functions       ###
#################################
#################################

# mr.countlines(Path("macaqueretina"))
# mr.countlines(Path("tests"))

# ###################################
# ###  Show spikes from gz files  ###
# ###################################

# filename_parents = mr.context.output_folder
# filename_offspring = f"Response_{gc_type}_{response_type}_tf6.gz"
# filename = Path(filename_parents).joinpath(filename_offspring)
# mr.viz.show_spikes_from_gz_file(
#     filename=filename,
#     savefigname=Path(filename).parent.name
#     + "_puhti_12s.png",  # None for no save, or string with type suffix
# )

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

#####################################################
##   Luminance and Photoisomerization calculator   ##
#####################################################

I_cone = 4000  # photoisomerizations per second per cone
A_pupil = 9.0

luminance = mr.retina_math.get_luminance_from_photoisomerizations(
    I_cone, A_pupil=A_pupil
)
print(f"{luminance:.2f} cd/m2")
print(f"{luminance * A_pupil} trolands")

luminance = 128  # Luminance in cd/m2
A_pupil = 9.0

I_cone = mr.retina_math.get_photoisomerizations_from_luminance(
    luminance, A_pupil=A_pupil
)
print(f"{I_cone:.2f} photoisomerizations per second per cone")
print(f"{luminance * A_pupil} trolands")

##########################################
#   Sample figure data from literature  ##
##########################################

# # # If possible, sample only temporal hemiretina
# # Third-party
# import numpy as np

# # Local
# from macaqueretina.project.project_utilities_module import DataSampler

# filename = "Derrington_1984b_Fig10B_magno_spatial.jpg"
# filename_full = git_repo_root_path.joinpath(r"retina/validation_data", filename)

# # # Fig lowest and highest tick values in the image, use these as calibration points
# # min_X, max_X, min_Y, max_Y = (0.5, 100, 1, 1000)  # DeValois 1974
# min_X, max_X, min_Y, max_Y = (0.1, 10, 1, 100)  # Derrington 1984b Fig10
# ds = DataSampler(filename_full, min_X, max_X, min_Y, max_Y, logX=True, logY=True)

# # ds.collect_and_save_points()
# # ds.quality_control(restore=True)

# x_data, y_data = ds.get_data_arrays()

# # parameters: K, K_c, r_c, k_s, r_s
# lower_bounds = [0, 0, 0, 0, 0]
# upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]

# # bounds = (lower_bounds, upper_bounds)
# bounds = (-np.inf, np.inf)

# mr.viz.show_data_and_fit(
#     mr.retina_math.enrothcugell_robson,
#     # mr.retina_math.temporal_contrast_sensitivity,
#     x_data,
#     y_data,
#     p0=(100, 10, 0.05, 5, 0.1),
#     xlim=(0.1, 100),
#     ylim=(1, 1000),
#     xlog=True,
#     ylog=True,
#     fit_in_log_space=False,
#     bounds=bounds,
#     savefigname=None,
#     # savefigname=f"Derrington_1984b_Fig10_magno_spatial_combined.eps",
#     # savefigname=f"{filename}.eps",
# )
# plt.show()
# # Built-in
# import sys

# sys.exit(0)  # Exit after sampling data
