# Third-party
import matplotlib.pyplot as plt

# Local
import macaqueretina as mr

mr.construct_retina()
mr.make_stimulus()
mr.simulate_retina()

###############################################
##   Show multiple units for single trial   ###
###############################################

# for all temporal models ##
mr.viz.show_all_gc_responses(savefigname=None)
# mr.viz.show_all_generator_potentials(savefigname=None)

# mr.viz.show_stimulus_with_gcs(
#     example_gc=None,  # [int,], None for all
#     frame_number=180,  # depends on fps and baseline lengths
#     show_rf_id=False,
#     savefigname=None,
# )

# mr.viz.show_single_gc_view(unit_index=2, frame_number=160, savefigname=None)
# mr.viz.show_gc_noise_hist_cov_mtx(savefigname=None)

# # for fixed temporal model only ##
# mr.viz.show_spatiotemporal_filter_sums(savefigname=None)
# mr.viz.show_spatiotemporal_filter(unit_index=1, savefigname=None)
# mr.viz.show_temporal_kernel_frequency_response(unit_index=2, savefigname=None)

# # for subunit temporal model only ##
# mr.viz.show_cone_responses(time_range=[0.0, 1.1], savefigname=None)


# ################################################################################
# #####     Interactive plot of spike frequency on stimulus video     ############
# ################################################################################
# video_file_name = mr.config.visual_stimulus_parameters.stimulus_video_name
# response_file_name = mr.config.simulation_parameters.gc_response_filenames[0] + ".gz"

# window_length = 0.1  # seconds
# rate_scale = 20  # Hz, Colorscale max amplitude
# mr.viz_spikes_with_stimulus(
#     video_file_name, response_file_name, window_length, rate_scale
# )


###############################################

plt.show()
