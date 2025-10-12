# Third-party
import matplotlib.pyplot as plt

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


###############################################
##   Show multiple units for single trial   ###
###############################################

# for all temporal models ##
mr.viz.show_all_gc_responses(savefigname=None)
mr.viz.show_all_generator_potentials(savefigname=None)

mr.viz.show_stimulus_with_gcs(
    example_gc=None,  # [int,], None for all
    frame_number=31,  # 31 depends on fps, and video and baseline lengths
    show_rf_id=True,
    savefigname=None,
)

mr.viz.show_single_gc_view(unit_index=2, frame_number=31, savefigname=None)
mr.viz.show_gc_noise_hist_cov_mtx(savefigname=None)

# # for fixed temporal model only ##
# mr.viz.show_spatiotemporal_filter_sums(savefigname=None)
# mr.viz.show_spatiotemporal_filter(unit_index=1, savefigname=None)
# mr.viz.show_temporal_kernel_frequency_response(unit_index=2, savefigname=None)

# # for subunit temporal model only ##
# mr.viz.show_cone_responses(time_range=[0.0, 1.1], savefigname=None)


###############################################

plt.show()
