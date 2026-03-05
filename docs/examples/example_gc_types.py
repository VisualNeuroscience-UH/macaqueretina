# Third-party
import matplotlib.pyplot as plt

# Local
import macaqueretina as mr

mr.load_parameters()
mr.visual_stimulus.make_stimulus_video()

gc_types = ["parasol", "midget"]
response_types = ["on", "off"]

for gc_type in gc_types:
    for response_type in response_types:
        mr.config.retina_parameters.gc_type = gc_type
        mr.config.retina_parameters.response_type = response_type
        mr.construct_retina.build_retina_client()
        mr.simulate_retina.client()
        mr.viz.show_all_gc_responses(savefigname=None)

print(f"Output folder: {mr.config.output_folder}")
###############################################

plt.show()
