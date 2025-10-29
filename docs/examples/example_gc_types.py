# Third-party
import matplotlib.pyplot as plt

# Local
import macaqueretina as mr

mr.make_stimulus()

gc_types = ["parasol", "midget"]
response_types = ["on", "off"]

for gc_type in gc_types:
    for response_type in response_types:
        mr.config.retina_parameters.gc_type = gc_type
        mr.config.retina_parameters.response_type = response_type
        mr.construct_retina()
        mr.simulate_retina()
        mr.viz.show_all_gc_responses(savefigname=None)

print(f"Output folder: {mr.config.output_folder}")
###############################################

plt.show()
