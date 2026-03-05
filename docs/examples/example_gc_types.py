# Third-party
import matplotlib.pyplot as plt

# Local
import macaqueretina as mr

mr.load_parameters()
mr.stimulus_factory.generate()

gc_types = ["parasol", "midget"]
response_types = ["on", "off"]

for gc_type in gc_types:
    for response_type in response_types:
        mr.config.retina_parameters.gc_type = gc_type
        mr.config.retina_parameters.response_type = response_type
        mr.retina_constructor.construct()
        mr.retina_simulator.simulate()
        mr.viz.show_all_gc_responses_after_simulate(savefigname=None)

print(f"Output folder: {mr.config.output_folder}")
###############################################

plt.show()
