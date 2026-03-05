## Run multiple GC types
This example shows how to run the four ganglion cell types using the same cone noise.


```python
import matplotlib.pyplot as plt
import macaqueretina as mr

mr.load_parameters()
```

You need to make one simulus first
```python
mr.stimulus_factory.generate()
```

#### Define types
```python
gc_types = ["parasol", "midget"]
response_types = ["on", "off"]
```

#### Construct, simulate and show spikes
```python

filename = "my_response.gz"

for gc_type in gc_types:
    for response_type in response_types:
        mr.config.retina_parameters.gc_type = gc_type
        mr.config.retina_parameters.response_type = response_type
        mr.retina_constructor.construct()
        mr.retina_simulator.simulate(filename=filename)
        mr.viz.show_all_gc_responses(savefigname=None)

print(f"Output folder: {mr.config.output_folder}")

plt.show()
```

