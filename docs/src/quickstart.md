
## Getting started with the MacaqueRetina simulator

The simulator should be imported as package. When imported as a package, you have the option of changing parameters at runtime.


### First, set the folder for your output data 

Navigate to your local MacaqueRetina git repository root.

Open macaqueretina/parameters/core_parameters.yaml and change the `model_root_path` to point to an existing directory in your system. This is where all output data will be written.


### Import as a package and run

Add the 'shell' plugin to your Poetry environment:

```bash
poetry self add poetry-plugin-shell
```

Activate your Poetry-managed environment:

```bash
poetry shell
```

Import into your Python environment:
```python
import macaqueretina as mr
```

Run the following for a quick example: 
```python
import macaqueretina as mr
import matplotlib.pyplot as plt

mr.load_parameters()
filename="my_response.gz"

mr.retina_constructor.construct()
mr.stimulus_factory.generate()
mr.retina_simulator.simulate(filename=filename)
mr.viz.show_all_gc_responses_after_simulate()
plt.show()
```

Variables saved to disk can be accessed with:
```python
 import macaqueretina as mr 
 
 my_data = mr.data_io.load_data(filename)
```
