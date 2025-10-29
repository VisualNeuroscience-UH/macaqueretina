
## Getting started with the MacaqueRetina simulator

The simulator can be run either directly or imported as package. When run directly, the parameters are read from all the yaml files in macaqueretina/parameters folder. When imported as a package, you have the option of changing parameters at runtime.


### First, set the folder for your output data 

Navigate to your local MacaqueRetina git repository root.

Open macaqueretina/parameters/core_parameters.yaml and change the `model_root_path` to point to an existing directory in your system. This is where all output data will be written.


### Run using parameters from the yaml files

Add the 'shell' plugin to your Poetry environment:

```bash
poetry self add poetry-plugin-shell
```

Activate your Poetry-managed environment:

```bash
poetry shell
```

Run the project:

```bash
python macaqueretina
```

This will run the simulator with the parameters read from the yaml files. 

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

mr.construct_retina()
mr.make_stimulus()
mr.simulate_retina()
mr.viz.show_all_gc_responses(savefigname=None)
plt.show()
```

Variables saved to disk can be accessed with:
```python
 import macaqueretina as mr 
 
 my_data = mr.load_data("filename")
```
