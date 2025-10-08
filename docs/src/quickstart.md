
## Getting started with MacaqueRetina simulator

Navigate to your local MacaqueRetina git repository root.

Open file macaqueretina/parameters/core_parameters.yaml and change the  `model_root_path` to point into an existing directory in your system. This is where all data will be written out.


### How to run using parameters from the yaml files

Activate your Poetry-managed environment:

```
poetry shell
```

Run the project:

```
python macaqueretina/project/project_conf_module.py
```

This will run the simulator with the parameters from the yaml files. 

### How to import and run

Activate your Poetry-managed environment:

```
poetry shell
```

Run the following: 
```
import macaqueretina as mr
import matplotlib.pyplot as plt

mr.construct_retina()
mr.make_stimulus()
mr.simulate_retina()
mr.viz.show_all_gc_responses(savefigname=None)
plt.show()
```