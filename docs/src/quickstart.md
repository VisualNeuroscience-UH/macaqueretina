
## Getting started with MacaqueRetina simulator

The simulator can be run either directly or imported as package. When run directly, all parameters are read from macaqueretina/parameters folder. When imported you have the option of changing parameters at runtime.


### First, update the model root path to match your system

Navigate to your local MacaqueRetina git repository root.

Open file macaqueretina/parameters/core_parameters.yaml and change the  `model_root_path` to point into an existing directory in your system. This is where all data will be written out.


### Run using parameters from the yaml files

Activate your Poetry-managed environment:

```
poetry shell
```

Run the project:

```
python macaqueretina/project/project_conf_module.py
```

This will run the simulator with the parameters from the yaml files. 

### Import as package and run

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

Variables saved to disk can be accessed after `import macaqueretina as mr` and using `my_data = mr.load_data("filename")`

