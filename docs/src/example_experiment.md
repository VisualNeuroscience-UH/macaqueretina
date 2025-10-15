## Build and run experiment
This example shows how to build, run, analyze and visualize the results of an experiment.


```
import matplotlib.pyplot as plt
import macaqueretina as mr
```

You need to build retina first
```python
mr.construct_retina()
```


### Short contrast response function experiment
The `exp_variables` are the variables to be changed in the experiment. 
See visual_stimulus_parameters.yaml, safe up to two variables
```python
exp_variables = ["contrast"]
mr.config.experiment_parameters = {
    "exp_variables": exp_variables,
    "min_max_values": [[0, 1.0]],
    "n_steps": [5], 
    "logarithmic": [False],
    "n_sweeps": 1,
    "distributions": {"uniform": None},
}
filename = mr.experiment.build_and_run(build_without_run=False)
```
You will need the `filename` downstream

#### Analyze short experiment
```python
my_analysis_options = {
    "exp_variables": exp_variables,
    "t_start_ana": 0.5, # 0.5 s baseline as default
    "t_end_ana": 1.5,
}
mr.analysis.analyze_experiment(filename, my_analysis_options)
```

#### Visualize results
```python
mr.viz.spike_raster_response(filename, sweeps_to_show=[0], savefigname=None)
mr.viz.fr_response(filename, exp_variables, xlog=False, savefigname=None)
mr.viz.F1F2_unit_response(filename, exp_variables, xlog=False, savefigname=None)
plt.show()
```

### Long experiment
Let's change the stimulus pattern and duration  
```python
mr.config.visual_stimulus_parameters.pattern = "sine_grating"
mr.config.visual_stimulus_parameters.duration_seconds = 6.0
```

```python
exp_variables = ["contrast", "spatial_frequency"]
mr.config.experiment_parameters = {
    "exp_variables": exp_variables,
    "min_max_values": [[0, 0.6], [0.1, 15.0]]
    "n_steps": [10, 16]
    "logarithmic": [True, True]
    "n_sweeps": 1,
    "distributions": {"uniform": None},
}
filename = mr.experiment.build_and_run(build_without_run=False)
```

#### Analyze long experiment
```python
my_analysis_options = {
    "exp_variables": exp_variables,
    "t_start_ana": 0.5, # 0.5 s baseline as default
    "t_end_ana": 6.5,
}
mr.analysis.analyze_experiment(filename, my_analysis_options)
```

#### Visualize results
```python
mr.viz.spike_raster_response(filename, sweeps_to_show=[0], savefigname=None)
mr.viz.fr_response(filename, exp_variables, xlog=False, savefigname=None)
mr.viz.F1F2_unit_response(filename, exp_variables, xlog=False, savefigname=None)
plt.show()
```

