Visualization can be accessed through mr.viz.[method_name].

During runtime, some variables are temporarily passed into a dictionary `project_data` for visualization at the end of the process. These are not saved to disk, and visualizing these variables requires that the corresponding process is run before calling visualization.

See docs/examples/example_visualize.py.

Try dir(mr.viz) or see macaqueretina/viz/viz_module.py for other implementations.