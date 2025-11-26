# Macaque retina simulator

This is a Python software to build a model of the primate retina and convert various visual stimuli to ganglion cell action potentials. 
The simulator is primarily for research, with the intention of providing biologically plausible spike trains for downstream visual cortex simulations.

The simulator can be run either directly or imported as a package.

When run directly, all parameters are read from all yaml files in the [your_repo_root]/macaqueretina/parameters folder.

When imported, all parameters are first read from the yaml files, but you have the option of changing the parameters at runtime.


## Short installation guide

We suggest using [Poetry](https://python-poetry.org/docs/main/) for creating and managing the project environment.

Navigate to your local macaqueretina git repository root and run `poetry install`.


## Documentation

See [ReadtheDocs](https://macaqueretina.readthedocs.io/en/latest/)  
For local documentation, install the package with `poetry install --with dev` and run `mkdocs serve` at repo root after installation.


## How to cite this project

Vanni S, Vedele F, Hokkanen H. Macaque retina simulator (in preparation)


## Contributing

Henri Hokkanen  
Simo Vanni  
Francescangelo Vedele  
