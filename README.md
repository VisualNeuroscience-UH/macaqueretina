# MacaqueRetina

Python software to build a model of the primate retina and convert various visual stimuli to ganglion cell action potentials. This is software for research, with the intention of providing biologically plausible spike trains for downstream visual cortex simulations.

The simulator can be run either directly or imported as a package.

When run directly, all parameters are read from all yaml files in the [your_repo_root]/macaqueretina/parameters folder.

When imported, all parameters are first read from the yaml files, but you have the option of changing the parameters at runtime.


## Short installation guide

We suggest using [Poetry](https://python-poetry.org/docs/main/) for creating and managing the project environment.

Navigate to your local MacaqueRetina git repository root and run `poetry install`.


## Documentation

For local documentation, run `mkdocs serve` at repo root after installation.


## How to cite this project

Vanni S, Vedele F, Hokkanen H. Macaque Retina Simulator (in preparation)


## Contributing

Henri Hokkanen  
Simo Vanni  
Francescangelo Vedele  
