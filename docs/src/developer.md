### Overview

We keep a modular code structure, to be able to add new features at a later phase. 

You can add new parameters to existing yaml files at the root level, which become Configuration objects under macaquretina.config as such. In case you wish for validation of such new parameters, you need to build them into parameter_validation.py within the pydantic framework.

The parameter_validation adds Brian2 units to some variables to avoid errors in unit magnitudes and conversions. We import the units as `import brian2.units as b2u` and the units can be applied as e.g. `x = 1.0 * b2u.second`. You can strip the unit by `x / b2u.second` e.g. for pyplot.

You can make any class pretty printable by inheriting from PrintableMixin in project_utilities_module. It shows you attributes by type and methods.

The retina construction and simulation become memory-heavy with larger retina patches. 

The numpy seed is a legacy function but still used in our code. The seed will be overridden in fit_module, _fit_with_retry method, in case the fit fails.

### Component diagram
In docs we have the software component diagram which helps navigating the dependency structure. When calling `import macaqueretina`, code at macaqueretina/__init__.py is executed. Because it imports the other key submodules, the submodule specific __init__.py files become executed. These modules call project_manager_module for lazy imports which keeps the software reactive. During the lazy import of a submodule, project manager actually imports the classes and inject dependencies across classes when necessary. Thus, all dependencies crossing subfolder boundaries are handled inside the project_manager_module which supports maintenance of the code.    


### Abbreviations in the code
ana : analysis  
cen : center  
col : column  
dd : dendritic diameter
ecc : eccentricity
exp : experimental  
full : full absolute path  
gc : ganglion cell  
gen : generated  
lit : literature  
mtx : matrix  
param : parameter  
sur : surround  
viz : visualization  

Custom suffixes:
_df : pandas dataframe  
_mm : millimeter  
_np : numpy array  
_pix : pixel  
_t : tensor  
_um : micrometer  