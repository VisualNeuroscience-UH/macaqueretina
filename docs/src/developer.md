### Overview

We keep modular code structure, to be able to add new features at later phase. 

You can add new parameters to existing dictionaries, and these will be passed into config object as such. In case you wish for validation of the new parameters, ie when somebody else will be using the software, you need to build it into the parameter_validation.py in pydantic framework.

The parameter_validation adds Brian2 units to some variables to avoid errors in unit magnitudes and conversions. We import the units as `import brian2.units as b2u` and the units can be applied as e.g. `x = 1.0 * b2u.second`. You can strip the unit by `x / b2u.second` e.g. for pyplot.

The retina buildup and simulation become memory-heavy with larger retina patches. 


### Abbreviations in the code
ana : analysis  
cen : center  
col : column  
dd : dendritic diameter  
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