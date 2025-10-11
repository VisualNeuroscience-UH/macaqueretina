This is code for building and using synthetic macaque monkey temporospatial ganglion cell (GC) receptive fields for four unit types (parasol & midget, ON & OFF). Name *unit* is used as distinction from biological *cell*.

In this introduction we introduce several valid parameter values as `code_style`

Constructing retina models, see the [Building](building.md) page.  
Making visual stimuli, see the [Stimuli](stimuli.md) page.  
Running a simulation, see the [Simulations](simulations.md) page.  

### Relation between the simulator and experimental data

The original spike triggered averaging RGC data was borrowed from Chichilnisky lab. The method and data are described in Chichilnisky_2001_Network, Chichilnisky_2002_JNeurosci Field_2010_Nature. Data was manually curated and bad cell indices were marked for omission before generating the spatial receptive field statistics. 

For many parameters, we extracted statistics either from the Chichilnisky receptive field data or from literature and build continuous functions from the statistics. Finally we sample these functions to produce new units. For the fixed spatial and temporal receptive fields, it is necessary to use multivariate statistics to get the correlations between parameters into account.



