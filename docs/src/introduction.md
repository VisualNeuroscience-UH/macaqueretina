This is code for building and using synthetic macaque monkey temporospatial ganglion cell (GC) receptive fields for four unit types (parasol & midget, ON & OFF). Name *unit* is used as distinction from biological *cell*.

In introduction we introduce several valid parameter values in `code_style`

Constructing retina models, see the [Building](building.md) page.  
Making visual stimuli, see the [Stimuli](stimuli.md) page.  
Running a simulation, see the [Simulations](simulations.md) page.  
Visualizing the retina and simulation results, see the [Visualization](visualization.md) page.  


### Relation between the simulator and experimental data

We had the opportunity to use spike triggered averaging GC receptive field data from Chichilnisky lab. The method and data are described in Chichilnisky_2001_Network, Chichilnisky_2002_JNeurosci Field_2010_Nature. Data was manually curated and bad cell indices were marked for omission before generating the spatial or temporal receptive field statistics. 

For parameters whose statistics was available either as receptive field data or reported in literature we aim to build continuous functions from the statistics. Next, we sample these functions to produce new units. For the `DOG` spatial and `fixed` temporal receptive fields, it is necessary to use multivariate statistics to get the correlations between parameters into account. In addition, when statistical distribution crossed 0-value in some cases it was necessary to implement hard limit to prevent switch of the unit ON-OFF polarity. 



