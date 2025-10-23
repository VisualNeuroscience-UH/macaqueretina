This is code for building and using synthetic macaque monkey temporospatial ganglion cell (GC) receptive fields for four unit types (parasol & midget, ON & OFF). Name *unit* is used as distinction from biological *cell*.


### Relation between the simulator and experimental data

We had the opportunity to use spike triggered averaging GC receptive field data from Chichilnisky lab. The method and data are described in 
[Chichilnisky 2001](references.md#chichilnisky-2001), [Chichilnisky and Kalmar 2002](references.md#chichilnisky-2002), [Field et al. 2010](references.md#field-2010). Data was manually curated and bad cell indices were marked for omission before generating the spatial or temporal receptive field statistics. 

We use the receptive field and literature data statistics to build continuous functions. Next, we sample these functions to produce new units. For the `DOG` spatial and `fixed` temporal receptive fields, it is necessary to use multivariate statistics to get the correlations between parameters into account. In addition, when statistical distribution crossed 0-value in some cases it was necessary to implement hard limit to prevent switch of the unit ON-OFF polarity. 


