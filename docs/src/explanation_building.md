
### Spatial layout of GC units

Visual angle (A) in degrees from previous studies ([Croner and Kaplan 1995](references.md#croner-1995), [Dacey and Petersen 1992](references.md#dacey-1992)) was approximated with relation ~4.4 deg/mm, or 1 degree / 0.229 mm. This works fine up to 20 deg ecc, but somewhat underestimates the distance thereafter. If more peripheral representations are necessary, the millimeters should be calculated by inverting the relation A = 0.1 + 4.21E + 0.038E^2 ([Dacey and Petersen 1992](references.md#dacey-1992), [Goodchild et al. 1996](references.md#goodchild-1996)). 

The density of many GC types is inversely proportional to their dendritic field coverage, suggesting constant coverage factor 
([Perry et al. 1984](references.md#perry-1984), [Wässle and Boycott 1991](references.md#wassle-1991)). Midget coverage factor is 1  ([Dacey, 1993](references.md#dacey-1993) for humans; [Wässle and Boycott 1991](references.md#wassle-1991), [Lee, 2010](references.md#lee-2010)). It is likely that coverage factor is 1 for all our unit types, which is also in line with [Doi, 2012](references.md#doi-2012) and [Field et al. 2010](references.md#field-2010). At the moment, we are not able to reach constant coverage factor of 1 due to technical limitations in the way we optimize the receptive field positions and overlap.  


### Spatial receptive field models

The difference-of-Gaussians `DOG` spatial receptive fields for the four unit types were modelled with separate center and surround. The DoG can have either center and surround fixed to same position and allowing the size and amplitude of the surround vary (`circular`, `ellipse_fixed`) or with independent center and surround (`ellipse_independent`). We use the fixed ellipse in all our demos.

Variational autoencoder `VAE` is a machine learning model which generates new spatial receptive field samples not restricted to ellipse form.


### Temporal receptive field models

The `fixed` temporal model comprises the sum of a faster positive and slower negative low-pass filters. Parameters are from the Chichilnisky data. 

Contrast gain control, or the `dynamic` model, is implemented according to [Victor 1987](references.md#victor-j-d). The parameters are from [Benardete and Kaplan 1999](references.md#benardete-1999) for parasol units and [Benardete and Kaplan 1997](references.md#benardete-1997) for midget units. We are sampling from Benardete Kaplan data assuming triangular distribution of the reported tables of statistics (original data points not shown). For a review of physiological mechanisms, see [Demb, 2008](references.md#demb-2008) and [Beaudoin, 2007](references.md#beaudoin-2007). 

The `subunit` temporal model is a combination of fast cone adaptation model followed by a center subunit nonlinearity model. The fast cone adaptation model is from [Clark et al. 2013](references.md#clark-2013) with parameters from [Angueyra, 2022](references.md#angueyra-2022). The center subunit nonlinearity is described in [Schwartz et al. 2012](references.md#schwartz-2012) and  [Turner and Rieke 2016](references.md#turner-2016) with parameters from [Turner et al. 2018](references.md#turner-2018).

For subunit model we assume constant bipolar to cone ratio as function of ecc. The bipolar cell type -dependent parameters are from  Boycott_1991_EurJNeurosci Table 1 which report the bipolar densities at 6-7 mm ecc.
Inputs to ganglion cells are coming from:  
 - OFF parasol: diffuse bipolars, DB2 and DB3 ([Jacoby, 2000](references.md#jacoby-2000))  
 - ON parasol:  diffuse bipolars, DB4 and DB5 ([Marshak, 2002](references.md#marshak-2002), [Boycott, 1991](references.md#boycott-1991))  
 - OFF midget: flat midget bipolars, FMB ([Wässle, 1994](references.md#wassle-1994), [Freeman, 2015](references.md#freeman-2015))  
 - ON midget: invaginating midget bipolars, IMB ([Wässle, 1994](references.md#wassle-1994)
)  
