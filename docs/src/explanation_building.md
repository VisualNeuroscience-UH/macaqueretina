
### Spatial layout of GC units

Visual angle (A) in degrees from previous studies (Croner and Kaplan, 1995; Dacey and Petersen, 1992) was approximated with relation ~4.4 deg/mm, or 1 degree / 0.229 mm. This works fine up to 20 deg ecc, but somewhat underestimates the distance thereafter. If more peripheral representations are necessary, the millimeters should be calculated by inverting the relation A = 0.1 + 4.21E + 0.038E^2 (Drasdo and Fowler, 1974; Dacey and Petersen, 1992, Goodchild et al. 1996). 

The density of many GC types is inversely proportional to their dendritic field coverage, suggesting constant coverage factor 
(Perry_1984_Neurosci, Wassle_1991_PhysRev). Midget coverage factor is 1  (Dacey_1993_JNeurosci for humans; Wassle_1991_PhysRev, 
Lee_2010_ProgRetEyeRes). It is likely that coverage factor is 1 for all our unit types, which is also in line with Doi_2012 JNeurosci, Field_2010_Nature. At the moment, we are not able to reach constant coverage factor of 1 due to technical limitations in the way we optimize the receptive field positions and overlap.  


### Spatial receptive field models

The difference-of-Gaussians `DOG` spatial receptive fields for the four unit types were modelled with separate center and surround. The DoG can have either center and surround fixed to same position and allowing the size and amplitude of the surround vary (`circular`, `ellipse_fixed`) or with independent center and surround (`ellipse_independent`). We use the fixed ellipse in all our demos.

Variational autoencoder `VAE` is a machine learning model which generates new spatial receptive field samples not restricted to ellipse form.


### Temporal receptive field models

The `fixed` temporal model comprises the sum of a faster positive and slower negative low-pass filters. Parameters are from the Chichilnisky data. 

Contrast gain control, or the `dynamic` model, is implemented according to Victor_1987_JPhysiol. The parameters are from Benardete_1999_VisNeurosci for parasol units and Benardete_1997_VisNeurosci_a for midget units. We are sampling from Benardete Kaplan data assuming triangular distribution of the reported tables of statistics (original data points not shown).
For a review of physiological mechanisms, see Demb_2008_JPhysiol and Beaudoin_2007_JNeurosci. 

The `subunit` temporal model is a combination of fast cone adaptation model followed by a center subunit nonlinearity model. The fast cone adaptation model is from Clark_2013_PLoSComputBiol with parameters from Angueyra_2022_JNeurosci. The center subunit nonlinearity is described in (Schwartz et al., 2012; Turner & Rieke, 2016) with parameters from (Turner et al., 2018).

For subunit model we assume constant bipolar to cone ratio as function of ecc. The bipolar cell type -dependent parameters are from  Boycott_1991_EurJNeurosci Table 1 which report the bipolar densities at 6-7 mm ecc.
Inputs to ganglion cells are coming from:  
 - OFF parasol: Diffuse Bipolars, DB2 and DB3 (Jacoby_2000_JCompNeurol, )  
 - ON parasol:  Diffuse Bipolars, DB4 and DB5 (Marshak_2002_VisNeurosci, Boycott_1991_EurJNeurosci)  
 - OFF midget: Flat Midget Bipolars, FMB (Wässle_1994_VisRes, Freeman_2015_eLife)  
 - ON midget: Invaginating Midget Bipolars, IMB (Wässle_1994_VisRes)  
