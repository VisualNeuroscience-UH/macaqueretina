In this page, keywords which are valid parameter values, are marked as `this`

## Relation between the simulator and experimental data

This is code for building and using synthetic macaque monkey temporospatial ganglion cell (GC) receptive fields for four unit types (parasol & midget, ON & OFF). Name *unit* is used as distinction from biological *cell*.

The original spike triggered averaging RGC data was borrowed from Chichilnisky lab. The method and data are described in Chichilnisky_2001_Network, Chichilnisky_2002_JNeurosci Field_2010_Nature. Data was manually curated and bad cell indices were marked for omission before generating the spatial receptive field statistics. 

For many parameters, we extracted statistics either from the Chichilnisky receptive field data or from literature and build continuous functions from the statistics. Finally we sample these functions to produce new units. For the fixed spatial and temporal receptive fields, it is necessary to use multivariate statistics to get the correlations between parameters into account.


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


### Spike generation

After summing the generator potential from the different temporal models, we apply firing rate gain (spikes/(second * unit contrast)), parameter *A* in the Victor 1987 model, values from Benardete_1999_VisNeurosci for parasol units and Benardete_1997_VisNeurosci_a for midget units. This gives us a distribution of gains. Next this value is calibrated across the different temporal and spatial model combinations to give a threshold firing rate for 3.5% contrast (parasol units) or 11.4% contrast (midget units). We run separately luminance contrast calibration (according to Lee et al., 1989) and drifting grating calibration (according to (Derrington & Lennie, 1984).

Next the signal goes through a threshold linear (rectified linear, ReLu) nonlinearity.

Finally the spike generation comes from either a `poisson` process or from a `refractory` model. For refractory model the recovery function is from Berry_1998_JNeurosci and absolute and relative refractory parameters were estimated from Uzzell_2004_JNeurophysiol,
Fig 7B, bottom row, inset. Currently we have only one set of fixed refracory model parameters, shared across all GC units.


### Noise model

Spontaneous firing rates in midgets and parasol units ranges from 5 to over 30 Hz (Appleby & Manookin, 2019; Sinha et al., 2017); we fixed the cone noise to induce ~25Hz baseline firing rates for the ON parasol and midget units, and ~5Hz for the OFF parasol and midget units (Raunak Sinha, personal communication). 

The noise type `shared` gets cone spectrum from Angueyra_2013_NatNeurosci, and this drives the ganglion cell background firing rates. You can use the same cone noise for all ganglion cell types, hypothetically resulting in shared downstream signal for a shared stimulus. The noise type `independent` fits Gaussian distribution to the GC noise firing rates from the shared noise and next samples indipendent noise for each GC unit from this distribution.


### Visual stimulus generation

Stimulus video will be saved on output_folder in mp4 format (viewing) and hdf5 format (reloading)
Valid stimulus_options include (overriding visual_stimulus_module.VideoBaseClass):

image_width: in pixels  
image_height: in pixels  
pix_per_deg: pixels per degree  
fps: frames per second  
duration_seconds: stimulus duration  
baseline_start_seconds: midgray at the beginning  
baseline_end_seconds: midgray at the end  
pattern:  
    `sine_grating`; `square_grating`; `colored_temporal_noise`; `white_gaussian_noise`;
    `natural_images`; `natural_video`; `temporal_sine_pattern`; `temporal_square_pattern`;
    `temporal_chirp_pattern`; `contrast_chirp_pattern`; `spatially_uniform_binary_noise`

stimulus_form: `circular`; `rectangular`; `annulus`

For stimulus_form annulus, additional arguments are:
size_inner: in degrees
size_outer: in degrees

stimulus_position: in degrees, (0,0) is the center of the stimulus on the image pixel grid.
stimulus_size: In degrees. Radius for circle and annulus, half-width for rectangle.
background: `mean`, `intensity_min`, `intensity_max` or value. This is the frame around stimulus in time and space, incl pre- and ppost-stimulus baselines.
contrast: between 0 and 1
mean: mean stimulus intensity in cd/m2

If intensity (min, max) is defined, it overrides contrast and mean becomes baseline.

For sine_grating and square_grating, additional arguments are:
temporal_frequency: in Hz
spatial_frequency: in cycles per degree
orientation: in degrees

For all temporal and spatial gratings, additional argument is
phase_shift: between 0 and 2pi

For spatially_uniform_binary_noise, additional argument is
on_proportion: between 0 and 1, proportion of on-stimulus, default 0.5
on_time: in seconds, duration of on-stimulus, default 0.1 seconds, floor division to full frames
direction: 'increment' or 'decrement'
stimulus_video_name: name of the stimulus video

With assuming rgb voltage = cd/m2, and average pupil diameter of 3 mm, the mean voltage of 128 in background
would mean ~ 905 Trolands. Td = lum * pi * (diam/2)^2, resulting in 128 cd/m2 = 128 * pi * (3/2)^2 ~ 905 Td.

NOTE: GC receptive fields outside the stimulus causes error. In other words, stimulus must be bigger than retina.