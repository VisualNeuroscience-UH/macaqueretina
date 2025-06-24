# Built-in
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

# Third-party
import brian2.units as b2u
import matplotlib.pyplot as plt

# Local
from macaqueretina.project.project_manager_module import ProjectManager

start_time = time.time()
warnings.simplefilter("ignore")

"""
This is code for building macaque retinal filters corresponding to midget and parasol unit responses
for temporal hemiretina. We keep modular code structure, to be able to add new features at later phase. Name
'unit' is used as distinction from biological cell.

Visual angle (A) in degrees from previous studies (Croner and Kaplan, 1995) was approximated with relation 5 deg/mm. 
This works fine up to 20 deg ecc, but underestimates the distance thereafter. If more peripheral representations are 
necessary, the millimeters should be calculated by inverting the relation A = 0.1 + 4.21E + 0.038E^2 (Drasdo and Fowler, 1974;
Dacey and Petersen, 1992). Current implementation uses one deg = 220um (Perry et al 1985). One mm retina is ~4.55 deg visual field.

We have extracted statistics of macaque ganglion cell receptive fields from literature and build continuous functions.

The density of many cell types is inversely proportional to dendritic field coverage, suggesting constant coverage factor 
(Perry_1984_Neurosci, Wassle_1991_PhysRev). Midget coverage factor is 1  (Dacey_1993_JNeurosci for humans; Wassle_1991_PhysRev, 
Lee_2010_ProgRetEyeRes). It is likely that coverage factor is 1 for midget and parasol 
ON- and OFF-center cells each, which is also in line with Doi_2012 JNeurosci, Field_2010_Nature

The spatiotemporal receptive fields for the four unit types (parasol & midget, ON & OFF) were modelled with double ellipsoid 
difference-of-Gaussians model. The original spike triggered averaging RGC data in courtesy of Chichilnisky lab. The method is 
described in Chichilnisky_2001_Network, Chichilnisky_2002_JNeurosci Field_2010_Nature.

Chichilnisky_2002_JNeurosci states that L-ON (parasol) cells have on average 21% larger RFs than L-OFF cells. He also shows that 
OFF cells have more nonlinear response to input, which is not implemented currently (a no-brainer to implement if necessary).

NOTE: bad cell indices and metadata hard coded from Chichilnisky apricot data at fit_module ApricotData class. 
For another data set change metadata, visualize fits and change the bad cells.
NOTE: Visual stimulus video default options are hard coded at visual_stimulus_module.VideoBaseClass class.
NOTE: If eccentricity stays under 20 deg, dendritic diameter data fitted up to 25 deg only (better fit close to fovea)

-center-surround response ratio (in vivo, anesthetized, recorded from LGN; Croner_1995_VisRes) PC: ; MC: ;
-Michelson contrast definition for sinusoidal gratings (Croner_1995_VisRes).
-optical quality probably poses no major limit to behaviorally measured spatial vision (Williams_1981_IOVS).
-spatial contrast sensitivity nonlinearity in the center subunits is omitted. This might reduce sensitivity to natural scenes Turner_2018_eLife.

-quality control: compare to Watanabe_1989_JCompNeurol
    -dendritic diameter scatter is on average (lower,upper quartile) 21.3% of the median diameter in the local area

    Parasol dendritic field diameter: temporal retina 51.8 microm + ecc(mm) * 20.6 microm/mm, nasal retina; 115.5 microm + ecc(mm) * 6.97 microm/mm

spatial_model_type DOG : Fit ellipse to center and surround
spatial_model_type : VAE : Variational autoencoder. The model reconstructs the full receptive field and generates new samples from the latent space.
    
Contrast gain control (CGC) is implemented according to Victor_1987_JPhysiol using numerical integration in discretized temporal domain.
The unit parameters are drawn from Benardete_1999_VisNeurosci for parasol units and Benardete_1997_VisNeurosci_a for midget units.
We are sampling from Benardete Kaplan data assuming triangular distribution of the reported tables of statistics (original data points not shown).
For a review of physiological mechanisms, see Demb_2008_JPhysiol and Beaudoin_2007_JNeurosci.

Parasol ON firing rate for spatially uniform monochromatic light of wavelength 561 (530, 430) nm and intensity 4300
(4200, 2400) photons/microm2/s, incident on the photoreceptors: Across all preparations examined firing rates varied from about 5
to 20 Hz. From Shlens_2009_JNeurosci/Methods.

The firing rate gain (spikes/(second * unit contrast)), parameter "A" in the Victor model, comes from Benardete_1999_VisNeurosci for parasol units and Benardete_1997_VisNeurosci_a for midget units.
The max amplitude the the center spatial DoG model in Victor_1987_JPhysiol is 1.0. 

The cone photoreceptor sampling is approximated as achromatic (single) compressive cone response(Baylor_1987_JPhysiol).

Most of the functions fitted have no theoretical significance but provide empirical expressions for interpolation.
"""

"""
Technical notes:
Use keyword substring "file" in filenames, and "folder" or "path" in folder names to assert that they are turned into pathlib objects. 
Path structure is assumed to be model_root_path/project/experiment/output_folder

Abbreviations:
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
"""

"""
Main paths
"""
# Update this to your model root path
model_root_path = "/opt3/projects"

"""
Project name
"""
project = "macaqueretina"


"""
Current experiment. Use distinct folders for distinct stimuli.
"""
experiment = "amplitude_calibration_testi"


"""
Input context
Existing retina model, any general files
"""
input_folder = "../in"  # input figs, videos, models

"""
Data context for output. 
"""
output_folder = "testi"
# output_folder_stem = f"{gc_type}_{response_type}_{spatial_model_type}_{temporal_model_type}_{contrast}_gain{signal_gain}_temporal_frequency"


"""
Stimulus context
Stimulus images and videos
"""

stimulus_folder = f"stim_{output_folder}"

testrun = False  # Stops execution before heavy calculations

"""
Remove random variations by setting the numpy random seed
"""
numpy_seed = 42  # random.randint(0, 1000000)  # 42

"""
Computing device
For small retinas cpu is faster. Use cpu if you do not have cuda (and NVidia GPU).
"""
# After training a VAE model with a device, the model must be loaded to same device. Pytorch quirk.
device = "cpu"  # "cpu" or "cuda"

"""
### Housekeeping ###. Do not comment out.
"""
project_conf_module_file_path = Path(__file__).resolve()
git_repo_root_path = project_conf_module_file_path.parent.parent
model_root_path = Path(model_root_path)
# Test if the paths exist, fallback to current working directory if not
if not model_root_path.exists():
    model_root_path = Path.cwd()

    print(
        f"\033[91m \nModel root path {model_root_path} does not exist. \nUpdate project/project_conf_module.py model_root_path. \nUsing current working directory. \033[0m\n"
    )
path = Path.joinpath(model_root_path, Path(project), experiment)

# When training or tuning generative VAE model, multiple hyperparameters are set at the RetinaVAE class.
# For training, see __init__ method. For tuning, the __init__ contains search space and
# _set_ray_tuner contains the starting point.

# For "load_model" training_mode, the model is loaded from model_file_name at output_folder (primary)
# or input_folder. The correct model name (including time stamp) must be given in the model_file_name.

# VAE RF is generated in experimental data space originating from macaque peripheral retina.
# VAE RF sizes need to be scaled according to eccentricity.
# This scaling is based on dendritic field diameter (DoG model diameter) comparison between
# experimental data fit and literature data on dendritic field diameter vs eccentricity.
# When the spatial model is VAE, the DoG model is fitted twice. First, the experimental data is fitted
# to get the RF scaling. Second, the scaled VAE RF is fitted to get a description of the final RF.
# The experimental (first) dog_model_type fit for VAE is automatically changed to to ellipse_fixed. This way
# the scaling is not dependent on the selected dog model. The final RF fit is what you call for in dog_model_type.

# The model_file_name must must be of correct gc_type and response_type.

# Note: DOG model ellipse independent does not correlate the center and surround parameters. Thus they are independent, which
# is not the case in the VAE model, and not very physiological.

gc_type = "parasol"  # "parasol", "midget"
response_type = "on"  # "on", "off"
spatial_model_type = "DOG"  # "DOG", "VAE"
temporal_model_type = "fixed"  # "fixed", "dynamic", "subunit"

print(f"{gc_type=}")
print(f"{response_type=}")


signal_gain = {
    "threshold": 10,  # 10 Hz
    "parasol": {
        "on": {
            "DOG": {"fixed": 1.51, "dynamic": 1.32, "subunit": 20.36},
            "VAE": {"fixed": 1.06, "dynamic": 1.33, "subunit": 19.17},
        },
        "off": {
            "DOG": {"fixed": 1.74, "dynamic": 1.44, "subunit": 37.03},
            "VAE": {"fixed": 1.27, "dynamic": 1.60, "subunit": 37.31},
        },
    },
    "midget": {
        "on": {
            "DOG": {"fixed": 5.33, "dynamic": 4.50, "subunit": 37.33},
            "VAE": {"fixed": 3.73, "dynamic": 4.97, "subunit": 37.39},
        },
        "off": {
            "DOG": {"fixed": 6.48, "dynamic": 4.58, "subunit": 74.98},
            "VAE": {"fixed": 3.76, "dynamic": 4.77, "subunit": 74.48},
        },
    },
}

# These values are used for building a new retina.
# Rebuilding is avoided by serializing and hashing these into save names.
# Less mutable values, not included into the hash, are set in retina_parameters_append below.
retina_parameters = {
    # Main model type variants
    "gc_type": gc_type,
    "response_type": response_type,
    "spatial_model_type": spatial_model_type,  # "DOG" for difference-of-Gaussians model, "VAE" for variational autoencoder. DOG data is used for VAE scaling.
    "dog_model_type": "ellipse_fixed",  # 'ellipse_fixed', 'ellipse_independent' or 'circular'
    "temporal_model_type": temporal_model_type,
    "signal_gain": (
        signal_gain[gc_type][response_type][spatial_model_type][temporal_model_type]
    ),
    # Auxiliary model type variants included into the hash
    "ecc_limits_deg": [4.5, 5.5],  # eccentricity in degrees MIDGET
    "pol_limits_deg": [-1.5, 1.5],  # polar angle in degrees
    # "ecc_limits_deg": [4.0, 6.0],  # eccentricity in degrees PARASOL
    # "pol_limits_deg": [-6, 6],  # polar angle in degrees
    # "ecc_limits_deg": [3.5, 6.5],  # eccentricity in degrees PARASOL
    # "pol_limits_deg": [-15, 15],  # polar angle in degrees
    "model_density": 1.0,  # 1.0 for 100% of the literature density of ganglion cells
    "retina_center": 5.0 + 0j,  # degrees, this is stimulus_position (0, 0)
    "force_retina_build": True,  # False or True. If True, rebuilds retina even if the hash matches
    "training_mode": "load_model",  # "load_model", "train_model" or "tune_model". Applies to VAE only
    "model_file_name": None,  # None for most recent or "model_[GC TYPE]_[RESPONSE TYPE]_[DEVICE]_[TIME_STAMP].pt" at input_folder. Applies to VAE "load_model" only
    "ray_tune_trial_id": None,  # Trial_id for tune, None for loading single run after "train_model". Applies to VAE "load_model" only
}

"""
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
    'sine_grating'; 'square_grating'; 'colored_temporal_noise'; 'white_gaussian_noise';
    'natural_images'; 'natural_video'; 'temporal_sine_pattern'; 'temporal_square_pattern';
    'temporal_chirp_pattern'; contrast_chirp_pattern; 'spatially_uniform_binary_noise'
stimulus_form: 'circular'; 'rectangular'; 'annulus'

For stimulus_form annulus, additional arguments are:
size_inner: in degrees
size_outer: in degrees

stimulus_position: in degrees, (0,0) is the center.
stimulus_size: In degrees. Radius for circle and annulus, half-width for rectangle. 0 for full image.
background: "mean", "intensity_min", "intensity_max" or value. The "frame", around stimulus in time and space, incl baseline times
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

VAE rf have different resolution from original RF data, if the estimated eccentricity is different from the original data.
VAE rf have different amplitude from original RF data, because the VAE model operates with values between 0 and 1. Later
the median is removed to get the zero level to approximately match the original data. 

Note that in experiments (below), tuple values are captured for varying each tuple value separately.
"""
# TÄHÄN JÄIT: REFRACTORY DYSSYNCRHRONY. DYNAMIC FR > MUUT. -- TÄSSÄKÖ SYY APPARENTTIIN DYSSYNKRONIAAN?
# GENERAATTORIPOTENTIAALI NÄYTTI NOUSEVAN HITAASTI AINAKIN MUISSA.
visual_stimulus_parameters = {
    "pattern": "sine_grating",  # One of the StimulusPatterns
    "image_width": 240,  # 752 for nature1.avi
    "image_height": 240,  # 432 for nature1.avi
    "pix_per_deg": 60,
    "dtype_name": "float16",  # low contrast needs "float16", for performance, use "uint8",
    "fps": 300,  # >=200 for good dynamic model integration
    "duration_seconds": 0.5,  # actual frames = floor(duration_seconds * fps)
    "baseline_start_seconds": 0.1,  # Total duration is duration + both baselines
    "baseline_end_seconds": 0.5,
    "stimulus_form": "circular",
    "size_inner": 0.1,  # deg, applies to annulus only
    "size_outer": 1,  # deg, applies to annulus only
    "stimulus_position": (0.0, 0.0),  # relative to stimuls center in retina
    "stimulus_size": 1.5,  # 2.2,  # deg, radius for circle, sidelen/2 for rectangle.
    "temporal_frequency": 0,  # 0.01,  # 4.0,  # 40,  # Hz
    "temporal_frequency_range": (0.5, 50),  # Hz, applies to temporal chirp only
    "spatial_frequency": 2,  # cpd
    "orientation": 90,  # degrees
    "phase_shift": 0,  # math.pi + 0.1,  # radians
    "stimulus_video_name": f"{stimulus_folder}.mp4",
    # "contrast": 0.035,  # mean +- contrast * mean
    "contrast": 1.0,  # mean +- contrast * mean
    "mean": 128,  # Mean luminance in  cd/m2 and adaptation level
    # "intensity": (0, 255),  # intensity overrides contrast and mean if not commented out
    "background": "mean",  # "mean", "intensity_min", "intensity_max" or value.
    "ND_filter": 0.0,  # 0.0, log10 neutral density filter factor, can be negative
}

# For external video and image input. See visual_stimulus_module.VideoBaseClass for more options.
stimulus_metadata_parameters = {
    "stimulus_file": "testi.jpg",  # nature1.avi, testi.jpg
    "pix_per_deg": 30,  # VanHateren_1998_ProcRSocLondB 2 arcmin per pixel
    "apply_cone_filter": False,
    "fps": 25,
}

# Each gc response file contain n_sweeps with independent cone noise and spike generator.
n_files = 1

# Running multiple trials on multiple units is not supported
run_parameters = {
    "n_sweeps": 1,  # For each of the response files
    "spike_generator_model": "poisson",  # poisson or refractory
    "save_data": True,
    "gc_response_filenames": [f"gc_response_{x:02}" for x in range(n_files)],
    "simulation_dt": 0.0001,  # in sec 0.001 = 1 ms
    # ["spikes", "cone_noise", "cone_bipo_gen_fir"],
    "save_variables": ["spikes", "cone_noise"],
}


############################
###  Semi-constant variables
############################

# Cone noise gain induces ~25Hz spontaneous firing rates for the ON parasol and
# midget gc units, and ~5Hz for the OFF parasol and midget units. (Raunak Sinha,
# personal communication).
noise_gain = {
    "on": {
        "parasol": 16.3,  # 16.3
        "midget": 5.4,  # 5.4
    },
    "off": {
        "parasol": 3.8,  # 3.8
        "midget": 1.5,  # 1.5
    },
}

# Options linear, quadratic, cubic, powerlaw.
# For midget < 20 deg, use quadratic; for parasol use powerlaw
dd_regr_model = {
    "parasol": "powerlaw",
    "midget": "quadratic",
}

retina_parameters_append = {
    "noise_type": "shared",  # "shared" or "independent"
    "noise_gain": noise_gain[response_type][gc_type],  # 0 for no noise
    "fixed_mask_threshold": 0.1,  # 0.1,  # Applies to rf volume normalization. This value is also the only surround mask threshold.
    "center_mask_threshold": 0.1,  # 0.1,  Limits rf center extent to values above this proportion of the peak values after volume normalization.
    "apricot_data_noise_mask": 0.1,  # 0.1,  # 0.1 for +-10% from abs max removed, 0 for no noise removal. Applies to DOG spatial model statistics.
    "fit_statistics": "multivariate",  # "univariate" or "multivariate" Applies only to DOG spatial and fixed temporal model statistics.
    "dd_regr_model": dd_regr_model[gc_type],
    "ecc_limit_for_dd_fit": 20,  # 20,  # degrees, math.inf for no limit
}

# apricot data is the original Chichilnisky data, and the metadata is used for filtering and fitting
apricot_metadata_parameters = {
    "data_microm_per_pix": 60,
    "data_spatialfilter_height": 13,
    "data_spatialfilter_width": 13,
    "data_fps": 30,
    "data_temporalfilter_samples": 15,
    "apricot_data_folder": git_repo_root_path.joinpath(r"retina/apricot_data"),
    "apricot_statistics_folder": git_repo_root_path.joinpath(
        r"retina/apricot_statistics"
    ),
    "mask_noise": retina_parameters_append["apricot_data_noise_mask"],
}

# Proportion from all ganglion cells. Density of all ganglion cells is given later as a function of ecc from literature.
proportion_of_parasol_gc_type = 0.08
proportion_of_midget_gc_type = 0.64

# Proportion of ON and OFF response type units, assuming ON rf diameter = 1.2 x OFF rf diameter, and
# coverage factor =1; Chichilnisky_2002_JNeurosci
proportion_of_ON_response_type = 0.40
proportion_of_OFF_response_type = 0.60

# Perry_1985_VisRes; 223 um/deg in the fovea, 169 um/deg at 90 deg ecc.
# With this relationship one mm retina is ~4.55 deg visual field.
# Constant (linear) approximation of quadratic formula from
# Goodchild_1996_JCompNeurol 0.038 * x**2 + 4.21 * x + 0.1
# gives 229 um/degree. The R2 (constant vs quadratic) is
# > 0.999 for ecc 0.1 - 4 mm, and > 0.97 for ecc 0.1 - 20 mm.
deg_per_mm = 1 / 0.229

optical_aberration = 2 / 60  # deg , 2 arcmin, Navarro 1993 JOSAA

# Parameters for cortical - visual coordinate transformation.
# a for macaques between 0.3 - 0.9, Schwartz 1994 citing Wilson et al 1990 "The perception of form"
# in Visual perception: The neurophysiological foundations, Academic Press
# k has been pretty open.
# However, if we relate 1/M = (a/k) + (1/k) * E and M = (1/0.077) + (1/(0.082 * E)), we get
# Andrew James, personal communication: k=1/.082, a=.077/.082
visual2cortical_params = {
    "a": 0.077 / 0.082,  # ~0.94
    "k": 1 / 0.082,  # ~12.2
}

# Compressing cone nonlinearity. Parameters are manually scaled to give dynamic cone ouput.
# Equation, data from Baylor_1987_JPhysiol
# Cone noise parameters from Angueyra_2013_NatNeurosci and Ala-laurila_2011_NatNeurosci

cone_general_parameters = {
    "rm": 25,  # pA
    "k": 2.77e-4,  # at 500 nm
    "sensitivity_min": 5e2,
    "sensitivity_max": 2e4,
    "cone2gc_midget": 9,  # um, 1 SD of Gaussian
    "cone2gc_parasol": 27,  # um 27
    "cone2gc_cutoff_SD": 1,  # 3 SD is 99.7% of Gaussian
    "cone2bipo_cutoff_SD": 1,
    "cone_noise_wc": [14, 160],  # lorenzian freqs, Angueyra_2013_NatNeurosci Fig1
}

# Parameters from Angueyra_2022_JNeurosci, model according to Clark_2013_PLoSComputBiol
# Light intensity photons/microm^2/second
cone_signal_parameters = {
    "unit": "pA",
    "A_pupil": 9.0,  # * b2u.mm2,  # mm^2
    "lambda_nm": 555,  # nm 555 monkey Clark models: DN 650
    "input_gain": 1.0,  # unitless
    "r_dark": -136 * b2u.pA,  # dark current
    "max_response": 116.8 * b2u.pA,  # "pA", measured for a strong flash
    # Angueyra: unitless; Clark: mV * microm^2 * ms / photon
    "alpha": 19.4 * b2u.pA * b2u.ms,
    "beta": 0.36 * b2u.ms,  # unitless
    "gamma": 0.448,  # unitless
    "tau_y": 4.49 * b2u.ms,
    "n_y": 4.33,  # unitless
    "tau_z": 166 * b2u.ms,
    "n_z": 1.0,  # unitless
    "tau_r": 4.78 * b2u.ms,
    "filter_limit_time": 3.0 * b2u.second,
}

bipolar_general_parameters = {
    "bipo2gc_div": 6,  # Divide GC dendritic diameter to get bipolar/subunit SD
    "bipo2gc_cutoff_SD": 2,  # Multiplier for above value
    "cone2bipo_cen_sd": 10,  # um, Turner_2018_eLife
    "cone2bipo_sur_sd": 150,
    # Surround / Center amplitude ratio.
    "bipo_sub_sur2cen": 0.9,  # 0.9 avoids cancellation for fullfield stimuli
}

# Recovery function from Berry_1998_JNeurosci, Uzzell_2004_JNeurophysiol
# abs and rel refractory estimated from Uzzell_2004_JNeurophysiol,
# Fig 7B, bottom row, inset. Parasol ON unit.
# Uzzell_2004_JNeurophysiol instataneous firing rates for binary noise go up to 320 Hz
refractory_parameters = {
    "abs_refractory": 1,
    "rel_refractory": 3,
    "p_exp": 4,
    "clip_start": 0,
    "clip_end": 100,
}

# If you see patterning, reduce unit_distance_threshold
# If you see clustering, increase increase diffusion_speed, unit_repulsion_stregth,
# or carefully unit_distance_threshold
# Algorithms are
# Voronoi-based Layout with Loyd's Relaxation
# "voronoi" (v) : better for big retinas, fast
# Force Based Layout Algorithm with Boundary Repulsion
# "force" (f) : better for small retinas, slow with cpu -- fast with cuda
# None : initial random placement. Nonvarying with fixed seed above.
# Good for testing, speed and memory.
gc_placement_parameters = {
    "algorithm": "force",  # "voronoi" or "force" or None
    "n_iterations": 30,  # v 20, f 5000
    "change_rate": 0.0005,  # f 0.001, v 0.5
    "unit_repulsion_stregth": 10,  # 10 f only
    "unit_distance_threshold": 0.2,  # f only, adjusted with ecc
    "diffusion_speed": 0.001,  # f only, adjusted with ecc
    "border_repulsion_stength": 0.2,  # f only
    "border_distance_threshold": 0.001,  # f only
    "show_placing_progress": False,  # True False
    "show_skip_steps": 1,  # v 1, f 100
}

cone_placement_parameters = {
    "algorithm": "force",  # "voronoi" or "force" or None
    "n_iterations": 15,  # v 20, f 300
    "change_rate": 0.0005,  # f 0.0005, v 0.5
    "unit_repulsion_stregth": 2,  # 10 f only
    "unit_distance_threshold": 0.25,  # f only, adjusted with ecc
    "diffusion_speed": 0.002,  # f only, adjusted with ecc
    "border_repulsion_stength": 5,  # f only
    "border_distance_threshold": 0.0001,  # f only
    "show_placing_progress": False,  # True False
    "show_skip_steps": 1,  # v 1, f 100
}

bipolar_placement_parameters = {
    "algorithm": "force",  # "voronoi" or "force" or None
    "n_iterations": 15,  # v 20, f 300
    "change_rate": 0.0005,  # f 0.0005, v 0.5
    "unit_repulsion_stregth": 2,  # 10 f only
    "unit_distance_threshold": 0.2,  # f only, adjusted with ecc
    "diffusion_speed": 0.005,  # f only, adjusted with ecc
    "border_repulsion_stength": 5,  # f only
    "border_distance_threshold": 0.0001,  # f only
    "show_placing_progress": False,  # True False
    "show_skip_steps": 5,
}


n_iterations = {
    "parasol": 400,
    "midget": 40,
}

# For VAE, this is enough to have good distribution between units.
receptive_field_repulsion_parameters = {
    "n_iterations": n_iterations[gc_type],  # 200 for parasol, 20 for midget
    "change_rate": 0.005,
    "cooling_rate": 0.999,  # each iteration change_rate = change_rate * cooling_rate
    "border_repulsion_stength": 5,
    "show_repulsion_progress": False,  # True False
    "show_only_unit": None,  # None or int for unit idx
    "show_skip_steps": 5,
    "savefigname": None,  # f"Layout_{gc_type}_{response_type}_FIT.eps",  # # string w/ type suffix or None, None for no save
}

# For subunit model we assume constant bipolar to cone ratio as function of ecc.
# Boycott_1991_EurJNeurosci Table 1 has bipolar densities at 6.5 mm ecc.
# Inputs to ganglion cells:
# -OFF parasol: Diffuse Bipolars, DB2 and DB3 (Jacoby_2000_JCompNeurol, )
# -ON parasol:  Diffuse Bipolars, DB4 and DB5 (Marshak_2002_VisNeurosci, Boycott_1991_EurJNeurosci)
# -OFF midget: Flat Midget Bipolars, FMB (Wässle_1994_VisRes, Freeman_2015_eLife)
# -ON midget: Invaginating Midget Bipolars, IMB (Wässle_1994_VisRes)

bipolar2gc_dict = {
    "midget": {"on": ["IMB"], "off": ["FMB"]},
    "parasol": {"on": ["DB4", "DB5"], "off": ["DB2", "DB3"]},
}

# Auxiliary retina parameters not included in the hash
retina_parameters_append_the_rest = {
    "mosaic_file": gc_type + "_" + response_type + "_mosaic.csv",
    "spatial_rfs_file": gc_type + "_" + response_type + "_spatial_rfs.npz",
    "proportion_of_parasol_gc_type": proportion_of_parasol_gc_type,
    "proportion_of_midget_gc_type": proportion_of_midget_gc_type,
    "proportion_of_ON_response_type": proportion_of_ON_response_type,
    "proportion_of_OFF_response_type": proportion_of_OFF_response_type,
    "deg_per_mm": deg_per_mm,
    "optical_aberration": optical_aberration,
    "cone_general_parameters": cone_general_parameters,
    "cone_signal_parameters": cone_signal_parameters,
    "bipolar_general_parameters": bipolar_general_parameters,
    "refractory_parameters": refractory_parameters,
    "gc_placement_parameters": gc_placement_parameters,
    "cone_placement_parameters": cone_placement_parameters,
    "bipolar_placement_parameters": bipolar_placement_parameters,
    "receptive_field_repulsion_parameters": receptive_field_repulsion_parameters,
    "visual2cortical_params": visual2cortical_params,
    "bipolar2gc_dict": bipolar2gc_dict,
}

retina_parameters_append.update(retina_parameters_append_the_rest)

literature_data_folder = git_repo_root_path.joinpath(r"retina/literature_data")

# Define digitized literature data files for gc density and dendritic diameters.
# Dendritic diameter is the diameter of the circle that has the
# same area as the dendritic field polygon, as defined in Watanabe 1989.
# Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
# Define literature data files for linear temporal models.
# Data from Benardete_1999_VisNeurosci and Benardete_1997_VisNeurosci
# Define literature data files for cone density.
# Data from Packer_1989_JCompNeurol

gc_density_1_datafile = "Wässle_1989_Nature_Fig2a_c.npz"
# foveal gc density within 3 deg is estimated from cone density, with a decaying exponential multiplier.
# See Wässle_1989_Nature Fig2a for details.
gc_density_1_datafile_scaling_data_and_function = [
    [0, 0.75, 3],
    [3.34, 2, 1],
    "single_exponential_func",
]
gc_density_2_datafile = "Wässle_1989_Nature_Fig3_gcData_c.npz"
gc_density_control_datafile = "Wässle_1989_Nature_Fig3_c.npz"

if retina_parameters["gc_type"] == "parasol":
    dendr_diam1_datafile = "Perry_1984_Neurosci_ParasolDendrDiam_Fig6A_c.npz"
    dendr_diam2_datafile = "Watanabe_1989_JCompNeurol_ParasolDendrDiam_Fig7_c.npz"
    dendr_diam3_datafile = "Goodchild_1996_JCompNeurol_Parasol_DendDiam_Fig2A_c.npz"
    temporal_BK_model_datafile = "Benardete_1999_VisNeurosci_parasol.csv"
    spatial_DoG_datafile = "Schottdorf_2021_JPhysiol_CenRadius_Fig4C_parasol_c.npz"
elif retina_parameters["gc_type"] == "midget":
    dendr_diam1_datafile = "Perry_1984_Neurosci_MidgetDendrDiam_Fig6B_c.npz"
    dendr_diam2_datafile = "Watanabe_1989_JCompNeurol_MidgetDendrDiam_Fig7_c.npz"
    dendr_diam3_datafile = "Goodchild_1996_JCompNeurol_Midget_DendDiam_Fig2B_c.npz"
    temporal_BK_model_datafile = "Benardete_1997_VisNeurosci_midget.csv"
    spatial_DoG_datafile = "Schottdorf_2021_JPhysiol_CenRadius_Fig4C_midget_c.npz"
dendr_diam_units = {
    "data1": ["mm", "um"],
    "data2": ["mm", "um"],
    "data3": ["deg", "um"],
}

cone_density1_datafile = "Packer_1989_JCompNeurol_ConeDensity_Fig6A_main_c.npz"
cone_density2_datafile = "Packer_1989_JCompNeurol_ConeDensity_Fig6A_insert_c.npz"
cone_noise_datafile = "Angueyra_2013_NatNeurosci_Fig6E_c.npz"
cone_response_datafile = "Angueyra_2013_NatNeurosci_Fig6B_c.npz"

bipolar_table_datafile = "Boycott_1991_EurJNeurosci_Table1.csv"

parasol_on_RI_values_datafile = "Turner_2018_eLife_Fig5C_ON_c.npz"
parasol_off_RI_values_datafile = "Turner_2018_eLife_Fig5C_OFF_c.npz"

temporal_pattern_datafile = "Angueyra_2022_JNeurosci_Fig2B_c.npz"

literature_data_files = {
    "gc_density_1_path": gc_density_1_datafile,
    "gc_density_2_path": gc_density_2_datafile,
    "gc_density_control_path": gc_density_control_datafile,
    "dendr_diam1_path": dendr_diam1_datafile,
    "dendr_diam2_path": dendr_diam2_datafile,
    "dendr_diam3_path": dendr_diam3_datafile,
    "temporal_BK_model_path": temporal_BK_model_datafile,
    "spatial_DoG_path": spatial_DoG_datafile,
    "cone_density1_path": cone_density1_datafile,
    "cone_density2_path": cone_density2_datafile,
    "cone_noise_path": cone_noise_datafile,
    "cone_response_path": cone_response_datafile,
    "bipolar_table_path": bipolar_table_datafile,
    "parasol_on_RI_values_path": parasol_on_RI_values_datafile,
    "parasol_off_RI_values_path": parasol_off_RI_values_datafile,
    "temporal_pattern_path": temporal_pattern_datafile,
}
literature_data_files = {
    key: f"{literature_data_folder}/{value}"
    for key, value in literature_data_files.items()
}
literature_data_files["dendr_diam_units"] = dendr_diam_units
literature_data_files["gc_density_1_scaling_data_and_function"] = (
    gc_density_1_datafile_scaling_data_and_function
)

profile = False

#############################################################################################################
#############################################################################################################
###                                      End of  module-level script                                      ###
#############################################################################################################
#############################################################################################################


if __name__ == "__main__":
    if profile is True:
        # Built-in
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()

    """
    Housekeeping. Do not comment out.

    All ProjectManager input parameters go to context. These are validated by the context object, and returned 
    to the class instance by set_context() method. They are available by class_instance.context.attribute. 
    """
    PM = ProjectManager(
        path=path,
        input_folder=input_folder,
        output_folder=output_folder,
        stimulus_folder=stimulus_folder,
        project=project,
        experiment=experiment,
        device=device,
        retina_parameters=retina_parameters,
        retina_parameters_append=retina_parameters_append,
        stimulus_metadata_parameters=stimulus_metadata_parameters,
        visual_stimulus_parameters=visual_stimulus_parameters,
        run_parameters=run_parameters,
        literature_data_files=literature_data_files,
        apricot_metadata_parameters=apricot_metadata_parameters,
        numpy_seed=numpy_seed,
        project_conf_module_file_path=project_conf_module_file_path,
    )

    # PM.countlines(Path("macaqueretina"))
    # PM.countlines(Path("tests"))

    # ###################################
    # ###  Show spikes from gz files  ###
    # ###################################

    # filename_parents = PM.context.output_folder
    # filename_offspring = f"Response_{gc_type}_{response_type}_tf6.gz"
    # filename = Path(filename_parents).joinpath(filename_offspring)
    # PM.viz.show_spikes_from_gz_file(
    #     filename=filename,
    #     savefigname=Path(filename).parent.name
    #     + "_puhti_12s.png",  # None for no save, or string with type suffix
    # )

    ############################
    ###  Measure model gain  ###
    ############################

    # PM.viz.show_gain_calibration(
    #     signal_gain["threshold"],
    #     f"parasol_on_DOG_subunit_c0p035_g*_temporal_frequency",
    #     # f"{gc_type}_{response_type}_{spatial_model_type}_{temporal_model_type}_c{contrast_str}_g*_temporal_frequency",
    #     signal_gain=10,
    #     savefigname=None,
    # )

    ###########################################
    ##   Luminance and Photoisomerizations   ##
    ###########################################

    # I_cone = 4000  # photoisomerizations per second per cone

    # luminance = PM.cones.get_luminance_from_photoisomerizations(I_cone)
    # print(f"{luminance:.2f} cd/m2")

    # luminance = 128  # Luminance in cd/m2

    # I_cone = PM.cones.get_photoisomerizations_from_luminance(luminance)
    # print(f"{I_cone:.2f} photoisomerizations per second per cone")

    # ##########################################
    # #   Sample figure data from literature  ##
    # ##########################################

    # # # If possible, sample only temporal hemiretina
    # # Local
    # from macaqueretina.project.project_utilities_module import DataSampler

    # filename = "Lee_1989_JPhysiol_Fig3B.jpg"
    # filename_full = git_repo_root_path.joinpath(r"retina/literature_data", filename)
    # # # Fig lowest and highest tick values in the image, use these as calibration points
    # min_X, max_X, min_Y, max_Y = (1.0, 40.0, 1.0, 50.0)  # Wässle 2a
    # ds = DataSampler(filename_full, min_X, max_X, min_Y, max_Y, logX=True, logY=True)
    # ds.collect_and_save_points()
    # ds.quality_control(restore=True)

    #################################
    #################################
    ###        Build retina       ###
    #################################
    #################################

    """
    Build and test your retina here, one gc type at a time. 
    """
    PM.construct_retina.build_retina_client()  # Main method for building the retina

    # The following visualizations are dependent on the ConstructRetina instance.
    # Thus, they are called after the retina is built.

    # For DOG and VAE
    # PM.viz.show_cones_linked_to_bipolars(n_samples=4, savefigname=None)
    # PM.viz.show_bipolars_linked_to_gc(gc_list=[10, 17], savefigname=None)
    # PM.viz.show_bipolars_linked_to_gc(n_samples=4, savefigname=None)
    # PM.viz.show_cones_linked_to_gc(gc_list=[10, 17], savefigname=None)
    # PM.viz.show_cones_linked_to_gc(n_samples=4, savefigname=None)
    # PM.viz.show_DoG_img_grid(gc_list=[0, 1, 5, 10], savefigname=None)
    # PM.viz.show_DoG_img_grid(n_samples=8)
    # PM.viz.show_cell_density_vs_ecc(unit_type="cone", savefigname=None)
    # PM.viz.show_cell_density_vs_ecc(unit_type="gc", savefigname=None)
    # PM.viz.show_cell_density_vs_ecc(unit_type="bipolar", savefigname=None)
    # PM.viz.show_connection_histograms(savefigname=None)

    # Subunit temporal model only
    # PM.viz.show_fan_in_out_distributions(savefigname=None)

    # PM.viz.show_experimental_data_DoG_fit(gc_list=[0, 3, 10], savefigname=None)
    # PM.viz.show_experimental_data_DoG_fit(gc_list=[69, 134, 167, 159], savefigname=None)
    # PM.viz.show_experimental_data_DoG_fit(n_samples=6, savefigname=None)
    # PM.viz.show_dendrite_diam_vs_ecc(log_x=False, log_y=True, savefigname=None)
    # PM.viz.show_retina_img(savefigname=None)
    # PM.viz.show_cone_noise_vs_freq(savefigname=None)
    # PM.viz.show_bipolar_nonlinearity(savefigname=None)

    # For DOG (DoG fits, temporal kernels and tonic drives)
    # spatial and temporal filter responses, ganglion cell positions and density,
    # mosaic layout, and dendrite diameter versus eccentricity.
    # PM.viz.show_exp_build_process(show_all_spatial_fits=True)

    # PM.viz.show_distribution_statistics(
    #     retina_parameters["fit_statistics"],
    #     distribution="spatial",
    #     correlation_reference="ampl_s",
    #     savefigname=None,
    # )

    # PM.viz.show_temporal_filter_response(n_samples=3, savefigname=None)
    # PM.viz.show_temporal_filter_response(gc_list=[0, 1, 5, 10], savefigname=None)

    # For VAE
    # PM.viz.show_gen_exp_spatial_rf(ds_name="train_ds", n_samples=5, savefigname=None)
    # PM.viz.show_latent_tsne_space()
    # PM.viz.show_gen_spat_post_hist()
    # PM.viz.show_latent_space_and_samples()
    # PM.viz.show_rf_imgs(n_samples=10, savefigname="parasol_on_vae_gen_rf.eps")
    # PM.viz.show_rf_violinplot()  # Pixel values for each unit

    # # "train_loss", "val_loss", "mse", "ssim", "kid_mean", "kid_std"
    # this_dep_var = "val_loss"
    # ray_exp_name = None  # "TrainableVAE_2023-04-20_22-17-35"  # None for most recent
    # highlight_trial = None  # "2199e_00029"  # or None
    # PM.viz.show_ray_experiment(
    #     ray_exp_name, this_dep_var, highlight_trial=highlight_trial
    # )

    # For both DOG and VAE. Estimated luminance for validation data (Schottdorf_2021_JPhysiol,
    # van Hateren_2002_JNeurosci) is 222.2 td / (np.pi * (4 mm diam / 2)**2) = 17.7 cd/m2
    # PM.viz.validate_gc_rf_size(savefigname="rf_size_vs_Schottdorf_data.eps")

    ###################################
    ###################################
    ###         Single Trial        ###
    ###################################
    ###################################

    ########################
    ### Create stimulus ###
    ########################

    # Based on visual_stimulus_parameters above
    PM.stimulate.make_stimulus_video()

    ####################################
    ### Run multiple trials or units ###
    ####################################

    PM.simulate_retina.client()

    ##########################################
    ### Show single ganglion cell features ###
    ##########################################
    # PM.viz.show_spatiotemporal_filter_summary(savefigname=None)

    # PM.viz.show_spatiotemporal_filter(unit_index=1, savefigname=None)
    # PM.viz.show_temporal_kernel_frequency_response(unit_index=2, savefigname=None)
    # PM.viz.plot_midpoint_contrast(unit_index=2, savefigname=None)
    # PM.viz.plot_local_rms_contrast(unit_index=2, savefigname=None)
    # PM.viz.plot_local_michelson_contrast(unit_index=2, savefigname=None)
    # PM.viz.show_single_gc_view(unit_index=2, frame_number=300, savefigname=None)

    # ################################################################################
    # #####     Interactive plot of spike frequency on stimulus video     ############
    # ################################################################################

    # video_file_name = visual_stimulus_parameters["stimulus_video_name"]
    # response_file_name = run_parameters["gc_response_filenames"][0] + ".gz"
    # window_length = 0.1  # seconds
    # rate_scale = 20  # Hz, Colorscale max amplitude
    # PM.viz_spikes_with_stimulus.client(
    #     video_file_name, response_file_name, window_length, rate_scale
    # )

    ################################################
    ###   Show multiple trials for single unit,  ###
    ###   or multiple units for single trial     ###
    ################################################

    # Based on run_parameters above
    PM.viz.show_all_gc_responses(savefigname=None)
    # PM.viz.show_all_generator_potentials(savefigname=None)
    # PM.viz.show_generator_potential_histogram(savefigname=None)
    # PM.viz.show_cone_responses(time_range=[0.4, 1.1], savefigname=None)
    # PM.viz.show_gc_noise(savefigname=None)

    # PM.viz.show_stimulus_with_gcs(
    #     example_gc=None,  # [int,], None for all
    #     frame_number=31,  # 31 depends on fps, and video and baseline lengths
    #     show_rf_id=True,
    #     savefigname=f"stimulus_with_gcs_{gc_type}_{response_type}.png",
    # )

    ##########################################
    ###       Show impulse response        ###
    ##########################################

    # # Contrast applies only for parasol units with dynamic model, use [1.0] for others
    # PM.context.run_parameters["contrasts_for_impulse"] = [1.0]
    # PM.simulate_retina.client(impulse=True)
    # PM.viz.show_impulse_response(savefigname="sununit_impulse_response.eps")

    #########################################
    ##          Show unity data           ###
    #########################################

    # Get uniformity data and exit
    # See Gauthier_2009_PLoSBiol for details
    # TÄHÄN JÄIT: TEE UNITY MAX EXPERIMENT, ASETA CENTER MASK TH TO UNITY MAX
    # EKAKSI TARKISTA DYNAMIC JA SUBUNIT MALLIEN NORMALISOINTI. NYT FIXED ASETETTU P1 AREA UNDER CURVE MUKAAN, MUIDEN PITÄISI TEHDÄ SAMANKALTAISESTI
    # PM.simulate_retina.client(unity=True)
    # PM.viz.show_unity(savefigname=None)

    #################################################################
    #################################################################
    ###   Experiment with multiple units, conditions and trials   ###
    #################################################################
    #################################################################

    ################################
    ### Build and run Experiment ###
    ################################

    # # Retina needs to be built for this to work.
    # # visual_stimulus_parameters above defines the stimulus. From that dictionary,
    # # defined keys' values are dynamically changed in the experiment.
    # # Note that tuple values from visual_stimulus_parameters are captured for varying each tuple value separately.

    # # from visual_stimulus_parameters, safe up to two variables
    exp_variables = ["temporal_frequency"]
    # experiment_parameters = {
    #     "exp_variables": exp_variables,
    #     # two vals below for each exp_variable, even is it is not changing
    #     # "min_max_values": [[5, 5]],
    #     "min_max_values": [[1, 32]],
    #     # "min_max_values": [[1, 32], [0.015, 0.5]],
    #     # "n_steps": [1],  # [6 ,10]
    #     "n_steps": [16],  # [6 ,10]
    #     "logarithmic": [True],
    #     # "logarithmic": [True, True],
    #     "n_sweeps": 1,
    #     # "distributions": {"gaussian": {"sweeps": 10, "mean": [-30, 30], "sd": [5, 5]}},
    #     "distributions": {"uniform": None},
    # }

    # # # # N trials or N units must be 1, and the other > 1. This is set above in run_parameters.
    # filename = PM.experiment.build_and_run(
    #     experiment_parameters, build_without_run=False, show_histogram=False
    # )
    # filename = "exp_metadata_orientation_spatial_frequency_edd770296d0e.csv"

    # # #########################
    # # ## Analyze Experiment ###
    # # #########################

    # my_analysis_options = {
    #     "exp_variables": exp_variables,
    #     "t_start_ana": 0.5,
    #     "t_end_ana": 6.5,
    # }

    # # PM.ana.analyze_experiment(filename, my_analysis_options)
    # # PM.ana.unit_correlation(
    # #     filename, my_analysis_options, gc_type, response_type, gc_units=None
    # # )
    # PM.ana.relative_gain(filename, my_analysis_options)
    # # PM.ana.response_vs_background(filename, my_analysis_options)

    # # ############################
    # # ### Visualize Experiment ###
    # # ############################
    # PM.viz.spike_raster_response(filename, sweeps_to_show=[0], savefigname="exp.png")
    # # # PM.viz.show_relative_gain(filename, exp_variables, savefigname=None)
    # # # PM.viz.show_response_vs_background_experiment(unit="cd/m2", savefigname=None)

    # # # PM.viz.show_unit_correlation(
    # # #     filename, exp_variables, time_window=[-0.2, 0.2], savefigname=None
    # # # )

    # PM.viz.F1F2_popul_response(
    #     filename,
    #     exp_variables,
    #     xlog=True,
    #     F1_only=False,
    #     savefigname=None,
    # )
    # # # PM.viz.F1F2_unit_response(filename, exp_variables, xlog=True, savefigname=None)
    # # # PM.viz.F1F2_unit_response(
    # # #     filename, exp_variables, xlog=True, hue="temporal_frequency", savefigname=None
    # # # )
    # # ### PM.viz.ptp_response(filename, exp_variables, x_of_interest=None, savefigname=None)
    # # ### PM.viz.fr_response(filename, exp_variables, xlog=True, savefigname=None)
    # # # PM.viz.tf_vs_fr_cg(
    # # #     filename,
    # # #     exp_variables,
    # # #     n_contrasts=10,
    # # #     xlog=True,
    # # #     ylog=False,
    # # #     savefigname=None,
    # # # )

    """
    ### Housekeeping ###. Do not comment out.
    """
    # End measuring time and print the time in HH hours MM minutes SS seconds format
    end_time = time.time()
    print(
        "Total time taken: ",
        time.strftime(
            "%H hours %M minutes %S seconds", time.gmtime(end_time - start_time)
        ),
    )

    plt.show()

    if profile is True:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("tottime")
        stats.print_stats(20)
