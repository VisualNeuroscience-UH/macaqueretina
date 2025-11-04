# Overview
Most parameters are available for modification in yaml files at [your_repo_root]/macaqueretina/parameters. They will be available in a Configuration object, accessed as macaqueretina.config.your_parameter (or as dict keys: macaqueretina.config["your_parameter"]). See [Tutorials, Utility methods, Print parameters](example_utility_methods.md#print-parameters) for examples on the mr.config use.

## core_parameters.yaml
These parameters include major path settings, seed number, cpu/cuda device selection, profiler flag. In addition, it contains a run pipeline to create retina, make stimulus, simulate, and show basic visualization. The run pipeline is executed if you run `python macaqueretina`. The profiler is available only if you run as pipeline, but not if you import macaqueretina into a script.

Project paths:
```text
model_root_path/
└── project/
    └── experiment/
        ├── input_folder/
        └── output_folder/
```

- ***model_root_path*** : **str** | This is the root directory for all large input and output files. 
- ***project*** : **str** | 1st level folder
- ***experiment*** : **str** | 2nd level folder
- ***input_folder*** : **str** | 3rd level folder for input like VAE models, images, videos, other data
- ***output_folder*** : **str** | 3rd level folder for output like spikes, retinas, experiment metadata
- ***numpy_seed*** : **int** or  `null` | integer as fixed seed to replicate simulation or null for random seed.
- ***device*** :  **str** `"cpu"` or `"cuda"` | If you have an NVIDIA GPU and CUDA installed, it is useful for speed. For large models the GPU may run out of memory. 
- ***profile*** : **bool** | false or true (lowercase in yaml format)


## Parameters for constructing retina
### retina_parameters.yaml
These are the core retina parameters. These are used to generate a unique hash in retina-related filenames, so that you can avoid rebuilding when you re-run the pipeline with the same retna parameters. Available options for each parameter are documented in the yaml file besides the keys.

The core parameters include the ganglion cell and response types, spatial and temporal model types, and dog model type (used both DOG model build and VAE model quantification). The retina segment is defined as `ecc_limits_deg: [start, stop]` and the polar segment size as `pol_limits_deg: [start, stop]`. At the moment polar rotation is not fully applied, so you need to keep your retina centered at the horizontal meridian eg by `pol_limits_deg: [-1.5, 1.5]`.

- ***model_density*** : **float** [0...1.0]  | 1.0 builds the retina patch with 100% density of ganglion cells according to literature.

- ***retina_center*** : **str** complex number in degrees | E.g. `"5.0+0j"` for 5 deg eccentricity at horizontal meridian. This corresponds to stimulus_position (0, 0).

- ***force_retina_build*** : **bool** | Flag enables retina rebuild even if the hash matches. This is handy if you experiment with code changes or change any parameter in the retina_parameters_extend.yaml file which affects the retina construction.

### retina_parameters_extend.yaml
These are additional retina parameters which are not changing so often and are not included into hash generation. Changing these will not be detected by the hash id. Instead, you need to set retina_parameters force_retina_build to true (overwrites retina files with the same hash) or change output or experiment directories. 

For documentation of these parameters, see the yaml file.

## visual_stimulus_parameters.yaml
This file contains the data for the following dictionaries:

### visual_stimulus_parameters
This comprises a complete definition of artificial stimuli and also affects external image and video stimuli. 
Valid parameters include (overriding visual_stimulus_module.VideoBaseClass):

- ***image_width***: **int** | in pixels  
- ***image_height***: **int** | in pixels  
- ***pix_per_deg***: **int** | pixels per degree  
- ***dtype_name***: **str** | low contrast steps need `"float16"`, for performance, use `"uint8"`  
- ***fps***: **int** | frames per second  
- ***duration_seconds***: **float** | stimulus duration  
- ***baseline_start_seconds***: **float** | duration of midgray at the beginning  
- ***baseline_end_seconds***: **float** | duration of midgray at the end  
- ***pattern***:  **str** |  
    `sine_grating` `square_grating` `colored_temporal_noise` `white_gaussian_noise`
    `natural_images` `natural_video` `temporal_sine_pattern` `temporal_square_pattern`
    `temporal_chirp_pattern` `contrast_chirp_pattern` `spatially_uniform_binary_noise`

- ***stimulus_form***: **str** |  
    `circular` `rectangular` `annulus`


- ***stimulus_position***: **[float, float]** | in degrees, [0, 0] is the center of the stimulus on the image pixel grid.  
- ***stimulus_size***: **float** | In degrees. Radius for circle and annulus, half-width for rectangle.  
- ***contrast***:  **float** | Value between 0 and 1. 
- ***mean***: **float** | Mean stimulus intensity in cd/m2. 

Assuming rgb voltage = cd/m2, and average pupil diameter of 3 mm, the mean voltage of 128 in background
would mean ~ 905 Trolands. Td = lum * pi * (diam/2)^2, resulting in 128 cd/m2 = 128 * pi * (3/2)^2 ~ 905 Td.

- ***intensity***: [**float**, **float**] or `null` | If not null, it overrides contrast and mean.

- ***stimulus_video_name***: **str** with suffix | Name of the stimulus video. if null, defaults to f"{stimulus_folder}_{hash}.mp4".  The hash is constructed from the visual stimulus parameter values.
- ***background***: **str** or **float** | options are `"mean"`, `"intensity_min"`, `"intensity_max"` or value in cd/m2. This is the frame around stimulus in time and space, incl pre- and post-stimulus baselines.  
- ***ND_filter***: **float** | Adds log10 neutral density filter factor, can be negative. This is handy if you want large retinal illumination changes without redoing the stimuli. Applies only to `subunit` temporal_model.

For sine_grating and square_grating, additional arguments are:  

- ***temporal_frequency***: **float** | in Hz  
- ***spatial_frequency***: **float** | in cycles per degree  
- ***orientation***: **float** | in degrees  

For all temporal and spatial gratings, additional argument is:  

- ***phase_shift***: **float** | between 0 and 2pi

For spatially_uniform_binary_noise, additional arguments are:  

- ***on_proportion***: **float** | between 0 and 1, proportion of on-stimulus
- ***on_time***: **float** | in seconds, duration of on-stimulus, default 0.1 seconds, floor division to full frames
- ***direction***: **str** | '"increment"' or '"decrement"'  

For stimulus_form annulus, additional arguments are:  

- ***size_inner***: **float** | in degrees  
- ***size_outer***: **float** | in degrees  

For chirp stimulus, additional parameter is:

- ***temporal_frequency_range***: [**float**, **float**] | [start, stop] frequency in Hz

***NOTE***: GC receptive fields outside the stimulus causes error. In other words, stimulus must be bigger than retina.  
***NOTE***: if you ask for too high spatial or temporal frequency in relation to pixels per degree or fps, respectively, will result in aliasing without warning

### external_stimulus_parameters
- ***stimulus_file***:  **str** | filename for an image or video file  
- ***pix_per_deg***: **int** | resolution for external stimuli  
- ***fps***. **int** | frames per second for external videos  
Put stimlus image and video files e.g. to the input_folder.

## simulation_parameters.yaml
These parameters guide simulation at runtime.  

- ***n_files***: **int** | make distinct files with same stimulus and **same cone noise** but changing randomization seed, thus creating distinct spike patterns.

- ***n_sweeps***: **int** | is faster than n_files and creates one response file but with multiple sweeps inside each response file. Each sweep will have **independent cone noise**. Correspondingly, the cone noise file size will be multiplied by about the n sweeps.

- ***spike_generator_model***: **str** | `"refractory"` or `"poisson"`
- ***save_data***: **bool** 
- ***simulation_dt***: **float** | in seconds, `0.0001` is 0.1 ms
- ***save_variables***: `["spikes", "cone_noise"]` as default, you can add `"cone_bipo_gen_fir"` to save the middle steps for subunit model and generator potentials for all temporal model types.

## gain_calibration.yaml
Here you find the gain calibration values for various model combinations for visual signals and for background noise.

## literature.yaml
Filenames and additional metadata for redigitized literature data. The files and corresponding .jpg images are in the main repo at macaqueretina/retina/literature_data. To visualize the datafiles, see example_utility_methods.py subsection "Sample and view figure data from literature".