# Overview
Most parameters are available for modification in yaml files at [your_repo_root]/macaqueretina/parameters. They will be converted into Configuration type, and will be available under the macaqueretina.config object either with dot notation or as a dict keys. See [Tutorials, Utility methods, Print parameters](example_utility_methods.md#print-parameters) for examples on the mr.config use.

## core_parameters.yaml
These parameters include major path settings, seed number, cpu/cuda device selection, profiler flag. In addition, it contains a runtime pipeline to create retina, make stimulus, simulate and show basic visualization.

Project paths:
```text
model_root_path/
└── experiment/
    ├── input_folder/
    └── output_folder/
```

## Parameters for constructing retina
### retina_parameters.yaml
These are the core retina parameters. These become the hash in retina related filenames so that you can avoid rebuilding when you rerunning the pipeline. Available options are documented besides the keys.

The core parameters include the ganglion cell and response types, spatial and temporal model types, dog model type (used both DOG model build and VAE model quantification). The retina segment is defined as `ecc_limits_deg: [start, stop]` and the polar segment size as `pol_limits_deg: [start, stop]`. At the moment polar rotation is not fully applied, so you need to keep your retina centered at the horizontal meridian eg by `pol_limits_deg: [-1.5, 1.5]`.

- ***model_density: 1.0*** : builds the retina patch with 100% density of ganglion cells according to literature.

- ***retina_center: "5.0+0j"*** : complex number in degrees. This corresponds to stimulus_position (0, 0).

- ***force_retina_build*** : flag enables retina rebuild even if the hash matches. This is handy if you experiment with code changes or change any parameter in the retina_parameters_extend.yaml file which affects the retina construction.

### retina_parameters_extend.yaml
These are additional retina parameters which are not changing so often and are not included into hash generation. Changing these will not be detected by the hash id. Instead, you need to set retina_parameters force_retina_build to true (overwrites retina files with the same hash) or change output or experiment directories. 

For documentation of these parameters, see the yaml file.

## visual_stimulus_parameters.yaml
This file contains the data for following dictionaries:

### visual_stimulus_parameters
This comprise complete definition of artificial stimuli and affect also external image and video stimuli. 
Valid parameters include (overriding visual_stimulus_module.VideoBaseClass):

- ***image_width***: in pixels  
- ***image_height***: in pixels  
- ***pix_per_deg***: pixels per degree  
- ***dtype_name***: low contrast steps need "float16", for performance, use "uint8"  
- ***fps***: frames per second  
- ***duration_seconds***: stimulus duration  
- ***baseline_start_seconds***: midgray at the beginning  
- ***baseline_end_seconds***: midgray at the end  
- ***pattern***:  
    `sine_grating` `square_grating` `colored_temporal_noise` `white_gaussian_noise`
    `natural_images` `natural_video` `temporal_sine_pattern` `temporal_square_pattern`
    `temporal_chirp_pattern` `contrast_chirp_pattern` `spatially_uniform_binary_noise`

- ***stimulus_form***: `circular` `rectangular` `annulus`


- ***stimulus_position***: in degrees, (0,0) is the center of the stimulus on the image pixel grid.  
- ***stimulus_size***: In degrees. Radius for circle and annulus, half-width for rectangle.  
- ***contrast***: between 0 and 1  
- ***mean***: mean stimulus intensity in cd/m2  

- ***intensity*** [min, max] or `null`. If defined, it overrides contrast and mean becomes baseline.

- ***stimulus_video_name***: name of the stimulus video. if null, defaults to f"{stimulus_folder}.mp4".  
- ***background***: `mean`, `intensity_min`, `intensity_max` or value in cd/m2. This is the frame around stimulus in time and space, incl pre- and post-stimulus baselines.  
- ***ND_filter*** adds log10 neutral density filter factor, can be negative. This is handy if you want large retinal illumination changes without redoing the stimuli. Applies only to ***temporal_model*** `subunit`

For sine_grating and square_grating, additional arguments are:  

- ***temporal_frequency***: in Hz  
- ***spatial_frequency***: in cycles per degree  
- ***orientation***: in degrees  

For all temporal and spatial gratings, additional argument is:  

- ***phase_shift***: between 0 and 2pi

For spatially_uniform_binary_noise, additional arguments are:  

- ***on_proportion***: between 0 and 1, proportion of on-stimulus, default 0.5
- ***on_time***: in seconds, duration of on-stimulus, default 0.1 seconds, floor division to full frames
- ***direction***: 'increment' or 'decrement'  

For stimulus_form annulus, additional arguments are:  

- ***size_inner***: in degrees  
- ***size_outer***: in degrees  

For chirp stimulus, additional parameter is:

- ***temporal_frequency_range***: [start, stop] frequency in Hz

With assuming rgb voltage = cd/m2, and average pupil diameter of 3 mm, the mean voltage of 128 in background
would mean ~ 905 Trolands. Td = lum * pi * (diam/2)^2, resulting in 128 cd/m2 = 128 * pi * (3/2)^2 ~ 905 Td.

***NOTE***: GC receptive fields outside the stimulus causes error. In other words, stimulus must be bigger than retina.  
***NOTE***: if you ask for too high spatial or temporal frequency in relation to pixels per degree or fps, respectively, will result in aliasing without warning

### external_stimulus_parameters
- ***stimulus_file***: filename for an image or video file  
- ***pix_per_deg***: resolution for external stimuli  
- ***fps***. frames per second for external videos  
Put stimlus image and video files e.g. to the input_folder.

## simulation_parameters.yaml
These parameters guide simulation at runtime.  

- ***n_files*** make distinct files with same stimulus and **same cone noise** but changing randomization seed, thus creating distinct spike patterns.

- ***n_sweeps*** is faster than n_files and creates one response file but with multiple sweeps inside each response file. Each sweep will have **independent cone noise**. Correspondingly, the cone noise file size will be multiplied by about the n sweeps.

- ***spike_generator_model***: "refractory"  # poisson or refractory
- ***save_data***: True
- ***simulation_dt***: in seconds, `0.0001` is 0.1 ms
- ***save_variables***: `["spikes", "cone_noise"]` as default, you can add "cone_bipo_gen_fir" to save the middle steps for subunit model and generator potentials for all temporal model types.

## gain_calibration.yaml
Here you find the gain calibration values for various model combinations for visual signals and for background noise.

## literature.yaml
Filenames and additional metadata for redigitized literature data. The files and corresponding jpg images are in the main repo at macaqueretina/retina/literature_data. To visualize the datafiles, see example_utility_methods.py subsection "Sample and view figure data from literature"