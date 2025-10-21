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
    `sine_grating` `square_grating` `colored_temporal_noise` `white_gaussian_noise`
    `natural_images` `natural_video` `temporal_sine_pattern` `temporal_square_pattern`
    `temporal_chirp_pattern` `contrast_chirp_pattern` `spatially_uniform_binary_noise`

stimulus_form: `circular` `rectangular` `annulus`

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
NOTE: if you ask for too high spatial or temporal frequency in relation to pixels per degree or fps, respectively, will result in aliasing without warning

### Own video stimuli

This is work in progress, it is possible but you need to get the video data and set the visual_stimulus_parameters to match the dimensions, duration, frame rate, etc of your video. 