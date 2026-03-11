### Visual stimulus generation

Stimulus video will be saved in the output_folder in mp4 format (viewing) and hdf5 format (reloading). See the  [visual_stimulus_parameters](parameters.md#visual_stimulus_parameters) for details on the available parameters.

### Own image and video stimuli

You can use own images in png or jpg formats, and videos in avi or mp4 formats. 

external_stimulus_parameters:

  ext_stimulus_file: "my_file.avi" # filename with extension  
  ext_pix_per_deg: 30  # Pixels per degree of visual field

  See stimulus_factory_module.VideoClass for more options you can control with mr.external_stimulus_parameters.[option].