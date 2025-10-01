""" 
Example implementation for reading data from an experimental dataset. 
This module is specific to the each dataset and needs to be adjusted 
accordingly.
"""

# Third-party
import numpy as np
import scipy.io as sio


class ExperimentalData:
    """
    Read data from external mat files.
    """

    def __init__(self, dog_metadata_parameters, gc_type, response_type):
        self.experimental_data_folder = dog_metadata_parameters[
            "experimental_data_folder"
        ]
        self.metadata = dog_metadata_parameters
        gc_type = gc_type.lower()
        response_type = response_type.lower()
        self.gc_type = gc_type
        self.response_type = response_type

        # Define filenames
        # Spatial data are read from a separate mat file that have been derived from the originals.
        # Non-spatial data are read from the original data files.
        if gc_type == "parasol" and response_type == "on":
            self.spatial_filename = "Parasol_ON_spatial.mat"
            self.known_bad_data_idx = [9, 15, 20, 25, 71, 86, 89]
            self.nonspatial_filename = "mosaicGLM_apricot_ONParasol-1-mat.mat"
        elif gc_type == "parasol" and response_type == "off":
            self.spatial_filename = "Parasol_OFF_spatial.mat"
            self.known_bad_data_idx = [6, 31, 71, 73]
            self.nonspatial_filename = "mosaicGLM_apricot_OFFParasol-1-mat.mat"
        elif gc_type == "midget" and response_type == "on":
            self.spatial_filename = "Midget_ON_spatial.mat"
            self.known_bad_data_idx = [13, 23]
            self.nonspatial_filename = "mosaicGLM_apricot_ONMidget-1-mat.mat"
        elif gc_type == "midget" and response_type == "off":
            self.spatial_filename = "Midget_OFF_spatial.mat"
            self.known_bad_data_idx = [39, 43, 50, 56, 65, 129, 137]
            self.nonspatial_filename = "mosaicGLM_apricot_OFFMidget-1-mat.mat"
        else:
            raise NotImplementedError("Unknown cell type or response type, aborting")

        # Make a key-value dictionary for labeling the data
        self.data_names2labels_dict = {
            "parasol_on": 0,
            "parasol_off": 1,
            "midget_on": 2,
            "midget_off": 3,
        }
        self.data_labels2names_dict = {
            v: k for k, v in self.data_names2labels_dict.items()
        }

        # Read nonspatial data. Data type is numpy nd array, but it includes a lot of metadata.
        filepath = self.experimental_data_folder / self.nonspatial_filename
        raw_data = sio.loadmat(filepath)
        self.data = raw_data["mosaicGLM"][0]

        self.n_cells = len(self.data)
        self.inverted_data_indices = self._get_inverted_indices()

    def _get_inverted_indices(self):
        """
        The rank-1 space and time matrices in the dataset may have bumps in an inconsistent way, but the
        outer product always produces a positive deflection first irrespective of on/off polarity.
        This method tells which cell indices you need to flip to get a spatial filter with positive central component.

        Returns
        -------
        inverted_data_indices : np.ndarray
        """

        temporal_filters = self.read_temporal_filter_data(flip_negs=False)
        # Based on 3 samples
        inverted_data_indices = np.argwhere(
            np.mean(temporal_filters[:, 1:3], axis=1) < 0
        ).flatten()

        return inverted_data_indices

    # Called from Fit
    def read_spatial_filter_data(self):
        filepath = self.experimental_data_folder / self.spatial_filename
        gc_spatial_data = sio.loadmat(filepath, variable_names=["c", "stafit"])
        spat_data_array = gc_spatial_data["c"]
        # Rotate dims to put n cells the first dim
        spat_data_array = np.moveaxis(spat_data_array, 2, 0)
        n_spatial_cells = len(spat_data_array[:, 0, 0])

        initial_center_values = gc_spatial_data["stafit"]

        # Pick out the initial guess for rotation of center ellipse
        cen_rot_rad_all = np.zeros(n_spatial_cells)
        for cell_idx in range(n_spatial_cells):
            cen_rot_rad = float(initial_center_values[0, cell_idx][4])
            if cen_rot_rad < 0:  # For negative angles, turn positive
                cen_rot_rad_all[cell_idx] = cen_rot_rad + 2 * np.pi

        n_bad = len(self.known_bad_data_idx)
        print("\n[%s %s]" % (self.gc_type, self.response_type))
        print(
            "Read %d cells from datafile and then removed %d bad cells (handpicked)"
            % (n_spatial_cells, n_bad)
        )

        return spat_data_array, cen_rot_rad_all

    def read_tonic_drive(self, remove_bad_data_idx=True):
        tonic_drive = np.array(
            [
                self.data[cellnum][0][0][0][0][0][1][0][0][0][0][0]
                for cellnum in range(self.n_cells)
            ]
        )
        if remove_bad_data_idx is True:
            tonic_drive[self.known_bad_data_idx] = 0.0

        return tonic_drive

    def read_temporal_filter_data(self, flip_negs=False, normalize=False):
        time_rk1 = np.array(
            [
                self.data[cellnum][0][0][0][0][0][3][0][0][3]
                for cellnum in range(self.n_cells)
            ]
        )
        temporal_filters = time_rk1[:, :, 0]

        # Flip temporal filters so that first deflection is always positive
        for i in range(self.n_cells):
            if temporal_filters[i, 1] < 0 and flip_negs is True:
                temporal_filters[i, :] = temporal_filters[i, :] * (-1)

        if normalize is True:
            assert (
                flip_negs is True
            ), "Normalization does not make sense without flip_negs"
            for i in range(self.n_cells):
                tf = temporal_filters[i, :]
                pos_sum = np.sum(tf[tf > 0])
                temporal_filters[i, :] = tf / pos_sum

        return temporal_filters
