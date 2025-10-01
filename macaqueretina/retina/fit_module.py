# Built-in
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats
from scipy.optimize import curve_fit, minimize
from tqdm import tqdm

# Local
from macaqueretina.project.project_utilities_module import Printable
from macaqueretina.retina.experimental_data_module import ExperimentalData
from macaqueretina.retina.retina_math_module import RetinaMath


@dataclass
class ReceptiveFieldSD:
    center_sd: float
    surround_sd: float


class FitDoGTemplate(ABC, RetinaMath, Printable):

    def fit_spatial_filters_template(
        self,
        spat_data_array,
        cen_rot_rad_all,
        bad_spatial_idx,
        mark_outliers_bad=False,
        mask_noise=0,
    ):
        """
        Fit difference of Gaussians (DoG) model spatial filters using receptive field image data.

        Parameters
        ----------
        spat_data_array : numpy.ndarray
            Array of shape `(n_cells, num_pix_y, num_pix_x)` containing the spatial data for each cell to fit.
        cen_rot_rad_all : numpy.ndarray or None, optional
            Array of shape `(n_cells,)` containing the rotation angle for each cell. If None, rotation is set to 0, by default None
        bad_spatial_idx : numpy.ndarray or None, optional
            Indices of cells to exclude from fitting, by default None
        mark_outliers_bad : bool, optional
            Whether to mark cells with large fit error (> 3SD from mean) as bad, by default False

        Returns
        -------
        tuple
            A dataframe with spatial parameters and errors for each cell, and a dictionary of spatial filters to show with visualization.
            The dataframe has shape `(n_cells, 13)` and columns:
            ['ampl_c', 'xoc_pix', 'yoc_pix', 'semi_xc_pix', 'semi_yc_pix', 'orient_cen_rad', 'ampl_s', 'xos_pix', 'yos_pix', 'semi_xs_pix',
            'semi_ys_pix', 'orient_sur_rad', 'offset'] if dog_model_type=ellipse_independent,
            or shape `(n_cells, 8)` and columns: ['ampl_c', 'xoc_pix', 'yoc_pix', 'semi_xc_pix', 'semi_yc_pix', 'orient_cen_rad',
            'ampl_s', 'relat_sur_diam', 'offset'] if dog_model_type=ellipse_fixed.
            ['ampl_c', 'xoc_pix', 'yoc_pix', 'rad_c_pix', 'ampl_s', 'rad_s_pix', 'offset'] if dog_model_type=circular.
            The dictionary spat_filt has keys:
                'x_grid': numpy.ndarray of shape `(num_pix_y, num_pix_x)`, X-coordinates of the grid points
                'y_grid': numpy.ndarray of shape `(num_pix_y, num_pix_x)`, Y-coordinates of the grid points
                'dog_model_type': str, the type of DoG model used ('ellipse_independent', 'ellipse_fixed' or 'circular')
                'num_pix_x': int, the number of pixels in the x-dimension
                'num_pix_y': int, the number of pixels in the y-dimension
                'filters': numpy.ndarray of shape `(n_cells, num_pix_y, num_pix_x)`, containing the fitted spatial filters for each cell
            good_mask: numpy.ndarray of shape `(n_cells,)`, containing a boolean mask of cells that were successfully fitted

        Raises
        ------
        ValueError
            If the shape of spat_data_array is not `(n_cells, num_pix_y, num_pix_x)`.
        """
        self.bad_spatial_idx = bad_spatial_idx
        self.mask_noise = mask_noise

        self.base_spatial_fitting(spat_data_array, cen_rot_rad_all)
        self.base_get_viable_units()
        self.require_parameter_names()
        self.require_dog_model_type()
        self.base_data_array()
        self.require_initial_guess()
        self.base_flip_negative_spatial_rf()

        print(("Fitting DoG model, surround is {0}".format(self.surround_status)))
        self.require_fit()
        self.require_relative_surround_volume()
        self.base_append_data_to_spat_filt()
        self.require_append_aspect_ratios()
        self.base_append_relative_surround_volume()
        data_df = self.base_return_data_df(mark_outliers_bad)

        return data_df, self.spat_filt, self.parameter_names

    def base_spatial_fitting(self, spat_data_array, cen_rot_rad_all):

        self.rot = 0.0
        self.spat_data_array = spat_data_array
        self.cen_rot_rad_all = cen_rot_rad_all
        self.n_cells = int(spat_data_array.shape[0])

        # Check indices: x horizontal, y vertical
        self.num_pix_y = spat_data_array.shape[1]
        self.num_pix_x = spat_data_array.shape[2]

        # Note: this index used to start from 1, creating a bug downstream.
        # The reason for exceptional 1-index was data coming from matlab with 1-index.
        x_position_indices = np.linspace(0, self.num_pix_x - 1, self.num_pix_x)
        y_position_indices = np.linspace(0, self.num_pix_y - 1, self.num_pix_y)
        self.x_grid, self.y_grid = np.meshgrid(x_position_indices, y_position_indices)

    def base_get_viable_units(self):
        self.all_viable_cells = np.setdiff1d(
            np.arange(self.n_cells), self.bad_spatial_idx
        )

    @abstractmethod
    def require_parameter_names(self):
        pass

    @abstractmethod
    def require_dog_model_type(self):
        pass

    def base_data_array(self):
        n_cells = self.n_cells
        self.data_all_viable_cells = np.zeros(
            np.array([n_cells, len(self.parameter_names)])
        )
        # Create error & other arrays
        self.error_all_viable_cells = np.zeros((n_cells, 1))
        self.aspect_ratios = np.zeros(n_cells)

        self.spat_filt = {
            "x_grid": self.x_grid,
            "y_grid": self.y_grid,
            "dog_model_type": self.dog_model_type,
            "num_pix_x": self.num_pix_x,
            "num_pix_y": self.num_pix_y,
        }

    @abstractmethod
    def require_initial_guess(self):
        pass

    def base_flip_negative_spatial_rf(self):
        # RetinaMath method
        self.spat_data_array = self.flip_negative_spatial_rf(
            self.spat_data_array, self.mask_noise
        )

    @abstractmethod
    def require_fit(self):
        pass

    @abstractmethod
    def require_relative_surround_volume(self):
        pass

    def _base_rotate_center(self, cell_idx):

        data = self.data_all_viable_cells

        # Set rotation angle between 0 and pi
        data[cell_idx, 5] = data[cell_idx, 5] % np.pi

        # Rotate fit so that semi_x is always the semimajor axis (longer radius)
        if data[cell_idx, 3] < data[cell_idx, 4]:
            sd_x = data[cell_idx, 3]
            sd_y = data[cell_idx, 4]
            rotation = data[cell_idx, 5]

            data[cell_idx, 3] = sd_y
            data[cell_idx, 4] = sd_x
            data[cell_idx, 5] = (rotation + np.pi / 2) % np.pi

        self.data_all_viable_cells = data

    def _base_rotate_surround(self, cell_idx):

        data = self.data_all_viable_cells
        if data[cell_idx, 9] < data[cell_idx, 10]:
            sd_x_sur = data[cell_idx, 9]
            sd_y_sur = data[cell_idx, 10]
            rotation = data[cell_idx, 11]

            data[cell_idx, 9] = sd_y_sur
            data[cell_idx, 10] = sd_x_sur
            data[cell_idx, 11] = (rotation + np.pi / 2) % np.pi

        self.data_all_viable_cells = data

    def _base_rotate_to_plus_minus_pi(self, cell_idx):
        data = self.data_all_viable_cells
        rotation = data[cell_idx, 5]
        if rotation > np.pi / 2:
            data[cell_idx, 5] = rotation - np.pi
        else:
            data[cell_idx, 5] = rotation

        self.data_all_viable_cells = data

    def _base_set_aspect_ratios(self, cell_idx):

        # Check position of semi_xc_pix and semi_yc_pix in parameter array
        semi_xc_idx = self.parameter_names.index("semi_xc_pix")
        semi_yc_idx = self.parameter_names.index("semi_yc_pix")
        self.aspect_ratios[cell_idx] = (
            self.data_all_viable_cells[cell_idx, semi_xc_idx]
            / self.data_all_viable_cells[cell_idx, semi_yc_idx]
        )

    def _base_compute_fitting_error(self, model, cell_idx, this_rf, popt):
        # Compute fitting error
        gc_img_fitted = model((self.x_grid, self.y_grid), *popt)
        gc_img_fitted = gc_img_fitted.reshape(self.num_pix_y, self.num_pix_x)
        fit_deviations = gc_img_fitted - this_rf

        # MSE
        fit_error = np.sum(fit_deviations**2) / np.prod(this_rf.shape)
        self.error_all_viable_cells[cell_idx, 0] = fit_error

    def base_append_data_to_spat_filt(self):
        # Append data to spat_filt
        self.spat_filt["filters"] = self.data_all_viable_cells

    @abstractmethod
    def require_append_aspect_ratios(self):
        pass

    def base_append_relative_surround_volume(self):
        self.parameter_names.append("relative_surround_volume")
        self.data_all_viable_cells = np.hstack(
            (
                self.data_all_viable_cells,
                self.relative_surround_volume.reshape(self.n_cells, 1),
            )
        )

    def base_return_data_df(self, mark_outliers_bad):
        # Finally build a dataframe of the fitted parameters
        fits_df = pd.DataFrame(self.data_all_viable_cells, columns=self.parameter_names)
        error_df = pd.DataFrame(self.error_all_viable_cells, columns=["spatialfit_mse"])
        good_mask = np.ones(len(self.data_all_viable_cells))

        # Remove hand picked bad indices or failed fits
        if self.bad_spatial_idx is not None:
            good_mask[self.bad_spatial_idx] = 0

        if mark_outliers_bad == True:
            print(
                f"Previously removed {len(self.bad_spatial_idx)} outliers: {', '.join(map(str, self.bad_spatial_idx))}"
            )
            old_bad_spatial_idx = self.bad_spatial_idx.copy()
            # identify outliers (> 3SD from mean) and mark them bad
            self.bad_spatial_idx = self._get_fit_outliers(
                error_df, self.bad_spatial_idx, error_df.columns
            )
            self.bad_spatial_idx = np.unique(self.bad_spatial_idx)

            diff_idx = np.setdiff1d(self.bad_spatial_idx, old_bad_spatial_idx)
            print(
                f"Removing {len(diff_idx)} cell(s) with error > 3SD from mean: {', '.join(map(str, diff_idx))}"
            )

            for i in diff_idx:
                good_mask[i] = 0

        elif mark_outliers_bad == False:
            # We need to mark failed fits even if we don't mark outliers
            self.nan_idx = fits_df[fits_df.isna().any(axis=1)].index.values
            if len(self.nan_idx) > 0:
                good_mask[self.nan_idx] = 0
                print(f"\Marked cells {self.nan_idx} with failed fits")

        good_mask_df = pd.DataFrame(good_mask, columns=["good_filter_data"])

        return pd.concat(
            [
                fits_df,
                error_df,
                good_mask_df,
            ],
            axis=1,
        )

    # Additional abstract methods
    @abstractmethod
    def calculate_center_surround_sd(self):
        pass

    def _get_fit_outliers(self, fits_df, bad_spatial_idx, columns):
        """
        Finds the outliers of the spatial filters.
        """
        for col in columns:
            out_data = fits_df[col].values
            mean = np.mean(out_data)
            std_dev = np.std(out_data)
            threshold = 3 * std_dev
            mask = np.abs(out_data - mean) > threshold
            idx = np.where(mask)[0]
            bad_spatial_idx += idx.tolist()
        bad_spatial_idx.sort()

        return bad_spatial_idx


class FitEllipseFixed(FitDoGTemplate):
    def require_parameter_names(self):
        self.parameter_names = [
            "ampl_c",
            "xoc_pix",
            "yoc_pix",
            "semi_xc_pix",
            "semi_yc_pix",
            "orient_cen_rad",
            "ampl_s",
            "relat_sur_diam",
            "offset",
        ]

    def require_dog_model_type(self):
        self.surround_status = "fixed"
        self.dog_model_type = "ellipse_fixed"

    def require_initial_guess(self):
        # Build initial guess for (ampl_c, xoc_pix, yoc_pix, semi_xc_pix, semi_yc_pix,
        # orient_cen_rad, ampl_s, relat_sur_diam, offset)

        num_pix_y = self.num_pix_y
        num_pix_x = self.num_pix_x
        self.p0 = np.array(
            [
                1,
                num_pix_y // 2,
                num_pix_x // 2,
                num_pix_y // 4,
                num_pix_x // 4,
                self.rot,
                0.1,
                3,
                0,
            ]
        )

        self.boundaries = (
            np.array([0.999, -np.inf, -np.inf, 0, 0, 0, 0, 1, 0]),
            np.array([1, np.inf, np.inf, np.inf, np.inf, 2 * np.pi, 1, np.inf, 0.001]),
        )

    def require_fit(self):
        # Go through all cells
        for cell_idx in tqdm(self.all_viable_cells, desc="Fitting spatial filters"):
            this_rf = self.spat_data_array[cell_idx, :, :]

            try:
                if np.sum(this_rf) < 0:
                    print(("Negative sum for cell {0}".format(str(cell_idx))))
                    self.data_all_viable_cells[cell_idx, :] = np.nan
                    self.bad_spatial_idx.append(cell_idx)
                    continue
                rot = self.cen_rot_rad_all[cell_idx]
                self.p0[5] = rot
                popt, pcov = opt.curve_fit(
                    self.DoG2D_fixed_surround,
                    (self.x_grid, self.y_grid),
                    this_rf.ravel(),
                    p0=self.p0,
                    bounds=self.boundaries,
                )
                self.data_all_viable_cells[cell_idx, :] = popt
            except:
                print(("Fitting failed for cell {0}".format(str(cell_idx))))
                self.data_all_viable_cells[cell_idx, :] = np.nan
                self.bad_spatial_idx.append(cell_idx)
                continue

            self._base_rotate_center(cell_idx)
            self._base_rotate_to_plus_minus_pi(cell_idx)
            self._base_set_aspect_ratios(cell_idx)
            self._base_compute_fitting_error(
                self.DoG2D_fixed_surround, cell_idx, this_rf, popt
            )

            # For visualization
            self.spat_filt[f"cell_ix_{cell_idx}"] = {
                "spatial_data_array": this_rf,
            }

    def require_relative_surround_volume(self):
        """
        Calculate the relative surround volume for the fixed surround model.
        """
        data = self.data_all_viable_cells

        # Given self.parameter_names ['ampl_c', 'xoc_pix', 'yoc_pix', 'semi_xc_pix',
        # 'semi_yc_pix', 'orient_cen_rad', 'ampl_s', 'relat_sur_diam', 'offset']

        # Extracting the necessary parameters
        ampl_c = data[:, 0]
        semi_xc_pix = data[:, 3]
        semi_yc_pix = data[:, 4]
        ampl_s = data[:, 6]
        relat_sur_diam = data[:, 7]

        # Calculating the surround standard deviations
        sigma_s_x = semi_xc_pix * relat_sur_diam
        sigma_s_y = semi_yc_pix * relat_sur_diam

        # Calculating the relative surround volume
        V_s = 2 * np.pi * ampl_s * sigma_s_x * sigma_s_y
        V_c = 2 * np.pi * ampl_c * semi_xc_pix * semi_yc_pix

        self.relative_surround_volume = np.nan_to_num(V_s / V_c)

    def require_append_aspect_ratios(self):
        self.parameter_names.append("xy_aspect_ratio")
        self.data_all_viable_cells = np.hstack(
            (self.data_all_viable_cells, self.aspect_ratios.reshape(self.n_cells, 1))
        )

    def calculate_center_surround_sd(
        self, df: pd.DataFrame, scale_factor_mm_per_pixel: float
    ) -> ReceptiveFieldSD:
        """
        Calculate the mean center and surround receptive field sizes in millimeters for the
        ellipse fixed surround model.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing the fitted receptive field parameters
        scale_factor_mm_per_pixel : float
            Scale factor to convert pixels to millimeters

        Returns
        -------
        ReceptiveFieldSD
            Dataclass containing the mean center and surround receptive field sizes in millimeters
        """
        # Calculate geometric mean of semi-axes for center RF
        center_area_pixels = np.sqrt(df.semi_xc_pix * df.semi_yc_pix)
        mean_center_sd = np.mean(center_area_pixels) * scale_factor_mm_per_pixel

        # Calculate surround RF using relative diameter
        surround_area_pixels = np.sqrt(
            df.relat_sur_diam**2 * df.semi_xc_pix * df.semi_yc_pix
        )
        mean_surround_sd = np.mean(surround_area_pixels) * scale_factor_mm_per_pixel

        return ReceptiveFieldSD(center_sd=mean_center_sd, surround_sd=mean_surround_sd)


class FitEllipseIndependent(FitDoGTemplate):
    def require_parameter_names(self):
        self.parameter_names = [
            "ampl_c",
            "xoc_pix",
            "yoc_pix",
            "semi_xc_pix",
            "semi_yc_pix",
            "orient_cen_rad",
            "ampl_s",
            "xos_pix",
            "yos_pix",
            "semi_xs_pix",
            "semi_ys_pix",
            "orient_sur_rad",
            "offset",
        ]

    def require_dog_model_type(self):
        self.surround_status = "independent"
        self.dog_model_type = "ellipse_independent"

    def require_initial_guess(self):
        # Build initial guess for (ampl_c, xoc_pix, yoc_pix, semi_xc_pix, semi_yc_pix,
        # orient_cen_rad, ampl_s, xos_pix, yos_pix, semi_xs_pix, semi_ys_pix, orient_sur_rad, offset)
        num_pix_y = self.num_pix_y
        num_pix_x = self.num_pix_x
        self.p0 = np.array(
            [
                1,
                num_pix_y // 2,
                num_pix_x // 2,
                num_pix_y // 4,
                num_pix_x // 4,
                self.rot,
                0.1,
                num_pix_y // 2,
                num_pix_x // 2,
                (num_pix_y // 4) * 3,
                (num_pix_x // 4) * 3,
                self.rot,
                0,
            ]
        )
        self.boundaries = (
            np.array(
                [
                    0.999,
                    -np.inf,
                    -np.inf,
                    0,
                    0,
                    -2 * np.pi,
                    0,
                    -np.inf,
                    -np.inf,
                    0,
                    0,
                    -2 * np.pi,
                    -np.inf,
                ]
            ),
            np.array(
                [
                    1,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    2 * np.pi,
                    1,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    2 * np.pi,
                    np.inf,
                ]
            ),
        )

    def require_fit(self):
        # Go through all cells
        for cell_idx in tqdm(self.all_viable_cells, desc="Fitting spatial filters"):
            this_rf = self.spat_data_array[cell_idx, :, :]

            try:
                if np.sum(this_rf) < 0:
                    print(("Negative sum for cell {0}".format(str(cell_idx))))
                    self.data_all_viable_cells[cell_idx, :] = np.nan
                    self.bad_spatial_idx.append(cell_idx)
                    continue

                rot = self.cen_rot_rad_all[cell_idx]
                self.p0[5] = rot
                self.p0[11] = rot
                popt, pcov = opt.curve_fit(
                    self.DoG2D_independent_surround,
                    (self.x_grid, self.y_grid),
                    this_rf.ravel(),
                    p0=self.p0,
                    bounds=self.boundaries,
                )
                self.data_all_viable_cells[cell_idx, :] = popt
            except:
                print(("Fitting failed for cell {0}".format(str(cell_idx))))
                self.data_all_viable_cells[cell_idx, :] = np.nan
                self.bad_spatial_idx.append(cell_idx)
                continue

            self._base_rotate_center(cell_idx)
            self._base_rotate_surround(cell_idx)
            self._base_rotate_to_plus_minus_pi(cell_idx)
            self._base_set_aspect_ratios(cell_idx)
            self._base_compute_fitting_error(
                self.DoG2D_independent_surround, cell_idx, this_rf, popt
            )

            # For visualization
            self.spat_filt[f"cell_ix_{cell_idx}"] = {
                "spatial_data_array": this_rf,
            }

    def require_relative_surround_volume(self):
        """
        Calculate the relative surround volume for the fixed surround model.
        """
        data = self.data_all_viable_cells
        # Given self.parameter_names ['ampl_c', 'xoc_pix', 'yoc_pix', 'semi_xc_pix',
        # 'semi_yc_pix', 'orient_cen_rad', 'ampl_s', 'xos_pix', 'yos_pix', 'semi_xs_pix',
        # 'semi_ys_pix', 'orient_sur_rad', 'offset']

        # Extracting the necessary parameters for center
        ampl_c = data[:, 0]
        semi_xc_pix = data[:, 3]
        semi_yc_pix = data[:, 4]

        # Extracting the necessary parameters for surround
        ampl_s = data[:, 6]
        semi_xs_pix = data[:, 9]
        semi_ys_pix = data[:, 10]

        # Calculating the volumes for center and surround
        V_c = 2 * np.pi * ampl_c * semi_xc_pix * semi_yc_pix
        V_s = 2 * np.pi * ampl_s * semi_xs_pix * semi_ys_pix

        # Calculating the relative surround volume
        self.relative_surround_volume = np.nan_to_num(V_s / V_c)

    def require_append_aspect_ratios(self):
        self.parameter_names.append("xy_aspect_ratio")
        self.data_all_viable_cells = np.hstack(
            (self.data_all_viable_cells, self.aspect_ratios.reshape(self.n_cells, 1))
        )

    def calculate_center_surround_sd(
        self, df: pd.DataFrame, scale_factor_mm_per_pixel: float
    ) -> ReceptiveFieldSD:
        """
        Calculate the mean center and surround receptive field sizes in millimeters for the
        ellipse independent surround model.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing the fitted receptive field parameters
        scale_factor_mm_per_pixel : float
            Scale factor to convert pixels to millimeters

        Returns
        -------
        ReceptiveFieldSD
            Dataclass containing the mean center and surround receptive field sizes in millimeters
        """
        # Calculate geometric mean of semi-axes for center RF
        center_area_pixels = np.sqrt(df.semi_xc_pix * df.semi_yc_pix)
        mean_center_sd = np.mean(center_area_pixels) * scale_factor_mm_per_pixel

        # Calculate geometric mean of semi-axes for surround RF
        surround_area_pixels = np.sqrt(df.semi_xs_pix * df.semi_ys_pix)
        mean_surround_sd = np.mean(surround_area_pixels) * scale_factor_mm_per_pixel

        return ReceptiveFieldSD(center_sd=mean_center_sd, surround_sd=mean_surround_sd)


class FitCircular(FitDoGTemplate):
    def require_parameter_names(self):
        self.parameter_names = [
            "ampl_c",
            "xoc_pix",
            "yoc_pix",
            "rad_c_pix",
            "ampl_s",
            "rad_s_pix",
            "offset",
        ]

    def require_dog_model_type(self):
        self.surround_status = "concentric"
        self.dog_model_type = "circular"

    def require_initial_guess(self):
        # Build initial guess for (ampl_c, xoc_pix, yoc_pix, rad_c_pix, ampl_s, rad_s_pix, offset)
        num_pix_y = self.num_pix_y
        num_pix_x = self.num_pix_x
        self.p0 = np.array(
            [
                1,
                num_pix_y // 2,
                num_pix_x // 2,
                num_pix_y // 4,
                0.1,
                num_pix_y // 2,
                0,
            ]
        )
        self.boundaries = (
            np.array([0.999, -np.inf, -np.inf, 0, 0, 0, 0]),
            np.array([1, np.inf, np.inf, np.inf, 1, np.inf, np.inf]),
        )

    def require_fit(self):
        # Go through all cells
        for cell_idx in tqdm(self.all_viable_cells, desc="Fitting spatial filters"):
            this_rf = self.spat_data_array[cell_idx, :, :]

            try:
                if np.sum(this_rf) < 0:
                    print(("Negative sum for cell {0}".format(str(cell_idx))))
                    self.data_all_viable_cells[cell_idx, :] = np.nan
                    self.bad_spatial_idx.append(cell_idx)
                    continue
                popt, pcov = opt.curve_fit(
                    self.DoG2D_circular,
                    (self.x_grid, self.y_grid),
                    this_rf.ravel(),
                    p0=self.p0,
                    bounds=self.boundaries,
                )
                self.data_all_viable_cells[cell_idx, :] = popt
            except:
                print(("Fitting failed for cell {0}".format(str(cell_idx))))
                self.data_all_viable_cells[cell_idx, :] = np.nan
                self.bad_spatial_idx.append(cell_idx)
                continue

            self._base_compute_fitting_error(
                self.DoG2D_circular, cell_idx, this_rf, popt
            )

            # For visualization
            self.spat_filt[f"cell_ix_{cell_idx}"] = {
                "spatial_data_array": this_rf,
            }

    def require_relative_surround_volume(self):
        """
        Calculate the relative surround volume for the fixed surround model.
        """
        data = self.data_all_viable_cells
        # Given self.parameter_names ['ampl_c', 'xoc_pix', 'yoc_pix', 'rad_c_pix',
        # 'ampl_s', 'rad_s_pix', 'offset']

        # Extracting the necessary parameters for center
        ampl_c = data[:, 0]
        rad_c_pix = data[:, 3]

        # Extracting the necessary parameters for surround
        ampl_s = data[:, 4]
        rad_s_pix = data[:, 5]

        # Calculating the volumes for center and surround
        V_c = np.pi * ampl_c * rad_c_pix**2
        V_s = np.pi * ampl_s * rad_s_pix**2

        # Calculating the relative surround volume
        self.relative_surround_volume = np.nan_to_num(V_s / V_c)

    def require_append_aspect_ratios(self):
        pass

    def calculate_center_surround_sd(
        self, df: pd.DataFrame, scale_factor_mm_per_pixel: float
    ) -> ReceptiveFieldSD:
        """
        Calculate the mean center and surround receptive field sizes in millimeters for the
        circular model.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing the fitted receptive field parameters
        scale_factor_mm_per_pixel : float
            Scale factor to convert pixels to millimeters

        Returns
        -------
        ReceptiveFieldSD
            Dataclass containing the mean center and surround receptive field sizes in millimeters
        """
        # Calculate mean center RF radius
        mean_center_sd = np.mean(df.rad_c_pix) * scale_factor_mm_per_pixel

        # Calculate mean surround RF radius
        mean_surround_sd = np.mean(df.rad_s_pix) * scale_factor_mm_per_pixel

        return ReceptiveFieldSD(center_sd=mean_center_sd, surround_sd=mean_surround_sd)


class FitDataTypeTemplate(ABC, Printable):

    def fit_data_type_template(self):

        self.require_spatial_fit()
        self.base_good_idx()
        self.hook_temporal_fit()
        self.hook_tonic_drive_fit()
        self.require_all_data_fits_df()
        self.base_clean_invalid_fits()
        self.base_get_param_distribution_dict()
        self.hook_spatial_statistics()
        self.hook_temporal_statistics()
        self.hook_tonic_drive_statistics()

    @abstractmethod
    def require_spatial_fit(self):
        pass

    def base_good_idx(self):
        self.good_idx = self.spat_fits_df[
            self.spat_fits_df["good_filter_data"] == 1
        ].index.values

    def hook_temporal_fit(self):
        pass

    def hook_tonic_drive_fit(self):
        pass

    @abstractmethod
    def require_all_data_fits_df(self):
        pass

    def base_clean_invalid_fits(self):
        """
        Filters out rows with invalid fit data (all zeros) from the fits DataFrame.
        Updates self.data_df with only the valid fit data.
        """
        df = self.all_data_fits_df

        # Identify rows where all values are zero
        is_invalid_fit = (df == 0.0).all(axis=1)

        # Create new DataFrame with only valid fits
        self.data_df = df[~is_invalid_fit].copy()

    def base_get_param_distribution_dict(self):
        # 1. Define the lists of parameters for skewnorm and vonmises distributions
        spatial_parameter_names = self.spat_DoG_fit_params
        vonmises_strings = ["orient_cen_rad", "orient_sur_rad"]
        skewnorm_params = ["relative_surround_volume"]
        gamma_strings = [
            "ampl_c",
            "ampl_s",
            "semi_xc_pix",
            "semi_yc_pix",
            "semi_xs_pix",
            "semi_ys_pix",
            "rad_c_pix",
            "rad_s_pix",
        ]
        vonmises_params = [
            param for param in spatial_parameter_names if param in vonmises_strings
        ]
        skewnorm_params = [
            param for param in spatial_parameter_names if param not in vonmises_strings
        ]

        # 2. Create a dictionary with parameter name: distribution
        param_distribution_dict = {param: "skewnorm" for param in skewnorm_params}
        param_distribution_dict.update({param: "vonmises" for param in vonmises_params})
        param_distribution_dict.update({param: "gamma" for param in gamma_strings})

        # Drop key:value pairs from param_distribution_dict which are not in data_df.columns
        param_distribution_dict = {
            param: dist
            for param, dist in param_distribution_dict.items()
            if param in self.data_df.columns
        }
        self.param_distribution_dict = param_distribution_dict

    def hook_spatial_statistics(self):
        pass

    def hook_temporal_statistics(self):
        pass

    def hook_tonic_drive_statistics(self):
        pass

    # Helper methods
    def _fit_univariate_statistics(self, param_distribution_dict, data_df):
        n_distributions = len(param_distribution_dict)
        loc = np.zeros([n_distributions])
        scale = np.zeros([n_distributions])
        experimental_data = np.zeros([data_df.shape[0], n_distributions])
        x_model_fit = np.zeros([100, n_distributions])
        y_model_fit = np.zeros([100, n_distributions])

        # Create dict for statistical parameters
        model_parameters = {}

        for index, (param, dist) in enumerate(param_distribution_dict.items()):
            experimental_data[:, index] = data_df[param]

            match dist:
                case "gamma":
                    shape, loc[index], scale[index] = stats.gamma.fit(
                        experimental_data[:, index], loc=0
                    )
                    x_model_fit[:, index] = np.linspace(
                        stats.gamma.ppf(
                            0.001, shape, loc=loc[index], scale=scale[index]
                        ),
                        stats.gamma.ppf(
                            0.999, shape, loc=loc[index], scale=scale[index]
                        ),
                        100,
                    )

                    y_model_fit[:, index] = stats.gamma.pdf(
                        x=x_model_fit[:, index],
                        a=shape,
                        loc=loc[index],
                        scale=scale[index],
                    )

                    model_parameters[param] = {
                        "shape": shape,
                        "loc": loc[index],
                        "scale": scale[index],
                        "distribution": "gamma",
                    }

                case "skewnorm":
                    shape, loc[index], scale[index] = stats.skewnorm.fit(
                        experimental_data[:, index], loc=0
                    )
                    x_model_fit[:, index] = np.linspace(
                        stats.skewnorm.ppf(
                            0.001, shape, loc=loc[index], scale=scale[index]
                        ),
                        stats.skewnorm.ppf(
                            0.999, shape, loc=loc[index], scale=scale[index]
                        ),
                        100,
                    )

                    y_model_fit[:, index] = stats.skewnorm.pdf(
                        x=x_model_fit[:, index],
                        a=shape,
                        loc=loc[index],
                        scale=scale[index],
                    )

                    model_parameters[param] = {
                        "shape": shape,
                        "loc": loc[index],
                        "scale": scale[index],
                        "distribution": "skewnorm",
                    }

                case "vonmises":

                    def neg_log_likelihood(params, data):
                        kappa, loc = params
                        return -np.sum(
                            stats.vonmises.logpdf(data, kappa, loc=loc, scale=np.pi)
                        )

                    guess = [1.0, 0.0]  # kappa, loc
                    result = minimize(
                        neg_log_likelihood, guess, args=(experimental_data[:, index],)
                    )
                    kappa, loc[index] = result.x
                    scale[index] = np.pi  # fixed

                    x_model_fit[:, index] = np.linspace(
                        stats.vonmises.ppf(
                            0.001, kappa, loc=loc[index], scale=scale[index]
                        ),
                        stats.vonmises.ppf(
                            0.999, kappa, loc=loc[index], scale=scale[index]
                        ),
                        100,
                    )
                    y_model_fit[:, index] = stats.vonmises.pdf(
                        x=x_model_fit[:, index],
                        kappa=kappa,
                        loc=loc[index],
                        scale=scale[index],
                    )

                    model_parameters[param] = {
                        "shape": kappa,
                        "loc": loc[index],
                        "scale": scale[index],
                        "distribution": "vonmises",
                    }

        return experimental_data, model_parameters, x_model_fit, y_model_fit

    def _fit_multivariate_statistics(self, param_distribution_dict, data_df):
        """
        Fits a multivariate normal distribution to the standardized parameters specified in param_distribution_dict.

        Parameters
        ----------
        param_distribution_dict : dict
            A dictionary where keys are parameter names and values are the names of distributions.
        data_df : pandas.DataFrame
            A pandas DataFrame containing the data for each parameter.

        Returns
        -------
        means : np.ndarray
            The mean vector for the multivariate normal distribution.
        covariance_matrix : np.ndarray
            The covariance matrix for the multivariate normal distribution.
        std_devs : np.ndarray
            The standard deviations used for standardization.
        """

        keys = param_distribution_dict.keys()

        n_params = len(keys)
        n_samples = data_df.shape[0]

        relevant_data = np.zeros((n_samples, n_params))
        for index, param in enumerate(keys):
            relevant_data[:, index] = data_df[param].values

        means = np.mean(relevant_data, axis=0)
        std_devs = np.std(relevant_data, axis=0)
        covariance_matrix = np.cov(relevant_data, rowvar=False)

        return means, covariance_matrix, std_devs, keys


class FitExperimental(FitDataTypeTemplate):
    def __init__(
        self,
        DoG_model,
        diff_of_lowpass_filters,
        metadata,
        gc_type: str,
        response_type: str,
    ):
        self.DoG_model = DoG_model
        self.diff_of_lowpass_filters = diff_of_lowpass_filters
        self.metadata = metadata
        self.gc_type = gc_type
        self.response_type = response_type

    def get_center_surround_sd(self):
        """ """
        df = self.all_data_fits_df.iloc[self.good_idx]
        data_mm_per_pix = self.metadata["data_microm_per_pix"] / 1000

        return self.DoG_model.calculate_center_surround_sd(df, data_mm_per_pix)

    def get_experimental_statistics(self):

        # Collect everything into one big dataframe
        experimental_univariate_stat = pd.concat(
            [
                self.spatial_univariate_stat,
                self.temporal_univariate_stat,
                self.tonic_univariate_stat,
            ],
            axis=0,
        )

        return (
            experimental_univariate_stat,
            self.spatial_multivariate_stat,
            self.temporal_multivariate_stat,
        )

    def require_spatial_fit(self):
        self.experimental_data = ExperimentalData(
            self.metadata, self.gc_type, self.response_type
        )
        self.bad_data_idx = self.experimental_data.known_bad_data_idx
        self.n_cells = self.experimental_data.n_cells

        # Read experimental data and manually picked bad data indices
        (
            spatial_data,
            cen_rot_rad_all,
        ) = self.experimental_data.read_spatial_filter_data()

        # Check that original experimental data spatial resolution match metadata given in project_conf_module.
        assert (
            spatial_data.shape[1] == self.metadata["data_spatialfilter_height"]
        ), "Spatial data height does not match metadata"
        assert (
            spatial_data.shape[2] == self.metadata["data_spatialfilter_width"]
        ), "Spatial data width does not match metadata"

        (
            self.spat_fits_df,
            self.spat_filt,
            self.spat_DoG_fit_params,
        ) = self.DoG_model.fit_spatial_filters_template(
            spatial_data,
            cen_rot_rad_all,
            self.bad_data_idx,
            mark_outliers_bad=False,
            mask_noise=self.metadata["mask_noise"],
        )

    def hook_temporal_fit(self):
        """
        Fits each temporal filter to a function consisting of the difference of two cascades of lowpass filters.
        This follows the method described by Chichilnisky & Kalmar in their 2002 JNeurosci paper, using retinal spike
        triggered average (STA) data.

        Sets the following attributes:
        - temp_fits_df : pd.DataFrame
            DataFrame containing the fitted temporal filter parameters
        - temp_filt : dict
            Dictionary containing the temporal filter data for visualization
        """

        # shape (n_cells, 15); 15 time points @ 30 Hz (500 ms)
        temporal_filters = self.experimental_data.read_temporal_filter_data(
            flip_negs=True
        )

        data_fps = self.metadata["data_fps"]
        data_n_samples = self.metadata["data_temporalfilter_samples"]

        """
        Parameters
        ----------
        - n (float): Order of the filters.
        - p1 (float): Normalization factor for the first filter.
        - p2 (float): Normalization factor for the second filter.
        - tau1 (float): Time constant of the first filter.
        - tau2 (float): Time constant of the second filter.
        """
        parameter_names = ["n", "p1", "p2", "tau1", "tau2"]
        bounds = (
            [0, 0, 0, 0.1, 3],
            [np.inf, 10, 10, 3, 6],
        )  # bounds when time points are 0...14

        fitted_parameters = np.zeros((self.n_cells, len(parameter_names)))
        error_array = np.zeros(self.n_cells)
        max_error = -0.1

        xdata = np.arange(15)
        xdata_finer = np.linspace(0, max(xdata), 100)
        exp_temp_filt = {
            "xdata": xdata,
            "xdata_finer": xdata_finer,
            "title": f"{self.gc_type}_{self.response_type}",
        }

        for cell_ix in tqdm(self.good_idx, desc="Fitting temporal filters"):
            ydata = temporal_filters[cell_ix, :]

            try:
                popt, pcov = curve_fit(
                    self.diff_of_lowpass_filters, xdata, ydata, bounds=bounds
                )
                fitted_parameters[cell_ix, :] = popt
                error_array[cell_ix] = (1 / data_n_samples) * np.sum(
                    (ydata - self.diff_of_lowpass_filters(xdata, *popt)) ** 2
                )  # MSE error
            except:
                print("Fitting for cell index %d failed" % cell_ix)
                fitted_parameters[cell_ix, :] = np.nan
                error_array[cell_ix] = max_error
                continue

            exp_temp_filt[f"cell_ix_{cell_ix}"] = {
                "ydata": ydata,
                "y_fit": self.diff_of_lowpass_filters(xdata_finer, *popt),
            }

        parameters_df = pd.DataFrame(fitted_parameters, columns=parameter_names)
        # Convert taus to milliseconds
        parameters_df["tau1"] = parameters_df["tau1"] * (1 / data_fps) * 1000
        parameters_df["tau2"] = parameters_df["tau2"] * (1 / data_fps) * 1000

        error_df = pd.DataFrame(error_array, columns=["temporalfit_mse"])

        self.temp_fits_df = pd.concat([parameters_df, error_df], axis=1)
        self.temp_filt = exp_temp_filt

    def hook_tonic_drive_fit(self):
        self.tonic_drives_df = pd.DataFrame(
            self.experimental_data.read_tonic_drive(), columns=["tonic_drive"]
        )

    def require_all_data_fits_df(self):
        """
        Collect everything into one big dataframe

        Sets the following attributes:
        - all_data_fits_df : pd.DataFrame
            DataFrame containing the fitted parameters for spatial, temporal, and tonic filters
        """
        all_data_fits_df = pd.concat(
            [
                self.spat_fits_df,
                self.temp_fits_df,
                self.tonic_drives_df,
            ],
            axis=1,
        )

        # Set all_data_fits_df rows which are not part of good_idx_experimental to zero
        all_data_fits_df.loc[~all_data_fits_df.index.isin(self.good_idx)] = 0.0

        self.all_data_fits_df = all_data_fits_df

    def hook_spatial_statistics(self):
        """
        Fits skewnorm distribution parameters for the 'semi_xc_pix', 'semi_yc_pix', 'xy_aspect_ratio', 'ampl_s',
        and 'relat_sur_diam' RF parameters, and fits vonmises distribution parameters for the 'orient_cen_rad'
        RF parameter.
        """

        param_distribution_dict = self.param_distribution_dict

        # For experimental data, we mark relative_surround_volume > 1 bad, because
        # it would result in inverted unit polarity
        self.data_df = self.data_df.drop(
            self.data_df[self.data_df["relative_surround_volume"] >= 1].index
        )

        # 3. Fit univariate statistics
        (
            experimental_data,
            model_parameters,
            x_model_fit,
            y_model_fit,
        ) = self._fit_univariate_statistics(param_distribution_dict, self.data_df)

        # 4. Fit multivariate statistics
        means, covariance_matrix, std_devs, keys = self._fit_multivariate_statistics(
            param_distribution_dict, self.data_df
        )

        # 5. Collect data for visualization
        spatial_data_and_model = {
            "experimental_data": experimental_data,
            "univariate_statistics": {
                "model_parameters": model_parameters,
                "model_fit_curves": (x_model_fit, y_model_fit),
            },
            "multivariate_statistics": {
                "means": means,
                "std_devs": std_devs,
                "covariance_matrix": covariance_matrix,
                "keys": keys,
            },
        }

        # 6. Collect data for receptive field creation
        spatial_univariate_stat = pd.DataFrame.from_dict(
            model_parameters, orient="index"
        )
        spatial_univariate_stat["domain"] = "spatial"

        spatial_multivariate_stat = pd.DataFrame(
            {
                "means": means,
                "std_devs": std_devs,
            },
            index=param_distribution_dict.keys(),
        )

        idxs, keys = spatial_multivariate_stat.index, param_distribution_dict.keys()
        if not all([i == k for (i, k) in zip(idxs, keys)]):
            raise ValueError("Covariance parameter order unclear. Aborting...")
        covariance_columns = "cov_" + spatial_multivariate_stat.index
        spatial_multivariate_stat[covariance_columns] = covariance_matrix

        self.spatial_univariate_stat = spatial_univariate_stat
        self.spatial_multivariate_stat = spatial_multivariate_stat
        self.spatial_data_and_model = spatial_data_and_model

    def hook_temporal_statistics(self):
        """
        Fit temporal statistics of the temporal filter parameters using the skewnorm distribution.

        Parameters
        ----------
        good_data_fit_idx : ndarray
            Boolean index array indicating which rows of `self.all_data_fits_df` to use for fitting.

        """

        temporal_parameter_names = ["n", "p1", "p2", "tau1", "tau2"]

        param_distribution_dict = {param: "gamma" for param in temporal_parameter_names}
        (
            experimental_data,
            model_parameters,
            x_model_fit,
            y_model_fit,
        ) = self._fit_univariate_statistics(param_distribution_dict, self.data_df)

        # 4. Fit multivariate statistics
        means, covariance_matrix, std_devs, keys = self._fit_multivariate_statistics(
            param_distribution_dict, self.data_df
        )

        temporal_data_and_model = {
            "experimental_data": experimental_data,
            "univariate_statistics": {
                "model_parameters": model_parameters,
                "model_fit_curves": (x_model_fit, y_model_fit),
            },
            "multivariate_statistics": {
                "means": means,
                "std_devs": std_devs,
                "covariance_matrix": covariance_matrix,
                "keys": keys,
            },
        }

        temporal_univariate_stat = pd.DataFrame.from_dict(
            model_parameters, orient="index"
        )
        temporal_univariate_stat["domain"] = "temporal"

        temporal_multivariate_stat = pd.DataFrame(
            {
                "means": means,
                "std_devs": std_devs,
            },
            index=param_distribution_dict.keys(),
        )

        idxs, keys = temporal_multivariate_stat.index, param_distribution_dict.keys()
        if not all([i == k for (i, k) in zip(idxs, keys)]):
            raise ValueError("Covariance parameter order unclear. Aborting...")
        covariance_columns = "cov_" + temporal_multivariate_stat.index
        temporal_multivariate_stat[covariance_columns] = covariance_matrix

        self.temporal_univariate_stat = temporal_univariate_stat
        self.temporal_multivariate_stat = temporal_multivariate_stat
        self.temporal_data_and_model = temporal_data_and_model

    def hook_tonic_drive_statistics(self):
        """ """
        param_distribution_dict = {"tonic_drive": "gamma"}

        (
            experimental_data,
            model_parameters,
            x_model_fit,
            y_model_fit,
        ) = self._fit_univariate_statistics(param_distribution_dict, self.data_df)

        tonic_data_and_model = {
            "experimental_data": experimental_data,
            "model_parameters": model_parameters,
            "model_fit_curves": (x_model_fit, y_model_fit),
        }

        tonic_univariate_stat = pd.DataFrame.from_dict(model_parameters, orient="index")
        tonic_univariate_stat["domain"] = "tonic"

        self.tonic_univariate_stat = tonic_univariate_stat
        self.tonic_data_and_model = tonic_data_and_model


class FitGenerated(FitDataTypeTemplate):
    def __init__(self, DoG_model, spatial_data, um_per_pix, mark_outliers_bad):
        self.DoG_model = DoG_model
        self.spatial_data = spatial_data
        self.um_per_pix = um_per_pix
        self.mark_outliers_bad = mark_outliers_bad

    def get_center_surround_sd(self):
        """ """

        df = self.all_data_fits_df.iloc[self.good_idx]
        data_mm_per_pix = self.um_per_pix / 1000

        return self.DoG_model.calculate_center_surround_sd(df, data_mm_per_pix)

    def get_generated_DoG_fits(self):

        (
            experimental_data,
            model_parameters,
            x_model_fit,
            y_model_fit,
        ) = self._fit_univariate_statistics(self.param_distribution_dict, self.data_df)

        # 6. Collect data for statistics
        gen_stat_df = pd.DataFrame.from_dict(model_parameters, orient="index")
        gen_stat_df["domain"] = "spatial"

        return (
            gen_stat_df,
            self.all_data_fits_df,
            self.good_idx,
        )

    def require_spatial_fit(self):
        """ """

        cen_rot_rad_all = np.zeros(self.spatial_data.shape[0])

        (
            self.spat_fits_df,
            self.spat_filt,
            self.spat_DoG_fit_params,
        ) = self.DoG_model.fit_spatial_filters_template(
            self.spatial_data,
            cen_rot_rad_all,
            bad_spatial_idx=[],
            mark_outliers_bad=self.mark_outliers_bad,
        )

    def require_all_data_fits_df(self):
        # Collect everything into one big dataframe
        all_data_fits_df = pd.concat([self.spat_fits_df], axis=1)

        # Set all_data_fits_df rows which are not part of good_idx_experimental to zero
        all_data_fits_df.loc[~all_data_fits_df.index.isin(self.good_idx)] = 0.0

        self.all_data_fits_df = all_data_fits_df

    def hook_spatial_statistics(self):
        pass


class Fit(RetinaMath):
    """
    This class contains methods for fitting elliptical and circularly symmetric
    difference of Gaussians (DoG) models to experimental  (Field_2010) and generated data.
    In addition, it contains methods for fitting experimental impulse response magnitude to
    a function consisting of cascade of two lowpass filters and for adding the tonic drive.
    """

    def __init__(self, project_data, dog_metadata_parameters):
        # Dependency injection at ProjectManager construction
        self._project_data = project_data

        self.metadata = dog_metadata_parameters

    @property
    def project_data(self):
        return self._project_data

    def _get_concrete_components(self) -> None:

        match self.dog_model_type:
            case "ellipse_fixed":
                DoG_model = FitEllipseFixed()
            case "ellipse_independent":
                DoG_model = FitEllipseIndependent()
            case "circular":
                DoG_model = FitCircular()

        match self.fit_type:
            case "experimental":
                fit_data_type = FitExperimental(
                    DoG_model,
                    self.diff_of_lowpass_filters,
                    self.metadata,
                    self.gc_type,
                    self.response_type,
                )
            case "generated":
                fit_data_type = FitGenerated(
                    DoG_model,
                    self.spatial_data,
                    self.um_per_pix,
                    self.mark_outliers_bad,
                )

        self.fit_data_type = fit_data_type
        self.DoG_model = DoG_model

    def client(
        self,
        gc_type,
        response_type,
        fit_type="experimental",
        dog_model_type="ellipse_fixed",
        spatial_data=None,
        um_per_pix=None,
        mark_outliers_bad=False,
    ):
        """
        Initialize the Fit object.

        Parameters
        ----------
        gc_type : str
            The type of ganglion cell.
        response_type : str
            The type of response.
        fit_type : str, optional
            The fit type, can be either 'experimental' or 'generated'. Default is 'experimental'.
        dog_model_type : str, optional
            The type of DoG model used ('ellipse_independent', 'ellipse_fixed' or 'circular').
        spatial_data : array_like, optional
            The spatial data. Default is None.
        um_per_pix : float, optional
            The new micrometers per pixel value, required when fit_type is 'generated'.
            Default is None.
        mark_outliers_bad : bool, optional
            Whether to mark cells with large fit error (> 3SD from mean) as bad. Default is False.

        Raises
        ------
        ValueError
            If fit_type is 'generated' and either um_per_pix or spatial_data is not provided.
        """

        if fit_type in ["generated"] and um_per_pix is None:
            raise ValueError("If fit_type is 'generated', um_per_pix must be provided")

        if fit_type in ["generated"] and spatial_data is None:
            raise ValueError(
                "If fit_type is 'generated', spatial_data must be provided"
            )

        self.gc_type = gc_type
        self.response_type = response_type
        self.fit_type = fit_type
        self.dog_model_type = dog_model_type
        self.spatial_data = spatial_data
        self.um_per_pix = um_per_pix
        self.mark_outliers_bad = mark_outliers_bad

        self._get_concrete_components()

        self.fit_data_type.fit_data_type_template()

        self.receptive_field_sd = self.fit_data_type.get_center_surround_sd()

    def get_experimental_statistics(self):

        self.project_data.fit["exp_spat_filt"] = self.fit_data_type.spat_filt
        self.project_data.fit["spatial_data_and_model"] = (
            self.fit_data_type.spatial_data_and_model
        )
        self.project_data.fit["exp_temp_filt"] = self.fit_data_type.temp_filt
        self.project_data.fit["temporal_data_and_model"] = (
            self.fit_data_type.temporal_data_and_model
        )
        self.project_data.fit["tonic_data_and_model"] = (
            self.fit_data_type.tonic_data_and_model
        )
        self.project_data.fit["all_data_fits_df"] = self.fit_data_type.all_data_fits_df
        self.project_data.fit["good_idx_experimental"] = self.fit_data_type.good_idx

        return self.fit_data_type.get_experimental_statistics()

    def get_generated_DoG_fits(self):
        gen_stat_df, all_data_fits_df, good_idx = (
            self.fit_data_type.get_generated_DoG_fits()
        )
        # Save data to project_data for vizualization
        self.project_data.fit["gen_spat_filt"] = self.fit_data_type.spat_filt
        return gen_stat_df, all_data_fits_df, good_idx

    def get_good_data_df(self):
        return self.fit_data_type.all_data_fits_df.loc[self.fit_data_type.good_idx, :]

    def get_good_data_idx(self):
        return self.fit_data_type.good_idx
