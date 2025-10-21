# Built-in
import math
import time
import traceback
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

# Third-party
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats
import torch
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.spatial import Voronoi
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm

# Local
from .rf_repulsion_utils import apply_rf_repulsion
from macaqueretina.project.project_utilities_module import PrintableMixin
from macaqueretina.retina.retina_math_module import RetinaMath


class Retina(PrintableMixin):
    """
    A class representing the biological and computational model of the macaque retina.

    This class houses retina-level parameters, manages the placement and properties of retinal cells,
    and collects the computational products of the retina simulation. Most parameters are loaded from
    the yaml files, but some are computed during initialization or simulation.

    Parameters
    ----------
    retina_parameters : dict
        A dictionary containing all parameters and settings for the retina model.
        This includes placement, spatial, temporal, and response properties for all cell types.
        Expected keys include:
        - "gc_type", "response_type", "spatial_model_type", "dog_model_type", "temporal_model_type"
        - "gc_placement_parameters", "cone_placement_parameters", "bipolar_placement_parameters"
        - "cone_general_parameters", "bipolar_general_parameters"
        - "dd_regr_model", "deg_per_mm", "ecc_limits_deg", "ecc_limit_for_dd_fit", "pol_limits_deg"
        - "fit_statistics", "center_mask_threshold", "bipolar2gc_dict", "receptive_field_repulsion_parameters"
        - "model_density", "proportion_of_parasol_gc_type", "proportion_of_midget_gc_type"
        - "proportion_of_ON_response_type", "proportion_of_OFF_response_type"
        - "experimental_archive"

    Attributes
    ----------
    whole_ret_img : np.ndarray or None
        A 2D array representing the whole retina image. Computed during simulation.
    whole_ret_lu_mm : np.ndarray or None
        Coordinates of the left upper corner of the whole retina image, in millimeters. Computed during simulation.
    cones_to_gcs_weights : np.ndarray or None
        Weights mapping cones to ganglion cells. Computed during simulation.
    experimental_archive : str
        Identifier for the experimental dataset or archive used for parameter fitting.
    gc_type : str
        Type of ganglion cell (e.g., "parasol", "midget").
    response_type : str
        Response type of the ganglion cell (e.g., "ON", "OFF").
    spatial_model_type : str
        Type of spatial model used for cell placement.
    dog_model_type : str
        Type of difference-of-Gaussians model used for receptive fields.
    temporal_model_type : str
        Type of temporal model used for cell responses.
    fit_statistics : dict
        Statistics related to the fitting of model parameters.
    mask_threshold : float
        Threshold for the center mask of the receptive field.
    gc_placement_parameters : dict
        Parameters for placing ganglion cells on the retina.
    cone_placement_parameters : dict
        Parameters for placing cone cells on the retina.
    bipolar_placement_parameters : dict
        Parameters for placing bipolar cells on the retina.
    cone_general_parameters : dict
        Parameters for natural stimulus filtering and cone-to-GC connections.
    bipolar_general_parameters : dict
        General parameters for bipolar cells.
    dd_regr_model : Any
        Regression model for dendritic diameter as a function of eccentricity.
    deg_per_mm : float
        Conversion factor: degrees of visual field per millimeter of retina.
    bipolar2gc_dict : dict
        Mapping of bipolar cell types to ganglion cell types.
    receptive_field_repulsion_parameters : dict
        Parameters controlling the repulsion between receptive fields.
    ecc_lim_mm : np.ndarray
        Eccentricity limits of the retina, in millimeters (converted from degrees).
    ecc_limit_for_dd_fit_mm : float
        Eccentricity limit for dendritic density fit, in millimeters (converted from degrees).
    polar_lim_deg : np.ndarray
        Polar angle limits of the retina, in degrees.
    model_density : float
        Density of the model cells on the retina (must be <= 1.0).
    proportion_of_parasol_gc_type : float
        Proportion of parasol ganglion cells in the model.
    proportion_of_midget_gc_type : float
        Proportion of midget ganglion cells in the model.
    proportion_of_ON_response_type : float
        Proportion of ON-response ganglion cells in the model.
    proportion_of_OFF_response_type : float
        Proportion of OFF-response ganglion cells in the model.
    """

    def __init__(self, retina_parameters: Dict[str, Any]) -> None:
        # Computed downstream
        self.whole_ret_img: np.ndarray | None = None
        self.whole_ret_lu_mm: np.ndarray | None = None
        self.cones_to_gcs_weights: np.ndarray | None = None

        # Literature and other metadata
        self.experimental_archive: str = retina_parameters["experimental_archive"]

        # Attributes
        self.gc_type: str = retina_parameters["gc_type"]
        self.response_type: str = retina_parameters["response_type"]
        self.spatial_model_type: str = retina_parameters["spatial_model_type"]
        self.dog_model_type: str = retina_parameters["dog_model_type"]
        self.temporal_model_type: str = retina_parameters["temporal_model_type"]

        self.fit_statistics: dict = retina_parameters["fit_statistics"]
        self.mask_threshold: float = retina_parameters["center_mask_threshold"]

        self.gc_placement_parameters: dict = retina_parameters[
            "gc_placement_parameters"
        ]
        self.cone_placement_parameters: dict = retina_parameters[
            "cone_placement_parameters"
        ]
        self.cone_general_parameters: dict = retina_parameters[
            "cone_general_parameters"
        ]
        self.bipolar_placement_parameters: dict = retina_parameters[
            "bipolar_placement_parameters"
        ]
        self.bipolar_general_parameters: dict = retina_parameters[
            "bipolar_general_parameters"
        ]

        self.dd_regr_model: Any = retina_parameters["dd_regr_model"]
        self.deg_per_mm: float = retina_parameters["deg_per_mm"]
        self.bipolar2gc_dict: dict = retina_parameters["bipolar2gc_dict"]
        self.receptive_field_repulsion_parameters: dict = retina_parameters[
            "receptive_field_repulsion_parameters"
        ]

        ecc_limits_deg: list[float] = retina_parameters["ecc_limits_deg"]
        ecc_limit_for_dd_fit: float = retina_parameters["ecc_limit_for_dd_fit"]
        pol_limits_deg: list[float] = retina_parameters["pol_limits_deg"]

        self.model_density: float = retina_parameters["model_density"]

        # Turn list to numpy array and deg to mm
        self.ecc_lim_mm: np.ndarray = (
            np.asarray(ecc_limits_deg).astype(float) / self.deg_per_mm
        )
        self.ecc_limit_for_dd_fit_mm: float = ecc_limit_for_dd_fit / self.deg_per_mm
        self.polar_lim_deg: np.ndarray = np.asarray(pol_limits_deg).astype(float)

        self.proportion_of_parasol_gc_type: float = retina_parameters[
            "proportion_of_parasol_gc_type"
        ]
        self.proportion_of_midget_gc_type: float = retina_parameters[
            "proportion_of_midget_gc_type"
        ]
        self.proportion_of_ON_response_type: float = retina_parameters[
            "proportion_of_ON_response_type"
        ]
        self.proportion_of_OFF_response_type: float = retina_parameters[
            "proportion_of_OFF_response_type"
        ]


class DistributionSampler:
    """
    A class to sample from univariate and multivariate distributions.
    """

    def filter_stat(self, available_stat, covariances_of_interest):
        stat = pd.DataFrame(
            available_stat[available_stat.index.isin(covariances_of_interest)]
        )

        stat_index = stat.index.tolist()

        # Maintain the order of covariances_of_interest in the intersection
        intersection = [item for item in covariances_of_interest if item in stat_index]
        covariance_columns = ["cov_" + item for item in intersection]

        columns_to_keep = ["means", "std_devs"] + covariance_columns
        filtered_stat = stat[columns_to_keep]

        return filtered_stat, intersection

    def sample_univariate(self, shape, loc, scale, n_cells, distribution):
        """
        Create random samples from a model distribution.

        Parameters
        ----------
        shape : float or array_like of floats
            The shape parameters of the distribution.
        loc : float or array_like of floats
            The location parameters of the distribution.
        scale : float or array_like of floats
            The scale parameters of the distribution.
        n_cells : int
            The number of units to generate samples for.
        distribution : str
            The distribution to sample from. Supported distributions: "gamma", "vonmises", "skewnorm", "triang".

        Returns
        -------
        random_samples : ndarray
            The generated random samples from the specified distribution.

        Raises
        ------
        ValueError
            If the specified distribution is not supported.
        """

        if distribution not in ["gamma", "vonmises", "skewnorm", "triang"]:
            raise ValueError(
                f"Distribution '{distribution}' not supported, aborting..."
            )

        # Check if any of the shape, loc, scale parameters are np.nan
        # If so, set random_samples to np.nan and return
        if np.isnan(shape) or np.isnan(loc) or np.isnan(scale):
            random_samples = np.nan * np.ones(n_cells)
            return random_samples

        match distribution:
            # Continuous probability distribution on the positive real line

            case "gamma":
                random_samples = stats.gamma.rvs(
                    a=shape, loc=loc, scale=scale, size=n_cells, random_state=None
                )

            # Continuous probability distribution on the circle
            case "vonmises":
                random_samples = stats.vonmises.rvs(
                    kappa=shape, loc=loc, scale=scale, size=n_cells, random_state=None
                )

            # Skewed normal distribution
            case "skewnorm":
                random_samples = stats.skewnorm.rvs(
                    a=shape, loc=loc, scale=scale, size=n_cells, random_state=None
                )

            # Triangular distribution when min max mean median and sd is available in literature
            case "triang":
                random_samples = stats.triang.rvs(
                    c=shape, loc=loc, scale=scale, size=n_cells, random_state=None
                )

        return random_samples

    def sample_multivariate(self, df, n_cells):

        means = df["means"].values
        cov_column_names = df.columns[df.columns.str.startswith("cov_")]

        # Assert column order = row order to get correct covariance matrix
        ordered_columns = []
        for idx_str in df.index:
            for col in cov_column_names:
                if idx_str in col:
                    ordered_columns.append(col)
                    break

        cov_matrix = df.loc[:, ordered_columns].values

        samples = np.random.multivariate_normal(means, cov_matrix, size=n_cells)

        multivariate_samples_df = pd.DataFrame(samples, columns=df.index)

        if not all(multivariate_samples_df.columns == df.index):
            raise ValueError("Covariance index and column name mismatch. Aborting...")

        return multivariate_samples_df


class GanglionCellBase(ABC, PrintableMixin):
    """
    Abstract base class for storing and processing data related to ganglion cell receptive field models.

    Attributes
    ----------
    n_units : int | None
        The number of units, computed later.
    um_per_pix : float | None
        Micrometers per pixel, computed later.
    pix_per_side : int | None
        Number of pixels per side, computed later.
    um_per_side : float | None
        Micrometers per side, computed later.
    img : np.ndarray | None
        Receptive field image, computed later.
    img_mask : np.ndarray | None
        Receptive field center mask, computed later.
    img_lu_pix : np.ndarray | None
        Left upper corner of the receptive field image in pixels, computed later.
    X_grid_cen_mm : np.ndarray | None
        X grid in millimeters, computed later.
    Y_grid_cen_mm : np.ndarray | None
        Y grid in millimeters, computed later.
    cones_to_gcs_weights : np.ndarray | None
        Weights mapping cones' noise to ganglion cells, necessary for models without bipolar cells.
    df : pd.DataFrame
        DataFrame containing parameters of the ganglion cell mosaic, with multiple columns depending on model type.

    Methods
    -------

    get_BK_parameter_names() -> list[str]
        Abstract method to retrieve parameter names.
    """

    def __init__(self) -> None:
        self.n_units: int | None = None
        self.um_per_pix: float | None = None
        self.pix_per_side: int | None = None
        self.um_per_side: float | None = None
        self.img: np.ndarray | None = None
        self.img_mask: np.ndarray | None = None
        self.img_lu_pix: np.ndarray | None = None
        self.X_grid_cen_mm: np.ndarray | None = None
        self.Y_grid_cen_mm: np.ndarray | None = None
        self.cones_to_gcs_weights: np.ndarray | None = None

        columns: list[str] = [
            # position
            "pos_ecc_mm",
            "pos_polar_deg",
            "xoc_pix",
            "yoc_pix",
            "ecc_group_idx",
            # size
            "gc_scaling_factors",
            "zoom_factor",
            "den_diam_um",
            "center_mask_area_mm2",
            "center_fit_area_mm2",
            # amplitude
            "ampl_c",
            "ampl_s",
            "ampl_c_norm",
            "ampl_s_norm",
            "relat_sur_ampl",
            "offset",
            "tonic_drive",
            "A",  # spike generation gain from Benardete & Kaplan
            "Mean",  # mean firing rate from Benardete & Kaplan
        ]
        self.df: pd.DataFrame = pd.DataFrame(columns=columns)

    @abstractmethod
    def get_BK_parameter_names(self) -> list[str]:
        pass


class GanglionCellParasol(GanglionCellBase):
    """
    A class to build parasol ganglion cells.

    Methods
    -------
    get_BK_parameter_names() -> list[str]
        Returns a list of parameter names specific to the Benardete & Kaplan (BK) parasol ganglion cell model.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_BK_parameter_names(self) -> list[str]:
        BK_parameter_names: list[str] = [
            "A",
            "NLTL",
            "NL",
            "TL",
            "HS",
            "T0",
            "Chalf",
            "D",
            "Mean",
        ]
        return BK_parameter_names


class GanglionCellMidget(GanglionCellBase):
    """
    A class to build midget ganglion cells.

    Methods
    -------
    get_BK_parameter_names() -> list[str]
        Returns a list of parameter names specific to the Benardete & Kaplan (BK) midget ganglion cell model.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_BK_parameter_names(self) -> list[str]:
        BK_parameter_names: list[str] = [
            "A_cen",
            "NLTL_cen",
            "NL_cen",
            "HS_cen",
            "TS_cen",
            "D_cen",
            "A_sur",
            "NLTL_sur",
            "NL_sur",
            "HS_sur",
            "TS_sur",
            "deltaNLTL_sur",
            "Mean",
        ]
        return BK_parameter_names


class DoGModelBase(ABC):
    """
    Abstract base class for building Difference of Gaussian (DoG) models.

    Parameters
    ----------
    ret : Retina instance.
        Includes necessary parameters for fetching receptive field data.
    fit : Fit
        Fit instance responsible for managing model fitting.
    retina_math : Any
        Instance for mathematical operations related to the retina model.
    """

    def __init__(self, ret: Any, fit: Any, retina_math: Any) -> None:
        self.ret: Any = ret
        self.fit: Any = fit
        self.retina_math: Any = retina_math

        dog_statistics = ret.experimental_archive["dog_statistics"]
        for key, value in dog_statistics.items():
            setattr(self, key, value)

    @abstractmethod
    def scale_to_mm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to scale receptive field data to millimeters.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the receptive field data.

        Returns
        -------
        pd.DataFrame
            Scaled DataFrame.
        """
        pass

    @abstractmethod
    def generate_fit_img(
        self, x_grid: np.ndarray, y_grid: np.ndarray, popt: np.ndarray
    ) -> np.ndarray:
        """
        Abstract method to generate a fit image using grid coordinates and optimized parameters.

        Parameters
        ----------
        x_grid : np.ndarray
            X coordinates of the grid.
        y_grid : np.ndarray
            Y coordinates of the grid.
        popt : np.ndarray
            Optimized parameters for the fit.

        Returns
        -------
        np.ndarray
            Generated fit image.
        """
        pass

    @abstractmethod
    def get_param_names(self, gc: Any) -> list[str]:
        """
        Abstract method to retrieve the parameter names for the ganglion cell model.

        Parameters
        ----------
        gc : Ganglion cell object.

        Returns
        -------
        list[str]
            List of parameter names.
        """
        pass

    @abstractmethod
    def transform_vae_dog_to_mm(
        self, df: pd.DataFrame, gc_df_in: pd.DataFrame, mm_per_pix: float
    ) -> pd.DataFrame:
        """
        Abstract method to transform VAE DoG data from pixels to millimeters.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the DoG data.
        gc_df_in : pd.DataFrame
            Input ganglion cell DataFrame.
        mm_per_pix : float
            Millimeters per pixel conversion factor.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame in millimeters.
        """
        pass

    def _add_center_fit_area_to_df(self, gc: Any) -> Any:
        """
        Adds the center fit area to the ganglion cell DataFrame.

        Parameters
        ----------
        gc : Ganglion cell object.

        Returns
        -------
        Any
            Updated ganglion cell object with the center fit area added.
        """
        gc.df["center_fit_area_mm2"] = np.pi * gc.df["semi_xc_mm"] * gc.df["semi_yc_mm"]
        return gc

    def _get_dd_in_um(self, gc: Any) -> Any:
        """
        Calculates the dendritic diameter in micrometers and adds it to the DataFrame.

        Parameters
        ----------
        gc : Ganglion cell object.

        Returns
        -------
        Any
            Updated ganglion cell object with dendritic diameter in micrometers.
        """
        den_diam_um_s = pd.Series(
            self.retina_math.ellipse2diam(
                gc.df["semi_xc_mm"].values * 1000,
                gc.df["semi_yc_mm"].values * 1000,
            )
        )
        gc.df["den_diam_um"] = den_diam_um_s
        return gc

    def _get_center_volume(self, gc: Any) -> np.ndarray:
        """
        Calculates the volume of the center of the receptive field in cubic millimeters.

        Parameters
        ----------
        gc : Ganglion cell object.

        Returns
        -------
        np.ndarray
            Array of center volumes in cubic millimeters.
        """
        cen_vol_mm3 = (
            2 * np.pi * gc.df["ampl_c"] * gc.df["semi_xc_mm"] * gc.df["semi_yc_mm"]
        )
        return cen_vol_mm3


class DoGModelEllipseFixed(DoGModelBase):
    """
    A class to build Difference of Gaussian (DoG) models with fixed ellipses.

    Methods
    -------
    scale_to_mm(df: pd.DataFrame, um_per_pixel: float) -> pd.DataFrame
        Scales the semi-major and semi-minor axes of ellipses from pixels to millimeters.

    generate_fit_img(x_grid: np.ndarray, y_grid: np.ndarray, popt: np.ndarray) -> np.ndarray
        Generates the fitted image for the ganglion cell model using grid coordinates and optimized parameters.

    get_param_names(gc: Any) -> Any
        Sets the parameter names, scaling, and zoom parameters for the ganglion cell model and returns the updated object.

    transform_vae_dog_to_mm(df: pd.DataFrame, gc_df_in: pd.DataFrame, mm_per_pix: float) -> pd.DataFrame
        Transforms VAE DoG model data from pixel to millimeter units, updating the DataFrame with additional parameters.
    """

    def __init__(self, ret: Any, fit: Any, retina_math: Any) -> None:
        super().__init__(ret, fit, retina_math)

    def scale_to_mm(self, df: pd.DataFrame, um_per_pixel: float) -> pd.DataFrame:
        """
        Scales the semi-major and semi-minor axes of ellipses from pixels to millimeters.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the receptive field data.
        um_per_pixel : float
            Micrometers per pixel for scaling.

        Returns
        -------
        pd.DataFrame
            Updated DataFrame with ellipses scaled to millimeters.
        """
        gc_scaling_factors = df["gc_scaling_factors"]

        # Scale semi_x to virtual pixels at its actual eccentricity
        semi_xc_pix_eccscaled = df["semi_xc_pix"] * gc_scaling_factors
        df["semi_xc_mm"] = semi_xc_pix_eccscaled * um_per_pixel / 1000

        # Scale semi_y to pixels at its actual eccentricity
        semi_yc_pix_eccscaled = df["semi_yc_pix"] * gc_scaling_factors
        df["semi_yc_mm"] = semi_yc_pix_eccscaled * um_per_pixel / 1000

        return df

    def generate_fit_img(
        self, x_grid: np.ndarray, y_grid: np.ndarray, popt: np.ndarray
    ) -> np.ndarray:
        """
        Generates the fitted image for the ganglion cell model using grid coordinates and optimized parameters.

        Parameters
        ----------
        x_grid : np.ndarray
            X coordinates of the grid.
        y_grid : np.ndarray
            Y coordinates of the grid.
        popt : np.ndarray
            Optimized parameters for the fit.

        Returns
        -------
        np.ndarray
            Generated fit image for the ganglion cell model.
        """
        gc_img_fitted = self.retina_math.DoG2D_fixed_surround((x_grid, y_grid), *popt)
        return gc_img_fitted

    def get_param_names(self, gc: Any) -> Any:
        """
        Sets the parameter names, scaling, and zoom parameters for the ganglion cell model.

        Parameters
        ----------
        gc : Ganglion cell object.

        Returns
        -------
        gc : Ganglion cell object
            Updated ganglion cell object with parameter names and scaling information.
        """
        gc.parameter_names = [
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
        gc.mm_scaling_params = ["semi_xc_pix", "semi_yc_pix"]
        gc.zoom_scaling_params = ["xoc_pix", "yoc_pix"]

        return gc

    def transform_vae_dog_to_mm(
        self, df: pd.DataFrame, gc_df_in: pd.DataFrame, mm_per_pix: float
    ) -> pd.DataFrame:
        """
        Transforms VAE DoG model data from pixel to millimeter units.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the VAE DoG model data.
        gc_df_in : pd.DataFrame
            Input ganglion cell DataFrame.
        mm_per_pix : float
            Millimeters per pixel conversion factor.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with ellipses and other parameters in millimeter units.
        """
        df["relat_sur_diam"] = gc_df_in["relat_sur_diam"]
        df["semi_xc_mm"] = gc_df_in["semi_xc_pix"] * mm_per_pix
        df["semi_yc_mm"] = gc_df_in["semi_yc_pix"] * mm_per_pix

        df["den_diam_um"] = self.retina_math.ellipse2diam(
            df["semi_xc_mm"].values * 1000, df["semi_yc_mm"].values * 1000
        )  # in micrometers

        df["orient_cen_rad"] = gc_df_in["orient_cen_rad"]
        df["xy_aspect_ratio"] = df["semi_yc_mm"] / df["semi_xc_mm"]

        return df

    def recalculate_ampl_s_from_relative_surround_volume(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Recalculates the surround amplitude based on the relative surround volume.
        """

        df["ampl_s"] = (
            df["relative_surround_volume"] * df["ampl_c"] / (df["relat_sur_diam"] ** 2)
        )

        return df


class DoGModelEllipseIndependent(DoGModelBase):
    """
    A class to build Difference of Gaussian (DoG) models with independent ellipses.

    Methods
    -------
    scale_to_mm(df: pd.DataFrame, um_per_pixel: float) -> pd.DataFrame
        Scales the semi-major and semi-minor axes of ellipses from pixels to millimeters for both center and surround.

    generate_fit_img(x_grid: np.ndarray, y_grid: np.ndarray, popt: np.ndarray) -> np.ndarray
        Generates the fitted image for the ganglion cell model using grid coordinates and optimized parameters.

    get_param_names(gc: Any) -> Any
        Sets the parameter names, scaling, and zoom parameters for the ganglion cell model and returns the updated object.

    transform_vae_dog_to_mm(df: pd.DataFrame, gc_df_in: pd.DataFrame, mm_per_pix: float) -> pd.DataFrame
        Transforms VAE DoG model data from pixel to millimeter units, updating the DataFrame with additional parameters.
    """

    def __init__(self, ret: Any, fit: Any, retina_math: Any) -> None:
        super().__init__(ret, fit, retina_math)

    def scale_to_mm(self, df: pd.DataFrame, um_per_pixel: float) -> pd.DataFrame:
        """
        Scales the semi-major and semi-minor axes of ellipses from pixels to millimeters for both center and surround.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the receptive field data.
        um_per_pixel : float
            Micrometers per pixel for scaling.

        Returns
        -------
        pd.DataFrame
            Updated DataFrame with ellipses scaled to millimeters.
        """
        gc_scaling_factors = df["gc_scaling_factors"]

        # Scale semi_x and semi_y for center
        semi_xc_pix_eccscaled = df["semi_xc_pix"] * gc_scaling_factors
        df["semi_xc_mm"] = semi_xc_pix_eccscaled * um_per_pixel / 1000

        semi_yc_pix_eccscaled = df["semi_yc_pix"] * gc_scaling_factors
        df["semi_yc_mm"] = semi_yc_pix_eccscaled * um_per_pixel / 1000

        # Scale semi_x and semi_y for surround
        semi_xs_pix_eccscaled = df["semi_xs_pix"] * gc_scaling_factors
        df["semi_xs_mm"] = semi_xs_pix_eccscaled * um_per_pixel / 1000

        semi_ys_pix_eccscaled = df["semi_ys_pix"] * gc_scaling_factors
        df["semi_ys_mm"] = semi_ys_pix_eccscaled * um_per_pixel / 1000

        return df

    def generate_fit_img(
        self, x_grid: np.ndarray, y_grid: np.ndarray, popt: np.ndarray
    ) -> np.ndarray:
        """
        Generates the fitted image for the ganglion cell model using grid coordinates and optimized parameters.

        Parameters
        ----------
        x_grid : np.ndarray
            X coordinates of the grid.
        y_grid : np.ndarray
            Y coordinates of the grid.
        popt : np.ndarray
            Optimized parameters for the fit.

        Returns
        -------
        np.ndarray
            Generated fit image for the ganglion cell model.
        """
        gc_img_fitted = self.retina_math.DoG2D_independent_surround(
            (x_grid, y_grid), *popt
        )
        return gc_img_fitted

    def get_param_names(self, gc: Any) -> Any:
        """
        Sets the parameter names, scaling, and zoom parameters for the ganglion cell model.

        Parameters
        ----------
        gc : Any
            Ganglion cell object.

        Returns
        -------
        gc
            Updated ganglion cell object with parameter names and scaling information.
        """
        gc.parameter_names = [
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
        gc.mm_scaling_params = [
            "semi_xc_pix",
            "semi_yc_pix",
            "semi_xs_pix",
            "semi_ys_pix",
        ]
        gc.zoom_scaling_params = ["xoc_pix", "yoc_pix", "xos_pix", "yos_pix"]

        return gc

    def transform_vae_dog_to_mm(
        self, df: pd.DataFrame, gc_df_in: pd.DataFrame, mm_per_pix: float
    ) -> pd.DataFrame:
        """
        Transforms VAE DoG model data from pixel to millimeter units.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the VAE DoG model data.
        gc_df_in : pd.DataFrame
            Input ganglion cell DataFrame.
        mm_per_pix : float
            Millimeters per pixel conversion factor.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with ellipses and other parameters in millimeter units.
        """
        df["semi_xs_mm"] = gc_df_in["semi_xs_pix"] * mm_per_pix
        df["semi_ys_mm"] = gc_df_in["semi_ys_pix"] * mm_per_pix
        df["orient_sur_rad"] = gc_df_in["orient_sur_rad"]
        df["xos_pix"] = gc_df_in["xos_pix"]
        df["yos_pix"] = gc_df_in["yos_pix"]

        df["semi_xc_mm"] = gc_df_in["semi_xc_pix"] * mm_per_pix
        df["semi_yc_mm"] = gc_df_in["semi_yc_pix"] * mm_per_pix
        df["den_diam_um"] = self.retina_math.ellipse2diam(
            df["semi_xc_mm"].values * 1000,
            df["semi_yc_mm"].values * 1000,
        )  # in micrometers

        df["orient_cen_rad"] = gc_df_in["orient_cen_rad"]

        df["xy_aspect_ratio"] = df["semi_yc_mm"] / df["semi_xc_mm"]

        return df

    def recalculate_ampl_s_from_relative_surround_volume(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Recalculates the surround amplitude based on the relative surround volume
        for independent center and surround ellipses.
        """
        df["ampl_s"] = (
            df["relative_surround_volume"]
            * (df["ampl_c"] * df["semi_xc_pix"] * df["semi_yc_pix"])
            / (df["semi_xs_pix"] * df["semi_ys_pix"])
        )

        return df


class DoGModelCircular(DoGModelBase):
    """
    A class to build Difference of Gaussian (DoG) models with circular shapes.

    Methods
    -------
    scale_to_mm(df: pd.DataFrame, um_per_pixel: float) -> pd.DataFrame
        Scales the radii of circular receptive fields from pixels to millimeters for both center and surround.

    generate_fit_img(x_grid: np.ndarray, y_grid: np.ndarray, popt: np.ndarray) -> np.ndarray
        Generates the fitted image for the ganglion cell model using grid coordinates and optimized parameters.

    get_param_names(gc: Any) -> Any
        Sets the parameter names, scaling, and zoom parameters for the ganglion cell model and returns the updated object.

    transform_vae_dog_to_mm(df: pd.DataFrame, gc_df_in: pd.DataFrame, mm_per_pix: float) -> pd.DataFrame
        Transforms VAE DoG model data from pixel to millimeter units, updating the DataFrame with additional parameters.
    """

    def __init__(self, ret: Any, fit: Any, retina_math: Any) -> None:
        super().__init__(ret, fit, retina_math)

    def _get_dd_in_um(self, gc: Any) -> Any:
        """
        Calculates the dendritic diameter in micrometers and adds it to the DataFrame.

        Parameters
        ----------
        gc : Ganglion cell object.

        Returns
        -------
        Any
            Updated ganglion cell object with dendritic diameter in micrometers.
        """
        gc.df["den_diam_um"] = gc.df["rad_c_mm"] * 2 * 1000
        return gc

    def _add_center_fit_area_to_df(self, gc: Any) -> Any:
        """
        Adds the center fit area to the ganglion cell DataFrame.

        Parameters
        ----------
        gc : Any
            Ganglion cell object.

        Returns
        -------
        Any
            Updated ganglion cell object with the center fit area added.
        """
        gc.df["center_fit_area_mm2"] = np.pi * gc.df["rad_c_mm"] ** 2
        return gc

    def _get_center_volume(self, gc: Any) -> np.ndarray:
        """
        Calculates the center volume of the receptive field in cubic millimeters.

        Parameters
        ----------
        gc : Ganglion cell object.

        Returns
        -------
        np.ndarray
            Array of center volumes in cubic millimeters.
        """
        cen_vol_mm3 = 2 * np.pi * gc.df["ampl_c"] * gc.df["rad_c_mm"] ** 2
        return cen_vol_mm3

    def scale_to_mm(self, df: pd.DataFrame, um_per_pixel: float) -> pd.DataFrame:
        """
        Scales the radii of circular receptive fields from pixels to millimeters for both center and surround.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the receptive field data.
        um_per_pixel : float
            Micrometers per pixel for scaling.

        Returns
        -------
        pd.DataFrame
            Updated DataFrame with radii scaled to millimeters.
        """
        gc_scaling_factors = df["gc_scaling_factors"]

        # Scale rad_c to virtual pixels at its actual eccentricity
        rad_c_pix_eccscaled = df["rad_c_pix"] * gc_scaling_factors
        df["rad_c_mm"] = rad_c_pix_eccscaled * um_per_pixel / 1000

        # Scale rad_s for surround
        rad_s_pix_eccscaled = df["rad_s_pix"] * gc_scaling_factors
        df["rad_s_mm"] = rad_s_pix_eccscaled * um_per_pixel / 1000

        return df

    def generate_fit_img(
        self, x_grid: np.ndarray, y_grid: np.ndarray, popt: np.ndarray
    ) -> np.ndarray:
        """
        Generates the fitted image for the ganglion cell model using grid coordinates and optimized parameters.

        Parameters
        ----------
        x_grid : np.ndarray
            X coordinates of the grid.
        y_grid : np.ndarray
            Y coordinates of the grid.
        popt : np.ndarray
            Optimized parameters for the fit.

        Returns
        -------
        np.ndarray
            Generated fit image for the ganglion cell model.
        """
        gc_img_fitted = self.retina_math.DoG2D_circular((x_grid, y_grid), *popt)
        return gc_img_fitted

    def get_param_names(self, gc: Any) -> Any:
        """
        Sets the parameter names, scaling, and zoom parameters for the ganglion cell model.

        Parameters
        ----------
        gc : Ganglion cell object.

        Returns
        -------
        Any
            Updated ganglion cell object with parameter names and scaling information.
        """
        gc.parameter_names = [
            "ampl_c",
            "xoc_pix",
            "yoc_pix",
            "rad_c_pix",
            "ampl_s",
            "rad_s_pix",
            "offset",
        ]
        gc.mm_scaling_params = ["rad_c_pix", "rad_s_pix"]
        gc.zoom_scaling_params = ["xoc_pix", "yoc_pix"]

        return gc

    def transform_vae_dog_to_mm(
        self, df: pd.DataFrame, gc_df_in: pd.DataFrame, mm_per_pix: float
    ) -> pd.DataFrame:
        """
        Transforms VAE DoG model data from pixel to millimeter units.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the VAE DoG model data.
        gc_df_in : pd.DataFrame
            Input ganglion cell DataFrame.
        mm_per_pix : float
            Millimeters per pixel conversion factor.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with radii and other parameters in millimeter units.
        """
        df["rad_c_mm"] = gc_df_in["rad_c_pix"] * mm_per_pix
        df["rad_s_mm"] = gc_df_in["rad_s_pix"] * mm_per_pix
        df["den_diam_um"] = df["rad_c_mm"] * 2 * 1000  # in micrometers

        return df

    def recalculate_ampl_s_from_relative_surround_volume(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Recalculates the surround amplitude based on the relative surround volume.
        """
        df["ampl_s"] = (
            df["relative_surround_volume"]
            * (df["ampl_c"] * df["rad_c_pix"] ** 2)
            / (df["rad_s_pix"] ** 2)
        )

        return df


class SpatialModelBase(ABC):
    """
    Abstract base class for building spatial receptive fields.

    Attributes
    ----------
    DoG_model : Any
        Difference of Gaussian model used to generate receptive fields.
    sampler : Any
        Distribution sampler for generating receptive fields.
    project_data : dict
        Stores project-related data.
    """

    def __init__(
        self,
        DoG_model: Any,
        distribution_sampler: Callable,
        retina_vae: Any,
        fit: Any,
        retina_math: Any,
        viz: Any,
    ) -> None:
        self.DoG_model: Any = DoG_model
        self.sampler: Callable = distribution_sampler
        self.retina_vae: Any = retina_vae
        self.fit: Any = fit
        self.retina_math: Any = retina_math
        self.viz: Any = viz
        self.project_data: dict = {}

    @abstractmethod
    def create(self) -> None:
        """
        Create spatial receptive fields.
        """
        pass

    def _apply_local_zoom_compensation(
        self, epps: float, pps: float, pix: float, zoom: float
    ) -> float:
        """
        Apply zoom compensation to pixel scaling, used when receptive fields are fit with experimental image grids.

        Parameters
        ----------
        epps : float
            Experimental pixels per side.
        pps : float
            Pixels per side in the grid.
        pix : float
            Pixel center position.
        zoom : float
            Zoom factor applied during receptive field generation.

        Returns
        -------
        float
            Scaled pixel center position.
        """
        pix_scaled = -zoom * ((epps / 2) - pix) + (pps / 2)
        return pix_scaled

    def _get_img_grid_from_selected_img_stack(
        self, ret: Any, gc: Any, mask: Any
    ) -> Any:
        """
        Get the image grid from the selected image stack.

        Parameters
        ----------
        gc : Any
            Ganglion cell object.
        mask : Any
            Mask to extract the grid from.

        Returns
        -------
        Any
            Extracted image grid.
        """

        rf_pix_y = mask.shape[1]
        rf_pix_x = mask.shape[2]

        Y_grid, X_grid = np.meshgrid(
            np.arange(rf_pix_y), np.arange(rf_pix_x), indexing="ij"
        )

        if isinstance(gc.um_per_pix, np.ndarray):
            um_per_pix = np.tile(
                gc.um_per_pix[:, np.newaxis, np.newaxis], (1, rf_pix_y, rf_pix_x)
            )
        else:
            um_per_pix = gc.um_per_pix

        mm_per_pix = um_per_pix / 1000
        _Y_grid = np.tile(Y_grid, (gc.n_units, 1, 1))
        _X_grid = np.tile(X_grid, (gc.n_units, 1, 1))

        Y_grid_local_mm = _Y_grid * mm_per_pix
        X_grid_local_mm = _X_grid * mm_per_pix

        _rf_lu_pix_x = np.tile(
            gc.img_lu_pix[:, 0, np.newaxis, np.newaxis], (1, rf_pix_y, rf_pix_x)
        )
        _rf_lu_pix_y = np.tile(
            gc.img_lu_pix[:, 1, np.newaxis, np.newaxis], (1, rf_pix_y, rf_pix_x)
        )

        _rf_lu_mm_x = _rf_lu_pix_x * mm_per_pix
        _rf_lu_mm_y = _rf_lu_pix_y * mm_per_pix

        X_grid_cen_mm = ret.whole_ret_lu_mm[0] + _rf_lu_mm_x + X_grid_local_mm
        Y_grid_cen_mm = ret.whole_ret_lu_mm[1] - _rf_lu_mm_y - Y_grid_local_mm

        # Rotate X_grid_cen_mm and Y_grid_cen_mm according to polar angle limits,
        # i.e. the rotation of the retina patch
        rot_deg = (ret.polar_lim_deg[1] + ret.polar_lim_deg[0]) / 2
        X_grid_rot_mm, Y_grid_rot_mm = self.DoG_model.retina_math.rotate_image_grids(
            X_grid_cen_mm, Y_grid_cen_mm, rot_deg, gc.n_units, rf_pix_x, rf_pix_y
        )

        X_grid_cen_mm = X_grid_rot_mm
        Y_grid_cen_mm = Y_grid_rot_mm

        return X_grid_cen_mm, Y_grid_cen_mm

    def _get_img_grid_mm(self, ret: Any, gc: Any) -> Any:
        """
        Get receptive field center x and y coordinate grids in millimeters for downstream distance calculations.

        Parameters
        ----------
        ret : Any
            Retina instance.
        gc : Any
            Ganglion cell object with the attributes `img_mask`, `um_per_pix`, `n_units`, and `img_lu_pix`.

        Returns
        -------
        Any
            Updated ganglion cell object with X and Y coordinate grids (X_grid_cen_mm and Y_grid_cen_mm) in millimeters.
        """

        # Get center mask grid
        gc.X_grid_cen_mm, gc.Y_grid_cen_mm = self._get_img_grid_from_selected_img_stack(
            ret, gc, gc.img_mask
        )

        # Get surround mask grid
        gc.X_grid_sur_mm, gc.Y_grid_sur_mm = self._get_img_grid_from_selected_img_stack(
            ret, gc, gc.img_mask_sur
        )

        return gc

    def _get_retina_corners(
        self, ret: Any, rot_deg: float, pol2cart: Callable[[float, float], np.ndarray]
    ) -> np.ndarray:
        """Calculate the four corners of the retina in mm using eccentricity and polar coordinates."""
        ecc_lim_mm = ret.ecc_lim_mm
        polar_lim_deg = ret.polar_lim_deg

        corners_mm = np.zeros((4, 2))
        corners_mm[0, :] = pol2cart(ecc_lim_mm[0], polar_lim_deg[1] - rot_deg)
        corners_mm[1, :] = pol2cart(ecc_lim_mm[0], polar_lim_deg[0] - rot_deg)
        corners_mm[2, :] = pol2cart(ecc_lim_mm[1], polar_lim_deg[0] - rot_deg)
        corners_mm[3, :] = pol2cart(ecc_lim_mm[1], polar_lim_deg[1] - rot_deg)

        return corners_mm

    def _calculate_retina_size(
        self, corners_mm: np.ndarray, mm_per_pix: float, pad_size_mm: float
    ) -> tuple[int, int, float, float]:
        """Calculate the pixel size of the retina image from its corner coordinates."""
        min_x_mm_im = np.min(corners_mm[:, 0]) - pad_size_mm
        max_x_mm_im = (
            np.max(corners_mm[:, 0]) + pad_size_mm * 1.5
        )  # extra space for convex periphery
        min_y_mm_im = np.min(corners_mm[:, 1]) - pad_size_mm
        max_y_mm_im = np.max(corners_mm[:, 1]) + pad_size_mm

        ret_pix_x = int(np.ceil((max_x_mm_im - min_x_mm_im) / mm_per_pix))
        ret_pix_y = int(np.ceil((max_y_mm_im - min_y_mm_im) / mm_per_pix))

        return ret_pix_x, ret_pix_y, min_x_mm_im, max_y_mm_im

    def _convert_to_pixel_positions(
        self,
        pos_ecc_mm: np.ndarray,
        pos_polar_deg: np.ndarray,
        rot_deg: float,
        min_x_mm_im: float,
        max_y_mm_im: float,
        mm_per_pix: float,
        pol2cart: Callable[
            [np.ndarray, np.ndarray, bool], tuple[np.ndarray, np.ndarray]
        ],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert polar coordinates to Cartesian and then to pixel coordinates."""
        x_mm, y_mm = pol2cart(pos_ecc_mm, pos_polar_deg - rot_deg, deg=True)

        # Convert to pixel coordinates
        x_pix_c = (x_mm - min_x_mm_im) / mm_per_pix
        y_pix_c = (max_y_mm_im - y_mm) / mm_per_pix

        return x_pix_c, y_pix_c

    def _apply_pixel_scaling(
        self, df: pd.DataFrame, apply_pix_scaler: bool, gc: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply pixel scaling if required, or return the original pixel positions."""
        y_pix = df["yoc_pix"].values.astype(float)
        x_pix = df["xoc_pix"].values.astype(float)

        if apply_pix_scaler:
            epps = gc.exp_pix_per_side
            pps = gc.pix_per_side
            zoom = df["zoom_factor"].values.astype(float)
            yoc_pix_scaled = self._apply_local_zoom_compensation(epps, pps, y_pix, zoom)
            xoc_pix_scaled = self._apply_local_zoom_compensation(epps, pps, x_pix, zoom)
        else:
            yoc_pix_scaled = y_pix
            xoc_pix_scaled = x_pix

        return xoc_pix_scaled, yoc_pix_scaled

    def _convert_center_positions(
        self,
        df: pd.DataFrame,
        rot_deg: float,
        min_x_mm_im: float,
        max_y_mm_im: float,
        mm_per_pix: float,
        apply_pix_scaler: bool,
        gc: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert ganglion cell center positions to pixel coordinates, and apply optional scaling."""

        # Step 1: Convert polar coordinates to pixel coordinates
        x_pix_c, y_pix_c = self._convert_to_pixel_positions(
            df["pos_ecc_mm"].values.astype(float),
            df["pos_polar_deg"].values.astype(float),
            rot_deg,
            min_x_mm_im,
            max_y_mm_im,
            mm_per_pix,
            self.DoG_model.retina_math.pol2cart,
        )

        # Step 2: Apply scaling to pixel positions
        xoc_pix_scaled, yoc_pix_scaled = self._apply_pixel_scaling(
            df, apply_pix_scaler, gc
        )

        return x_pix_c, y_pix_c, xoc_pix_scaled, yoc_pix_scaled

    def _place_gc_images_on_retina(
        self,
        df: pd.DataFrame,
        gc_img: np.ndarray,
        ret_img_pix: np.ndarray,
        pix_per_side: int,
        x_pix_c: np.ndarray,
        y_pix_c: np.ndarray,
        xoc_pix_scaled: np.ndarray,
        yoc_pix_scaled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Place the ganglion cell images on the full retina image."""
        gc_img_lu_pix = np.zeros((gc_img.shape[0], 2), dtype=int)

        for i in range(len(df)):
            y_pix_lu = int(np.round(y_pix_c[i] - yoc_pix_scaled[i]))
            x_pix_lu = int(np.round(x_pix_c[i] - xoc_pix_scaled[i]))
            ret_img_pix[
                y_pix_lu : y_pix_lu + pix_per_side,
                x_pix_lu : x_pix_lu + pix_per_side,
            ] += gc_img[i, :, :]
            gc_img_lu_pix[i, :] = [x_pix_lu, y_pix_lu]

        return ret_img_pix, gc_img_lu_pix

    def _get_full_retina_with_rf_images(
        self, ret: Any, gc: Any, gc_img: np.ndarray, apply_pix_scaler: bool = False
    ) -> tuple:
        """
        Build a full retina image containing all receptive fields and return updated retina
        and ganglion cell objects.

        Parameters
        ----------
        ret : Any
            Retina object.
        gc : Any
            Ganglion cell object.
        gc_img : np.ndarray
            Ganglion cell images to be laid onto retina. Eg mask and img separately, thus the distinct input from gc.

        Returns
        -------
        tuple
            Updated retina and ganglion cell objects, and the full retina image.
        """

        # Step 1: Calculate mm_per_pix and rotation
        mm_per_pix = gc.um_per_pix / 1000
        rot_deg = np.mean(ret.polar_lim_deg)

        # Step 2: Calculate retina corners and dimensions
        corners_mm = self._get_retina_corners(
            ret, rot_deg, self.DoG_model.retina_math.pol2cart
        )
        pad_size_mm = gc.pix_per_side * mm_per_pix
        ret_pix_x, ret_pix_y, min_x_mm_im, max_y_mm_im = self._calculate_retina_size(
            corners_mm, mm_per_pix, pad_size_mm
        )

        # Initialize the retina image
        ret_img_pix = np.zeros((ret_pix_y, ret_pix_x))

        # Step 3: Convert center positions and apply scaling
        x_pix_c, y_pix_c, xoc_pix_scaled, yoc_pix_scaled = (
            self._convert_center_positions(
                gc.df,
                rot_deg,
                min_x_mm_im,
                max_y_mm_im,
                mm_per_pix,
                apply_pix_scaler,
                gc,
            )
        )

        # Step 4: Place ganglion cell images on the retina
        ret_img_pix, gc_img_lu_pix = self._place_gc_images_on_retina(
            gc.df,
            gc_img,
            ret_img_pix,
            gc.pix_per_side,
            x_pix_c,
            y_pix_c,
            xoc_pix_scaled,
            yoc_pix_scaled,
        )

        # Update ganglion cell object with the positions
        gc.img_lu_pix = gc_img_lu_pix
        ret.whole_ret_lu_mm = np.array([min_x_mm_im, max_y_mm_im])

        return ret, gc, ret_img_pix

    def _generate_center_masks(self, ret: Any, gc: Any) -> Any:
        """
        Extract contours around the receptive field center based on the mask threshold.

        Parameters
        ----------
        ret : Any
            Retina instance.
        gc : Any
            Ganglion cell object with the attribute `img`.

        Returns
        -------
        Any
            Updated ganglion cell object with center masks added.
        """
        img_stack = gc.img
        mask_threshold = ret.mask_threshold
        assert 0 <= mask_threshold <= 1, "mask_threshold must be between 0 and 1."

        masks = []
        for img in img_stack:
            max_val = np.max(img)
            mask = img >= max_val * mask_threshold
            labeled_mask, _ = ndimage.label(mask)
            max_label = labeled_mask[np.unravel_index(np.argmax(img), img.shape)]
            mask = labeled_mask == max_label
            masks.append(mask)

        gc.img_mask = np.array(masks)
        return gc

    def _generate_surround_masks(self, ret: Any, gc: Any) -> Any:
        """
        Extract contours around the receptive field surround based on the mask threshold.

        Parameters
        ----------
        ret : Any
            Retina instance.
        gc : Any
            Ganglion cell object with the attribute `img` containing the receptive fields.

        Returns
        -------
        Any
            Updated ganglion cell object with surround masks added.
        """
        img_stack = gc.img
        mask_threshold = ret.mask_threshold
        assert 0 <= mask_threshold <= 1, "mask_threshold must be between 0 and 1."

        masks = []
        for img in img_stack:
            min_val = np.min(img)
            mask = img <= min_val * mask_threshold

            labeled_mask, _ = ndimage.label(mask)
            min_label = labeled_mask[np.unravel_index(np.argmin(img), img.shape)]
            mask = labeled_mask == min_label

            masks.append(mask)
        gc.img_mask_sur = np.array(masks)

        return gc

    def _add_center_mask_area_to_df(self, gc: Any) -> Any:
        """
        Add the area of the center mask to the ganglion cell DataFrame in mm^2.

        Parameters
        ----------
        gc : Any
            Ganglion cell object with attributes `img_mask` and `um_per_pix`.

        Returns
        -------
        Any
            Updated ganglion cell object with center mask areas added to the DataFrame.
        """
        center_mask_area_mm2 = np.sum(gc.img_mask, axis=(1, 2)) * gc.um_per_pix**2 / 1e6
        gc.df["center_mask_area_mm2"] = center_mask_area_mm2
        return gc

    def _update_vae_gc_df(
        self,
        ret: Any,
        gc: Any,
        gc_df_in: pd.DataFrame,
    ) -> Any:
        """
        Update the ganglion cell DataFrame with new values in millimeter units.

        Parameters
        ----------
        ret : Any
            Retina object containing retinal parameters.
        gc : Any
            GanglionCell object containing ganglion cell data.
        gc_df_in : pd.DataFrame
            DataFrame containing new ganglion cell data.

        Returns
        -------
        gc : Any
            Updated GanglionCell object with updated DataFrame.
        """
        _df = gc_df_in.reindex(columns=gc.df.columns)
        mm_per_pix = gc.um_per_pix / 1000

        # Calculate the eccentricity and polar angle of the receptive field center
        xoc_mm = gc_df_in.xoc_pix * mm_per_pix
        yoc_mm = gc_df_in.yoc_pix * mm_per_pix
        rf_lu_mm = gc.img_lu_pix * mm_per_pix

        # Calculate positions
        x_mm = ret.whole_ret_lu_mm[0] + rf_lu_mm[:, 0] + xoc_mm
        y_mm = ret.whole_ret_lu_mm[1] - rf_lu_mm[:, 1] - yoc_mm
        (pos_ecc_mm, pos_polar_deg) = self.retina_math.cart2pol(
            x_mm.values, y_mm.values
        )

        _df["pos_ecc_mm"] = pos_ecc_mm
        _df["pos_polar_deg"] = pos_polar_deg

        _df = self.DoG_model.transform_vae_dog_to_mm(_df, gc_df_in, mm_per_pix)

        _df["ampl_c"] = gc_df_in["ampl_c"]
        _df["ampl_s"] = gc_df_in["ampl_s"]

        _df["xoc_pix"] = gc_df_in["xoc_pix"]
        _df["yoc_pix"] = gc_df_in["yoc_pix"]

        gc.df = _df

        return gc


class SpatialModelDOG(SpatialModelBase):
    """
    A class to build spatial receptive fields using the DOG model.
    """

    def __init__(
        self, DoG_model, distribution_sampler, retina_vae, fit, retina_math, viz
    ):
        super().__init__(
            DoG_model, distribution_sampler, retina_vae, fit, retina_math, viz
        )

    def _generate_DoG_with_rf_from_literature(
        self, ret: Any, gc: Any, experimental_metadata: dict
    ) -> Any:
        """
        Generate Difference of Gaussians (DoG) model with dendritic field sizes from literature.

        Parameters:
        ret: Object containing retina information
        gc: Object for ganglion cell data
        experimental_metadata: Metadata containing image scaling information

        Returns:
        Updated gc object with spatial parameters in millimeters.
        """
        n_cells = len(gc.df)
        stat = self.DoG_model.exp_univariate_stat
        spatial_df = stat[stat["domain"] == "spatial"]

        for param_name, row in spatial_df.iterrows():
            shape, loc, scale, distribution, _ = row
            gc.df[param_name] = self.sampler.sample_univariate(
                shape, loc, scale, n_cells, distribution
            )

        if ret.fit_statistics == "multivariate":
            covariances_of_interest = self._get_spatial_covariances_of_interest()
            available_stat = self.DoG_model.spatial_multivariate_stat
            filtered_stat, intersection = self.sampler.filter_stat(
                available_stat, covariances_of_interest
            )
            self.project_data["spatial_covariances_of_interest"] = intersection
            samples = self.sampler.sample_multivariate(filtered_stat, n_cells)
            missing_columns = [
                col for col in samples.columns if col not in gc.df.columns
            ]
            gc.df[missing_columns] = np.nan
            gc.df.update(samples)

        # fix offset at zero
        gc.df["offset"] = 0.0

        # Safequard for polarity inversion (ON becoming OFF and vice versa)
        gc.df["relative_surround_volume"] = np.clip(
            gc.df["relative_surround_volume"], a_min=0, a_max=0.99
        )

        gc.df = self.DoG_model.recalculate_ampl_s_from_relative_surround_volume(gc.df)

        um_per_pixel = experimental_metadata["data_microm_per_pix"]
        gc.df = self.DoG_model.scale_to_mm(gc.df, um_per_pixel)

        return gc

    def _get_gc_fit_img(self, gc: Any) -> Any:
        """
        Generate receptive field images from the DOG model parameters.

        Parameters:
        gc: Ganglion cell object containing parameters for image generation

        Returns:
        Updated gc object with generated receptive field images.
        """
        n_units = gc.n_units
        pix_per_side = gc.pix_per_side
        exp_pix_per_side = gc.exp_pix_per_side
        pix_scaler = pix_per_side / exp_pix_per_side
        mm_per_pix = gc.um_per_pix / 1000
        grid_indices = np.linspace(0, pix_per_side - 1, pix_per_side)
        y_grid, x_grid = np.meshgrid(grid_indices, grid_indices, indexing="ij")

        gc = self.DoG_model.get_param_names(gc)
        parameters = gc.df[gc.parameter_names].values.astype(float)
        parameters_scaled = parameters.copy()
        pix_per_mm = 1 / mm_per_pix
        zoom = gc.df["zoom_factor"].astype(float)

        for idx, param_name in enumerate(gc.parameter_names):
            if param_name in gc.mm_scaling_params:
                parameters_scaled[:, idx] = gc.df[param_name[:-3] + "mm"] * pix_per_mm
            elif param_name in gc.zoom_scaling_params:
                parameters_scaled[:, idx] = self._apply_local_zoom_compensation(
                    exp_pix_per_side, pix_per_side, parameters[:, idx], zoom
                )
                parameters[:, idx] = parameters_scaled[:, idx]

        gc.df[gc.parameter_names] = parameters
        gc_fit_img = np.zeros((n_units, pix_per_side, pix_per_side))

        for idx in range(n_units):
            popt = parameters_scaled[idx, :]
            gc_img_fitted = self.DoG_model.generate_fit_img(x_grid, y_grid, popt)

            gc_fit_img[idx, :, :] = gc_img_fitted.reshape(pix_per_side, pix_per_side)

        gc.img = gc_fit_img

        return gc

    def _get_spatial_covariances_of_interest(self) -> List[str]:
        """
        Get the list of spatial covariances of interest.

        Returns:
        List of spatial covariances for multivariate fitting.
        """
        return [
            "semi_xc_pix",
            "semi_yc_pix",
            "ampl_s",
            "semi_xs_pix",
            "semi_ys_pix",
            "rad_c_pix",
            "relative_surround_volume",
        ]

    def create(self, ret: Any, gc: Any) -> Tuple[Any, Any, Any]:
        """
        Create spatial receptive fields using the DOG model.

        Parameters:
        ret: Object containing retina information
        gc: Object for ganglion cell data

        Returns:
        Tuple containing updated retina, ganglion cell data, and visualization image.
        """
        experimental_metadata = ret.experimental_archive["experimental_metadata"]

        # Step 1: Generate DoG parameters
        gc = self._generate_DoG_with_rf_from_literature(ret, gc, experimental_metadata)

        # Step 2: Calculate dendritic diameter
        gc = self.DoG_model._get_dd_in_um(gc)

        print("\nGenerating RF images for DOG model...")

        # Step 3: Generate RF images
        gc = self._get_gc_fit_img(gc)

        # Step 4: Generate center masks
        gc = self._generate_center_masks(ret, gc)
        ret, gc, ret.whole_ret_img = self._get_full_retina_with_rf_images(
            ret, gc, gc.img
        )

        viz_whole_ret_img = ret.whole_ret_img

        ret, gc, ret.whole_ret_img_mask = self._get_full_retina_with_rf_images(
            ret, gc, gc.img_mask, apply_pix_scaler=False
        )

        # Step 5: Apply RF repulsion (if applicable)
        print("\nApplying repulsion between the receptive fields...")
        ret, gc = apply_rf_repulsion(ret, gc, self.viz)

        # 6) Redo the good fits for final statistics
        print("\nFinal DoG fit to generated rfs...")
        self.DoG_model.fit.client(
            ret.gc_type,
            ret.response_type,
            fit_type="generated",
            dog_model_type=ret.dog_model_type,
            spatial_data=gc.img,
            um_per_pix=gc.um_per_pix,
            mark_outliers_bad=False,
        )
        self.gen_spat_cen_sd = self.fit.receptive_field_sd.center_sd
        self.gen_spat_sur_sd = self.fit.receptive_field_sd.surround_sd
        (
            self.gen_stat_df,
            _gc_vae_df,
            final_good_idx,
        ) = self.fit.get_generated_DoG_fits()

        # 7) Update gc.df to include new positions and DoG fits after repulsion
        print("\nUpdating ganglion cell dataframe...")
        gc = self._update_vae_gc_df(ret, gc, _gc_vae_df)

        # Check that all fits are good after repulsion.
        if gc.n_units != np.sum(final_good_idx):
            print("Removing bad fits from final generated data...")
            gc.img = gc.img[final_good_idx]
            gc.img_mask = gc.img_mask[final_good_idx]
            gc.img_lu_pix = gc.img_lu_pix[final_good_idx]
            gc.df = gc.df.iloc[final_good_idx]
            # Reset gc.df index
            gc.df.reset_index(drop=True, inplace=True)
            gc.n_units = len(gc.df)

        # 8) Get final center masks for the generated spatial rfs
        print("\nGetting final masked rfs and retina...")
        gc = self._generate_center_masks(ret, gc)
        gc = self._generate_surround_masks(ret, gc)
        # Add center mask area (mm^2) to gc_vae_df for visualization
        gc = self._add_center_mask_area_to_df(gc)

        # 9) Sum separate rf center masks onto one retina pixel matrix.
        ret, gc, ret.whole_ret_img_mask = self._get_full_retina_with_rf_images(
            ret, gc, gc.img_mask
        )
        gc = self._get_img_grid_mm(ret, gc)
        return ret, gc, viz_whole_ret_img


class SpatialModelVAE(SpatialModelBase):
    """
    SpatialModelVAE builds spatial receptive fields using a Variational Autoencoder (VAE) model.

    This class utilizes a trained VAE model to generate spatial receptive fields (RFs) for ganglion cells (GCs),
    ensuring that they have good Difference-of-Gaussians (DoG) fits. It handles the generation of RFs,
    rescaling, fitting, and updates to the ganglion cell data.

    Parameters
    ----------
    DoG_model : Any
        Model used for fitting Difference-of-Gaussians to RFs.
    distribution_sampler : Callable
        A function or callable object used to sample distributions.
    retina_vae : Any
        Trained VAE model for generating RFs.
    fit : Any
        An object for fitting models to data.
    retina_math : Any
        An object providing mathematical utilities for retinal computations.
    viz : Any
        Visualization object for plotting and displaying data.

    Attributes
    ----------
    project_data : dict
        Dictionary for storing data during the RF generation process.
    """

    def __init__(
        self,
        DoG_model: Any,
        distribution_sampler: Callable,
        retina_vae: Any,
        fit: Any,
        retina_math: Any,
        viz: Any,
    ) -> None:
        """
        Initialize the SpatialModelVAE.

        Parameters
        ----------
        DoG_model : Any
            Model used for fitting Difference-of-Gaussians to RFs.
        distribution_sampler : Callable
            A function or callable object used to sample distributions.
        retina_vae : Any
            Trained VAE model for generating RFs.
        fit : Any
            An object for fitting models to data.
        retina_math : Any
            An object providing mathematical utilities for retinal computations.
        viz : Any
            Visualization object for plotting and displaying data.
        """
        super().__init__(
            DoG_model, distribution_sampler, retina_vae, fit, retina_math, viz
        )

    def _get_resampled_scaled_gc_img(
        self,
        rfs: np.ndarray,
        pix_per_side: int,
        zoom_factor: np.ndarray,
    ) -> np.ndarray:
        """
        Resample and scale ganglion cell images.

        Parameters
        ----------
        rfs : np.ndarray
            Array of receptive field images, shape (n_units, H, W).
        pix_per_side : int
            Target number of pixels per side after resampling.
        zoom_factor : np.ndarray
            Array of zoom factors for each RF, shape (n_units,).

        Returns
        -------
        img_upsampled : np.ndarray
            Array of upsampled images, shape (n_units, pix_per_side, pix_per_side).
        """
        # Resample all images to new img stack. Use scipy.ndimage.zoom,
        img_upsampled = np.zeros((len(rfs), pix_per_side, pix_per_side))

        orig_pix_per_side = rfs[0, ...].shape[0]
        is_even = (pix_per_side - orig_pix_per_side) % 2 == 0

        if is_even:
            padding = int((pix_per_side - orig_pix_per_side) / 2)
            crop_length = pix_per_side / 2
        else:
            padding = (
                int((pix_per_side - orig_pix_per_side) / 2),
                int((pix_per_side - orig_pix_per_side) / 2) + 1,
            )  # (before, after)
            crop_length = (pix_per_side - 1) / 2

        for i, img in enumerate(rfs):
            # Pad the image with zeros to achieve the new dimensions
            img_padded = np.pad(
                img, pad_width=padding, mode="constant", constant_values=0
            )

            # Upsample the padded image
            img_temp = ndimage.zoom(
                img_padded, zoom_factor[i], grid_mode=False, order=3
            )
            # Correct for uneven dimensions after upsampling
            if not is_even:
                img_temp = ndimage.shift(img_temp, 0.5)

            # Crop the upsampled image to the new dimensions
            if is_even:
                img_cropped = img_temp[
                    int(img_temp.shape[0] / 2 - crop_length) : int(
                        img_temp.shape[0] / 2 + crop_length
                    ),
                    int(img_temp.shape[1] / 2 - crop_length) : int(
                        img_temp.shape[1] / 2 + crop_length
                    ),
                ]
            else:
                img_cropped = img_temp[
                    int(img_temp.shape[0] / 2 - crop_length) : int(
                        img_temp.shape[0] / 2 + crop_length + 1
                    ),
                    int(img_temp.shape[1] / 2 - crop_length) : int(
                        img_temp.shape[1] / 2 + crop_length + 1
                    ),
                ]

            img_upsampled[i] = img_cropped
        return img_upsampled

    def _get_vae_imgs_with_good_fits(self, ret: Any, gc: Any, retina_vae: Any) -> Any:
        """
        Generate eccentricity-scaled spatial receptive fields from the VAE model with good DoG fits.

        Parameters
        ----------
        ret : Any
            Retina object containing retinal parameters.
        gc : Any
            GanglionCell object containing ganglion cell data.
        retina_vae : Any
            Variational Autoencoder model for creating spatial receptive fields.

        Returns
        -------
        gc : Any
            Updated GanglionCell object with new RF images.

        Notes
        -----
        The VAE generates a number of RFs larger than the required samples to account for outliers that are not accepted.
        """
        nsamples = gc.n_units

        # Get samples. We take 50% extra samples to cover the bad fits
        nsamples_extra = int(nsamples * 1.5)  # 50% extra to account for outliers
        img_processed_extra, img_raw_extra = self._get_generated_rfs(
            retina_vae, n_samples=nsamples_extra
        )

        idx_to_process = np.arange(nsamples)
        gc_vae_img = np.zeros((nsamples, gc.pix_per_side, gc.pix_per_side))
        available_idx_mask = np.ones(nsamples_extra, dtype=bool)
        available_idx_mask[idx_to_process] = False
        img_to_resample = img_processed_extra[idx_to_process, :, :]
        good_mask_compiled = np.zeros(nsamples, dtype=bool)
        _gc_vae_df = pd.DataFrame(
            index=np.arange(nsamples),
            columns=["xoc_pix", "yoc_pix"],
        )
        zoom_factor = gc.df["zoom_factor"].values
        # Loop until there are no bad fits
        for _ in range(100):
            # Upsample according to smallest rf diameter
            img_after_resample = self._get_resampled_scaled_gc_img(
                img_to_resample[idx_to_process, :, :],
                gc.pix_per_side,
                zoom_factor[idx_to_process],
            )

            # Fit elliptical gaussians to the img[idx_to_process]
            # This is dependent metrics, not affecting the spatial RFs
            # other than quality assurance (below)
            # Fixed DoG model type excludes the model effect on unit selection
            # Note that this fits the img_after_resample and thus the
            # xoc_pix and yoc_pix are veridical for the upsampled data.
            self.fit.client(
                ret.gc_type,
                ret.response_type,
                fit_type="generated",
                dog_model_type="ellipse_fixed",
                spatial_data=img_after_resample,
                um_per_pix=gc.um_per_pix,
                mark_outliers_bad=True,
            )
            # Discard bad fits
            good_idx_this_iter = self.fit.get_good_data_idx()
            good_idx_generated = idx_to_process[good_idx_this_iter]
            # Save the good RFs
            gc_vae_img[good_idx_generated, :, :] = img_after_resample[
                good_idx_this_iter, :, :
            ]

            good_df = self.fit.get_good_data_df()
            _gc_vae_df.loc[good_idx_generated, ["yoc_pix", "xoc_pix"]] = good_df.loc[
                :, ["yoc_pix", "xoc_pix"]
            ].values

            good_mask_compiled[good_idx_generated] = True

            # Update idx_to_process for the loop
            idx_to_process = np.setdiff1d(idx_to_process, good_idx_generated)
            print(f"Bad fits to replace by new RFs: {idx_to_process}")

            if len(idx_to_process) > 0:
                for this_miss in idx_to_process:
                    # Get next possible replacement index
                    this_replace = np.where(available_idx_mask)[0][0]
                    # Replace the bad RF with the reserve RF
                    img_to_resample[this_miss, :, :] = img_processed_extra[
                        this_replace, :, :
                    ]
                    # Remove replacement from available indices
                    available_idx_mask[this_replace] = False
            else:
                break

        # For visualization of the construction process early steps
        good_idx_compiled = np.where(good_mask_compiled)[0]
        assert (
            len(good_idx_compiled) == nsamples
        ), "Bad fit loop did not remove all bad fits, aborting..."

        self.project_data["gen_spat_img"] = {
            "img_processed": img_processed_extra[good_idx_compiled, :, :],
            "img_raw": img_raw_extra[good_idx_compiled, :, :],
        }

        gc.df.loc[:, ["xoc_pix", "yoc_pix"]] = _gc_vae_df
        gc.img = gc_vae_img

        return gc

    def _get_generated_rfs(
        self, retina_vae: Any, n_samples: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the spatial data generated by the retina VAE.

        Parameters
        ----------
        retina_vae : Any
            A RetinaVAE object.
        n_samples : int, optional
            Number of samples to generate (default is 10).

        Returns
        -------
        img_flipped : np.ndarray
            Processed image stack (median removed, flipped to positive max), shape (n_samples, img_size, img_size).
        img_reshaped : np.ndarray
            Raw image stack, shape (n_samples, img_size, img_size).
        """
        # 1. Make a probability density function of the latent space
        vae_latent_stats = retina_vae.vae_latent_stats

        # Make a probability density function of the vae_latent_stats
        # Both uniform and normal distributions during learning are sampled
        # using Gaussian KDE estimate.
        latent_pdf = stats.gaussian_kde(vae_latent_stats.T)

        # 2. Sample from the pdf
        latent_samples = torch.tensor(latent_pdf.resample(n_samples).T).to(
            retina_vae.device
        )
        # Change the dtype to float32
        latent_samples = latent_samples.type(torch.float32)
        latent_dim = retina_vae.latent_dim

        self.project_data["gen_latent_space"] = {
            "samples": latent_samples.to("cpu").numpy(),
            "dim": latent_dim,
            "data": vae_latent_stats,
        }

        # 3. Decode the samples
        img_stack_np = retina_vae.vae.decoder(latent_samples)

        # Reshape to (n_samples, img_size, img_size)
        img_reshaped = np.reshape(
            img_stack_np.detach().cpu().numpy(),
            (n_samples, img_stack_np.shape[2], img_stack_np.shape[3]),
        )

        # Remove median and flip images if necessary
        medians = np.median(img_reshaped, axis=(1, 2))
        img_median_removed = img_reshaped - medians[:, None, None]

        img_flipped = img_median_removed
        for i in range(img_flipped.shape[0]):
            if abs(np.min(img_flipped[i])) > abs(np.max(img_flipped[i])):
                img_flipped[i] = -img_flipped[i]

        return img_flipped, img_reshaped

    def create(self, ret: Any, gc: Any) -> Tuple[Any, Any, np.ndarray]:
        """
        Create spatial receptive fields using the VAE model.

        Parameters
        ----------
        ret : Any
            Retina object containing retinal parameters.
        gc : Any
            GanglionCell object containing ganglion cell data.

        Returns
        -------
        ret : Any
            Updated Retina object.
        gc : Any
            Updated GanglionCell object with new RFs.
        viz_whole_ret_img : np.ndarray
            Visualization image of the whole retina.
        """
        # Endow units with spatial receptive fields using the generative variational autoencoder model

        # 1) Get variational autoencoder to generate receptive fields
        print("\nGetting VAE model...")
        # self.vae_latent_stats = ret.experimental_archive["vae_statistics"]

        self.retina_vae.client()

        # 2) "Bad fit loop", provides eccentricity-scaled vae rfs with good DoG fits (error < 3SD from mean).
        print("\nBad fit loop: Generating receptive fields with good DoG fits...")
        gc = self._get_vae_imgs_with_good_fits(ret, gc, self.retina_vae)

        # 3) Get center masks
        gc = self._generate_center_masks(ret, gc)

        viz_gc_vae_img = gc.img.copy()
        viz_gc_vae_img_mask = gc.img_mask.copy()

        # 4) Sum separate rf images onto one retina pixel matrix.
        ret, gc, ret.whole_ret_img = self._get_full_retina_with_rf_images(
            ret, gc, gc.img, apply_pix_scaler=True
        )
        viz_whole_ret_img = ret.whole_ret_img

        # 5) Apply repulsion adjustment to the receptive fields.
        print("\nApplying repulsion between the receptive fields...")
        ret, gc = apply_rf_repulsion(ret, gc, self.viz)

        # 6) Redo the good fits for final statistics
        print("\nFinal DoG fit to generated rfs...")
        self.DoG_model.fit.client(
            ret.gc_type,
            ret.response_type,
            fit_type="generated",
            dog_model_type=ret.dog_model_type,
            spatial_data=gc.img,
            um_per_pix=gc.um_per_pix,
            mark_outliers_bad=False,
        )
        self.gen_spat_cen_sd = self.fit.receptive_field_sd.center_sd
        self.gen_spat_sur_sd = self.fit.receptive_field_sd.surround_sd
        (
            self.gen_stat_df,
            _gc_vae_df,
            final_good_idx,
        ) = self.fit.get_generated_DoG_fits()

        # 7) Update gc.df to include new positions and DoG fits after repulsion
        print("\nUpdating ganglion cell dataframe...")
        gc = self._update_vae_gc_df(ret, gc, _gc_vae_df)

        # Check that all fits are good after repulsion.
        if gc.n_units != np.sum(final_good_idx):
            print("Removing bad fits from final generated data...")
            gc.img = gc.img[final_good_idx]
            gc.img_mask = gc.img_mask[final_good_idx]
            gc.img_lu_pix = gc.img_lu_pix[final_good_idx]
            gc.df = gc.df.iloc[final_good_idx]
            # Reset gc.df index
            gc.df.reset_index(drop=True, inplace=True)
            gc.n_units = len(gc.df)

        # 8) Get final center masks for the generated spatial rfs
        print("\nGetting final masked rfs and retina...")
        gc = self._generate_center_masks(ret, gc)
        gc = self._generate_surround_masks(ret, gc)

        # Add center mask area (mm^2) to gc_vae_df for visualization
        gc = self._add_center_mask_area_to_df(gc)

        # 9) Sum separate rf center masks onto one retina pixel matrix.
        ret, gc, ret.whole_ret_img_mask = self._get_full_retina_with_rf_images(
            ret, gc, gc.img_mask
        )
        gc = self._get_img_grid_mm(ret, gc)

        # 10) Set vae data to project_data for later visualization
        self.project_data["retina_vae"] = self.retina_vae

        self.project_data["gen_rfs"] = {
            "gc_vae_img": viz_gc_vae_img,
            "gc_vae_img_mask": viz_gc_vae_img_mask,
            "final_gc_vae_img": gc.img,
            "centre_of_mass_x": gc.df["com_x_pix"],
            "centre_of_mass_y": gc.df["com_y_pix"],
        }

        return ret, gc, viz_whole_ret_img


class TemporalModelBase(ABC):
    """
    Abstract base class for building temporal models of retinal ganglion cells.

    This class provides a framework for creating temporal models, connecting units according to the model type,
    and performing various operations related to the temporal aspects of the retinal model.

    Attributes
    ----------
    ganglion_cell : GanglionCell
        The ganglion cell model instance.
    DoG_model : DoGModel
        The Difference of Gaussians (DoG) model instance.
    sampler : DistributionSampler
        The sampler used for sampling from distributions.
    retina_math : RetinaMath
        An instance containing mathematical functions and utilities.
    project_data : dict
        Dictionary to store project-specific data.
    """

    def __init__(
        self,
        ganglion_cell: Any,
        DoG_model: Any,
        distribution_sampler: DistributionSampler,
        retina_math: RetinaMath,
        device: str = "cpu",
    ):
        """
        Initialize the TemporalModelBase instance.

        Parameters
        ----------
        ganglion_cell : GanglionCell
            The ganglion cell model instance.
        DoG_model : DoGModel
            The Difference of Gaussians (DoG) model instance.
        distribution_sampler : DistributionSampler
            The sampler used for sampling from distributions.
        retina_math : RetinaMath
            An instance containing mathematical functions and utilities.
        """
        self.ganglion_cell = ganglion_cell
        self.DoG_model = DoG_model
        self.sampler = distribution_sampler
        self.retina_math = retina_math
        self.project_data = {}
        self.device = device

    @abstractmethod
    def create(self, ret: Retina, gc: Any) -> None:
        """
        Create temporal models.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        gc : GanglionCell
            The ganglion cell model instance.
        """
        pass

    @abstractmethod
    def connect_units(self, ret: Retina, gc: Any) -> None:
        """
        Connect units according to temporal model type.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        gc : GanglionCell
            The ganglion cell model instance.
        """
        pass

    def _link_cone_noise_units_to_gcs(self, ret: Retina, gc: Any) -> Retina:
        """
        Connect cones to ganglion cells for shared cone noise.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        gc : GanglionCell
            The ganglion cell model instance.

        Returns
        -------
        Retina
            The updated retina model instance with cones linked to ganglion cells.
        """
        print("Connecting cones to ganglion cells for shared cone noise...")
        cone_pos_mm: np.ndarray = ret.cone_optimized_pos_mm
        sd_cone: float = ret.cone_general_parameters[f"cone2gc_{ret.gc_type}"] / 1000
        cutoff_distance: float = (
            ret.cone_general_parameters["cone2gc_cutoff_SD"] * sd_cone
        )

        # Normalize center activation to probability distribution
        img_cen: np.ndarray = gc.img * gc.img_mask  # N, H, W
        img_prob: np.ndarray = img_cen / np.sum(img_cen, axis=(1, 2))[:, None, None]

        n_cones = cone_pos_mm.shape[0]
        n_gcs = img_prob.shape[0]

        # Convert inputs to PyTorch tensors and move to the appropriate device
        device = self.device
        cone_pos_mm = torch.tensor(cone_pos_mm, dtype=torch.float32).to(device)
        gc_X_grid_mm = torch.tensor(gc.X_grid_cen_mm, dtype=torch.float32).to(device)
        gc_Y_grid_mm = torch.tensor(gc.Y_grid_cen_mm, dtype=torch.float32).to(device)
        img_prob = torch.tensor(img_prob, dtype=torch.float32).to(device)
        weights = torch.zeros((n_cones, n_gcs), dtype=torch.float32).to(device)

        # Define batch size
        batch_size = 10  # Adjust this based on your GPU memory

        # Process in batches
        for i in range(0, n_cones, batch_size):
            batch_end = min(i + batch_size, n_cones)
            batch_indices = torch.arange(i, batch_end).to(device)
            batch_cone_pos = cone_pos_mm[i:batch_end]

            # Vectorize the distance calculation for the current batch
            dist_x_mtx = gc_X_grid_mm.unsqueeze(0) - batch_cone_pos[:, 0].unsqueeze(
                1
            ).unsqueeze(2).unsqueeze(3)
            dist_y_mtx = gc_Y_grid_mm.unsqueeze(0) - batch_cone_pos[:, 1].unsqueeze(
                1
            ).unsqueeze(2).unsqueeze(3)
            dist_mtx = torch.sqrt(dist_x_mtx**2 + dist_y_mtx**2)

            # Drop weight as a Gaussian function of distance with sd = sd_cone
            probability = torch.exp(-((dist_mtx / sd_cone) ** 2))
            probability[dist_mtx > cutoff_distance] = 0

            # Vectorize the weight calculation for the current batch
            weights_mtx = probability * img_prob.unsqueeze(0)
            weights[i:batch_end, :] = weights_mtx.sum(dim=(2, 3))

        weights = weights.cpu().numpy()

        ret.cones_to_gcs_weights = weights

        return ret

    def _get_BK_statistics(self, ret: Retina) -> pd.DataFrame:
        """
        Fit temporal statistics of the temporal parameters using the triangular distribution.

        Data from Benardete & Kaplan Visual Neuroscience 16 (1999) 355-368 (parasol cells),
        and Benardete & Kaplan Visual Neuroscience 14 (1997) 169-185 (midget cells).

        Parameters
        ----------
        ret : Retina
            The retina model instance.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the temporal statistics of the temporal filter parameters,
            including the shape, loc, and scale parameters of the fitted gamma distribution,
            as well as the name of the distribution and the domain.
        """
        temporal_model_parameters = self.ganglion_cell.get_BK_parameter_names()
        col_names = ["Minimum", "Maximum", "Median", "Mean", "SD", "SEM"]
        distrib_params = np.zeros((len(temporal_model_parameters), 3))
        response_type = ret.response_type.upper()

        temp_params_df = ret.experimental_archive["temporal_parameters_BK"]
        for i, param_name in enumerate(temporal_model_parameters):
            condition = (temp_params_df["Parameter"] == param_name) & (
                temp_params_df["Type"] == response_type
            )

            param_df = temp_params_df[condition].loc[:, col_names]

            if param_df.empty:
                continue

            minimum, maximum, median, mean, sd, sem = param_df.values[0]

            c, loc, scale = self.retina_math.get_triangular_parameters(
                minimum, maximum, median, mean, sd, sem
            )
            distrib_params[i, :] = [c, loc, scale]

        temporal_exp_univariate_stat = pd.DataFrame(
            distrib_params,
            index=temporal_model_parameters,
            columns=["shape", "loc", "scale"],
        )
        temporal_exp_univariate_stat["distribution"] = "triang"
        temporal_exp_univariate_stat["domain"] = "temporal_BK"
        all_data_fits_df = pd.concat(
            [self.DoG_model.exp_univariate_stat, temporal_exp_univariate_stat], axis=0
        )

        self.project_data["exp_temp_BK_model"] = {
            "temporal_model_parameters": temporal_model_parameters,
            "distrib_params": distrib_params,
            "suptitle": ret.gc_type + " " + ret.response_type,
            "all_data_fits_df": all_data_fits_df,
        }

        self.DoG_model.exp_univariate_stat = all_data_fits_df

        return temporal_exp_univariate_stat

    def _sample_temporal_rfs(self, gc: Any, stat_df: pd.DataFrame) -> Any:
        """
        Sample temporal receptive fields for ganglion cells based on provided statistics.

        Parameters
        ----------
        gc : GanglionCell
            The ganglion cell model instance.
        stat_df : pd.DataFrame
            A DataFrame containing the statistics for sampling temporal parameters.

        Returns
        -------
        GanglionCell
            The updated ganglion cell model instance with sampled temporal receptive fields.
        """
        n_cells = len(gc.df)
        for param_name, row in stat_df.iterrows():
            shape, loc, scale, distribution, _ = row
            gc.df[param_name] = self.sampler.sample_univariate(
                shape, loc, scale, n_cells, distribution
            )
        return gc

    def _fit_cone_noise_vs_freq(self, ret: Retina) -> Retina:
        """
        Fit the cone noise data as a function of frequency using interpolated cone response and two Lorentzian functions.

        This method extracts cone response and noise data, performs interpolation, and fits the data using
        a double Lorentzian function. It handles data transformation to logarithmic scale for fitting
        and converts fitted parameters back to linear scale for output.

        Parameters
        ----------
        ret : Retina
            The retina model instance to store the fitted parameters, frequency data, power data, and fitted noise power.

        Returns
        -------
        Retina
            The updated retina model instance populated with the fitting results, including cone noise parameters
            in linear scale, raw frequency and power data, and the fitted cone noise power spectrum.
        """
        # Interpolate cone response at 400 k photoisomerization background
        ret.cone_frequency_data = ret.experimental_archive["cone_frequency_data"]
        ret.cone_power_data = ret.experimental_archive["cone_power_data"]

        # Set interpolation function and parameters for double Lorentzian fit
        self.cone_interp_function = self.retina_math.interpolate_data(
            ret.cone_frequency_data, ret.cone_power_data
        )
        self.cone_noise_wc = ret.experimental_archive["cone_noise_wc"]

        noise_frequency_data = ret.experimental_archive["noise_frequency_data"]
        noise_power_data = ret.experimental_archive["noise_power_data"]

        log_frequency_data = np.log(noise_frequency_data)
        log_power_data = np.log(noise_power_data)

        # Linear scale initial parameters a0, a1, a2
        initial_guesses = [4e5, 1e-2, 1e-3]
        log_initial_guesses = [np.log(p) for p in initial_guesses]

        # Loose bounds for a0, a1, a2
        lower_bounds = [0, 0, 0]
        upper_bounds = [np.inf, np.inf, np.inf]

        # Take the log of bounds
        log_lower_bounds = [np.log(low) if low > 0 else -np.inf for low in lower_bounds]
        log_upper_bounds = [np.log(up) for up in upper_bounds]

        log_bounds = (log_lower_bounds, log_upper_bounds)

        self.retina_math.set_metaparameters_for_log_interp_and_double_lorenzian(
            self.cone_interp_function, self.cone_noise_wc
        )

        # Fit in log space to equalize errors across the power range
        log_popt, _ = opt.curve_fit(
            self.retina_math.fit_log_interp_and_double_lorenzian,
            log_frequency_data,
            log_power_data,
            p0=log_initial_guesses,
            bounds=log_bounds,
        )

        ret.cone_noise_parameters = np.exp(log_popt)  # Convert params to linear space
        ret.noise_frequency_data = noise_frequency_data
        ret.noise_power_data = noise_power_data

        # Get the final noise power fit
        ret.cone_noise_power_fit = np.exp(
            self.retina_math.fit_log_interp_and_double_lorenzian(
                log_frequency_data, *log_popt
            )
        )

        return ret


class TemporalModelFixed(TemporalModelBase):
    """
    A class to build fixed temporal models for retinal ganglion cells.

    This class implements methods to create fixed temporal models, sample temporal receptive fields using statistical data,
    and connect units according to the fixed temporal model type. It uses both univariate and multivariate statistics
    to sample parameters, depending on the configuration.

    Attributes
    ----------
    (Inherited from TemporalModelBase)
    """

    def __init__(
        self,
        ganglion_cell: Any,
        DoG_model: Any,
        distribution_sampler: DistributionSampler,
        retina_math: RetinaMath,
        device: str = "cpu",
    ):
        """
        Initialize the TemporalModelFixed instance.

        Parameters
        ----------
        ganglion_cell : GanglionCell
            The ganglion cell model instance.
        DoG_model : DoGModel
            The Difference of Gaussians (DoG) model instance.
        distribution_sampler : DistributionSampler
            The sampler used for sampling from distributions.
        retina_math : RetinaMath
            An instance containing mathematical functions and utilities.
        """
        super().__init__(
            ganglion_cell, DoG_model, distribution_sampler, retina_math, device
        )

    def _get_temporal_covariances_of_interest(self) -> list[str]:
        """
        Get the list of temporal covariances of interest for multivariate statistics.

        Returns
        -------
        list[str]
            A list of parameter names representing the covariances of interest.
        """
        temporal_covariances_of_interest = ["n", "p1", "p2", "tau1", "tau2"]
        return temporal_covariances_of_interest

    def create(self, ret: Retina, gc: Any) -> Any:
        """
        Create fixed temporal models by sampling temporal receptive fields.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        gc : GanglionCell
            The ganglion cell model instance.

        Returns
        -------
        GanglionCell
            The updated ganglion cell model instance with sampled temporal receptive fields.
        """
        C_statistics = self.DoG_model.exp_univariate_stat[
            self.DoG_model.exp_univariate_stat["domain"] == "temporal"
        ]
        BK_statistics = self._get_BK_statistics(ret)
        gain_name = "A_cen" if ret.gc_type == "midget" else "A"
        gain_and_mean = BK_statistics.loc[[gain_name, "Mean"]]
        stat_df = pd.concat([C_statistics, gain_and_mean])
        gc = self._sample_temporal_rfs(gc, stat_df)

        n_cells = len(gc.df)
        if ret.fit_statistics == "multivariate":
            covariances_of_interest = self._get_temporal_covariances_of_interest()

            available_stat = self.DoG_model.temporal_multivariate_stat
            filtered_stat, intersection = self.sampler.filter_stat(
                available_stat, covariances_of_interest
            )

            self.project_data["temporal_covariances_of_interest"] = intersection

            samples = self.sampler.sample_multivariate(filtered_stat, n_cells)
            missing_columns = [
                col for col in samples.columns if col not in gc.df.columns
            ]
            gc.df[missing_columns] = np.nan
            gc.df.update(samples)

        return gc

    def connect_units(self, ret: Retina, gc: Any) -> Retina:
        """
        Connect units according to the fixed temporal model type.
        Only noise units are connected.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        gc : GanglionCell
            The ganglion cell model instance.

        Returns
        -------
        Retina
            The updated retina model instance with connected units.
        """
        ret = self._link_cone_noise_units_to_gcs(ret, gc)
        return ret


class TemporalModelDynamic(TemporalModelBase):
    """
    A class to build dynamic temporal models for retinal ganglion cells.

    This class implements methods to create dynamic temporal models by sampling temporal receptive fields
    using Benardete & Kaplan (BK) statistics. It also connects units according to the dynamic temporal model type,
    focusing on noise units.

    Attributes
    ----------
    (Inherited from TemporalModelBase)
    """

    def __init__(
        self,
        ganglion_cell: Any,
        DoG_model: Any,
        distribution_sampler: DistributionSampler,
        retina_math: RetinaMath,
        device: str = "cpu",
    ):
        """
        Initialize the TemporalModelDynamic instance.

        Parameters
        ----------
        ganglion_cell : GanglionCell
            The ganglion cell model instance.
        DoG_model : DoGModel
            The Difference of Gaussians (DoG) model instance.
        distribution_sampler : DistributionSampler
            The sampler used for sampling from distributions.
        retina_math : RetinaMath
            An instance containing mathematical functions and utilities.
        """
        super().__init__(
            ganglion_cell, DoG_model, distribution_sampler, retina_math, device
        )

    def create(self, ret: Retina, gc: Any) -> Any:
        """
        Create dynamic temporal models by sampling temporal receptive fields.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        gc : GanglionCell
            The ganglion cell model instance.

        Returns
        -------
        GanglionCell
            The updated ganglion cell model instance with sampled temporal receptive fields.
        """
        BK_statistics = self._get_BK_statistics(ret)
        gc = self._sample_temporal_rfs(gc, BK_statistics)

        return gc

    def connect_units(self, ret: Retina, gc: Any) -> Retina:
        """
        Connect units according to the dynamic temporal model type.
        Only noise units are connected.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        gc : GanglionCell
            The ganglion cell model instance.

        Returns
        -------
        Retina
            The updated retina model instance with connected units.
        """
        ret = self._link_cone_noise_units_to_gcs(ret, gc)

        return ret


class TemporalModelSubunit(TemporalModelBase):
    """
    A class to build subunit temporal models for retinal ganglion cells.

    This class implements methods to create subunit temporal models, fit the bipolar rectification index,
    link cones to bipolar cells, and link bipolar units to ganglion cells. It extends the `TemporalModelBase` class.

    Attributes
    ----------
    (Inherited from TemporalModelBase)
    """

    def __init__(
        self,
        ganglion_cell: Any,
        DoG_model: Any,
        distribution_sampler: DistributionSampler,
        retina_math: RetinaMath,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the TemporalModelSubunit instance.

        Parameters
        ----------
        ganglion_cell : GanglionCell
            The ganglion cell model instance.
        DoG_model : DoGModel
            The Difference of Gaussians (DoG) model instance.
        distribution_sampler : DistributionSampler
            The sampler used for sampling from distributions.
        retina_math : RetinaMath
            An instance containing mathematical functions and utilities.
        """
        super().__init__(
            ganglion_cell, DoG_model, distribution_sampler, retina_math, device
        )

    def _fit_bipolar_rectification_index(self, ret: Retina) -> Retina:
        """
        Fit the rectification index (RI) data from Turner_2018_eLife assuming a parabolic function.

        This method fits the rectification index using data from Turner et al. (2018),
        scaling the surround activation values and fitting a parabola to obtain the bipolar nonlinearity parameters.

        Parameters
        ----------
        ret : Retina
            The retina model instance.

        Returns
        -------
        Retina
            The updated retina model instance with fitted bipolar nonlinearity parameters.
        """
        unit_type: str = ret.gc_type
        response_type: str = ret.response_type

        # Load the parasol data, but set target_RI_values to 0 for midgets (always linear).
        g_sur_values: np.ndarray = ret.experimental_archive["g_sur_values"]
        target_RI_values: np.ndarray = ret.experimental_archive["target_RI_values"]

        if unit_type == "midget":
            target_RI_values = target_RI_values * 0

        RI_function = self.retina_math.parabola

        # Define the target range for parasol on, which in Turner 2018 Fig 5C has the larger range
        # Measured for max contrast square grating at 4Hz temporal and spatial frequencies
        g_sur_range = (-0.15, 0.15)

        # Scale g_sur_values to g_sur_range
        g_sur_min: float = np.min(g_sur_values)
        g_sur_max: float = np.max(g_sur_values)
        g_sur_scaled: np.ndarray = (g_sur_values - g_sur_min) / (
            g_sur_max - g_sur_min
        ) * (g_sur_range[1] - g_sur_range[0]) + g_sur_range[0]

        # Exclude the first value as an outlier.
        popt, _ = opt.curve_fit(RI_function, g_sur_scaled[1:], target_RI_values[1:])

        ret.bipolar_nonlinearity_parameters = popt
        ret.g_sur_scaled = g_sur_scaled
        ret.target_RI_values = target_RI_values
        ret.bipolar_nonlinearity_fit = RI_function(g_sur_scaled, *popt)

        return ret

    def _link_cones_to_bipolars(self, ret: Retina) -> Retina:
        """
        Connect cones to bipolar cells.

        This method calculates the weights between cones and bipolar cells based on distances and
        other parameters, and updates the retina model instance with these weights.

        Parameters
        ----------
        ret : Retina
            The retina model instance.

        Returns
        -------
        Retina
            The updated retina model instance with cones linked to bipolar cells.
        """

        def _extract_range_and_average_from(
            df: pd.DataFrame,
        ) -> Tuple[float, float, float]:
            """
            Extract range and average values from the DataFrame.

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame containing the necessary data.

            Returns
            -------
            Tuple[float, float, float]
                A tuple containing the minimum range, maximum range, and average.
            """
            range_strings = df.loc["Cone_contacts_Range", :].values
            average_strings = df.loc["Cone_contacts_Average", :].values
            density_strings = df.loc["Estimated_density_mm^-2", :].values

            # Get range min and max.
            min_range = np.array(
                [float(x.split("-")[0]) for x in range_strings], dtype=float
            )
            max_range = np.array(
                [float(x.split("-")[1]) for x in range_strings], dtype=float
            )
            means = np.array([float(x) for x in average_strings], dtype=float)
            sizes = np.array([float(x) for x in density_strings], dtype=float)
            average = self.retina_math.weighted_average(means, sizes)

            return min(min_range), max(max_range), average

        print("Connecting cones to bipolar cells...")

        cone_pos_mm: np.ndarray = ret.cone_optimized_pos_mm
        bipo_pos_mm: np.ndarray = ret.bipolar_optimized_pos_mm
        selected_bipolars_df: pd.DataFrame = ret.selected_bipolars_df
        bipo_cen_sd_mm: float = (
            ret.bipolar_general_parameters["cone2bipo_cen_sd"] / 1000
        )
        bipo_sur_sd_mm: float = (
            ret.bipolar_general_parameters["cone2bipo_sur_sd"] / 1000
        )
        bipo_sur2cen_amp_ratio: float = ret.bipolar_general_parameters[
            "bipo_sub_sur2cen"
        ]
        cutoff_SD_sur: float = (
            ret.cone_general_parameters["cone2bipo_cutoff_SD"] * bipo_sur_sd_mm
        )

        cone_reshaped: np.ndarray = cone_pos_mm[:, np.newaxis, :]
        bipo_reshaped: np.ndarray = bipo_pos_mm[np.newaxis, :, :]

        squared_diffs: np.ndarray = (cone_reshaped - bipo_reshaped) ** 2
        squared_distances: np.ndarray = squared_diffs.sum(axis=2)
        distances: np.ndarray = np.sqrt(squared_distances).astype(np.float64)

        min_range, max_range, average = _extract_range_and_average_from(
            selected_bipolars_df
        )

        n_bipolar_dendritic_contacts = (
            self.retina_math.get_sample_from_range_and_average(
                min_range, max_range, average, len(bipo_pos_mm)
            )
        )

        # Get indices of ascending distances for each bipolar cell.
        ascending_distances_from_bipolars: np.ndarray = np.argsort(distances, axis=0)

        # Initialize null_idx with True (exclude all initially).
        null_idx: np.ndarray = np.ones_like(distances, dtype=bool)

        # Mark the top n_bipolar_dendritic_contacts shortest distances as False (include them).
        for j in range(ascending_distances_from_bipolars.shape[1]):
            null_column = ascending_distances_from_bipolars[
                : n_bipolar_dendritic_contacts[j], j
            ]
            null_idx[null_column, j] = False

        G_cen: np.ndarray = np.exp(-((distances / bipo_cen_sd_mm) ** 2))
        G_sur: np.ndarray = np.exp(-((distances / bipo_sur_sd_mm) ** 2))

        G_cen[null_idx] = 0
        G_sur[distances > cutoff_SD_sur] = 0

        # Assert no column-wise zero sums.
        G_cen_sum: np.ndarray = G_cen.sum(axis=0)
        G_sur_sum: np.ndarray = G_sur.sum(axis=0)
        assert not any(
            G_cen_sum == 0
        ), "Zero sum in cone to bipolar center, aborting..."
        assert not any(
            G_sur_sum == 0
        ), "Zero sum in cone to bipolar surround, aborting..."

        G_cen_probability: np.ndarray = G_cen / G_cen_sum[np.newaxis, :]
        G_sur_probability: np.ndarray = G_sur / G_sur_sum[np.newaxis, :]

        G_cen_weight: np.ndarray = G_cen_probability
        G_sur_weight: np.ndarray = G_sur_probability * bipo_sur2cen_amp_ratio

        ret.cones_to_bipolars_center_weights = G_cen_weight
        ret.cones_to_bipolars_surround_weights = G_sur_weight

        return ret

    def _link_bipo_to_gc(
        self,
        ret: Retina,
        gc: Any,
        mask: np.ndarray,
        X_grid_cen_mm: np.ndarray,
        Y_grid_cen_mm: np.ndarray,
    ) -> np.ndarray:
        """
        Link bipolar units to ganglion cells worker function.
        """
        bipo_pos_mm: np.ndarray = ret.bipolar_optimized_pos_mm
        n_bipos: int = bipo_pos_mm.shape[0]

        rf_div: float = ret.bipolar_general_parameters["bipo2gc_div"]
        sd_bipo: np.ndarray = gc.df.den_diam_um / rf_div
        sd_bipo = sd_bipo / 1000  # Convert from micrometers to millimeters.
        sd_bipo = sd_bipo.values[:, None, None]  # Shape: (N, 1, 1).
        cutoff_SD: np.ndarray = (
            ret.bipolar_general_parameters["bipo2gc_cutoff_SD"] * sd_bipo
        )

        weights: np.ndarray = np.zeros((n_bipos, gc.n_units))

        # Normalize center activation to probability distribution.
        img_masked: np.ndarray = gc.img * mask  # Shape: (N, H, W).

        desc_str = f"Calculating {n_bipos} x {gc.n_units} connections"
        for this_bipo in tqdm(range(n_bipos), desc=desc_str):
            this_bipo_pos = bipo_pos_mm[this_bipo]
            dist_x_mtx = X_grid_cen_mm - this_bipo_pos[0]
            dist_y_mtx = Y_grid_cen_mm - this_bipo_pos[1]
            dist_mtx = np.sqrt(dist_x_mtx**2 + dist_y_mtx**2)

            # Drop weight as a Gaussian function of distance with sd = sd_bipo.
            probability = np.exp(-((dist_mtx**2) / (2 * sd_bipo**2)))
            probability[dist_mtx > cutoff_SD] = 0

            weights_mtx = probability * img_masked
            # weights_mtx = probability * img_prob
            weights[this_bipo, :] = weights_mtx.sum(axis=(1, 2))

        return weights

    def _link_bipolar_units_to_gcs(self, ret: Retina, gc: Any) -> Retina:
        """
        Connect bipolar units to ganglion cell units for shared subunit model.

        This method calculates the weights between bipolar units and ganglion cells based on distances
        and other parameters, and updates the retina model instance with these weights.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        gc : GanglionCell
            The ganglion cell model instance.

        Returns
        -------
        Retina
            The updated retina model instance with bipolar units linked to ganglion cells.
        """
        print("Connecting bipolar units to ganglion cells...")

        # link gc center
        weights_cen = self._link_bipo_to_gc(
            ret, gc, gc.img_mask, gc.X_grid_cen_mm, gc.Y_grid_cen_mm
        )

        # Normalize weights so that the input to each ganglion cell center sums to 1.0.
        weights_out_cen: np.ndarray = weights_cen / weights_cen.sum(axis=0)[None, :]

        ret.bipolar_to_gcs_cen_weights = weights_out_cen

        # link gc surround
        weights_sur = self._link_bipo_to_gc(
            ret, gc, gc.img_mask_sur, gc.X_grid_sur_mm, gc.Y_grid_sur_mm
        )

        # Coming from RF img, where surround is negative. We want positive weights.
        weights_sur = weights_sur * -1

        # Normalize to center weight = 1
        weights_out_sur: np.ndarray = weights_sur / weights_cen.sum(axis=0)[None, :]
        ret.bipolar_to_gcs_sur_weights = weights_out_sur

        return ret

    def create(self, ret: Retina, gc: Any) -> Any:
        """
        Create subunit temporal models.

        This method samples temporal receptive fields and fits the bipolar rectification index.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        gc : GanglionCell
            The ganglion cell model instance.

        Returns
        -------
        GanglionCell
            The updated ganglion cell model instance.
        """
        BK_statistics = self._get_BK_statistics(ret)
        gain_name = "A_cen" if ret.gc_type == "midget" else "A"
        gain_and_mean = BK_statistics.loc[[gain_name, "Mean"]]
        gc = self._sample_temporal_rfs(gc, gain_and_mean)

        ret = self._fit_bipolar_rectification_index(ret)

        return gc

    def connect_units(self, ret: Retina, gc: Any) -> Retina:
        """
        Connect units from cones to bipolars and further to ganglion cells.

        The noise units are connected separately to maintain consistency with other temporal models.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        gc : GanglionCell
            The ganglion cell model instance.

        Returns
        -------
        Retina
            The updated retina model instance with units connected.
        """
        ret = self._link_cones_to_bipolars(ret)
        ret = self._link_bipolar_units_to_gcs(ret, gc)
        ret = self._link_cone_noise_units_to_gcs(ret, gc)

        return ret


class RetinaBuildInterface(ABC):
    """
    Abstract base class for building the retina.

    Logic here as abstract methods and properties.

    """

    @property
    @abstractmethod
    def retina(self):
        """
        The retina product under construction.
        """
        pass

    @property
    @abstractmethod
    def ganglion_cell(self):
        """
        The ganglion cell image parameters under construction.
        """
        pass

    @property
    @abstractmethod
    def retina_math(self):
        """
        The retina math instance.
        """
        pass

    @abstractmethod
    def get_concrete_components(self):
        """
        Get major alternative concrete components for the retina builder. The are defined in the retina_parameters dict.
        """
        pass

    @abstractmethod
    def fit_cell_density_data(self):
        """
        Read literature data from file and fit ganglion cell and cone density with respect to eccentricity.
        """
        pass

    @abstractmethod
    def place_units(self):
        """
        Place ganglion cells, cones and bipolar cells.
        """
        pass

    @abstractmethod
    def create_spatial_receptive_fields(self):
        """
        Create spatial receptive fields.
        """
        pass

    @abstractmethod
    def connect_units(self):
        """
        Connect cones to bipolars, bipolars to ganglion cells, and cone noise units to ganglion cells.
        """
        pass

    @abstractmethod
    def create_temporal_receptive_fields(self):
        """
        Create temporal receptive fields.
        """
        pass

    @abstractmethod
    def create_tonic_drive(self):
        """
        Create tonic drive for ganglion cells.
        """
        pass


class ConcreteRetinaBuilder(RetinaBuildInterface):
    """
    Compilation of the retina components into one concrete builder instance.
    """

    def __init__(self, retina, retina_math, fit, retina_vae, device, viz) -> None:

        self._retina = retina
        self._ganglion_cell = None
        self._retina_math = retina_math
        self._fit = fit
        self._retina_vae = retina_vae
        self._device = device
        self._viz = viz
        self.experimental_archive = retina.experimental_archive

        self.project_data = {}

    @property
    def retina(self):
        return self._retina

    @retina.setter
    def retina(self, value):
        self._retina = value

    @property
    def ganglion_cell(self):
        return self._ganglion_cell

    @ganglion_cell.setter
    def ganglion_cell(self, value):
        self._ganglion_cell = value

    @property
    def retina_math(self):
        return self._retina_math

    @property
    def fit(self):
        return self._fit

    @property
    def retina_vae(self):
        return self._retina_vae

    @property
    def device(self):
        return self._device

    # For testing where device is set to cpu
    @device.setter
    def device(self, value):
        self._device = value

    @property
    def viz(self):
        return self._viz

    @property
    def spatial_model(self):
        return self._spatial_model

    @property
    def DoG_model(self):
        return self._DoG_model

    @property
    def temporal_model(self):
        return self._temporal_model

    @property
    def sampler(self):
        return self._sampler

    # fit_cell_density_data helper functions
    def _check_boundaries(
        self,
        node_positions: torch.Tensor,
        ecc_lim_mm: torch.Tensor,
        polar_lim_deg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce boundary constraints on node positions based on eccentricity and polar limits.

        Parameters
        ----------
        node_positions : torch.Tensor
            A tensor of shape (N, 2) containing the x and y positions of nodes.
        ecc_lim_mm : torch.Tensor
            A tensor containing the minimum and maximum eccentricity limits in mm.
        polar_lim_deg : torch.Tensor
            A tensor containing the minimum and maximum polar angle limits in degrees.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, 2) containing the adjustments to be applied to node positions.
        """
        x, y = node_positions[:, 0], node_positions[:, 1]
        min_eccentricity, max_eccentricity = ecc_lim_mm
        min_polar, max_polar = polar_lim_deg

        r, theta = self._cart2pol_torch(x, y)
        # Guarding eccentricity boundaries
        r = torch.clamp(r, min=min_eccentricity, max=max_eccentricity)
        # Guarding polar boundaries
        theta = torch.clamp(theta, min=min_polar, max=max_polar)

        new_x, new_y = self._pol2cart_torch(r, theta)

        delta_x = new_x - x
        delta_y = new_y - y

        return torch.stack([delta_x, delta_y], dim=1)

    def _pol2cart_torch(
        self, radius: torch.Tensor, phi: torch.Tensor, deg: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert polar coordinates to Cartesian coordinates using PyTorch tensors.

        Parameters
        ----------
        radius : torch.Tensor
            Tensor representing the radius values in mm.
        phi : torch.Tensor
            Tensor representing the polar angle values.
        deg : bool, optional
            If True, the angle is given in degrees; if False, in radians. Default is True.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tensors representing the x and y Cartesian coordinates.
        """

        if deg:
            theta = phi * torch.pi / 180
        else:
            theta = phi

        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        return (x, y)

    def _cart2pol_torch(
        self, x: torch.Tensor, y: torch.Tensor, deg: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian coordinates to polar coordinates using PyTorch tensors.

        Parameters
        ----------
        x : torch.Tensor
            Tensor representing the x-coordinate in Cartesian coordinates.
        y : torch.Tensor
            Tensor representing the y-coordinate in Cartesian coordinates.
        deg : bool, optional
            If True, the returned angle is in degrees; if False, in radians. Default is True.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tensors representing the radius and angle in polar coordinates.
        """

        radius = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)

        if deg:
            phi = theta * 180 / torch.pi
        else:
            phi = theta

        return radius, phi

    # place_units helper functions
    def _get_gc_proportion(self, ret: Retina) -> Retina:
        """
        Calculate the proportion of ganglion cells based on cell type and response type.

        Parameters
        ----------
        ret : Retina
            The retina model instance.

        Returns
        -------
        Retina
            The updated retina model instance with gc_proportion set.
        """
        if ret.gc_type == "parasol":
            gc_proportion = ret.proportion_of_parasol_gc_type
        elif ret.gc_type == "midget":
            gc_proportion = ret.proportion_of_midget_gc_type
        else:
            raise ValueError(f"Unknown ganglion cell type: {ret.gc_type}")

        if ret.response_type == "on":
            gc_proportion *= ret.proportion_of_ON_response_type
        elif ret.response_type == "off":
            gc_proportion *= ret.proportion_of_OFF_response_type
        else:
            raise ValueError(f"Unknown response type: {ret.response_type}")

        gc_proportion *= ret.model_density
        ret.gc_proportion = gc_proportion

        return ret

    def _hexagonal_positions_group(
        self,
        min_ecc: float,
        max_ecc: float,
        n_units: int,
        polar_lim_deg: Tuple[float, float],
    ) -> np.ndarray:
        """
        Generate hexagonal positions for a group of units within specified eccentricity and polar angle limits.

        Parameters
        ----------
        min_ecc : float
            Minimum eccentricity in mm.
        max_ecc : float
            Maximum eccentricity in mm.
        n_units : int
            Number of units to place within the specified area.
        polar_lim_deg : Tuple[float, float]
            Tuple containing the minimum and maximum polar angle in degrees.

        Returns
        -------
        np.ndarray
            An array of positions in polar coordinates (eccentricity, angle).
        """

        delta_ecc = max_ecc - min_ecc
        mean_ecc = (max_ecc + min_ecc) / 2

        # Calculate polar coords in mm at mean ecc
        x0, y0 = self.retina_math.pol2cart(mean_ecc, polar_lim_deg[0])
        x1, y1 = self.retina_math.pol2cart(mean_ecc, polar_lim_deg[1])
        delta_pol = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        mean_dist_between_units = np.sqrt((delta_ecc * delta_pol) / n_units)
        n_ecc = int(np.round(delta_ecc / mean_dist_between_units))
        n_ecc = max(n_ecc, 1)
        n_pol = int(np.round(n_units / n_ecc))

        # Generate evenly spaced values for eccentricity and angle
        eccs = np.linspace(
            min_ecc + mean_dist_between_units / 2,
            max_ecc - mean_dist_between_units / 2,
            n_ecc,
            endpoint=True,
        )
        angles = np.linspace(polar_lim_deg[0], polar_lim_deg[1], n_pol, endpoint=False)

        # Create a meshgrid of all combinations of eccs and angles
        eccs_grid, angles_grid = np.meshgrid(eccs, angles, indexing="ij")

        # Offset every other row by half the distance between columns
        delta_angle = np.diff(angles).mean()
        for i in range(n_ecc):
            if i % 2 == 1:  # Check if the row is odd
                angles_grid[i, :] += delta_angle / 2
        # Finally turn every row 1/4 delta_angle more to get grid centered within borders
        angles_grid += delta_angle / 4

        # Reshape the grids and combine them into a single array
        positions = np.column_stack((eccs_grid.ravel(), angles_grid.ravel()))

        return positions

    def _initialize_positions_by_group(self, ret: Retina) -> Tuple:
        """
        Initialize unit positions based on grouped eccentricities.

        Parameters
        ----------
        ret : Retina
            The retina model instance.

        Returns
        -------
        Tuple
            A tuple containing eccentricity groups, sector surface areas, initial positions,
            and densities for ganglion cells, cones, and bipolars.
        """
        gc_density_params = ret.gc_density_params
        cone_density_params = ret.cone_density_params
        bipolar_density_params = ret.bipolar_density_params

        # Loop for reasonable delta ecc to get correct density in one hand and good unit distribution from the algo on the other
        # Lets fit close to 0.1 mm intervals, which makes sense up to some 15 deg. Thereafter longer jumps would do fine.
        assert (
            ret.ecc_lim_mm[0] < ret.ecc_lim_mm[1]
        ), "Radii in wrong order, give [min max], aborting"
        eccentricity_in_mm_total = ret.ecc_lim_mm
        fit_interval = 0.1  # mm
        n_steps = math.ceil(np.ptp(eccentricity_in_mm_total) / fit_interval)
        eccentricity_steps = np.linspace(
            eccentricity_in_mm_total[0], eccentricity_in_mm_total[1], 1 + n_steps
        )

        angle_deg = np.ptp(ret.polar_lim_deg)  # The angle_deg is now == max theta_deg

        eccentricity_groups = []
        areas_all_mm2 = []
        gc_initial_pos = []
        gc_density_all = []
        cone_initial_pos = []
        cone_density_all = []
        bipolar_initial_pos = []
        bipolar_density_all = []
        for group_idx in range(len(eccentricity_steps) - 1):
            min_ecc = eccentricity_steps[group_idx]
            max_ecc = eccentricity_steps[group_idx + 1]
            avg_ecc = (min_ecc + max_ecc) / 2

            gc_density_group = self.gc_fit_function(avg_ecc, *gc_density_params)

            cone_density_group = self.cone_fit_function(avg_ecc, *cone_density_params)

            bipolar_density_group = self.bipolar_fit_function(
                avg_ecc, *bipolar_density_params
            )

            # Calculate area for this eccentricity group
            sector_area_remove = self.retina_math.sector2area_mm2(min_ecc, angle_deg)
            sector_area_full = self.retina_math.sector2area_mm2(max_ecc, angle_deg)
            sector_surface_area = sector_area_full - sector_area_remove  # in mm2

            # collect sector area for each ecc step
            areas_all_mm2.append(sector_surface_area)

            gc_units = math.ceil(
                sector_surface_area * gc_density_group * ret.gc_proportion
            )
            gc_positions = self._hexagonal_positions_group(
                min_ecc, max_ecc, gc_units, ret.polar_lim_deg
            )
            # After hexagonal positions, we know the number of units
            gc_units = gc_positions.shape[0]
            eccentricity_groups.append(np.full(gc_units, group_idx))
            gc_density_all.append(
                np.full(gc_units, gc_density_group * ret.gc_proportion)
            )
            gc_initial_pos.append(gc_positions)

            cone_units = math.ceil(sector_surface_area * cone_density_group)
            cone_positions = self._hexagonal_positions_group(
                min_ecc, max_ecc, cone_units, ret.polar_lim_deg
            )

            cone_units = cone_positions.shape[0]
            cone_density_all.append(np.full(cone_units, cone_density_group))
            cone_initial_pos.append(cone_positions)

            bipolar_units = math.ceil(sector_surface_area * bipolar_density_group)
            bipolar_positions = self._hexagonal_positions_group(
                min_ecc, max_ecc, bipolar_units, ret.polar_lim_deg
            )
            bipolar_units = bipolar_positions.shape[0]

            bipolar_density_all.append(np.full(bipolar_units, bipolar_density_group))
            bipolar_initial_pos.append(bipolar_positions)

        gc_density_per_unit = np.concatenate(gc_density_all)
        cone_density_per_unit = np.concatenate(cone_density_all)
        bipolar_density_per_unit = np.concatenate(bipolar_density_all)

        return (
            eccentricity_groups,
            areas_all_mm2,
            gc_initial_pos,
            gc_density_per_unit,
            cone_initial_pos,
            cone_density_per_unit,
            bipolar_initial_pos,
            bipolar_density_per_unit,
        )

    def _boundary_force(
        self,
        positions: torch.Tensor,
        rep: float,
        dist_th: float,
        ecc_lim_mm: torch.Tensor,
        polar_lim_deg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate boundary repulsive forces for given positions based on both
        eccentricity (left-right) and polar (bottom-top) constraints.

        Parameters
        ----------
        positions : torch.Tensor
            A tensor of positions (shape: [N, 2], where N is the number of nodes).
        rep : float
            Repulsion coefficient for boundary force.
        dist_th : float
            Distance threshold beyond which no force is applied.
        ecc_lim_mm : torch.Tensor
            A tensor representing the eccentricity limits in millimeters for
            left and right boundaries (shape: [2]).
        polar_lim_deg : torch.Tensor
            A tensor representing the polar angle limits in degrees for
            bottom and top boundaries (shape: [2]).

        Returns
        -------
        torch.Tensor
            A tensor of forces (shape: [N, 2]) for each position.
        """

        forces = torch.zeros_like(positions)
        clamp_min = 1e-5

        # Polar angle-based calculations for bottom and top boundaries
        bottom_x, bottom_y = self._pol2cart_torch(
            ecc_lim_mm, polar_lim_deg[0].expand_as(ecc_lim_mm)
        )
        top_x, top_y = self._pol2cart_torch(
            ecc_lim_mm, polar_lim_deg[1].expand_as(ecc_lim_mm)
        )
        m_bottom = (bottom_y[1] - bottom_y[0]) / (bottom_x[1] - bottom_x[0])
        c_bottom = bottom_y[0] - m_bottom * bottom_x[0]
        m_top = (top_y[1] - top_y[0]) / (top_x[1] - top_x[0])
        c_top = top_y[0] - m_top * top_x[0]

        # Calculating distance from the line for each position
        bottom_distance = torch.abs(
            m_bottom * positions[:, 0] - positions[:, 1] + c_bottom
        ) / torch.sqrt(m_bottom**2 + 1)
        top_distance = torch.abs(
            m_top * positions[:, 0] - positions[:, 1] + c_top
        ) / torch.sqrt(m_top**2 + 1)

        # Computing repulsive forces based on these distances
        bottom_force = rep / (bottom_distance.clamp(min=clamp_min) ** 3)
        bottom_force[bottom_distance > dist_th] = 0
        forces[:, 1] -= bottom_force

        top_force = rep / (top_distance.clamp(min=clamp_min) ** 3)
        top_force[top_distance > dist_th] = 0
        forces[:, 1] += top_force

        # Eccentricity arc-based calculations for min and max arcs
        distances_to_center = torch.norm(positions, dim=1)
        min_ecc_distance = torch.abs(distances_to_center - ecc_lim_mm[0])
        max_ecc_distance = torch.abs(distances_to_center - ecc_lim_mm[1])

        # Compute forces based on these distances
        min_ecc_force = rep / (min_ecc_distance.clamp(min=clamp_min) ** 3)
        min_ecc_force[min_ecc_distance > dist_th] = 0

        max_ecc_force = rep / (max_ecc_distance.clamp(min=clamp_min) ** 3)
        max_ecc_force[max_ecc_distance > dist_th] = 0

        # Calculate direction for the forces (from point to the origin)
        directions = positions / distances_to_center.unsqueeze(1)

        # Update forces using the computed repulsive forces and their directions
        forces -= directions * min_ecc_force.unsqueeze(1)
        forces += directions * max_ecc_force.unsqueeze(1)

        return forces

    def _apply_force_based_layout(
        self,
        ret: Retina,
        all_positions: np.ndarray,
        cell_density: float,
        unit_placement_params: dict[str, Any],
    ) -> np.ndarray:
        """
        Apply a force-based layout on the given positions.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        all_positions : np.ndarray
            Initial positions of nodes.
        cell_density : float
            Local density according to eccentricity group.
        unit_placement_params : dict[str, Any]
            Parameters for unit placement.

        Returns
        -------
        np.ndarray
            New positions of nodes after the force-based optimization.

        Notes
        -----
        This method applies a force-based layout to optimize node positions.
        It visualizes the progress of the layout optimization.
        """
        n_iterations = unit_placement_params["n_iterations"]
        change_rate = unit_placement_params["change_rate"]
        unit_repulsion_stregth = unit_placement_params["unit_repulsion_stregth"]
        unit_distance_threshold = unit_placement_params["unit_distance_threshold"]
        diffusion_speed = unit_placement_params["diffusion_speed"]
        border_repulsion_stength = unit_placement_params["border_repulsion_stength"]
        border_distance_threshold = unit_placement_params["border_distance_threshold"]
        show_placing_progress = unit_placement_params["show_placing_progress"]
        show_skip_steps = unit_placement_params["show_skip_steps"]

        if show_placing_progress:
            # Init plotting
            fig_args = self.viz.show_unit_placement_progress(
                all_positions,
                ecc_lim_mm=ret.ecc_lim_mm,
                polar_lim_deg=ret.polar_lim_deg,
                init=True,
            )

        device = self.device
        dtype = torch.float32

        unit_distance_threshold = torch.tensor(
            unit_distance_threshold, device=device, dtype=dtype
        )
        unit_repulsion_stregth = torch.tensor(
            unit_repulsion_stregth, device=device, dtype=dtype
        )
        diffusion_speed = torch.tensor(diffusion_speed, device=device, dtype=dtype)
        n_iterations = torch.tensor(n_iterations, device=device, dtype=torch.int32)
        cell_density = torch.tensor(cell_density, device=device, dtype=dtype)

        rep = torch.tensor(border_repulsion_stength, device=device, dtype=dtype)
        dist_th = torch.tensor(border_distance_threshold, device=device, dtype=dtype)

        original_positions = deepcopy(all_positions)
        positions = torch.tensor(
            all_positions, requires_grad=True, dtype=dtype, device=device
        )
        change_rate = torch.tensor(change_rate, device=device, dtype=dtype)
        optimizer = torch.optim.Adam(
            [positions], lr=change_rate.item(), betas=(0.95, 0.999)
        )

        ecc_lim_mm = torch.tensor(ret.ecc_lim_mm, device=device, dtype=dtype)
        polar_lim_deg = torch.tensor(ret.polar_lim_deg, device=device, dtype=dtype)
        boundary_polygon = self.viz.boundary_polygon(
            ecc_lim_mm.cpu().numpy(), polar_lim_deg.cpu().numpy()
        )

        # Adjust unit_distance_threshold and diffusion speed with density of the units
        adjusted_distance_threshold = unit_distance_threshold * (952 / cell_density)
        adjusted_diffusion_speed = diffusion_speed * (952 / cell_density)

        for iteration in range(n_iterations.item()):
            optimizer.zero_grad()
            # Repulsive force between nodes
            diff = positions[None, :, :] - positions[:, None, :]
            dist = torch.norm(diff, dim=-1, p=2) + 1e-9

            # Clip minimum distance to avoid very high repulsion
            dist = torch.clamp(dist, min=0.00001)
            # Clip max to inf (zero repulsion) above a certain distance
            dist[dist > adjusted_distance_threshold] = torch.inf
            # Using inverse cube for repulsion
            repulsive_force = unit_repulsion_stregth * torch.sum(
                diff / (dist[..., None] ** 3), dim=1
            )

            # After calculating repulsive_force:
            boundary_forces = self._boundary_force(
                positions, rep, dist_th, ecc_lim_mm, polar_lim_deg
            )

            total_force = repulsive_force + boundary_forces

            # Use the force as the "loss"
            loss = torch.norm(total_force, p=2)

            loss.backward()
            optimizer.step()

            # Update positions in-place
            positions_delta = self._check_boundaries(
                positions, ecc_lim_mm, polar_lim_deg
            )

            gc_diffusion_speed_reshaped = adjusted_diffusion_speed.view(-1, 1)
            new_data = (
                torch.randn_like(positions) * gc_diffusion_speed_reshaped
                + positions_delta
            )

            positions.data = positions + new_data

            if show_placing_progress is True:
                # Update the visualization every 100 iterations for performance (or adjust as needed)
                if iteration % show_skip_steps == 0:
                    positions_cpu = positions.detach().cpu().numpy()
                    self.viz.show_unit_placement_progress(
                        original_positions=original_positions,
                        positions=positions_cpu,
                        iteration=iteration,
                        boundary_polygon=boundary_polygon,
                        **fig_args,
                    )

        del diff, dist, repulsive_force, boundary_forces, total_force

        if show_placing_progress:
            plt.ioff()  # Turn off interactive mode

        return positions.detach().cpu().numpy()

    def _apply_voronoi_layout(
        self,
        ret: Retina,
        all_positions: np.ndarray,
        unit_placement_params: dict[str, Any],
    ) -> np.ndarray:
        """
        Apply a Voronoi-based layout on the given positions.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        all_positions : np.ndarray
            Initial positions of nodes.
        unit_placement_params : dict[str, Any]
            Parameters for unit placement.

        Returns
        -------
        np.ndarray
            New positions of nodes after the Voronoi-based optimization.

        Notes
        -----
        This method applies a Voronoi diagram to optimize node positions.
        It uses Lloyd's relaxation for iteratively adjusting seed points.
        """

        # Extract parameters from config
        n_iterations = unit_placement_params["n_iterations"]
        change_rate = unit_placement_params["change_rate"]
        show_placing_progress = unit_placement_params["show_placing_progress"]
        show_skip_steps = unit_placement_params["show_skip_steps"]

        if show_placing_progress:
            fig_args = self.viz.show_unit_placement_progress(
                all_positions,
                ecc_lim_mm=ret.ecc_lim_mm,
                polar_lim_deg=ret.polar_lim_deg,
                init=True,
            )

        def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
            """Compute the centroid of a polygon."""
            A = 0.5 * np.sum(
                polygon[:-1, 0] * polygon[1:, 1] - polygon[1:, 0] * polygon[:-1, 1]
            )
            C_x = (1 / (6 * A)) * np.sum(
                (polygon[:-1, 0] + polygon[1:, 0])
                * (polygon[:-1, 0] * polygon[1:, 1] - polygon[1:, 0] * polygon[:-1, 1])
            )
            C_y = (1 / (6 * A)) * np.sum(
                (polygon[:-1, 1] + polygon[1:, 1])
                * (polygon[:-1, 0] * polygon[1:, 1] - polygon[1:, 0] * polygon[:-1, 1])
            )
            return np.array([C_x, C_y])

        ecc_lim_mm = ret.ecc_lim_mm
        polar_lim_deg = ret.polar_lim_deg
        boundary_polygon = self.viz.boundary_polygon(ecc_lim_mm, polar_lim_deg)
        original_positions = all_positions.copy()
        positions = all_positions.copy()
        boundary_polygon_shape = ShapelyPolygon(boundary_polygon)

        for iteration in range(n_iterations):
            vor = Voronoi(positions)
            new_positions = []
            old_positions = []
            intersected_polygons = []

            for region, original_seed in zip(vor.regions, original_positions):
                if not -1 in region and len(region) > 0:
                    polygon = np.array([vor.vertices[i] for i in region])
                    voronoi_cell_shape = ShapelyPolygon(polygon)

                    # Find the intersection between the Voronoi unit and the boundary polygon
                    intersection_shape = voronoi_cell_shape.intersection(
                        boundary_polygon_shape
                    )

                    if intersection_shape.is_empty:
                        new_positions.append(original_seed)
                        continue

                    intersection_polygon = np.array(intersection_shape.exterior.coords)

                    # Wannabe centroid
                    new_seed = polygon_centroid(intersection_polygon)

                    if show_placing_progress and iteration % show_skip_steps == 0:
                        # Take polygons for viz.
                        intersected_polygons.append(intersection_polygon)
                        old_positions.append(new_seed)

                    # We cool things down a bit by moving the centroid only the change_rate of the way
                    diff = new_seed - original_seed
                    partial_diff = diff * change_rate
                    new_seed = original_seed + partial_diff
                    new_positions.append(new_seed)

                else:
                    new_positions.append(original_seed)

            # Convert the list of numpy arrays to a single numpy array, then to a torch tensor
            positions_np = np.array(new_positions)
            positions_torch = torch.from_numpy(positions_np).float().to("cpu")

            # Check boundaries and adjust positions if needed
            position_deltas = self._check_boundaries(
                positions_torch,
                torch.tensor(ret.ecc_lim_mm),
                torch.tensor(ret.polar_lim_deg),
            )

            positions = (positions_torch + position_deltas).numpy()

            if show_placing_progress and iteration % show_skip_steps == 0:
                self.viz.show_unit_placement_progress(
                    original_positions=original_positions,
                    positions=np.array(old_positions),
                    iteration=iteration,
                    intersected_polygons=intersected_polygons,
                    boundary_polygon=boundary_polygon,
                    **fig_args,
                )

                # wait = input("Press enter to continue")

        if show_placing_progress:
            plt.ioff()

        return positions

    def _optimize_positions(
        self,
        ret: Retina,
        initial_positions: List[np.ndarray],
        cell_density: np.ndarray,
        unit_placement_params: dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize positions for units using specified placement algorithm.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        initial_positions : List[np.ndarray]
            List of initial positions in polar coordinates.
        cell_density : np.ndarray
            Array of cell densities.
        unit_placement_params : dict[str, Any]
            Parameters for unit placement.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Optimized positions in polar and Cartesian coordinates.
        """
        # Merge the Groups
        all_positions = np.vstack(initial_positions).astype(float)
        all_positions_tuple = self.retina_math.pol2cart(
            all_positions[:, 0], all_positions[:, 1]
        )
        all_positions_mm = np.column_stack(all_positions_tuple)

        # Optimize positions for ganglion cells
        optim_algorithm = unit_placement_params["algorithm"]

        if optim_algorithm == None:
            # Initial random placement.
            # Use this for testing/speed/nonvarying placements.
            optimized_positions = all_positions
            optimized_positions_mm = all_positions_mm
        else:
            if optim_algorithm == "force":
                # Apply Force Based Layout Algorithm with Boundary Repulsion
                optimized_positions_mm = self._apply_force_based_layout(
                    ret, all_positions_mm, cell_density, unit_placement_params
                )
            elif optim_algorithm == "voronoi":
                # Apply Voronoi-based Layout with Loyd's Relaxation
                optimized_positions_mm = self._apply_voronoi_layout(
                    ret, all_positions_mm, unit_placement_params
                )
            optimized_positions_tuple = self.retina_math.cart2pol(
                optimized_positions_mm[:, 0], optimized_positions_mm[:, 1]
            )
            optimized_positions = np.column_stack(optimized_positions_tuple)

        return optimized_positions, optimized_positions_mm

    # create_spatial_receptive_fields helper functions
    def _fit_dd_vs_ecc(self, ret: Retina, gc: Any) -> dict:
        """
        Fit dendritic field diameter with respect to eccentricity.

        Fits linear, quadratic, cubic, exponential, or log-log models to the relationship between dendritic field diameter and eccentricity.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        gc : GanglionCell
            The ganglion cell model instance.

        Returns
        -------
        dict
            Dictionary containing dendritic diameter parameters and related data for visualization.
        """
        dd_regr_model = ret.dd_regr_model
        ecc_limit_for_dd_fit_mm = ret.ecc_limit_for_dd_fit_mm

        # Read dendritic field data and return linear fit with scipy.stats.linregress
        dendr_diam_parameters = {}

        lit = self.experimental_archive
        dendr_diam1 = lit["dendr_diam1"]
        dendr_diam2 = lit["dendr_diam2"]
        dendr_diam3 = lit["dendr_diam3"]
        dendr_diam_units = lit["dendr_diam_units"]

        # Quality control. Datasets separately for visualization
        assert dendr_diam_units["data1"] == ["mm", "um"]
        data_set_1_x = np.squeeze(dendr_diam1["Xdata"])
        data_set_1_y = np.squeeze(dendr_diam1["Ydata"])
        assert dendr_diam_units["data2"] == ["mm", "um"]
        data_set_2_x = np.squeeze(dendr_diam2["Xdata"])
        data_set_2_y = np.squeeze(dendr_diam2["Ydata"])
        assert dendr_diam_units["data3"] == ["deg", "um"]
        data_set_3_x = np.squeeze(dendr_diam3["Xdata"]) / ret.deg_per_mm
        data_set_3_y = np.squeeze(dendr_diam3["Ydata"])

        # Both datasets together
        data_all_x = np.concatenate((data_set_1_x, data_set_2_x, data_set_3_x))
        data_all_y = np.concatenate((data_set_1_y, data_set_2_y, data_set_3_y))

        # Limit eccentricities for central visual field studies to get better approximation at about 5 deg ecc (1mm)
        # x is eccentricity in mm
        # y is dendritic field diameter in micrometers
        data_all_x_index = data_all_x <= ecc_limit_for_dd_fit_mm
        data_all_x = data_all_x[data_all_x_index]
        data_all_y = data_all_y[
            data_all_x_index
        ]  # Don't forget to truncate values, too

        # Sort to ascending order
        data_all_x_index = np.argsort(data_all_x)
        data_all_x = data_all_x[data_all_x_index]
        data_all_y = data_all_y[data_all_x_index]

        # Get rf diameter vs eccentricity
        # dd_regr_model is 'linear'  'quadratic' or cubic
        dict_key = "{0}_{1}".format(ret.gc_type, dd_regr_model)

        if dd_regr_model == "linear":
            polynomial_order = 1
            fit_parameters = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {
                "intercept": fit_parameters[1],
                "slope": fit_parameters[0],
            }
        elif dd_regr_model == "quadratic":
            polynomial_order = 2
            fit_parameters = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {
                "intercept": fit_parameters[2],
                "slope": fit_parameters[1],
                "square": fit_parameters[0],
            }
        elif dd_regr_model == "cubic":
            polynomial_order = 3
            fit_parameters = np.polyfit(data_all_x, data_all_y, polynomial_order)
            dendr_diam_parameters[dict_key] = {
                "intercept": fit_parameters[3],
                "slope": fit_parameters[2],
                "square": fit_parameters[1],
                "cube": fit_parameters[0],
            }
        elif dd_regr_model == "exponential":

            def exp_func(x, a, b):
                return a + np.exp(x / b)

            fit_parameters, pcov = opt.curve_fit(
                exp_func, data_all_x, data_all_y, p0=[0, 1]
            )
            dendr_diam_parameters[dict_key] = {
                "constant": fit_parameters[0],
                "lamda": fit_parameters[1],
            }
        elif dd_regr_model == "powerlaw":

            def power_func(E, a, b):
                return a * np.power(E, b)

            fit_parameters, pcov = opt.curve_fit(
                power_func, data_all_x, data_all_y, p0=[1, 1]
            )

            a = fit_parameters[0]
            b = fit_parameters[1]

            # Save the parameters
            dendr_diam_parameters[dict_key] = {
                "a": a,
                "b": b,
            }

        dd_model_caption = f"All data {dd_regr_model} fit"

        self.project_data["dd_vs_ecc"] = {
            "data_all_x": data_all_x,
            "data_all_y": data_all_y,
            "fit_parameters": fit_parameters,
            "dd_model_caption": dd_model_caption,
            "title": f"DF diam wrt ecc for {ret.gc_type} type, {dd_model_caption} dataset",
        }

        return dendr_diam_parameters

    def _get_gc_pixel_parameters(
        self,
        ret: Retina,
        gc: Any,
        ecc2dd_params: dict,
    ) -> Any:
        """
        Place RF images to pixel space.

        Calculates the dendritic diameter as a function of eccentricity for each ganglion cell unit.
        This dendritic diameter is then used to calculate the pixel size in micrometers for each unit.
        The minimum pixel size is used to determine the new image stack sidelength.

        Notes
        -----
        `gc_pos_ecc_mm` is expected to be slightly different each time because of the placement optimization process.

        Parameters
        ----------
        ret : Retina
            The retina model instance.
        gc : GanglionCellModel
            The ganglion cell model instance.
        ecc2dd_params : dict
            Dendritic diameter parameters obtained from fitting.

        Returns
        -------
        GanglionCellModel
            Updated ganglion cell model with pixel parameters.
        """
        gc_pos_ecc_mm = np.array(gc.df.pos_ecc_mm.values)

        experimental_metadata = self.experimental_archive["experimental_metadata"]
        exp_um_per_pix = experimental_metadata["data_microm_per_pix"]
        # Mean fitted dendritic diameter for the original experimental data

        exp_dd_um = self.DoG_model.exp_cen_radius_mm * 2 * 1000  # in micrometers
        exp_pix_per_side = experimental_metadata["data_spatialfilter_height"]

        # Get rf diameter vs eccentricity
        dict_key = "{0}_{1}".format(ret.gc_type, ret.dd_regr_model)
        parameters = ecc2dd_params[dict_key]

        match ret.dd_regr_model:
            case "linear" | "quadratic" | "cubic":
                lit_dd_at_gc_ecc_um = np.polyval(
                    [
                        parameters.get("cube", 0),
                        parameters.get("square", 0),
                        parameters.get("slope", 0),
                        parameters.get("intercept", 0),
                    ],
                    gc_pos_ecc_mm,
                )
            case "exponential":
                lit_dd_at_gc_ecc_um = parameters.get("constant", 0) + np.exp(
                    gc_pos_ecc_mm / parameters.get("lamda", 0)
                )
            case "powerlaw":
                # Calculate dendritic diameter from the power law relationship
                # D = a * E^b, where E is the eccentricity and D is the dendritic diameter
                a = parameters["a"]
                b = parameters["b"]
                # Eccentricity in mm, dendritic diameter in um
                lit_dd_at_gc_ecc_um = a * np.power(gc_pos_ecc_mm, b)
            case _:
                raise ValueError(f"Unknown dd_regr_model: {ret.dd_regr_model}")

        # Assuming the experimental data reflects the eccentricity for
        # VAE mtx generation
        gc_scaling_factors = lit_dd_at_gc_ecc_um / exp_dd_um
        gc_um_per_pix = gc_scaling_factors * exp_um_per_pix

        # Get min and max values of gc_um_per_pix
        new_um_per_pix = np.min(gc_um_per_pix)
        max_um_per_pix = np.max(gc_um_per_pix)

        # Get new img stack sidelength whose pixel size = min(gc_um_per_pix),
        new_pix_per_side = int(
            np.round((max_um_per_pix / new_um_per_pix) * exp_pix_per_side)
        )

        # Save scaling factors to gc_df for VAE model type
        gc.df["gc_scaling_factors"] = gc_scaling_factors
        # The pixel grid will be fixed for all units, but the unit eccentricities vary.
        # Thus we need to zoom units to the same size.
        gc.df["zoom_factor"] = gc_um_per_pix / new_um_per_pix

        # Set gc img parameters.
        gc.um_per_pix = new_um_per_pix
        gc.pix_per_side = new_pix_per_side
        gc.um_per_side = new_um_per_pix * new_pix_per_side

        gc.exp_pix_per_side = exp_pix_per_side

        return gc

    def _scale_DoG_amplitudes(self, gc: Any) -> Any:
        """
        Scale DoG amplitudes for each unit based on the center volume.

        Parameters
        ----------
        gc : GanglionCellModel
            The ganglion cell model instance.

        Returns
        -------
        GanglionCellModel
            Updated ganglion cell model with scaled amplitude values.
        """
        cen_vol_mm3 = self.DoG_model._get_center_volume(gc)

        gc.df["relat_sur_ampl"] = gc.df["ampl_s"] / gc.df["ampl_c"]

        # This normalizing factor sets center volume to one (sum of all pixel values in data)
        ampl_c_norm = 1 / cen_vol_mm3
        ampl_s_norm = gc.df["relat_sur_ampl"] / cen_vol_mm3

        gc.df["ampl_c_norm"] = ampl_c_norm
        gc.df["ampl_s_norm"] = ampl_s_norm

        return gc

    def _get_ecc_from_dd(
        self, dendr_diam_parameters: dict, dd_regr_model: str, dd: float
    ) -> float:
        """
        Given the parameters of a polynomial and a dendritic diameter (dd), find the corresponding eccentricity.

        Parameters
        ----------
        dendr_diam_parameters : dict
            Dictionary containing dendritic diameter parameters.
        dd_regr_model : str
            The model used for fitting the dendritic diameter vs eccentricity relationship.
        dd : float
            Dendritic diameter.

        Returns
        -------
        float

        """
        # Get the parameters of the polynomial
        params = dendr_diam_parameters[f"{self.gc_type}_{dd_regr_model}"]

        if dd_regr_model == "linear":
            # For a linear equation, we can solve directly
            # y = mx + c => x = (y - c) / m
            return (dd - params["intercept"]) / params["slope"]
        elif dd_regr_model in ["quadratic", "cubic"]:
            # For quadratic and cubic equations, we need to solve numerically
            # Set up the polynomial equation
            def equation(x):
                if dd_regr_model == "quadratic":
                    return (
                        params["square"] * x**2
                        + params["slope"] * x
                        + params["intercept"]
                        - dd
                    )
                elif dd_regr_model == "cubic":
                    return (
                        params["cube"] * x**3
                        + params["square"] * x**2
                        + params["slope"] * x
                        + params["intercept"]
                        - dd
                    )

            # Solve the equation numerically and return the root
            # We use 1 as the initial guess
            return opt.root(equation, 1).x[0]

        elif dd_regr_model == "powerlaw":
            # For the powerlaw (power law) model, we can solve directly using the inversion
            # D = aE^b => E = (D/a)^(1/b)
            a = params["a"]
            b = params["b"]
            return np.power(dd / a, 1 / b)

    def _merge_density_data(self, lit: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge ganglion cell density data from literature.

        Parameters
        ----------
        lit : dict
            Dictionary containing literature data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Merged eccentricity and density data.

        Notes
        -----
        The ganglion cell density data is merged from two datasets.
        The first dataset contains cone density data for the foveal region.
        The cone density data is multiplied with a decaying exponential multiplier
        to estimate the foveal ganglion cell density.
        The second dataset contains data for the peripheral region.
        The data from the second dataset is filtered to include only values
        above 3 mm eccentricity.

        """
        # Get merged eccentricity. For dataset 2 (ganglion cell density),
        # we take only values over 3 mm eccentricity, outside the foveal displacement.
        gc_eccentricity_1 = lit["gc_eccentricity_1"]
        gc_eccentricity_2_mask = lit["gc_eccentricity_2"] > 3  # 3 mm
        gc_eccentricity_2 = lit["gc_eccentricity_2"][gc_eccentricity_2_mask]
        gc_eccentricity = np.concatenate((gc_eccentricity_1, gc_eccentricity_2))

        # Get foveal gc density according to Wässle_1989_Nature
        # This is estimated from cone density with a varying multiplier
        cone_density_1 = lit["gc_density_1"]
        x_points, y_scaler, f_name = lit["gc_density_1_scaling_data_and_function"]
        func = eval(f"self.retina_math.{f_name}")
        popt, _ = opt.curve_fit(func, x_points, y_scaler, p0=[3, -1, 1])
        A, B, C = popt

        gc_density_1 = func(gc_eccentricity_1, A, B, C) * cone_density_1
        gc_density = np.concatenate(
            (gc_density_1, lit["gc_density_2"][gc_eccentricity_2_mask])
        )

        # Sort according to eccentricity
        sort_index = np.argsort(gc_eccentricity)
        gc_eccentricity = gc_eccentricity[sort_index]
        gc_density = gc_density[sort_index]

        return gc_eccentricity, gc_density

    # Interface methods
    def get_concrete_components(self) -> None:
        """
        Compile builder components from gc type, response type,
        spatial model, temporal model, and DoG model.
        """

        retina = self.retina

        match retina.gc_type:
            case "parasol":
                ganglion_cell = GanglionCellParasol()
            case "midget":
                ganglion_cell = GanglionCellMidget()

        match retina.dog_model_type:
            case "ellipse_fixed":
                DoG_model = DoGModelEllipseFixed(retina, self.fit, self.retina_math)
            case "ellipse_independent":
                DoG_model = DoGModelEllipseIndependent(
                    retina, self.fit, self.retina_math
                )
            case "circular":
                DoG_model = DoGModelCircular(retina, self.fit, self.retina_math)

        distribution_sampler = DistributionSampler()

        match retina.spatial_model_type:
            case "DOG":
                spatial_model = SpatialModelDOG(
                    DoG_model,
                    distribution_sampler,
                    self.retina_vae,
                    self.fit,
                    self.retina_math,
                    self.viz,
                )
            case "VAE":
                spatial_model = SpatialModelVAE(
                    DoG_model,
                    distribution_sampler,
                    self.retina_vae,
                    self.fit,
                    self.retina_math,
                    self.viz,
                )

        match retina.temporal_model_type:
            case "fixed":
                temporal_model = TemporalModelFixed(
                    ganglion_cell,
                    DoG_model,
                    distribution_sampler,
                    self.retina_math,
                    self.device,
                )
            case "dynamic":
                temporal_model = TemporalModelDynamic(
                    ganglion_cell,
                    DoG_model,
                    distribution_sampler,
                    self.retina_math,
                    self.device,
                )
            case "subunit":
                temporal_model = TemporalModelSubunit(
                    ganglion_cell,
                    DoG_model,
                    distribution_sampler,
                    self.retina_math,
                    self.device,
                )

        self._ganglion_cell = ganglion_cell
        self._spatial_model = spatial_model
        self._DoG_model = DoG_model
        self._temporal_model = temporal_model
        self._sampler = distribution_sampler

    def fit_cell_density_data(self) -> None:
        """
        Read literature data and fit ganglion cell and cone density with respect to eccentricity.
        """
        ret = self.retina
        lit = self.experimental_archive

        def _fit_density(
            eccentricity: np.ndarray, density: np.ndarray, cell_type: str
        ) -> np.ndarray:
            match cell_type:
                case "gc" | "gc_control":
                    this_function = self.retina_math.double_exponential_func
                    # More reasonable initial guesses for ganglion cells
                    p0 = [np.max(density), -0.3, np.max(density) / 10, -0.05]
                case "cone" | "bipolar":
                    this_function = self.retina_math.triple_exponential_func
                    # More reasonable initial guesses for cones/bipolar cells
                    p0 = [
                        np.max(density),
                        -0.3,
                        np.max(density) / 10,
                        -0.05,
                        np.max(density) / 100,
                        -0.01,
                    ]
                case _:
                    raise ValueError(f"Unknown cell type: {cell_type}")

            # Define the objective function in log space that works with vectors
            def log_objective(x: np.ndarray, *params) -> np.ndarray:
                y_pred = this_function(x, *params)
                # Handle potential negative or zero values
                # Set minimum value to positive number
                y_pred = np.maximum(y_pred, 1e-10)
                return np.log(y_pred)

            # Make sure we have numpy arrays
            eccentricity = np.asarray(eccentricity)
            density = np.asarray(density)

            # Fit in log space with better optimization parameters
            fit_parameters, _ = opt.curve_fit(
                log_objective,
                eccentricity,
                np.log(density),
                p0=p0,
                maxfev=10000,
                ftol=1e-6,
                method="lm",
            )

            # Save fit function and data for visualization
            setattr(self, f"{cell_type}_fit_function", this_function)
            self.project_data[f"{cell_type}_n_vs_ecc"] = {
                "fit_parameters": fit_parameters,
                "cell_eccentricity": eccentricity,
                "cell_density": density,
                "function": this_function,
            }
            return fit_parameters

        # ganglion cell density parameters
        gc_eccentricity, gc_density = self._merge_density_data(lit)
        gc_params = _fit_density(gc_eccentricity, gc_density, "gc")

        # ganglion cell density control data for visualization
        _ = _fit_density(
            lit["gc_control_eccentricity"], lit["gc_control_density"], "gc_control"
        )

        # Cone density parameters
        cone_params = _fit_density(
            lit["cone_eccentricity"], lit["cone_density"], "cone"
        )

        # Bipolar to cone ratio data at 6.5 mm eccentricity, from Boycott_1991_EurJNeurosci
        # We assume constant bipolar-to-cone ration across eccentricity

        bipolar_df = lit["bipolar_df"]
        bipolar_df = bipolar_df.set_index(bipolar_df.columns[0])
        b2c_ratio_s = bipolar_df.loc["Bipolar_cone_ratio"]

        # Pick correct bipolar types, then bipolar to cone ratio at 6.5 mm.
        bipolar_types = ret.bipolar2gc_dict[ret.gc_type][ret.response_type]
        b2c_ratio_str = b2c_ratio_s[bipolar_types].values
        b2c_ratios = np.array([float(x) for x in b2c_ratio_str])

        # Take the sum of relevant bipolar values.
        # If we later learn that the two types are not equal, we need to change this.
        b2c_ratio = np.sum(b2c_ratios)
        # The bipolar mock density follows cone density, assuming constant ratio
        bipolar_mock_density = lit["cone_density"] * b2c_ratio
        bipolar_fit_parameters = _fit_density(
            lit["cone_eccentricity"], bipolar_mock_density, "bipolar"
        )
        ret.gc_density_params = gc_params
        ret.cone_density_params = cone_params
        ret.bipolar_density_params = bipolar_fit_parameters
        ret.selected_bipolars_df = bipolar_df[bipolar_types]

        self.retina = ret

    def place_units(self) -> None:
        """
        Place ganglion cells, cones, and bipolar cells.
        """

        ret = self.retina
        gc = self.ganglion_cell

        ret = self._get_gc_proportion(ret)

        # Initial Positioning by Group
        print("\nPlacing units...\n")
        (
            eccentricity_groups,
            sector_surface_areas_mm2,
            gc_initial_pos,
            gc_density,
            cone_initial_pos,
            cone_density,
            bipolar_initial_pos,
            bipolar_density,
        ) = self._initialize_positions_by_group(ret)

        # Optimize positions
        gc_optimized_pos_pol, gc_optimized_pos_mm = self._optimize_positions(
            ret, gc_initial_pos, gc_density, ret.gc_placement_parameters
        )
        print("Optimized positions for ganglion cells")
        cone_optimized_pos_pol, cone_optimized_pos_mm = self._optimize_positions(
            ret, cone_initial_pos, cone_density, ret.cone_placement_parameters
        )
        print("Optimized positions for cones")
        bipolar_optimized_pos_pol, bipolar_optimized_pos_mm = self._optimize_positions(
            ret, bipolar_initial_pos, bipolar_density, ret.bipolar_placement_parameters
        )
        print("Optimized positions for bipolar cells")

        # Assign ganglion cell positions to gc_df
        gc.df["pos_ecc_mm"] = gc_optimized_pos_pol[:, 0]
        gc.df["pos_polar_deg"] = gc_optimized_pos_pol[:, 1]
        gc.df["ecc_group_idx"] = np.concatenate(eccentricity_groups)
        gc.n_units = gc.df.shape[0]

        ret.sector_surface_areas_mm2 = sector_surface_areas_mm2

        # Cones will be attached to gcs after the final position of gcs is known after
        # repulsion.
        ret.cone_optimized_pos_mm = cone_optimized_pos_mm
        ret.cone_optimized_pos_pol = cone_optimized_pos_pol
        ret.bipolar_optimized_pos_mm = bipolar_optimized_pos_mm

        self.ganglion_cell = gc
        self.retina = ret

    def create_spatial_receptive_fields(self) -> None:
        """
        Create spatial receptive fields (RFs) for the retinal ganglion cells (RGCs).

        The RFs are generated using either a generative variational autoencoder (VAE) model or
        a fit to the data from the literature. The VAE model is trained on the data from
        the literature and generates RFs that are similar to the literature data.

        The RFs are generated in the following steps:
        1) Get the VAE model to generate receptive fields.
        2) "Bad fit loop", provides eccentricity-scaled vae rfs with good DoG fits (error < 3SD from mean).
        3) Get center masks.
        4) Sum separate rf images onto one retina pixel matrix.
        5) Apply repulsion adjustment to the receptive fields. Note that this will
        change the positions of the receptive fields.
        6) Redo the good fits for final statistics.


        RF become resampled, and the resolution will change if
        eccentricity is different from eccentricity of the original data.
        """
        ret = self.retina
        gc = self.ganglion_cell

        # Get fit parameters for dendritic field diameter (dd) with respect to eccentricity (ecc).
        # Data from Watanabe_1989_JCompNeurol, Perry_1984_Neurosci and Goodchild_1996_JCompNeurol
        ecc2dd_params = self._fit_dd_vs_ecc(ret, gc)

        # # Quality control: check that the fitted dendritic diameter is close to the original data
        # # Frechette_2005_JNeurophysiol datasets: 9.7 mm (45°); 9.0 mm (41°); 8.4 mm (38°)
        # # Estimate the orginal data eccentricity from the fit to full eccentricity range
        # # TODO: move to integration tests
        # exp_rad = self.exp_cen_radius_mm * 2 * 1000
        # self.ecc_limit_for_dd_fit_mm = np.inf
        # dd_ecc_params_full = self._fit_dd_vs_ecc()
        # data_ecc_mm = self._get_ecc_from_dd(dd_ecc_params_full, dd_regr_model, exp_rad)
        # data_ecc_deg = data_ecc_mm * self.deg_per_mm  # 37.7 deg

        # Endow units with spatial elliptical receptive fields.
        # Units become mm unless specified in column names.
        # gc.df may be updated silently below

        gc = self._get_gc_pixel_parameters(ret, gc, ecc2dd_params)

        ret, gc, viz_whole_ret_img = self.spatial_model.create(ret, gc)

        self.project_data.update(self.spatial_model.project_data)

        self.project_data["ret"] = {
            "img_ret": viz_whole_ret_img,
            "img_ret_masked": ret.whole_ret_img_mask,
            "img_ret_adjusted": ret.whole_ret_img,
        }

        # Add fitted DoG center area to gc_df for visualization
        gc = self.DoG_model._add_center_fit_area_to_df(gc)

        # Scale center and surround amplitude: center Gaussian volume in pixel space becomes one
        # Surround amplitude is scaled relative to center volume of one
        gc = self._scale_DoG_amplitudes(gc)

        # Set more project_data for later visualization
        self.project_data["dd_vs_ecc"]["dd_DoG_x"] = gc.df.pos_ecc_mm.values
        self.project_data["dd_vs_ecc"]["dd_DoG_y"] = gc.df.den_diam_um.values

        self.ganglion_cell = gc
        self.retina = ret

    def connect_units(self) -> None:
        """
        Connect units according to the model.
        """
        ret = self.retina
        gc = self.ganglion_cell

        ret = self.temporal_model.connect_units(ret, gc)

        self.retina = ret

    def create_temporal_receptive_fields(self) -> None:
        """
        Create temporal receptive fields.
        """
        ret = self.retina
        gc = self.ganglion_cell

        gc = self.temporal_model.create(ret, gc)
        ret = self.temporal_model._fit_cone_noise_vs_freq(ret)

        self.ganglion_cell = gc
        self.retina = ret

    def create_tonic_drive(self) -> None:
        """
        Create tonic drive for ganglion cells.
        """
        gc = self.ganglion_cell

        tonic_df = self.DoG_model.exp_univariate_stat[
            self.DoG_model.exp_univariate_stat["domain"] == "tonic"
        ]
        for param_name, row in tonic_df.iterrows():
            shape, loc, scale, distribution, _ = row
            gc.df[param_name] = self.sampler.sample_univariate(
                shape, loc, scale, len(gc.df), distribution
            )

        self.ganglion_cell = gc


class RetinaBuildDirector:
    """
    A class that directs the construction of the retina by coordinating the builder instance.

    The director follows the RetinaBuildInterface and coordinates the execution of the steps
    to build the retina, such as fitting cell density data, placing units, and generating
    receptive fields.

    Attributes
    ----------
    builder : RetinaBuildInterface
        An instance of a class implementing RetinaBuildInterface to construct the retina.
    """

    def __init__(self, builder: RetinaBuildInterface) -> None:
        """
        Initialize the RetinaBuildDirector with a builder.

        Parameters
        ----------
        builder : RetinaBuildInterface
            The builder instance used to construct the retina.
        """
        self.builder = builder

    def construct_retina(self) -> None:
        """
        Direct the builder to construct the retina by performing the following steps:
        1) Retrieve the concrete components of the retina.
        2) Fit the ganglion cell and cone density data.
        3) Place the units (ganglion cells, cones, bipolars).
        4) Create spatial receptive fields.
        5) Connect the units based on the model.
        6) Create temporal receptive fields.
        7) Create tonic drive for the ganglion cells.
        """
        self.builder.get_concrete_components()
        self.builder.fit_cell_density_data()
        self.builder.place_units()
        self.builder.create_spatial_receptive_fields()
        self.builder.connect_units()
        self.builder.create_temporal_receptive_fields()
        self.builder.create_tonic_drive()

    def get_retina(self) -> tuple[Retina, Any]:
        """
        Retrieve the constructed retina and ganglion cell.

        Returns
        -------
        tuple[Retina, GanglionCell]
            A tuple containing the constructed retina and the associated ganglion cell.
        """
        ret = self.builder.retina
        gc = self.builder.ganglion_cell
        return ret, gc


class ConstructRetina(PrintableMixin):
    """
    Constructs the ganglion cell mosaic and associated retinal components.

    This class builds the retina model by coordinating various components such as
    data I/O, visualization, fitting functions, and mathematical operations.
    All spatial parameters are saved to the DataFrame `gc.df`.

    Attributes
    ----------
    config : config
        The config object containing configuration and parameters.
    data_io : DataIO
        An object for handling data input/output operations.
    viz : Visualization
        An object for handling visualization tasks.
    fit : Fit
        An object providing fitting functions.
    retina_vae : RetinaVAE
        An object for variational autoencoder-related operations.
    retina_math : RetinaMath
        An object providing mathematical operations for retina modeling.
    project_data : dict
        A dictionary for storing project-related data.
    get_xy_from_npz : Callable[[Any], Tuple[np.ndarray, np.ndarray]]
        Function to extract X and Y data from a .npz file.
    device : torch.device
        The device (CPU/GPU) to use for computations.
    """

    def __init__(
        self,
        config: Any,
        data_io: Any,
        viz: Any,
        fit: Any,
        retina_vae: Any,
        retina_math: RetinaMath,
        project_data: dict,
        get_xy_from_npz: Callable[[Any], Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """
        Initialize the ConstructRetina instance with dependencies.

        Parameters
        ----------
        config : Configuration
            The config object containing configuration and parameters.
        data_io : DataIO
            An object for handling data input/output operations.
        viz : Visualization
            An object for handling visualization tasks.
        fit : Fit
            An object providing fitting functions.
        retina_vae : RetinaVAE
            An object for variational autoencoder-related operations.
        retina_math : RetinaMath
            An object providing mathematical operations for retina modeling.
        project_data : dict
            A dictionary for storing project-related data.
        get_xy_from_npz : Callable[[Any], Tuple[np.ndarray, np.ndarray]]
            Function to extract X and Y data from a .npz file.
        """
        # Dependency injection at ProjectManager construction
        self._config = config
        self._data_io = data_io
        self._viz = viz
        self._fit = fit
        self._retina_vae = retina_vae
        self._retina_math = retina_math
        self._project_data = project_data

        # Set additional methods
        self.get_xy_from_npz = get_xy_from_npz

        # Set attributes
        self._device = self.config.device

        # Make or read fits
        retina_parameters = self.config.retina_parameters

        if "spatial_model_type" in retina_parameters and retina_parameters[
            "spatial_model_type"
        ] in ["VAE"]:
            self.vae_run_mode = self.config.vae_train_parameters["vae_run_mode"]

        self.spatial_rfs_file_filename = []
        self.ret_filename = []
        self.mosaic_filename = []

    @property
    def config(self):
        """Configuration object containing configuration and parameters."""
        return self._config

    @property
    def data_io(self):
        """DataIO object for handling data input/output operations."""
        return self._data_io

    @property
    def viz(self):
        """Visualization object for handling visualization tasks."""
        return self._viz

    @property
    def fit(self):
        """Fit object providing fitting functions."""
        return self._fit

    @property
    def retina_vae(self):
        """RetinaVAE object for variational autoencoder operations."""
        return self._retina_vae

    @property
    def retina_math(self):
        """RetinaMath object providing mathematical operations for retina modeling."""
        return self._retina_math

    @property
    def device(self) -> torch.device:
        match self._device:
            case "cuda":
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            case "cpu":
                self._device = torch.device("cpu")
            case _:
                self._device = torch.device("cpu")
        return self._device

    @property
    def project_data(self) -> dict:
        """Dictionary for storing project-related data."""
        return self._project_data

    def _build_exists(self, retina_parameters: dict) -> bool:
        """
        Check if the build exists by verifying if the hash exists in project directories.

        Parameters
        ----------
        retina_parameters : dict
            Dictionary containing retina configuration parameters.

        Returns
        -------
        bool
            True if the build exists, False otherwise.
        """
        if retina_parameters["force_retina_build"]:
            print("Forcing the build of the retina.")
            return False

        string_keys = [k for k, i in retina_parameters.items() if isinstance(i, str)]
        retina_parameters_hash = retina_parameters["retina_parameters_hash"]
        requested_file_name_keys = [
            n
            for n in string_keys
            if "file" in n and retina_parameters_hash in retina_parameters[n]
        ]

        hash_files = [
            self.data_io.parse_path(retina_parameters[this_file])
            for this_file in requested_file_name_keys
        ]
        if all(hash_files):
            print(
                "Retina construction hash exists. Continuing without building the retina."
            )
            return True
        else:
            print("Retina construction hash does not exist. Building the retina.")
            return False

    def _get_density_from(self, filepaths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get density data from given filepaths.

        Parameters
        ----------
        filepaths : List[str]
            List of file paths to density data files.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Arrays of cell eccentricity and cell density.
        """
        cell_eccentricity = np.array([])
        cell_density = np.array([])
        for filepath in filepaths:
            density = self.data_io.load_data(filepath)
            _eccentricity = np.squeeze(density["Xdata"])
            _density = np.squeeze(density["Ydata"])
            cell_eccentricity = np.concatenate((cell_eccentricity, _eccentricity))
            cell_density = np.concatenate((cell_density, _density))

        # Sort and scale data
        index = np.argsort(cell_eccentricity)
        cell_eccentricity, cell_density = (
            cell_eccentricity[index],
            cell_density[index],
        )
        return cell_eccentricity, cell_density

    def _get_literature_data(self) -> dict:
        """
        Read the literature data from the data files.

        Returns
        -------
        dict
            A dictionary containing literature data for retina modeling.
        """
        data_io = self.data_io

        # Read literature data paths from the config
        files = self.config.literature_data_files
        literature = {}

        # Get unit density data
        gc_ecc_1, gc_density_1 = self._get_density_from([files["gc_density_1_path"]])
        literature["gc_eccentricity_1"] = gc_ecc_1
        literature["gc_density_1"] = gc_density_1
        literature["gc_density_1_scaling_data_and_function"] = (
            self.config.literature_data_files["gc_density_1_scaling_data_and_function"]
        )
        gc_ecc_2, gc_density_2 = self._get_density_from([files["gc_density_2_path"]])
        literature["gc_eccentricity_2"] = gc_ecc_2
        literature["gc_density_2"] = gc_density_2

        gc_control_ecc, gc_control_density = self._get_density_from(
            [files["gc_density_control_path"]]
        )
        literature["gc_control_eccentricity"] = gc_control_ecc
        literature["gc_control_density"] = gc_control_density

        cone_filepaths = [
            self.config.literature_data_files["cone_density1_path"],
            self.config.literature_data_files["cone_density2_path"],
        ]
        cone_eccentricity, cone_density = self._get_density_from(cone_filepaths)
        literature["cone_eccentricity"] = cone_eccentricity
        literature["cone_density"] = cone_density

        bipolar_df = self.data_io.load_data(files["bipolar_table_path"])
        literature["bipolar_df"] = bipolar_df

        # Get dendritic diameter data
        dendr_diam1 = self.data_io.load_data(files["dendr_diam1_path"])
        literature["dendr_diam1"] = dendr_diam1

        dendr_diam2 = self.data_io.load_data(files["dendr_diam2_path"])
        literature["dendr_diam2"] = dendr_diam2

        dendr_diam3 = self.data_io.load_data(files["dendr_diam3_path"])
        literature["dendr_diam3"] = dendr_diam3

        literature["dendr_diam_units"] = files["dendr_diam_units"]

        # Get Benardete & Kaplan model parameters
        temporal_params_BK = self.data_io.load_data(files["temporal_BK_model_path"])
        literature["temporal_parameters_BK"] = temporal_params_BK

        # Get cone response and noise data
        cone_response = self.data_io.load_data(
            self.config.literature_data_files["cone_response_path"]
        )
        cone_frequency_data, cone_power_data = self.get_xy_from_npz(cone_response)
        literature["cone_frequency_data"] = cone_frequency_data
        literature["cone_power_data"] = cone_power_data

        cone_noise_wc = self.config.retina_parameters["cone_general_parameters"][
            "cone_noise_wc"
        ]
        literature["cone_noise_wc"] = cone_noise_wc

        cone_noise = self.data_io.load_data(
            self.config.literature_data_files["cone_noise_path"]
        )
        noise_frequency_data, noise_power_data = self.get_xy_from_npz(cone_noise)
        literature["noise_frequency_data"] = noise_frequency_data
        literature["noise_power_data"] = noise_power_data

        # Get bipolar rectification index data
        response_type = self.config.retina_parameters["response_type"]
        RI_values_npz = self.data_io.load_data(
            self.config.literature_data_files[f"parasol_{response_type}_RI_values_path"]
        )
        g_sur_values, target_RI_values = self.get_xy_from_npz(RI_values_npz)
        literature["g_sur_values"] = g_sur_values
        literature["target_RI_values"] = target_RI_values

        return literature

    def _append_dog_metadata_parameters(self, data: dict) -> dict:
        """
        Append dog metadata to the data dictionary.

        Parameters
        ----------
        data : dict
            The data dictionary to which metadata will be appended.

        Returns
        -------
        dict
            The updated data dictionary with metadata.
        """
        data["experimental_metadata"] = self.config.experimental_metadata
        return data

    def _get_statistics_config(
        self,
        gc_type: str,
        response_type: str,
        dog_model_type: str,
        experimental_archive: dict,
    ) -> None:

        path = experimental_archive["experimental_metadata"]["exp_rf_stat_folder"]
        filename_stem = f"{gc_type}_{response_type}_{dog_model_type}"

        filetypes = [
            "exp_univariate_stat.csv",
            "spatial_multivariate_stat.csv",
            "temporal_multivariate_stat.csv",
            "exp_cen_radius_mm.npy",
        ]
        self.dog_config = {
            "path": path,
            "filename_stem": filename_stem,
            "filetypes": filetypes,
        }

        filename_stem = f"{gc_type}_{response_type}"

        self.vae_config = {
            "path": path,
            "filename_stem": filename_stem,
        }

    def _update_dog_statistics_on_disc(self, dog_statistics) -> None:
        """
        Updates the dog statistics on disk with the current experimental statistics.
        """
        path = self.dog_config["path"]
        filename_stem = self.dog_config["filename_stem"]
        filetypes = self.dog_config["filetypes"]

        for filetype in filetypes:
            filename = f"{filename_stem}_{filetype}"
            filepath = path / filename
            my_object = dog_statistics[filetype[:-4]]
            if isinstance(my_object, pd.DataFrame):
                my_object.to_csv(filepath, index_label=my_object.index.name)
            elif isinstance(my_object, np.float64):
                np.save(filepath, my_object)
            print(f"Updated {filetype} statistics on disk.")

    def _get_dog_statistics(self, retina_parameters) -> None:
        """
        Loads the dog statistics from disk into the current instance.
        If not found, recalculates the statistics by fitting the model to
        experimental data and saves the statistics on disk.
        """
        path = self.dog_config["path"]
        filename_stem = self.dog_config["filename_stem"]
        filetypes = self.dog_config["filetypes"]

        dog_statistics = {}
        try:
            for filetype in filetypes:
                filename = f"{filename_stem}_{filetype}"
                filepath = path / filename
                if filetype.endswith(".csv"):
                    df = self.data_io.load_data(full_path=filepath)
                    dog_statistics[filetype[:-4]] = df
                elif filetype.endswith(".npy"):
                    exp_cen_radius_mm = self.data_io.load_data(full_path=filepath)
                    exp_cen_radius_mm = np.float64(exp_cen_radius_mm.item())
                    dog_statistics[filetype[:-4]] = exp_cen_radius_mm
                print(f"Loaded {filetype} statistics from disk.")
        except:
            self.fit.client(
                retina_parameters["gc_type"],
                retina_parameters["response_type"],
                fit_type="experimental",
                dog_model_type=retina_parameters["dog_model_type"],
                mark_outliers_bad=False,
            )

            dog_statistics["exp_cen_radius_mm"] = self.fit.receptive_field_sd.center_sd
            (
                dog_statistics["exp_univariate_stat"],
                dog_statistics["spatial_multivariate_stat"],
                dog_statistics["temporal_multivariate_stat"],
            ) = self.fit.get_experimental_statistics()

            self._update_dog_statistics_on_disc(dog_statistics)

        return dog_statistics

    def _update_latent_stats(self, vae_latent_stats: np.ndarray) -> None:

        filepath = self._get_vae_statistics_filepath()
        np.save(filepath, vae_latent_stats)

    def _get_vae_statistics_filepath(self) -> str:
        """
        Returns the file path for the VAE statistics based on the config.
        """
        path = self.vae_config["path"]
        filename_stem = self.vae_config["filename_stem"]
        return str(path / f"{filename_stem}_vae_latent_stats.npy")

    def _get_vae_statistics(self, retina_vae: Any) -> np.ndarray:
        """
        Loads the VAE statistics from disk. If not found, recalculates the statistics
        by augmenting the original data and getting the encoded samples from the VAE model.
        Saves the statistics on disk.
        """

        filepath = self._get_vae_statistics_filepath()
        retina_vae.client()

        try:
            raise FileNotFoundError("Bypassing VAE statistics loading for testing.")
            vae_latent_stats = np.load(filepath)
            print("Loaded VAE statistics from disk.")

        except FileNotFoundError:
            augmentation_dict = self.config.vae_train_parameters.get(
                "augmentation_dict", {}
            )

            retina_vae.get_and_split_experimental_data()
            retina_vae.train_loader = retina_vae.augment_and_get_dataloader(
                data_type="train", augmentation_dict=augmentation_dict
            )
            retina_vae.val_loader = retina_vae.augment_and_get_dataloader(
                data_type="val", augmentation_dict=augmentation_dict
            )
            retina_vae.test_loader = retina_vae.augment_and_get_dataloader(
                data_type="test", shuffle=False
            )

            # Get the latent space data
            train_df = retina_vae.get_encoded_samples(
                dataset=retina_vae.train_loader.dataset
            )
            valid_df = retina_vae.get_encoded_samples(
                dataset=retina_vae.val_loader.dataset
            )
            test_df = retina_vae.get_encoded_samples(
                dataset=retina_vae.test_loader.dataset
            )
            latent_df = pd.concat(
                [train_df, valid_df, test_df], axis=0, ignore_index=True
            )

            # Extract data from latent_df into a numpy array from columns whose title include "EncVariable"
            vae_latent_stats = latent_df.filter(regex="EncVariable").to_numpy()

            self._update_latent_stats(vae_latent_stats)

        return vae_latent_stats

    def _get_all_experimental_statistics(
        self, experimental_archive: dict, retina_parameters: dict
    ) -> dict:
        """
        Initializes fit and retrieves experimental fits for the DoG model.
        If fit statistics are not found on disk, it fits the model and saves the statistics.

        Parameters
        ----------
        dog_model_type : str
            The type of DoG model to be used for fitting.
        """

        self._get_statistics_config(
            retina_parameters["gc_type"],
            retina_parameters["response_type"],
            retina_parameters["dog_model_type"],
            experimental_archive,
        )

        try:
            dog_statistics = self._get_dog_statistics(retina_parameters)
        except FileNotFoundError:
            print("No experimental statistics or data found on disk.")
            raise

        experimental_archive["dog_statistics"] = dog_statistics

        # # VAE statistics
        # try:
        #     vae_latent_stats = self._get_vae_statistics(self.retina_vae)
        # except FileNotFoundError:
        #     print("No VAE statistics found on disk.")
        #     raise

        # experimental_archive["vae_statistics"] = vae_latent_stats

        return experimental_archive

    def _get_retina_hash_from_core_parameters(self) -> str:
        """
        Calculate the retina hash from core parameters.
        """

        retina_core_parameters = {
            key: self.config.retina_parameters[key]
            for key in self.config.retina_core_parameter_keys
        }

        # Get hash from core parameters which may be updated after import
        self.config.retina_core_parameters = retina_core_parameters
        hashstr = self.config.retina_core_parameters.hash()

        # # delete the retina_core_parameters attribute to avoid confusion
        # del self.config.retina_core_parameters

        return hashstr

    def _set_retina_parameters(self):
        """Sets some retina parameters, specific to the current build."""

        hashstr = self._get_retina_hash_from_core_parameters()
        self.config.retina_parameters["retina_parameters_hash"] = hashstr

        gc_type = self.config.retina_parameters["gc_type"]
        response_type = self.config.retina_parameters["response_type"]

        self.config.retina_parameters["mosaic_filename"] = (
            gc_type + "_" + response_type + "_" + hashstr + "_mosaic.csv"
        )
        self.config.retina_parameters["spatial_rfs_file"] = (
            gc_type + "_" + response_type + "_" + hashstr + "_spatial_rfs.npz"
        )
        self.config.retina_parameters["ret_file"] = (
            gc_type + "_" + response_type + "_" + hashstr + "_ret.npz"
        )
        self.config.retina_parameters["retina_metadata_file"] = (
            gc_type + "_" + response_type + "_" + hashstr + "_metadata.yaml"
        )

        self.mosaic_filename = self.config.retina_parameters["mosaic_filename"]
        self.config.retina_parameters["mosaic_file"] = self.config.retina_parameters[
            "mosaic_filename"
        ]
        self.spatial_rfs_file_filename = self.config.retina_parameters[
            "spatial_rfs_file"
        ]
        self.ret_filename = self.config.retina_parameters["ret_file"]

    def _save_minimal_config_yaml(self):
        """Saves a minimal configuration in a YAML file."""

        main_retina_parameters_list = [
            "gc_type",
            "response_type",
            "spatial_model_type",
            "temporal_model_type",
            "dog_model_type",
            "ecc_limits_deg",
            "pol_limits_deg",
            "model_density",
            "retina_center",
            "force_retina_build",
            "signal_gain",
        ]

        main_retina_parameters = {
            key: value
            for key, value in self.config.retina_parameters.items()
            if key in main_retina_parameters_list
        }

        yaml_filename = self.config.retina_parameters["retina_metadata_file"]
        yaml_filename_full = self.config.output_folder.joinpath(yaml_filename)

        # yaml does not support complex numbers, so we convert to string
        self.config.retina_parameters["retina_center"] = str(
            self.config.retina_parameters["retina_center"]
        )

        self.data_io.save_dict_to_yaml(
            yaml_filename_full,
            main_retina_parameters,
            overwrite=False,
        )

        # And then we change it back to complex number
        self.config.retina_parameters["retina_center"] = complex(
            self.config.retina_parameters["retina_center"]
        )

    def _set_cone_noise_hash(self):
        """Calculate the cone noise hash."""
        cone_noise_dict = self.config.retina_parameters["cone_general_parameters"]

        retina_limits = {
            key: self.config.retina_parameters[key]
            for key in ["ecc_limits_deg", "pol_limits_deg"]
        }
        cone_noise_dict.update(retina_limits)

        stim_duration = {
            key: self.config.visual_stimulus_parameters[key]
            for key in [
                "fps",
                "duration_seconds",
                "baseline_start_seconds",
                "baseline_end_seconds",
            ]
        }

        cone_noise_dict.update(stim_duration)
        self.config.retina_parameters["cone_noise_hash"] = cone_noise_dict.hash()

    def build_retina_client(self) -> None:
        """
        Build the retina using the builder pattern.

        For each call, a new retina is built with the parameters in the `retina_parameters` dictionary.
        After the build, the retina is saved to the output directory.
        """

        # Set build-specific retina parameters and cone noise hash
        self._set_retina_parameters()
        self._set_cone_noise_hash()

        # Save a minimal configuration in a YAML file
        self._save_minimal_config_yaml()

        retina_parameters = self.config.retina_parameters

        if self._build_exists(retina_parameters):
            return

        experimental_archive = self._get_literature_data()
        experimental_archive = self._append_dog_metadata_parameters(
            experimental_archive
        )
        experimental_archive = self._get_all_experimental_statistics(
            experimental_archive, retina_parameters
        )

        retina_parameters["experimental_archive"] = experimental_archive

        retina = Retina(retina_parameters)
        builder = ConcreteRetinaBuilder(
            retina,
            self.retina_math,
            self.fit,
            self.retina_vae,
            self.device,
            self.viz,
        )
        director = RetinaBuildDirector(builder)
        director.construct_retina()
        retina, ganglion_cell = director.get_retina()

        self.save_retina(retina, ganglion_cell)
        self.project_data.construct_retina.update(builder.project_data)

    def save_retina(self, ret: "Retina", gc: Any) -> None:
        """
        Save the constructed retina and ganglion cell data to files.

        Parameters
        ----------
        ret : Retina
            The constructed retina object.
        gc : GanglionCell
            The ganglion cell object associated with the retina.
        """
        print("\nSaving gc and ret data...")
        output_path = self.config.output_folder

        # Save the generated receptive field pixel images, masks, and locations
        spatial_rfs_dict = {
            "gc_img": gc.img,
            "gc_img_mask": gc.img_mask,
            "X_grid_cen_mm": gc.X_grid_cen_mm,
            "Y_grid_cen_mm": gc.Y_grid_cen_mm,
            "um_per_pix": gc.um_per_pix,
            "pix_per_side": gc.pix_per_side,
        }

        self.data_io.save_np_dict_to_npz(
            spatial_rfs_dict, output_path, filename_stem=self.spatial_rfs_file_filename
        )

        # Save retinal attributes
        ret_attributes = [
            "cone_optimized_pos_mm",
            "cone_optimized_pos_pol",
            "cones_to_gcs_weights",
            "cone_noise_parameters",
            "noise_frequency_data",
            "noise_power_data",
            "cone_frequency_data",
            "cone_power_data",
            "cone_noise_power_fit",
            "cones_to_bipolars_center_weights",
            "cones_to_bipolars_surround_weights",
            "bipolar_to_gcs_cen_weights",
            "bipolar_to_gcs_sur_weights",
            "bipolar_optimized_pos_mm",
            "bipolar_nonlinearity_parameters",
            "bipolar_nonlinearity_fit",
            "g_sur_scaled",
            "target_RI_values",
        ]

        ret_dict = {k: getattr(ret, k) for k in ret_attributes if hasattr(ret, k)}

        self.data_io.save_np_dict_to_npz(
            ret_dict, output_path, filename_stem=self.ret_filename
        )

        filepath = output_path.joinpath(self.mosaic_filename)
        print(f"Saving model mosaic to {filepath}")
        gc.df.to_csv(filepath)
