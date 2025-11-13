# Built-in
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import brian2 as b2
import brian2.units as b2u
import brian2cuda  # noqa: F401
import numpy as np
import pandas as pd
import scipy.fftpack as fftpack
import torch
from brian2 import BrianLogger
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.signal import convolve, fftconvolve
from scipy.spatial import Delaunay
from scipy.special import factorial
from scipy.special import gamma as gamma_function
from skimage.transform import resize
from tqdm import tqdm

# Local
from macaqueretina.project.project_utilities_module import PrintableMixin
from macaqueretina.retina.retina_math_module import RetinaMath

BrianLogger.log_level_error()


class GanglionCellBase(ABC):
    """
    Base class for ganglion cells, providing core methods and properties
    for simulating retinal ganglion cell behavior.

    Attributes
    ----------
    device : str
        Device used for computation, e.g., 'cuda' or 'cpu'.
    standalone_tmp_dir : str
        Temporary directory used for Brian2 standalone mode.
    """

    def __init__(self, device: str) -> None:
        self._device = device

    @property
    def device(self) -> str:
        """
        Get the device being used for computation and adjust it
        for Brian2 standalone mode if necessary.

        Returns
        -------
        str
            The adjusted device string, e.g., 'cuda_standalone' or 'cpp_standalone'.
        """
        match self._device:
            case "cuda":
                self._device = "cuda_standalone"
            case "cpu":
                self._device = "cpp_standalone"
        return self._device

    @abstractmethod
    def get_BK_parameters(self) -> np.ndarray:
        pass

    @abstractmethod
    def create_dynamic_temporal_signal(self) -> object:
        pass

    def _create_lowpass_response(
        self, tvec: np.ndarray, NL: int, TL: float
    ) -> np.ndarray:
        """
        Lowpass filter kernel for convolution using NumPy.

        Parameters
        ----------
        tvec : np.ndarray
            Time vector for the response.
        NL : int
            Order of the lowpass filter.
        TL : float
            Time constant for the lowpass filter.

        Returns
        -------
        np.ndarray
            Lowpass filtered response.
        """
        h = (1 / factorial(NL)) * (tvec / TL) ** (NL - 1) * np.exp(-tvec / TL)

        # Set inf and nan values of h to zero to avoid issues with large NL
        h[np.isinf(h)] = 0
        h[np.isnan(h)] = 0

        if h.ndim == 1:
            h = h / np.sum(h)
        else:
            h = h / np.expand_dims(np.sum(h, axis=1), axis=1)

        return h

    def _set_brian_standalone_device(self, build_on_run: bool = True) -> None:
        """
        Set up Brian2 in standalone mode with the appropriate device.

        Parameters
        ----------
        build_on_run : bool, optional
            Whether to build the model on run, by default True.
        """
        self.standalone_tmp_dir = tempfile.mkdtemp(dir="/tmp")
        b2.set_device(
            self.device, directory=self.standalone_tmp_dir, build_on_run=build_on_run
        )
        if b2.get_device().has_been_run:
            b2.device.reinit()
            b2.device.activate(build_on_run=build_on_run)

    def _teardown_brian_standalone_device(self) -> None:
        """
        Tear down Brian2 standalone device and clean up temporary directories.
        """
        shutil.rmtree(self.standalone_tmp_dir)
        b2.set_device("runtime")

    def _add_delay(
        self,
        yvecs: np.ndarray,
        D: np.ndarray,
        _dt: float,
        n_units: int,
        n_timepoints: int,
    ) -> np.ndarray:
        """
        Add delay to the generator potential.

        Parameters
        ----------
        yvecs : np.ndarray
            The generator potential vector.
        D : np.ndarray
            Delay vector.
        _dt : float
            Time step for the simulation.
        n_units : int
            Number of units.
        n_timepoints : int
            Number of timepoints.

        Returns
        -------
        np.ndarray
            Generator potential with delay applied.
        """
        delay_timepoints = np.int16(np.median(np.round(D / (_dt / b2u.ms))))
        generator_potential = np.zeros((n_units, n_timepoints))
        generator_potential[:, delay_timepoints:] = yvecs[
            :, : n_timepoints - delay_timepoints
        ]
        return generator_potential


class GanglionCellParasol(GanglionCellBase):
    """
    A class representing parasol ganglion cells, inheriting from GanglionCellBase.

    Attributes
    ----------
    _device : str
        Device used for computation (e.g., 'cuda', 'cpu').
    standalone_tmp_dir : str
        Temporary directory used for Brian2 standalone mode.
    """

    def __init__(self, device) -> None:
        super().__init__(device)

    def _validate_svec(
        self, svec: np.ndarray, n_units: int, n_timepoints: int, params: np.ndarray
    ) -> np.ndarray:
        """
        Validate the size and shape of the stimulus vector.

        Parameters
        ----------
        svec : np.ndarray
            The stimulus vector or array.
        n_units : int
            Number of units (e.g., cells).
        n_timepoints : int
            Number of timepoints in the stimulus.
        params : np.ndarray
            Array of parameters for the parasol units.

        Returns
        -------
        np.ndarray
            Reshaped stimulus vector to match the expected format.

        Raises
        ------
        ValueError
            If the dimensions of the stimulus vector are invalid.
        """
        if params.ndim != 2:
            raise ValueError("params array must be 2D for parasol units, aborting...")

        if svec.size == n_units * n_timepoints:
            # Visual stimulus
            svecs = svec
        elif svec.size == n_timepoints:
            # Impulse response
            svecs = np.tile(svec, (n_units, 1))
        elif svec.size == n_units * n_timepoints * 2:
            # 2D inputs for each unit
            svecs = svec.sum(axis=-1)
        else:
            raise ValueError(
                "svec size matches neither visual simulus or impulse, aborting..."
            )

        return svecs

    def get_BK_parameters(self, gcs: object) -> np.ndarray:
        """
        Get BK parameters for the parasol ganglion cells.

        Parameters
        ----------
        gcs : object
            Object containing ganglion cell data, including unit indices.

        Returns
        -------
        np.ndarray
            Array of BK parameters (e.g., NL, TL, HS, T0, Chalf, D).
        """
        columns = ["NL", "TL", "HS", "T0", "Chalf", "D"]
        # columns = ["NL", "TL", "HS", "T0", "Chalf", "D", "A"]
        params = gcs.df.loc[gcs.unit_indices, columns].values
        return params

    def create_dynamic_temporal_signal(
        self,
        tvec: np.ndarray,
        svec: np.ndarray,
        _dt: float,
        params: np.ndarray,
        show_impulse: bool = False,
        impulse_contrast: float = 1.0,
    ) -> np.ndarray:
        """
        Generate a dynamic temporal signal with contrast gain control.

        Parameters
        ----------
        tvec : np.ndarray
            Time vector for the simulation.
        svec : np.ndarray
            Stimulus vector or array.
        _dt : float
            Time step for the simulation (in ms).
        params : np.ndarray
            Array of parameters for the parasol units.
        show_impulse : bool, optional
            Whether to show the impulse response, by default False.
        impulse_contrast : float, optional
            Impulse contrast value, by default 1.0.

        Returns
        -------
        np.ndarray
            The generator potential with added delays or the impulse response if `show_impulse` is True.
        """
        n_units = params.shape[0]
        n_timepoints = tvec.shape[-1]

        svecs = self._validate_svec(svec, n_units, n_timepoints, params)

        # Time constant for dynamical variable c(t), ms. Victor_1987_JPhysiol
        _Tc = np.array(15) * b2u.ms

        # parameter_names for parasol gain control ["NL", "TL", "HS", "T0", "Chalf", "D", "A"]
        NL = np.int32(params[:, 0])
        TL = params[:, 1]  # ms, control multiplier 0.25
        _HS = params[:, 2]
        _T0 = params[:, 3] * b2u.ms
        _Chalf = params[:, 4]
        D = params[:, 5]

        # Correct dimensions to  [n units, n timepoints].
        tvec = np.expand_dims(tvec, axis=0)
        NL = np.expand_dims(NL, axis=-1)
        TL = np.expand_dims(TL, axis=-1)

        h = self._create_lowpass_response(tvec, NL, TL)

        if show_impulse is True:
            svecs = svecs.astype(np.float32)
            _c = np.array(impulse_contrast)

        padding_size = n_timepoints - 1
        svec_padded = np.pad(svecs, ((0, 0), (padding_size, 0)), mode="edge")
        _dt = _dt * b2u.ms  # Adjust as needed
        simulation_duration = n_timepoints * _dt

        self._set_brian_standalone_device(build_on_run=False)

        # Low-pass stage by convolution
        x_vec = np.empty((n_units, n_timepoints))
        for this_unit in range(n_units):
            x_vec[this_unit, :] = fftconvolve(
                svec_padded[this_unit, :], h[this_unit, :], mode="valid"
            )
        x_input = b2.TimedArray(x_vec.T, dt=_dt)  # noqa: F841

        # Define and run the high-pass stage. x_input is precalculated, thus "manual" derivative.
        eqs = """
        dy/dt = (-y + Ts * x_derivative + (1 - HS) * x_input(t, i)) / Ts : 1
        x_derivative = (x_input(t, i) - x_input(t-dt, i)) / dt : Hz
        Ts = T0 / (1 + (c / Chalf)) : second
        # Parameters
        HS : 1
        T0 : second
        Chalf : 1
        Tc : second
        """
        if show_impulse is True:
            eqs += """
            c = _c : 1
            """
        else:
            eqs += """
            dc/dt = (abs(y) - c) / Tc : 1
            """

        # TODO: For efficiency vs integration stability, consider lower-freq video, but upsampling for Brian2
        # Currently _dt is video dt, not simulation dt
        neuron_group = b2.NeuronGroup(n_units, eqs, method="rk4", dt=_dt)

        neuron_group.y = 0
        if show_impulse is not True:
            neuron_group.c = np.zeros(n_units, dtype=np.float32)

        neuron_group.HS = _HS
        neuron_group.T0 = _T0
        neuron_group.Chalf = _Chalf
        neuron_group.Tc = _Tc

        state_monitor = b2.StateMonitor(neuron_group, ["y"], record=True)

        b2.run(simulation_duration)
        b2.device.build(directory=self.standalone_tmp_dir, compile=True, run=True)

        yvecs = state_monitor.y

        self._teardown_brian_standalone_device()

        if show_impulse is True:
            return yvecs

        generator_potential = self._add_delay(yvecs, D, _dt, n_units, n_timepoints)

        return generator_potential


class GanglionCellMidget(GanglionCellBase):
    """
    Class for midget ganglion cells, inheriting from GanglionCellBase.
    Provides methods for validating stimulus vectors, retrieving BK parameters,
    and generating dynamic temporal signals for midget cells.
    """

    def __init__(self, device: str) -> None:
        super().__init__(device)

    def _validate_svec(
        self, svec: np.ndarray, n_units: int, n_timepoints: int, params: np.ndarray
    ) -> np.ndarray:
        """
        Validate the size and shape of the stimulus vector.

        Parameters
        ----------
        svec : np.ndarray
            The stimulus vector or array.
        n_units : int
            Number of units (e.g., cells).
        n_timepoints : int
            Number of timepoints in the stimulus.
        params : np.ndarray
            Array of parameters for the midget units.

        Returns
        -------
        np.ndarray
            Reshaped stimulus vector to match the expected format.

        Raises
        ------
        ValueError
            If the dimensions of the stimulus vector are invalid.
        """
        if params.ndim != 3:  # params is [n_units, n_features, n_domains]
            raise ValueError("params array must be 3D for midget units, aborting...")

        if svec.size == n_units * n_timepoints * 2:
            # Visual stimulus, cen and sur stimulus for each unit
            svecs = svec
        elif svec.size == n_timepoints:
            # Impulse response, only one vector
            svecs = np.tile(
                np.expand_dims(svec, -1), (params.shape[0], 1, params.shape[2])
            )
        elif svec.size == n_units * n_timepoints:
            # 1D inputs for each unit
            svecs = np.tile(np.expand_dims(svec, -1), (1, 1, params.shape[2]))
        else:
            raise ValueError(
                "svec size matches neither visual simulus or impulse, aborting..."
            )

        return svecs

    def get_BK_parameters(self, gcs: object) -> np.ndarray:
        """
        Retrieve BK parameters for the midget ganglion cells.

        Parameters
        ----------
        gcs : object
            Object containing ganglion cell data, including unit indices.

        Returns
        -------
        np.ndarray
            Array of BK parameters with dimensions [n_units, n_features, n_domains].
        """

        columns_cen = ["NL_cen", "NLTL_cen", "TS_cen", "HS_cen", "D_cen", "A_cen"]
        params_cen = gcs.df.loc[gcs.unit_indices, columns_cen].values
        columns_sur = ["NL_sur", "NLTL_sur", "TS_sur", "HS_sur", "D_cen", "A_sur"]
        params_sur = gcs.df.loc[gcs.unit_indices, columns_sur].values

        params = np.stack((params_cen, params_sur), axis=-1)

        return params

    def create_dynamic_temporal_signal(
        self,
        tvec: np.ndarray,
        svec: np.ndarray,
        _dt: float,
        params: np.ndarray,
        show_impulse: bool = False,
        impulse_contrast: float = 1.0,
    ) -> np.ndarray:
        """
        Generate a dynamic temporal signal for midget units, including center and surround responses.

        Parameters
        ----------
        tvec : np.ndarray
            Time vector for the simulation.
        svec : np.ndarray
            Stimulus vector or array.
        _dt : float
            Time step for the simulation (in ms).
        params : np.ndarray
            Array of parameters for the midget units.
        show_impulse : bool, optional
            Whether to show the impulse response, by default False.
        impulse_contrast : float, optional
            Impulse contrast value, by default 1.0.

        Returns
        -------
        np.ndarray
            The generator potential with added delays or the impulse response if `show_impulse` is True.
        """

        # params is [n units, n_features, n_domains] array, domains are the [center, surround].
        n_units = params.shape[0]
        n_timepoints = tvec.shape[-1]

        svecs = self._validate_svec(svec, n_units, n_timepoints, params)

        tvec = np.expand_dims(tvec, axis=0)
        lp = np.zeros((n_units, n_timepoints, 2))

        # parameter name order for midget ["NL", "NLTL", "TS", "HS", "D", "A"]
        for domain_idx in range(2):
            NL = np.int32(params[:, 0, domain_idx])
            TL = params[:, 1, domain_idx] / params[:, 0, domain_idx]  # TL = NLTL / NL
            NL = np.expand_dims(NL, axis=-1)
            TL = np.expand_dims(TL, axis=-1)

            lp[:, :, domain_idx] = self._create_lowpass_response(tvec, NL, TL)

        lp_total = np.sum(lp, axis=(-2))
        # h_cen and h_sur are [n_units, n_timepoints]
        h_cen = lp[:, :, 0] / np.expand_dims(lp_total[:, 0], -1)
        h_sur = lp[:, :, 1] / np.expand_dims(lp_total[:, 1], -1)

        _HS = params[:, 3]
        _TS = params[:, 2]
        D = params[:, 4]

        padding_size = n_timepoints - 1
        svec_padded = np.pad(svecs, ((0, 0), (padding_size, 0), (0, 0)), mode="edge")
        _dt = _dt * b2u.ms
        simulation_duration = n_timepoints * _dt

        self._set_brian_standalone_device(build_on_run=False)

        # Low-pass stage by convolution
        x_vec = np.empty((n_units, n_timepoints, 2))
        for domain_idx, h in enumerate([h_cen, h_sur]):
            # Convolve the stimulus with the low-pass kernel for all units
            for this_unit in range(n_units):
                x_vec[this_unit, :, domain_idx] = fftconvolve(
                    svec_padded[this_unit, :, domain_idx], h[this_unit, :], mode="valid"
                )
        x_input = b2.TimedArray(np.zeros_like(x_vec[:, :, 0].T), dt=_dt)

        # Define and run the high-pass stage
        eqs = """
        dy/dt = (-y + ts * x_derivative + (1 - hs) * x_input(t, i)) / ts : 1
        x_derivative = (x_input(t, i) - x_input(t-dt, i)) / dt : Hz
        # Parameters
        hs : 1
        ts : second
        """

        yvecs = np.zeros([n_units, n_timepoints, 2])

        neuron_group = b2.NeuronGroup(n_units, eqs, method="rk4", dt=_dt)

        neuron_group.hs = _HS[:, 0]
        neuron_group.ts = _TS[:, 0] * b2u.ms

        state_monitor = b2.StateMonitor(neuron_group, ["y"], record=True)

        b2.run(simulation_duration)
        b2.device.build(directory=self.standalone_tmp_dir, compile=True, run=False)

        # Separate run for the center and surround domains
        for domain_idx in range(2):
            this_x_vec = np.ascontiguousarray(x_vec[:, :, domain_idx].T)
            this_HS = np.ascontiguousarray(_HS[:, domain_idx])
            this_TS = np.ascontiguousarray(_TS[:, domain_idx])

            b2.device.run(
                run_args={
                    x_input: this_x_vec,
                    neuron_group.hs: this_HS,
                    neuron_group.ts: this_TS * b2u.ms,
                }
            )

            yvecs[:, :, domain_idx] = state_monitor.y

        self._teardown_brian_standalone_device()

        # Sum center and surround responses
        yvecs = np.sum(yvecs, axis=-1)

        if show_impulse is True:
            return yvecs

        D = np.sum(D, axis=-1)
        generator_potential = self._add_delay(yvecs, D, _dt, n_units, n_timepoints)

        return generator_potential


class ResponseTypeBase(ABC, PrintableMixin):
    @abstractmethod
    def get_contrast_by_response_type(
        self, visual_signal: int | float | np.ndarray
    ) -> int | float | np.ndarray:
        pass


class ResponseTypeON(ResponseTypeBase):
    def get_contrast_by_response_type(
        self, visual_signal: int | float | np.ndarray
    ) -> int | float | np.ndarray:
        return visual_signal


class ResponseTypeOFF(ResponseTypeBase):
    def get_contrast_by_response_type(
        self, visual_signal: int | float | np.ndarray
    ) -> int | float | np.ndarray:
        return -visual_signal


class DoGModelBase(ABC):
    def __init__(self, retina_math: object) -> None:
        self._retina_math = retina_math

    # @abstractmethod
    # def get_spatial_kernel(self):
    #     pass

    @abstractmethod
    def get_surround_params(self):
        pass

    @property
    def retina_math(self):
        return self._retina_math


class DoGModelEllipseFixed(DoGModelBase):
    """
    Difference of Gaussians (DoG) model with fixed elliptical surround.
    """

    def get_surround_params(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate surround parameters from a DataFrame of ganglion cell properties.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ganglion cell properties.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing semi-major axis (x), semi-minor axis (y), and orientation
            of the surround ellipse.
        """
        semi_x = df["semi_xc"].values * df["relat_sur_diam"].values
        semi_y = df["semi_yc"].values * df["relat_sur_diam"].values
        ori = df["orient_cen_rad"].values
        return semi_x, semi_y, ori


class DoGModelEllipseIndependent(DoGModelBase):
    """
    Difference of Gaussians (DoG) model with independent elliptical surround.
    """

    def get_surround_params(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract surround parameters from a DataFrame of ganglion cell properties.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ganglion cell properties.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing semi-major axis (x), semi-minor axis (y), and orientation
            of the surround ellipse.
        """
        semi_x = df["semi_xs"].values
        semi_y = df["semi_ys"].values
        ori = df["orient_sur_rad"].values
        return semi_x, semi_y, ori


class DoGModelCircular(DoGModelBase):
    """
    Difference of Gaussians (DoG) model with circular center and surround.
    """

    def get_surround_params(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract surround parameters from a DataFrame of ganglion cell properties.

        For circular DoG model, semi-major and semi-minor axes are both set to the surround radius.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ganglion cell properties.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing semi-major axis (x), semi-minor axis (y), and orientation
            of the surround. For circular model, x and y are identical and equal to rad_s.
        """
        semi_x = df["rad_s"].values
        semi_y = df["rad_s"].values
        ori = df["orient_cen_rad"].values
        return semi_x, semi_y, ori


class SpatialModelBase(ABC):
    """
    Abstract base class for spatial models in retinal processing.

    This class provides a foundation for implementing spatial models,
    including methods for creating spatial filters and cropping stimuli.

    Parameters
    ----------
    DoG_model : object
        Difference of Gaussians model object.

    Attributes
    ----------
    _DoG_model : object
        Stored Difference of Gaussians model object.
    """

    def __init__(self, DoG_model: object) -> None:
        self._DoG_model = DoG_model

    @abstractmethod
    def create_spatial_filter(self):
        pass

    @property
    def DoG_model(self) -> object:
        return self._DoG_model

    def _get_crop_pixels(
        self, gcs: object, unit_index: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get pixel coordinates for a stimulus crop matching the spatial filter size.

        Parameters
        ----------
        gcs : object
            Ganglion cell object containing spatial filter information.
        unit_index : int or array-like of int
            Index or indices of the unit(s) for which to retrieve crop coordinates.

        Returns
        -------
        qmin, qmax, rmin, rmax : np.ndarray
            Pixel coordinates defining the crop's bounding box.
            qmin and qmax specify the range in the q-dimension (horizontal),
            and rmin and rmax specify the range in the r-dimension (vertical).

        Notes
        -----
        The crop size is determined by the spatial filter's sidelength.
        """
        if isinstance(unit_index, (int, np.int32, np.int64)):
            unit_index = np.array([unit_index])
        df_stimpix = gcs.df_stimpix.iloc[unit_index]
        q_center = np.round(df_stimpix.q_pix).astype(int).values
        r_center = np.round(df_stimpix.r_pix).astype(int).values
        side_halflen = (gcs.spatial_filter_sidelen - 1) // 2
        qmin = q_center - side_halflen
        qmax = q_center + side_halflen
        rmin = r_center - side_halflen
        rmax = r_center + side_halflen
        return qmin, qmax, rmin, rmax

    def create_spatial_filter(self, gcs: object, unit_index: int) -> np.ndarray:
        """
        Create the spatial component of the spatiotemporal filter.

        This method generates a spatial filter for a given unit based on
        pre-computed spatial receptive fields.

        Parameters
        ----------
        gcs : object
            Ganglion cell object containing spatial filter information and
            pre-computed spatial receptive fields.
        unit_index : int
            Index of the unit in the dataframe.

        Returns
        -------
        np.ndarray
            2D array representing the spatial filter for the given unit.
        """
        s = gcs.spatial_filter_sidelen
        spatial_kernel = resize(
            gcs.spat_rf[unit_index, :, :], (s, s), anti_aliasing=True
        )
        return spatial_kernel


class SpatialModelDOG(SpatialModelBase):
    def __init__(self, DoG_model: object) -> None:
        super().__init__(DoG_model)


class SpatialModelVAE(SpatialModelBase):
    def __init__(self, DoG_model: object) -> None:
        super().__init__(DoG_model)


class TemporalModelBase(ABC):
    """
    Base class for temporal models in retinal processing.

    Parameters
    ----------
    retina_math : object
        Object containing retinal math calculations.
    ganglion_cell : object
        Object representing ganglion cell properties.
    response_type : object
        Object defining the response type characteristics.

    Attributes
    ----------
    retina_math : object
        Stored retinal math object.
    ganglion_cell : object
        Stored ganglion cell object.
    response_type : object
        Stored response type object.
    project_data : dict
        Dictionary to store project-related data.
    """

    def __init__(
        self,
        retina_math: object,
        ganglion_cell: object,
        response_type: object,
        device: str,
    ) -> None:
        self.retina_math = retina_math
        self.ganglion_cell = ganglion_cell
        self.response_type = response_type
        self.device = device
        self.project_data: Dict = {}

    @abstractmethod
    def impulse_response(self):
        """
        Abstract method to compute impulse response.
        """
        pass

    @abstractmethod
    def create_generator_potential(self):
        """
        Abstract method to create generator potential.
        """
        pass

    def _initialize_impulse(
        self, vs: object, gcs: object, dt_ms: float
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Initialize impulse response parameters.

        Parameters
        ----------
        vs : object
            Visual signal object.
        gcs : object
            Ganglion cell object.
        dt_ms : float
            Time step in milliseconds.

        Returns
        -------
        tvec : np.ndarray
            Time vector.
        svec : np.ndarray
            Stimulus vector.
        idx_100_ms : int
            Index corresponding to 100ms delay.
        """
        total_duration_ms = gcs.data_filter_timesteps * (1000 / gcs.data_filter_fps)
        stim_len_tp = np.round(total_duration_ms / dt_ms)
        tvec = np.arange(stim_len_tp) * dt_ms
        svec = np.zeros(len(tvec))
        start_delay = 100  # ms
        idx_100_ms = int(np.round(start_delay / dt_ms))
        svec[idx_100_ms] = 1.0
        svec = self.response_type.get_contrast_by_response_type(svec)
        svec = np.expand_dims(svec, axis=0)
        return tvec, svec, idx_100_ms

    def _create_dynamic_contrast(self, vs: object, gcs: object) -> object:
        """
        Create dynamic contrast signal.

        Parameters
        ----------
        vs : object
            Visual signal object containing the transduction cascade.
        gcs : object
            Ganglion cell object containing receptive fields.

        Returns
        -------
        vs : object
            Updated visual signal object with dynamic contrast signal.

        Notes
        -----
        This method multiplies the stimulus with spatial filter masks to create
        center and surround dynamic contrast signals.
        """
        spatial_filters = gcs.spatial_filters_flat.copy()
        spatial_filters_reshaped = np.expand_dims(spatial_filters, axis=2)

        masks_sur = gcs.surround_masks_flat[:, :, np.newaxis]

        vs.svecs_sur = np.einsum(
            "ijk,ijk->ik",
            spatial_filters_reshaped * masks_sur,
            vs.stimulus_cropped_adapted,
        )

        masks_cen = gcs.center_masks_flat[:, :, np.newaxis]
        vs.svecs_cen = np.einsum(
            "ijk,ijk->ik",
            spatial_filters_reshaped * masks_cen,
            vs.stimulus_cropped_adapted,
        )

        return vs


class TemporalModelFixed(TemporalModelBase):
    """
    A fixed temporal model for retinal processing.

    This class implements a temporal model with fixed parameters for simulating
    retinal responses.

    Parameters
    ----------
    retina_math : object
        Object containing retinal math methods.
    ganglion_cell : object
        Object representing ganglion cell properties.
    response_type : object
        Object defining the response type characteristics.
    """

    def __init__(
        self,
        retina_math: object,
        ganglion_cell: object,
        response_type: object,
        device: str,
    ) -> None:
        super().__init__(retina_math, ganglion_cell, response_type, device)

    def _create_temporal_filter_vectorized(
        self, gcs: object, filter_params: np.ndarray, tvec: np.ndarray
    ) -> np.ndarray:
        """
        Create the temporal component of the spatiotemporal filter in a vectorized manner.

        Parameters
        ----------
        gcs : object
            Ganglion cell object containing filter parameters.
        filter_params : np.ndarray
            Array of filter parameters [n, p1, p2, tau1, tau2].
        tvec : np.ndarray
            Time points at which to evaluate the filters.

        Returns
        -------
        np.ndarray
            Temporal filter for the given unit.
        """
        n, p1, p2, tau1, tau2 = filter_params

        temporal_filter = self.retina_math.diff_of_lowpass_filters(
            tvec, n, p1, p2, tau1, tau2
        )
        # Get p1 volume by setting p2 to zero
        temporal_filter_p1 = self.retina_math.diff_of_lowpass_filters(
            tvec, n, p1, 0.0, tau1, tau2
        )
        # Normalize the filter according to P1 area under curve
        temporal_filter_normalized = temporal_filter / np.sum(temporal_filter_p1)

        return temporal_filter_normalized

    def _get_linear_temporal_filters(self, gcs: object) -> object:
        """
        Retrieve temporal filters for an array of units.

        Parameters
        ----------
        gcs : object
            Ganglion cell object containing unit indices and filter parameters.

        Returns
        -------
        object
            Updated ganglion cell object with temporal filters.
        """
        temporal_filters = np.zeros((len(gcs.unit_indices), gcs.temporal_filter_len))
        tvec = np.linspace(0, gcs.data_filter_duration, gcs.temporal_filter_len)

        # Extract all filter parameters at once
        filter_params = gcs.df.loc[
            gcs.unit_indices, ["n", "p1", "p2", "tau1", "tau2"]
        ].values

        # Vectorize the creation of temporal filters
        for idx, params in enumerate(filter_params):
            temporal_filters[idx, :] = self._create_temporal_filter_vectorized(
                gcs, params, tvec
            )

        gcs.temporal_filters = temporal_filters
        return gcs

    def _get_linear_spatiotemporal_filters(self, gcs: object) -> object:
        """
        Generate spatiotemporal filters for given unit indices.

        Parameters
        ----------
        gcs : object
            Ganglion cell object containing spatial and temporal filters.

        Returns
        -------
        object
            Updated ganglion cell object with spatiotemporal filters.
        """
        spatiotemporal_filters = (
            gcs.spatial_filters_flat[:, :, None] * gcs.temporal_filters[:, None, :]
        )
        gcs.spatiotemporal_filters = spatiotemporal_filters

        return gcs

    def _create_fixed_generator_potential(self, vs: object, gcs: object) -> object:
        """
        Convolve the stimulus with the spatiotemporal filter for a given set of units.

        Parameters
        ----------
        vs : object
            Visual signal object containing stimulus information.
        gcs : object
            Ganglion cell object containing filter information.

        Returns
        -------
        object
            Updated visual signal object with generator potentials.

        Raises
        ------
        AssertionError
            If there is a mismatch between the duration of the stimulus and the duration of the generator potential.
        """
        start_time = time.time()

        print("Using PyTorch for convolution...")
        device = self.device
        num_units_t = torch.tensor(gcs.n_units, device=device)
        stim_len_tp_t = torch.tensor(vs.stim_len_tp, device=device)

        # Convert to float32 to save memory
        stimulus_cropped_adapted = torch.tensor(
            vs.stimulus_cropped_adapted, dtype=torch.float32
        ).to(device)

        spatiotemporal_filter = torch.tensor(
            gcs.spatiotemporal_filters, dtype=torch.float32
        ).to(device)

        # Convolving two signals involves "flipping" one signal and then sliding it across the other signal.
        # PyTorch, however, does not flip the kernel, so we need to do it manually.
        spatiotemporal_filter_flipped = torch.flip(spatiotemporal_filter, dims=[2])

        # Calculate padding size
        filter_length = spatiotemporal_filter_flipped.shape[2]
        padding_size = filter_length - 1

        # Initialize output tensor
        output = torch.empty(
            (num_units_t, stim_len_tp_t),
            device=device,
            dtype=torch.float32,
        )

        # Define batch size
        batch_size = 100  # Adjust this based on your GPU memory

        # Process in batches
        tqdm_desc = "Preparing fixed generator potential..."
        for i in tqdm(range(0, num_units_t, batch_size), desc=tqdm_desc):
            batch_end = min(i + batch_size, num_units_t)
            batch_indices = torch.arange(i, batch_end, device=device, dtype=torch.int32)

            # Extract the current batch of stimulus
            stimulus_batch = stimulus_cropped_adapted[batch_indices]

            # Pad the current batch of stimulus
            stimulus_batch_padded = torch.nn.functional.pad(
                stimulus_batch, (padding_size, 0), mode="replicate"
            )

            # Extract the current batch of filter
            filter_batch = spatiotemporal_filter_flipped[batch_indices]
            # Perform convolution for the current batch
            for idx, this_unit in enumerate(batch_indices):
                # Perform convolution on the current batch
                output[this_unit] = torch.nn.functional.conv1d(
                    stimulus_batch_padded[idx].unsqueeze(0),
                    filter_batch[idx].unsqueeze(0),
                    padding=0,
                )

        # Move back to CPU and convert to numpy
        generator_potential = output.cpu().squeeze().numpy()

        print(f"Convolution time: {time.time() - start_time:.2f} s")

        # Internal test for convolution operation
        generator_potential_duration_tp = generator_potential.shape[-1]
        assert (
            vs.stim_len_tp == generator_potential_duration_tp
        ), "Duration mismatch, check convolution operation, aborting..."

        vs.generator_potentials = generator_potential

        return vs

    def impulse_response(self, vs: object, gcs: object, contrasts: np.ndarray) -> dict:
        """
        Generate impulse response for a given set of units and contrasts.

        Parameters
        ----------
        vs : object
            Visual signal object.
        gcs : object
            Ganglion cell object.
        contrasts : np.ndarray
            Array of contrast values.

        Returns
        -------
        dict
            Dictionary containing impulse response data.
        """
        dt_ms = vs.video_dt / b2u.ms
        tvec, svec, idx_start_delay = self._initialize_impulse(vs, gcs, dt_ms)

        gcs = self._get_linear_temporal_filters(gcs)
        impulse_responses = np.tile(
            gcs.temporal_filters[:, :, np.newaxis],
            (1, 1, len(contrasts)),
        )

        impulse_responses = self.response_type.get_contrast_by_response_type(
            impulse_responses
        )

        # append zeros to the start of the impulse response
        impulse_responses = np.pad(
            impulse_responses,
            ((0, 0), (idx_start_delay, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        # cut the impulse response to the desired length
        impulse_responses = impulse_responses[:, :-idx_start_delay, :]

        impulse_to_show = {"tvec": tvec / 1000}  # convert from ms to seconds
        impulse_to_show["svec"] = svec
        impulse_to_show["idx_start_delay"] = idx_start_delay
        impulse_to_show["contrasts"] = contrasts
        impulse_to_show["impulse_responses"] = impulse_responses

        return impulse_to_show

    def create_generator_potential(
        self, vs: object, gcs: object
    ) -> tuple[object, object]:
        """
        Create generator potential for the given visual signal and ganglion cells.

        Parameters
        ----------
        vs : object
            Visual signal object.
        gcs : object
            Ganglion cell object.

        Returns
        -------
        tuple[object, object]
            Updated visual signal and ganglion cell objects.
        """
        gcs = self._get_linear_temporal_filters(gcs)
        gcs = self._get_linear_spatiotemporal_filters(gcs)
        vs = self._create_fixed_generator_potential(vs, gcs)

        return vs, gcs


class TemporalModelDynamic(TemporalModelBase):
    def __init__(
        self,
        retina_math: object,
        ganglion_cell: object,
        response_type: str,
        device: str,
    ) -> None:
        super().__init__(retina_math, ganglion_cell, response_type, device)

    def impulse_response(
        self, vs: object, gcs: object, contrasts: list[float]
    ) -> dict[str, Any]:
        """
        Calculate the impulse response based on input parameters.

        Parameters
        ----------
        vs : object
            An object containing video stimulus information.
        gcs : object
            An object representing ganglion cells.
        contrasts : list of float
            A list of contrast values to be applied.

        Returns
        -------
        dict[str, Any]
            A dictionary containing time vector, spatial vector,
            indices of start delay, contrasts, and impulse responses.
        """
        dt_ms = vs.video_dt / b2u.ms
        tvec, svec, idx_start_delay = self._initialize_impulse(vs, gcs, dt_ms)
        stim_len_tp = len(tvec)

        # Expand n units dim for svec to match the shape of params
        params = self.ganglion_cell.get_BK_parameters(gcs)

        impulse_responses = np.empty((params.shape[0], stim_len_tp, len(contrasts)))

        for contrast in contrasts:
            yvec = self.ganglion_cell.create_dynamic_temporal_signal(
                tvec,
                svec,
                dt_ms,
                params,
                show_impulse=True,
                impulse_contrast=contrast,
            )
            impulse_responses[:, :, contrasts.index(contrast)] = yvec

        impulse_to_show = {"tvec": tvec / 1000}  # convert from ms to seconds
        impulse_to_show["svec"] = svec
        impulse_to_show["idx_start_delay"] = idx_start_delay
        impulse_to_show["contrasts"] = contrasts
        impulse_to_show["impulse_responses"] = impulse_responses

        return impulse_to_show

    def create_generator_potential(
        self, vs: object, gcs: object
    ) -> tuple[object, object]:
        """
        Create the generator potential based on dynamic contrast.

        Parameters
        ----------
        vs : object
            An object containing video stimulus information.
        gcs : object
            An object representing ganglion cells.

        Returns
        -------
        tuple[object, object]
            A tuple containing updated video stimulus and ganglion cells objects.
        """
        vs = self._create_dynamic_contrast(vs, gcs)
        dt_ms = vs.video_dt / b2u.ms
        stim_len_tp = vs.stim_len_tp
        params = self.ganglion_cell.get_BK_parameters(gcs)
        tvec = np.arange(stim_len_tp) * dt_ms

        # Create an np array whose last dimension is cen, sur
        svecs = np.stack((vs.svecs_cen, vs.svecs_sur), axis=-1)

        vs.generator_potentials = self.ganglion_cell.create_dynamic_temporal_signal(
            tvec,
            svecs,
            dt_ms,
            params,
        )

        return vs, gcs


class TemporalModelSubunit(TemporalModelBase):
    """
    A subunit-based temporal model for retinal processing.

    This class implements a temporal model that incorporates cone and bipolar cell
    processing stages.

    Parameters
    ----------
    retina_math : Any
        Object containing retinal math calculations.
    ganglion_cell : Any
        Object representing ganglion cell properties.
    response_type : Any
        Object defining the response type characteristics.
    cones : Any
        Object representing cone photoreceptors.
    bipolars : Any
        Object representing bipolar cells.

    Attributes
    ----------
    cones : Any
        Stored cone photoreceptor object.
    bipolars : Any
        Stored bipolar cell object.
    """

    def __init__(
        self,
        retina_math: Any,
        ganglion_cell: Any,
        response_type: Any,
        device: str,
        cones: Any,
        bipolars: Any,
        stimulate: Any,
    ) -> None:
        super().__init__(retina_math, ganglion_cell, response_type, device)
        self.cones = cones
        self.bipolars = bipolars
        self.stimulate = stimulate

    def impulse_response(
        self, vs: object, gcs: object, contrasts: list[float]
    ) -> dict[str, Any]:
        """
        Calculate the impulse response based on input parameters.

        Parameters
        ----------
        vs : object
            An object containing video stimulus information.
        gcs : object
            An object representing ganglion cells.
        contrasts : list of float
            A list of contrast values to be applied.

        Returns
        -------
        dict[str, Any]
            A dictionary containing time vector, spatial vector,
            indices of start delay, contrasts, and impulse responses.
        """
        dt_ms = vs.video_dt / b2u.ms
        tvec, svec, idx_start_delay = self._initialize_impulse(vs, gcs, dt_ms)
        stim_len_tp = len(tvec)

        # duration_tp = 50
        duration_tp = svec.sum()

        options = self.stimulate.options
        options["fps"] = 1000 / dt_ms
        options["duration_seconds"] = duration_tp * dt_ms / 1000
        options["pattern"] = "temporal_square_pattern"

        options["background"] = "intensity_min"
        options["baseline_start_seconds"] = idx_start_delay * dt_ms / 1000
        options["baseline_end_seconds"] = (
            (stim_len_tp - duration_tp - idx_start_delay) * dt_ms / 1000
        )
        options["'temporal_frequency'"] = 0
        options["image_width"] = 240
        options["image_height"] = 240
        options["stimulus_video_name"] = "impulse_response_video.mp4"

        vs.video_dt = (1 / options["fps"]) * b2u.second  # input
        vs.stim_len_tp = stim_len_tp
        vs.duration = stim_len_tp * vs.video_dt
        vs.tvec = range(stim_len_tp) * vs.video_dt
        vs.baseline_len_tp = idx_start_delay

        vs.mean_luminance = bg_min = 128

        impulse_responses = np.empty((gcs.n_units, stim_len_tp, len(contrasts)))
        for idx, contrast in enumerate(contrasts):
            options["intensity"] = (
                bg_min,
                np.clip(int(bg_min * (1 + contrast)), 1, 255),
            )
            self.stimulate.options = options
            vs.stimulus_video = self.stimulate.make_stimulus_video(options=options)
            vs.options_from_videofile = vs.stimulus_video.options
            vs = self.cones.create_signal(vs)
            vs = self.bipolars.create_signal(vs)
            impulse_responses[..., idx] = vs.generator_potentials

        impulse_to_show = {"tvec": tvec / 1000}  # convert from ms to seconds
        impulse_to_show["svec"] = svec
        impulse_to_show["idx_start_delay"] = idx_start_delay
        impulse_to_show["contrasts"] = contrasts
        impulse_to_show["impulse_responses"] = impulse_responses

        return impulse_to_show

    def create_generator_potential(self, vs: Any, gcs: Any) -> Tuple[Any, Any]:
        """
        Create generator potential for the subunit model.

        This method processes the visual signal through dynamic contrast creation,
        cone signal generation, and bipolar cell signal generation.

        Parameters
        ----------
        vs : Any
            Visual signal object.
        gcs : Any
            Ganglion cell object.

        Returns
        -------
        Tuple[Any, Any]
            A tuple containing the updated visual signal object and the ganglion cell object.
        """
        vs = self._create_dynamic_contrast(vs, gcs)
        vs = self.cones.create_signal(vs)
        vs = self.bipolars.create_signal(vs)
        return vs, gcs


class SimulationBuildInterface(ABC):
    @property
    @abstractmethod
    def vs(self):
        pass

    @vs.setter
    @abstractmethod
    def vs(self, vs):
        pass

    @property
    @abstractmethod
    def gcs(self):
        pass

    @property
    @abstractmethod
    def cones(self):
        pass

    @property
    @abstractmethod
    def bipolars(self):
        pass

    @abstractmethod
    def get_concrete_components(self):
        pass

    @abstractmethod
    def get_impulse_response(self):
        pass

    @abstractmethod
    def create_spatial_filters(self):
        pass

    @abstractmethod
    def get_uniformity_index(self):
        pass

    @abstractmethod
    def apply_optical_aberration(self):
        pass

    @abstractmethod
    def get_spatially_cropped_video(self):
        pass

    @abstractmethod
    def get_noise(self):
        pass

    @abstractmethod
    def get_generator_potentials(self):
        pass

    @abstractmethod
    def generate_spikes(self):
        pass


class ConcreteSimulationBuilder(SimulationBuildInterface):
    """
    Concrete implementation of the SimulationBuildInterface for building retina simulations.

    This class handles the construction of various components needed for simulating
    retinal ganglion cell responses to visual stimuli.

    Parameters
    ----------
    vs : VisualSignal
        The visual signal object containing stimulus information.
    gcs : GanglionCellProduct
        The ganglion cell data object containing the receptive fields.
    cones : ConeCells
        The cone cells object for photoreceptor simulations.
    bipolars : BipolarCells
        The bipolar cells object for inner retina simulations.
    retina_math : RetinaMath
        Utility object for retina-related calculations.
    device : Any
        The device on which computations will be performed.
    n_sweeps : int
        Number of simulation trials to run.

    Attributes
    ----------
    _vs : VisualSignal
        The visual signal object.
    _gcs : GanglionCellProduct
        The ganglion cell data object.
    _cones : ConeCells
        The cone cells object.
    _bipolars : BipolarCells
        The bipolar cells object.
    _retina_math : RetinaMath
        The retina math utility object.
    _device : Any
        The computation device.
    n_sweeps : int
        Number of simulation trials.
    _project_data : Dict[str, Any]
        Dictionary for storing project-related data.
    """

    def __init__(
        self,
        vs: "VisualSignal",
        gcs: "GanglionCellProduct",
        cones: "ConeProduct",
        bipolars: "BipolarProduct",
        retina_math: RetinaMath,
        device: Any,
        n_sweeps: int,
        stimulate: Any,
    ) -> None:
        self._vs = vs
        self._gcs = gcs
        self._cones = cones
        self._bipolars = bipolars

        self._retina_math = retina_math
        self._device = device
        self.n_sweeps = n_sweeps
        self._stimulate = stimulate

        self._project_data = {}

    @property
    def vs(self) -> "VisualSignal":
        """Get the visual signal object."""
        return self._vs

    @vs.setter
    def vs(self, vs: "VisualSignal") -> None:
        """Set the visual signal object."""
        self._vs = vs

    @property
    def gcs(self):
        return self._gcs

    @gcs.setter
    def gcs(self, gcs):
        self._gcs = gcs

    @property
    def cones(self):
        return self._cones

    @property
    def bipolars(self):
        return self._bipolars

    @property
    def retina_math(self):
        return self._retina_math

    @property
    def device(self):
        print(f"KUKKUU3 {self._device=}")

        return self._device

    @property
    def stimulate(self):
        return self._stimulate

    @property
    def ganglion_cell(self):
        return self._ganglion_cell

    @property
    def response_type(self):
        return self._response_type

    @property
    def DoG_model(self):
        return self._DoG_model

    @property
    def spatial_model(self):
        return self._spatial_model

    @property
    def temporal_model(self):
        return self._temporal_model

    @property
    def project_data(self) -> Dict[str, Any]:
        """Dictionary for storing project-related data."""
        return self._project_data

    @project_data.setter
    def project_data(self, value: Dict[str, Any]) -> None:
        self._project_data = value

    def _get_center_masks(
        self, img_stack: np.ndarray, mask_threshold: float
    ) -> np.ndarray:
        """
        Generate center masks for the given image stack.

        Parameters
        ----------
        img_stack : np.ndarray
            Stack of images to process.
        mask_threshold : float
            Threshold for mask generation.

        Returns
        -------
        np.ndarray
            Array of center masks.
        """
        assert (
            mask_threshold >= 0 and mask_threshold <= 1
        ), "mask_threshold must be between 0 and 1, aborting..."

        masks = []
        for img in img_stack:
            max_val = np.max(img)
            mask = img >= max_val * mask_threshold

            # Label the distinct regions in the mask
            labeled_mask, num_labels = ndimage.label(mask)

            # Find the label of the region that contains the maximum value
            max_label = labeled_mask[np.unravel_index(np.argmax(img), img.shape)]

            # Keep only the region in the mask that contains the maximum value
            mask = labeled_mask == max_label

            masks.append(mask)

        return np.array(masks)

    def _get_surround_masks(
        self, gcs: "GanglionCellProduct", img_stack: np.ndarray, mask_threshold: float
    ) -> np.ndarray:
        """
        Generate surround masks for the given receptive fields and image stack.

        Parameters
        ----------
        gcs : GanglionCellProduct
            The ganglion cell data object containing the receptive fields.
        img_stack : np.ndarray
            Stack of images to process.

        Returns
        -------
        np.ndarray
            Array of surround masks.

        Notes
        -----
        Surround mask does not need to be continuous, as the center mask. However, it is limited to
        the area of the surround DoG model.
        """
        df = gcs.df_stimpix
        xo = df["xos_pix"].values
        yo = df["yos_pix"].values

        semi_x, semi_y, ori = self.DoG_model.get_surround_params(df)

        s = gcs.spatial_filter_sidelen
        n_sd = 2
        masks = []
        for idx, img in enumerate(img_stack):
            ellipse_mask = self.retina_math.create_ellipse_mask(
                xo[idx],
                yo[idx],
                semi_x[idx] * n_sd,
                semi_y[idx] * n_sd,
                -ori[idx],
                s,
            )

            min_val = np.min(img)
            mask = img < min_val * mask_threshold
            final_mask = np.logical_and(mask, ellipse_mask)
            masks.append(final_mask)

        return np.array(masks)

    def _generator_to_firing_rate_noise(
        self, vs: "VisualSignal", gcs: "GanglionCellProduct", n_sweeps: int
    ) -> "VisualSignal":
        """
        Convert generator potentials to firing rates with added noise.

        Parameters
        ----------
        vs : VisualSignal
            The visual signal object.
        gcs : GanglionCellProduct
            The ganglion cell data object.
        n_sweeps : int
            Number of trials.

        Returns
        -------
        VisualSignal
            Updated visual signal object with firing rates.
        """
        params_all = gcs.df.loc[gcs.unit_indices]

        gc_noise_mean = params_all.Mean.values
        gain_name = "A_cen" if gcs.gc_type == "midget" else "A"
        gc_gain_raw = params_all[gain_name].values
        gc_gain_adjusted = gc_gain_raw * gcs.gc_gain_adjustment
        firing_rates_light = vs.generator_potentials * gc_gain_adjusted[:, np.newaxis]
        firing_rates_light = firing_rates_light[:, :, np.newaxis]
        firing_rates_cone_noise = (
            vs.gc_synaptic_noise * gc_noise_mean[:, np.newaxis, np.newaxis]
        )

        firing_rates = np.maximum(firing_rates_light + firing_rates_cone_noise, 0)

        vs.firing_rates = firing_rates

        return vs

    def _firing_rates2brian_timed_arrays(
        self, firing_rates, tvec_original, duration, simulation_dt
    ) -> "VisualSignal":
        """
        Convert firing rates to Brian2 TimedArray objects.

        Parameters
        ----------
        vs : VisualSignal
            The visual signal object.

        Returns
        -------
        VisualSignal
            Updated visual signal object with Brian2 TimedArrays.
        """
        # Let's interpolate the rate to vs.video_dt intervals
        rates_func = interp1d(
            tvec_original,
            firing_rates,
            axis=1,
            fill_value=0,
            bounds_error=False,
        )
        tvec_new = np.arange(0, duration, simulation_dt)

        # This needs to be 2D array for Brian
        interpolated_rates_array = rates_func(tvec_new)

        # Identical rates array for every trial; rows=time, columns=unit index
        inst_rates = b2.TimedArray(interpolated_rates_array.T * b2u.Hz, simulation_dt)

        return tvec_new, inst_rates

    def _brian_spike_generation(
        self,
        n_units: int,
        inst_rates: b2.TimedArray,
        spike_generator_model: str,
        refractory_parameters: dict,
        simulation_dt: b2u,
        duration: b2u,
    ) -> Tuple[b2.NeuronGroup, b2.SpikeMonitor, b2.Network]:
        """
        Generate spikes using Brian2 simulator.

        Parameters
        ----------
        vs : VisualSignal
            The visual signal object.
        gcs : GanglionCellProduct
            The ganglion cell data object.


        Returns
        -------
        VisualSignal
            Updated visual signal object with generated spikes.
        """

        # Set inst_rates to locals() for Brian equation access
        inst_rates = eval("inst_rates")  # noqa: F841

        # units in parallel (NG), trial iterations (repeated runs)

        if spike_generator_model == "refractory":
            abs_refractory = refractory_parameters["abs_refractory"] * b2u.ms  # noqa: F841
            rel_refractory = refractory_parameters["rel_refractory"] * b2u.ms  # noqa: F841
            p_exp = refractory_parameters["p_exp"]  # noqa: F841
            clip_start = refractory_parameters["clip_start"] * b2u.ms  # noqa: F841
            clip_end = refractory_parameters["clip_end"] * b2u.ms  # noqa: F841
            neuron_group = b2.NeuronGroup(
                n_units,
                model="""
                lambda_ttlast = inst_rates(t, i) * dt * w: 1
                t_diff = clip(t - lastspike - abs_refractory, clip_start, clip_end) : second
                w = t_diff**p_exp / (t_diff**p_exp + rel_refractory**p_exp) : 1
                """,
                threshold="rand()<lambda_ttlast",
                refractory="(t-lastspike) < abs_refractory",
                dt=simulation_dt,
            )

            spike_monitor = b2.SpikeMonitor(neuron_group)
            net = b2.Network(neuron_group, spike_monitor)

        elif spike_generator_model == "poisson":
            poisson_group = b2.PoissonGroup(n_units, rates="inst_rates(t, i)")
            spike_monitor = b2.SpikeMonitor(poisson_group)
            net = b2.Network(poisson_group, spike_monitor)
        else:
            raise ValueError(
                "Missing valid spike_generator_model, check simulation_parameters parameters, aborting..."
            )

        # Save brian state
        net.store()
        # all_spiketrains = []
        spikearrays = []
        t_start = []
        t_end = []

        net.restore()  # Restore the initial state
        t_start.append(net.t)
        net.run(duration)
        t_end.append(net.t)

        spiketrains = list(spike_monitor.spike_trains().values())
        # all_spiketrains.extend(spiketrains)

        # Cxsystem spikemon save natively supports multiple monitors
        spikearrays.append(deepcopy(spike_monitor.it[0].__array__()))
        spikearrays.append(deepcopy(spike_monitor.it[1].__array__()))

        return spikearrays, spiketrains

    def get_concrete_components(self) -> None:
        """
        Compile simulation components from gc type, response type,
        spatial model, temporal model, and DoG model.
        """

        gcs = self.gcs

        match gcs.gc_type:
            case "parasol":
                ganglion_cell = GanglionCellParasol(self.device)
            case "midget":
                ganglion_cell = GanglionCellMidget(self.device)

        match gcs.response_type:
            case "on":
                response_type = ResponseTypeON()
            case "off":
                response_type = ResponseTypeOFF()

        match gcs.dog_model_type:
            case "ellipse_fixed":
                DoG_model = DoGModelEllipseFixed(self.retina_math)
            case "ellipse_independent":
                DoG_model = DoGModelEllipseIndependent(self.retina_math)
            case "circular":
                DoG_model = DoGModelCircular(self.retina_math)

        match gcs.spatial_model_type:
            case "DOG":
                spatial_model = SpatialModelDOG(DoG_model)
            case "VAE":
                spatial_model = SpatialModelVAE(DoG_model)

        match gcs.temporal_model_type:
            case "fixed":
                temporal_model = TemporalModelFixed(
                    self.retina_math, ganglion_cell, response_type, self.device
                )
            case "dynamic":
                temporal_model = TemporalModelDynamic(
                    self.retina_math, ganglion_cell, response_type, self.device
                )
            case "subunit":
                temporal_model = TemporalModelSubunit(
                    self.retina_math,
                    ganglion_cell,
                    response_type,
                    self.device,
                    self.cones,
                    self.bipolars,
                    self.stimulate,
                )

        self._ganglion_cell = ganglion_cell
        self._response_type = response_type
        self._DoG_model = DoG_model
        self._spatial_model = spatial_model
        self._temporal_model = temporal_model

    def get_impulse_response(self, contrasts: List[float]) -> None:
        """
        Calculate and store impulse responses for given contrasts.

        Parameters
        ----------
        contrasts : List[float]
            List of contrast values to use for impulse response calculation.
        """
        assert contrasts is not None and isinstance(
            contrasts, list
        ), "Impulse must specify contrasts as list, aborting..."

        vs, gcs = self.vs, self.gcs

        impulse_to_show = self.temporal_model.impulse_response(vs, gcs, contrasts)

        impulse_to_show["Unit idx"] = list(gcs.unit_indices)
        impulse_to_show["gc_type"] = gcs.gc_type
        impulse_to_show["response_type"] = gcs.response_type
        impulse_to_show["temporal_model_type"] = gcs.temporal_model_type

        self.project_data["impulse_to_show"] = impulse_to_show

    def create_spatial_filters(self) -> None:
        """
        Generate spatial filters for given unit indices.
        """
        gcs = self.gcs

        s = gcs.spatial_filter_sidelen
        spatial_filters = np.zeros((gcs.n_units, s, s))
        for idx, unit_index in enumerate(gcs.unit_indices):
            spatial_filters[idx, ...] = self.spatial_model.create_spatial_filter(
                gcs, unit_index
            )

        fixed_mask_threshold = gcs.fixed_mask_threshold

        # Get center masks for volume normalization. This must be done in 2D.
        center_masks = self._get_center_masks(spatial_filters, fixed_mask_threshold)
        surround_masks = self._get_surround_masks(
            gcs, spatial_filters, fixed_mask_threshold
        )

        # Reshape to N units, s**2 pixels
        center_masks_flat = center_masks.reshape((gcs.n_units, s**2))
        spatial_filters_flat = spatial_filters.reshape((gcs.n_units, s**2))

        # Scale spatial filters to sum one of centers for each unit to get veridical max contrast
        spatial_filters_flat_norm = (
            spatial_filters_flat
            / np.sum(spatial_filters_flat * center_masks_flat, axis=1)[:, None]
        )

        # Recalculate center masks in 2D for unity optimization. These are the actual center pixels.
        center_mask_threshold = gcs.mask_threshold
        center_masks = self._get_center_masks(spatial_filters, center_mask_threshold)
        center_masks_flat = center_masks.reshape((gcs.n_units, s**2))

        # We need to invert them back to max downwards for simulation.
        if gcs.response_type == "off":
            spatial_filters_flat_norm = -spatial_filters_flat_norm

        gcs.spatial_filters_flat = spatial_filters_flat_norm
        gcs.center_masks_flat = center_masks_flat
        gcs.surround_masks_flat = surround_masks.reshape((gcs.n_units, s**2))

        self.gcs = gcs

    def get_uniformity_index(self) -> None:
        """
        Calculate the uniformity index for retinal ganglion cell receptive fields.
        """
        vs = self.vs
        gcs = self.gcs

        height = vs.stimulus_height_pix
        width = vs.stimulus_width_pix
        unit_indices = gcs.unit_indices.copy()

        qmin, qmax, rmin, rmax = self.spatial_model._get_crop_pixels(gcs, unit_indices)

        stim_region = np.zeros((gcs.n_units, height, width), dtype=np.int32)
        center_region = np.zeros((gcs.n_units, height, width), dtype=np.int32)

        # Create the r and q indices for each unit, ensure they're integer type
        sidelen = gcs.spatial_filter_sidelen
        r_indices = (
            (np.arange(sidelen) + rmin[:, np.newaxis])
            .astype(int)
            .reshape(-1, 1, sidelen)
        )
        q_indices = (
            (np.arange(sidelen) + qmin[:, np.newaxis])
            .astype(int)
            .reshape(-1, sidelen, 1)
        )

        # Create r_matrix and q_matrix by broadcasting r_indices and q_indices
        r_matrix, q_matrix = np.broadcast_arrays(r_indices, q_indices)

        # create a unit index array
        unit_region_idx = np.arange(gcs.n_units).astype(np.int32).reshape(-1, 1, 1)

        # expand the indices arrays to the shape of r_matrix and q_matrix using broadcasting
        unit_region_idx = unit_region_idx + np.zeros_like(r_matrix, dtype=np.int32)

        # use the index arrays to select the elements from video_copy
        stim_region[unit_region_idx, r_matrix, q_matrix] = 1

        center_masks = gcs.center_masks_flat.copy()
        center_masks = center_masks.astype(bool).reshape(
            (gcs.n_units, sidelen, sidelen)
        )

        center_region[
            unit_region_idx * center_masks,
            r_matrix * center_masks,
            q_matrix * center_masks,
        ] = 1

        unit_region = np.sum(center_region, axis=0)

        # Delaunay triangulation for the total region
        gc = gcs.df_stimpix.iloc[unit_indices]
        q_center = np.round(gc.q_pix).astype(int).values
        r_center = np.round(gc.r_pix).astype(int).values

        # Create points array for Delaunay triangulation from r_center and q_center
        points = np.vstack((q_center, r_center)).T  # Shape should be (N, 2)

        # Perform Delaunay triangulation
        tri = Delaunay(points)

        # Initialize total area
        total_area = 0
        delaunay_mask = np.zeros((height, width), dtype=np.uint8)

        # Calculate the area of each triangle and sum it up
        for triangle in tri.simplices:
            # Get the vertices of the triangle
            vertices = points[triangle]

            # Use the vertices to calculate the area of the triangle
            # Area formula for triangles given coordinates: 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
            total_area += 0.5 * abs(
                vertices[0, 0] * (vertices[1, 1] - vertices[2, 1])
                + vertices[1, 0] * (vertices[2, 1] - vertices[0, 1])
                + vertices[2, 0] * (vertices[0, 1] - vertices[1, 1])
            )

            # Get the bounding box of the triangle to minimize the area to check
            min_x = np.min(vertices[:, 0])
            max_x = np.max(vertices[:, 0])
            min_y = np.min(vertices[:, 1])
            max_y = np.max(vertices[:, 1])

            # Generate a grid of points representing pixels in the bounding box
            x_range = np.arange(min_x, max_x + 1)
            y_range = np.arange(min_y, max_y + 1)
            grid_x, grid_y = np.meshgrid(x_range, y_range)

            # Use the points in the grid to check if they are inside the triangle
            grid_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
            indicator = tri.find_simplex(grid_points) >= 0

            # Reshape the indicator back to the shape of the bounding box grid
            indicator = indicator.reshape(grid_x.shape)

            # Place the indicator in the mask image
            delaunay_mask[min_y : max_y + 1, min_x : max_x + 1] = np.logical_or(
                delaunay_mask[min_y : max_y + 1, min_x : max_x + 1], indicator
            )

        # Unity region is where exactly one unit centre overlaps with the retina region
        unity_region = (unit_region * delaunay_mask) == 1

        uniformify_index = np.sum(unity_region) / np.sum(delaunay_mask)

        uniformify_data = {
            "uniformify_index": uniformify_index,
            "total_region": delaunay_mask,
            "unity_region": unity_region,
            "unit_region": unit_region,
            "mask_threshold": gcs.mask_threshold,
        }

        self.project_data["uniformify_data"] = uniformify_data

    def apply_optical_aberration(self) -> None:
        """
        Apply optical aberration to the visual signal.
        """

        match self.vs.optical_aberration:
            case None | 0.0:
                return

        vs = self.vs
        frames = vs.stimulus_video.frames.astype(np.float32)

        frames = self.cones.blur_stimulus_with_optical_aberration(
            frames,
            vs.optical_aberration,
            vs.pix_per_deg,
        )
        dtype = eval(f"np.{vs.stimulus_video.options['dtype_name']}")

        vs.stimulus_video.frames = frames.astype(dtype)
        self.vs = vs

    def get_spatially_cropped_video(self) -> None:
        """
        Crop the video to the surroundings of the specified retinal ganglion cells (RGCs).
        """
        vs = self.vs
        gcs = self.gcs

        sidelen = gcs.spatial_filter_sidelen
        unit_indices = gcs.unit_indices.copy()
        video_copy = vs.stimulus_video.frames.copy()
        # Original frames become [height, width, time points]
        video_copy = np.transpose(video_copy, (1, 2, 0))
        video_copy = video_copy[np.newaxis, ...]  # Add new axis for broadcasting

        qmin, qmax, rmin, rmax = self.spatial_model._get_crop_pixels(gcs, unit_indices)

        # Adjust the creation of r_indices and q_indices for proper broadcasting
        r_indices = np.arange(sidelen) + rmin[:, np.newaxis]
        q_indices = np.arange(sidelen) + qmin[:, np.newaxis]

        # Ensure r_indices and q_indices are repeated for each unit
        r_indices = r_indices.reshape(-1, 1, sidelen, 1)
        q_indices = q_indices.reshape(-1, sidelen, 1, 1)

        # Broadcasting to create compatible shapes for indexing
        r_matrix, q_matrix = np.broadcast_arrays(r_indices, q_indices)
        time_points_indices = np.arange(video_copy.shape[-1])

        # Convert NumPy arrays to PyTorch tensors
        device = self.device
        print(f"KUKKUU4 {device=}")
        video_copy_tensor = torch.tensor(video_copy, dtype=torch.float32, device=device)
        r_matrix_tensor = torch.tensor(r_matrix, dtype=torch.long, device=device)
        q_matrix_tensor = torch.tensor(q_matrix, dtype=torch.long, device=device)
        time_points_indices_tensor = torch.tensor(
            time_points_indices, dtype=torch.long, device=device
        )

        # Determine batch size
        batch_size = 100  # Adjust this based on your GPU memory

        # Calculate the number of batches
        num_batches = len(r_matrix_tensor) // batch_size + (
            len(r_matrix_tensor) % batch_size != 0
        )

        # Process the first batch to determine the shape of the final array
        r_batch = r_matrix_tensor[0:batch_size]
        q_batch = q_matrix_tensor[0:batch_size]

        # Perform slicing and arithmetic operations on the first batch
        stimulus_cropped_batch = video_copy_tensor[
            0, r_batch, q_batch, time_points_indices_tensor
        ]
        stimulus_cropped_batch = stimulus_cropped_batch  # / 128 - 1.0

        # Determine the shape of the final array
        final_shape = (len(r_matrix_tensor),) + stimulus_cropped_batch.shape[1:]

        # Preallocate a NumPy array to store the results
        stimulus_cropped = np.empty(final_shape, dtype=np.float32)

        # Store the result of the first batch in the preallocated array
        stimulus_cropped[0:batch_size] = stimulus_cropped_batch.cpu().numpy()

        # Process the remaining data in batches
        for i in range(1, num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            # Get the current batch
            r_batch = r_matrix_tensor[start_idx:end_idx]
            q_batch = q_matrix_tensor[start_idx:end_idx]

            # Perform slicing and arithmetic operations on the current batch
            stimulus_cropped_batch = video_copy_tensor[
                0, r_batch, q_batch, time_points_indices_tensor
            ]
            stimulus_cropped_batch = stimulus_cropped_batch  # / 127.5 - 1.0

            # Store the result in the preallocated array
            stimulus_cropped[start_idx:end_idx] = stimulus_cropped_batch.cpu().numpy()

        n_frames = vs.stimulus_video.frames.shape[0]
        stimulus_cropped = stimulus_cropped.reshape((gcs.n_units, sidelen**2, n_frames))

        vs.stimulus_cropped = stimulus_cropped
        vs.qr_min_max = (qmin, qmax, rmin, rmax)

        self.vs = vs

    def adapt_to_stimulus_video(self) -> None:
        """
        Adapt the simulation parameters to the stimulus video.

        This method adjusts the visual signal to mean zero
        and contrast 1 reaching [-1, 1].
        """
        vs = self.vs
        vs.stimulus_cropped_adapted = vs.stimulus_cropped / vs.mean_luminance - 1.0
        self.vs = vs

    def get_noise(self) -> None:
        """
        Create noise for the simulation.
        """
        vs = self.vs
        ndim_cones = (self.cones.n_units, vs.stim_len_tp, self.n_sweeps)
        ndim_gc = (self.gcs.n_units, vs.stim_len_tp, self.n_sweeps)

        if not hasattr(vs, "cone_noise"):
            self.vs = self.cones.create_noise(vs, self.n_sweeps)
        if not hasattr(vs, "gc_synaptic_noise"):
            self.vs = self.cones.connect_cone_noise_to_gcs(vs, self.n_sweeps)

        # Dimension checks
        if not vs.cone_noise.shape == ndim_cones:
            raise ValueError("Cone noise shape mismatch")
        if not vs.gc_synaptic_noise.shape == ndim_gc:
            raise ValueError("GC synaptic noise shape mismatch")

    def get_generator_potentials(self) -> None:
        """
        Calculate generator potentials for the current temporal model.
        """
        self.vs, self.gcs = self.temporal_model.create_generator_potential(
            self.vs, self.gcs
        )

    def generate_spikes(self) -> None:
        """
        Generate spikes based on the calculated generator potentials.
        """
        vs = self.vs
        gcs = self.gcs
        n_sweeps = self.n_sweeps

        vs = self._generator_to_firing_rate_noise(vs, gcs, n_sweeps)

        video_n_frames = vs.stimulus_video.video_n_frames
        video_dt = vs.video_dt
        duration = vs.duration
        simulation_dt = vs.simulation_dt

        n_units = gcs.n_units
        spike_generator_model = gcs.spike_generator_model
        refractory_parameters = gcs.refractory_parameters
        tvec_original = np.arange(1, video_n_frames + 1) * video_dt

        spikearrays = []
        all_spiketrains = []
        for this_sweep in range(n_sweeps):
            firing_rates = vs.firing_rates[..., this_sweep]
            tvec_new, inst_rates = self._firing_rates2brian_timed_arrays(
                firing_rates, tvec_original, duration, simulation_dt
            )
            spikearrays_this_sweep, spiketrains_this_sweep = (
                self._brian_spike_generation(
                    n_units,
                    inst_rates,
                    spike_generator_model,
                    refractory_parameters,
                    simulation_dt,
                    duration,
                )
            )
            # Store the results in the visual signal object
            spikearrays.append(spikearrays_this_sweep)
            all_spiketrains.append(spiketrains_this_sweep)

        vs.n_units = n_units
        vs.tvec_new = tvec_new
        vs.inst_rates = inst_rates
        vs.spikearrays = spikearrays
        vs.all_spiketrains = all_spiketrains

        self.vs = vs


class SimulationDirector:
    """
    Directs the simulation process using a builder pattern.

    This class orchestrates the steps of various simulation processes
    by delegating to a builder object.

    Parameters
    ----------
    builder : Any
        The builder object responsible for constructing simulation components.

    Attributes
    ----------
    builder : Any
        The builder object used for simulation construction.
    """

    def __init__(self, builder: Any) -> None:
        """
        Initialize the SimulationDirector with a builder.

        Parameters
        ----------
        builder : Any
            The builder object for simulation construction.
        """
        self.builder: Any = builder

    def run_simulation(self) -> None:
        """
        Run the full simulation process.

        This method orchestrates the steps of the simulation in sequence.
        """
        self.builder.get_concrete_components()
        self.builder.create_spatial_filters()
        self.builder.apply_optical_aberration()
        self.builder.get_spatially_cropped_video()
        self.builder.adapt_to_stimulus_video()
        self.builder.get_noise()
        self.builder.get_generator_potentials()
        self.builder.generate_spikes()

    def run_impulse_response(self, contrasts: List[float]) -> None:
        """
        Run the impulse response simulation.

        Parameters
        ----------
        contrasts : List[float]
            List of contrast values for the impulse response simulation.
        """
        self.builder.get_concrete_components()
        self.builder.get_impulse_response(contrasts)

    def run_uniformity_index(self) -> None:
        """
        Run the uniformity index calculation.

        This method orchestrates the steps to calculate the uniformity index.
        """
        self.builder.get_concrete_components()
        self.builder.create_spatial_filters()
        self.builder.get_uniformity_index()

    def get_simulation_result(self) -> Tuple[Any, Any]:
        """
        Retrieve the simulation results.

        Returns
        -------
        Tuple[Any, Any]
            A tuple containing the vs and gcs simulation results.
        """
        vs = self.builder.vs
        gcs = self.builder.gcs
        return vs, gcs


class ReceptiveFieldsBase(ABC, PrintableMixin, RetinaMath):
    """
    Base class for receptive fields information.

    Contains information associated with receptive fields, including
    retina parameters, spatial and temporal filters.

    Parameters
    ----------
    retina_parameters : Dict[str, Any]
        Dictionary containing retina parameters.

    Attributes
    ----------
    retina_parameters : Dict[str, Any]
        Dictionary of retina parameters.
    spatial_filter_sidelen : int
        Side length of the spatial filter.
    microm_per_pix : float
        Micrometers per pixel.
    temporal_filter_len : int
        Length of the temporal filter.
    gc_type : str
        Type of ganglion cell.
    response_type : str
        Type of response.
    deg_per_mm : float
        Degrees per millimeter.
    dog_model_type : str
        Type of Difference of Gaussians model.
    spatial_model_type : str
        Type of spatial model.
    temporal_model_type : str
        Type of temporal model.
    """

    def __init__(self, retina_parameters: Dict[str, Any]) -> None:
        """
        Initialize the ReceptiveFieldsBase instance.

        Parameters
        ----------
        retina_parameters : Dict[str, Any]
            Dictionary containing retina parameters.
        """
        self.retina_parameters: Dict[str, Any] = retina_parameters
        self.spatial_filter_sidelen: int = 0
        self.microm_per_pix: float = 0.0
        self.temporal_filter_len: int = 0
        self.gc_type: str = self.retina_parameters["gc_type"]
        self.response_type: str = self.retina_parameters["response_type"]
        self.deg_per_mm: float = self.retina_parameters["deg_per_mm"]
        self.dog_model_type: str = self.retina_parameters["dog_model_type"]
        self.spatial_model_type: str = self.retina_parameters["spatial_model_type"]
        self.temporal_model_type: str = self.retina_parameters["temporal_model_type"]


class ConeProduct(ReceptiveFieldsBase):
    """
    A class representing cone photoreceptors in the retina.

    This class inherits from ReceptiveFieldsBase and handles the processing
    of visual signals through cone photoreceptors, including signal generation
    and noise creation.

    Parameters
    ----------
    retina_parameters : Dict[str, Any]
        Dictionary containing retina parameters.
    ret_npz : Dict[str, np.ndarray]
        Dictionary containing retina data from NPZ file.
    device : Any
        Device for computation .
    ND_filter : float
        Neutral Density filter value.
    interpolate_data : callable
        Function to interpolate data.
    lin_interp_and_double_lorenzian : callable
        Function for linear interpolation and double Lorenzian.
    target_gc_for_multiple_trials : Optional[int], optional
        Index of the target ganglion cell for multiple trials, by default None.

    Attributes
    ----------
    cones_to_gcs_weights : np.ndarray
        Weights connecting cones to ganglion cells.
    cone_noise_parameters : np.ndarray
        Parameters for cone noise generation.
    cone_general_parameters : Dict[str, Any]
        General parameters for cones.
    n_units : int
        Number of cone units.
    target_gc_for_multiple_trials : Optional[int]
        Index of the target ganglion cell for multiple trials.
    """

    def __init__(
        self,
        retina_parameters: Dict[str, Any],
        ret_npz: Dict[str, np.ndarray],
        device: Any,
        ND_filter: float,
        interpolate_data: callable,
        lin_interp_and_double_lorenzian: callable,
        target_gc_for_multiple_trials: Optional[int] = None,
    ) -> None:
        super().__init__(retina_parameters)

        self.retina_parameters = retina_parameters
        self.ret_npz = ret_npz
        self.device = device
        self.ND_filter = ND_filter
        self.interpolate_data = interpolate_data
        self.lin_interp_and_double_lorenzian = lin_interp_and_double_lorenzian

        self.cones_to_gcs_weights = ret_npz["cones_to_gcs_weights"]
        self.cone_noise_parameters = ret_npz["cone_noise_parameters"]
        self.cone_general_parameters = retina_parameters["cone_general_parameters"]
        self.n_units = self.cones_to_gcs_weights.shape[0]
        self.target_gc_for_multiple_trials = target_gc_for_multiple_trials

    def _cornea_photon_flux_density_to_luminance(
        self, F: float, lambda_nm: float = 555
    ) -> float:
        """
        Convert photon flux density at cornea to luminance using human photopic vision V(lambda).

        Parameters
        ----------
        F : float
            Photon flux density at the cornea in photons/mm²/s.
        lambda_nm : float, optional
            Wavelength of the monochromatic light in nanometers, default is 555 nm.

        Returns
        -------
        float
            Luminance in cd/m².
        """
        # Constants
        h = 6.626e-34  # Planck's constant in J·s
        c = 3.00e8  # Speed of light in m/s
        lambda_m = lambda_nm * 1e-9  # Convert wavelength from nm to m
        kappa = 683  # Luminous efficacy of monochromatic radiation in lm/W at 555 nm

        # Energy of a photon at wavelength lambda in joules
        E_photon = (h * c) / lambda_m

        # Convert photon flux density F to luminance L in cd/m²
        F_m2 = F * 1e6  # Convert photon flux density from mm² to m²
        L = F_m2 * E_photon * kappa / np.pi

        return L

    def _luminance_to_cornea_photon_flux_density(
        self, L: float, lambda_nm: float = 555
    ) -> float:
        """
        Convert luminance to photon flux density at cornea using human photopic vision V(lambda).

        Parameters
        ----------
        L : float
            Luminance in cd/m².
        lambda_nm : float, optional
            Wavelength of the monochromatic light in nanometers, default is 555 nm.

        Returns
        -------
        float
            Photon flux density at the cornea in photons/mm²/s.
        """
        # Constants
        h = 6.626e-34  # Planck's constant in J·s
        c = 3.00e8  # Speed of light in m/s
        lambda_m = lambda_nm * 1e-9  # Convert wavelength from nm to m
        kappa = 683  # Luminous efficacy of monochromatic radiation in lm/W
        # at 555 nm (peak human photopic sensitivity)
        # Luminous efficacy of typical sunlight is usually estimated to be around 93 lumens per watt

        # Energy of a photon at wavelength lambda in joules
        E_photon = (h * c) / lambda_m  # ok, see table1 Shapley_1984_ProgRetRes_chapter9

        # Convert luminance L to photon flux density F in photons/mm²/s. The np.pi * L is the luminous flux in lumens
        # approximated from the luminance L in cd/m². The kappa is the luminous efficacy of monochromatic radiation
        F_m2 = np.pi * L / (E_photon * kappa)
        F = F_m2 / 1e6  # Convert from m² to mm²

        return F

    def _create_cone_noise(
        self, tvec: np.ndarray, n_cones: int, *params: float
    ) -> np.ndarray:
        """
        Create cone noise based on given parameters.

        Parameters
        ----------
        tvec : np.ndarray
            Time vector.
        n_cones : int
            Number of cones.
        *params : float
            Additional parameters for noise generation.

        Returns
        -------
        np.ndarray
            Generated cone noise.
        """
        tvec = tvec / b2u.second
        freqs = fftpack.fftfreq(len(tvec), d=(tvec[1] - tvec[0]))

        white_noise = np.random.normal(0, 1, (len(tvec), n_cones))
        noise_fft = fftpack.fft(white_noise, axis=0)

        # Generate the asymmetric concave function for scaling
        f_scaled = np.abs(freqs)
        # Prevent division by zero for zero frequency
        f_scaled[f_scaled == 0] = 1e-10

        # Transfer to log scale and
        # combine the fitted amplitudes with fixed corner frequencies
        a0 = params[0]
        L1_params = np.array([params[1], self.cone_noise_wc[0]])
        L2_params = np.array([params[2], self.cone_noise_wc[1]])

        asymmetric_scale = self.lin_interp_and_double_lorenzian(
            f_scaled, a0, L1_params, L2_params, self.cone_interp_function
        )

        noise_fft = noise_fft * asymmetric_scale[:, np.newaxis]

        # Transform back to time domain
        cone_noise = np.real(fftpack.ifft(noise_fft, axis=0))

        return cone_noise

    def _create_cone_signal_clark(
        self,
        cone_input: np.ndarray,
        p: Dict[str, Any],
        dt: float,
        duration: float,
        tvec: np.ndarray,
        pad_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create cone signal using Brian2. Works in video time domain.

        Parameters
        ----------
        cone_input : np.ndarray
            The cone input of shape (n_cones, n_timepoints).
        p : Dict[str, Any]
            The cone signal parameters.
        dt : float
            The video time step.
        duration : float
            The duration of input video.
        tvec : np.ndarray
            The time vector of input video.
        pad_value : float, optional
            Padding value, by default 0.0.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The cone signal and the cone signal with units.

        Notes
        -----
        The Clark 2013 PLoSComputBiol model output is in mV, referring to photoreceptor membrane potential.
        The Angueyra 2022 JNeurosci model output is in pA, referring to photoreceptor outer segment current.
        """

        alpha = p["alpha"]
        beta = p["beta"]
        gamma = p["gamma"]
        tau_y = p["tau_y"]
        n_y = p["n_y"]
        tau_z = p["tau_z"]
        n_z = p["n_z"]
        tau_r = p["tau_r"]  # Goes into Brian # noqa: F841
        filter_limit_time = p["filter_limit_time"]
        input_gain = p["input_gain"]
        max_response = p["max_response"]
        r_dark = p["r_dark"]

        def simple_filter(t, n, tau):
            norm_coef = gamma_function(n + 1) * np.power(tau, n + 1)
            values = (t**n / norm_coef) * np.exp(-t / tau)
            return values

        tvec_idx = tvec < filter_limit_time
        tvec_filter = tvec

        Ky = simple_filter(tvec_filter / b2u.second, n_y, tau_y / b2u.second)
        Kz_prime = simple_filter(tvec_filter / b2u.second, n_z, tau_z / b2u.second)
        Kz = gamma * Ky + (1 - gamma) * Kz_prime

        # Cut filters for computational efficiency
        Ky = Ky[tvec_idx]
        Kz = Kz[tvec_idx]
        Kz_prime = Kz_prime[tvec_idx]

        # Normalize filters to full filter = 1.0
        Ky = Ky / Ky.sum()
        Kz = Kz / Kz.sum()

        # Prepare 2D convolution for the filters,
        Ky_2D_kernel = Ky.reshape(1, -1)
        Kz_2D_kernel = Kz.reshape(1, -1)

        pad_length = len(Ky) - 1

        assert all(cone_input[:, 0] == pad_value), "Padding failed..."

        cone_input_u = cone_input * input_gain
        pad_value_u = pad_value * input_gain
        # Pad cone input start with the initial value to avoid edge effects. Use filter limit time.
        cone_input_padded = np.pad(
            cone_input_u,
            ((0, 0), (pad_length, 0)),
            mode="constant",
            constant_values=((0, 0), (pad_value_u, 0)),
        )

        print("\nConvolving cone signal matrices...")
        y_mtx = convolve(cone_input_padded, Ky_2D_kernel, mode="full", method="fft")[
            :, pad_length : pad_length + len(tvec)
        ]
        z_mtx = convolve(cone_input_padded, Kz_2D_kernel, mode="full", method="fft")[
            :, pad_length : pad_length + len(tvec)
        ]

        # # Add units to the matrices. Photoisomerizations per sec per um2
        y_mtx_u = y_mtx * b2u.second**-1
        z_mtx_u = z_mtx * b2u.second**-1

        print("\nRunning Brian code for cones...")
        y_mtx_ta = b2.TimedArray(y_mtx_u.T, dt=dt)  # noqa: F841
        z_mtx_ta = b2.TimedArray(z_mtx_u.T, dt=dt)  # noqa: F841
        # In original Clark model, the response is defined as r(t) = V(t) - Vrest
        # r(t) is the photoreceptor response (mV), V(t) is the photoreceptor membrane potential
        # Vrest is the depolarized cone membrane potential in the dark.
        # Negative r means hyperpolarization from the resting potential in millivolts.
        # In Angueyra_2022_JNeurosci model application of CLark model, the response is defined as
        # photoreceptor current (pA) and the resting dark current is measured in pA. The response
        # was scaled "so it matched the dark current for the naturalistic stimulus".

        if p["unit"] == "mV":
            eqs = b2.Equations(
                """
                dr/dt = (alpha * y_mtx_ta(t,i) - r - beta * z_mtx_ta(t,i) * r) / tau_r : volt
                """
            )
        elif p["unit"] == "pA":
            eqs = b2.Equations(
                """
                dr/dt = (alpha * y_mtx_ta(t,i) - r - beta * z_mtx_ta(t,i) * r) / tau_r : amp
                """
            )
        # Assuming dr/dt is zero at t=0, a.k.a. steady illumination
        r_initial_value = alpha * y_mtx_u[0, 0] / (1 + beta * z_mtx_u[0, 0])

        G = b2.NeuronGroup(self.n_units, eqs, dt=dt, method="exact")
        G.r = r_initial_value
        M = b2.StateMonitor(G, ("r"), record=True)
        b2.run(duration)

        # Synaptic vesicle release assumed linearly coupled to negative current.
        # It reduces with light and increases with dark.
        # Burkhardt 1994 and Clark 2013 response scaling is 0-1, but
        # vesicle release is 1-0. So, we invert the response to keep vesicle release
        # as the cone_output.
        response = M.r
        r0 = r_initial_value
        r0 = np.array(r0)[np.newaxis, np.newaxis]
        if p["unit"] == "mV":
            # np drops unit and interprets value as float, eg 7 mV -> 0.007 V
            r0 = r0 * b2u.volt
            # Shifted raw response
            cone_output_u = (r_dark + response) / b2u.mV
            cone_output = 1 + response / np.abs(max_response)
        elif p["unit"] == "pA":
            r0 = r0 * b2u.amp
            cone_output_u = (r_dark + response) / b2u.pA
            cone_output = 1 - response / np.abs(max_response)

        return cone_output, cone_output_u

    @staticmethod
    def blur_stimulus_with_optical_aberration(
        image: np.ndarray, optical_aberration: float, pix_per_deg: float
    ) -> np.ndarray:
        """
        Gaussian smoothing of images with optical aberration.
                optical_aberration = self.config.retina_parameters["optical_aberration"]

        """

        # Turn the optical aberration of 2 arcmin FWHM to Gaussian function sigma
        sigma_deg = optical_aberration / (2 * np.sqrt(2 * np.log(2)))
        sigma_pix = pix_per_deg * sigma_deg

        # Apply Gaussian blur to each frame in the image array
        if len(image.shape) == 3:
            image_after_optics = ndimage.gaussian_filter(
                image, sigma=[0, sigma_pix, sigma_pix]
            )
        elif len(image.shape) == 2:
            image_after_optics = ndimage.gaussian_filter(
                image, sigma=[sigma_pix, sigma_pix]
            )
        else:
            raise ValueError("Image must be 2D or 3D, aborting...")

        return image_after_optics

    # Detached internal legacy functions
    def _luminance2cone_response(self):
        """
        Cone nonlinearity. Equation from Baylor_1987_JPhysiol.
        """

        # Range
        response_range = np.ptp([self.cone_sensitivity_min, self.cone_sensitivity_max])

        # Scale. Image should be between 0 and 1
        image_at_response_scale = self.image * response_range
        cone_input = image_at_response_scale + self.cone_sensitivity_min

        # Cone nonlinearity
        cone_response = self.rm * (1 - np.exp(-self.k * cone_input))

        self.cone_response = cone_response

        # Save the cone response to output folder
        filename = self.config.external_stimulus_parameters["stimulus_file"]
        self.data_io.save_cone_response_to_hdf5(filename, cone_response)

    # Public functions
    def create_signal(self, vs: "VisualSignal") -> "VisualSignal":
        """
        Generates cone signal.

        Parameters
        ----------
        vs : VisualSignal
            The simulation product object containing the transduction cascade from stimulus video
            to RGC unit spike response.

        Returns
        -------
        VisualSignal
            The updated visual signal object with cone signal.
        """

        video_copy = vs.stimulus_video.frames.copy()
        video_copy = np.transpose(
            video_copy, (1, 2, 0)
        )  # Original frames are now [height, width, time points]
        video_copy = video_copy[np.newaxis, ...]  # Add new axis for broadcasting

        cone_pos_mm = self.ret_npz["cone_optimized_pos_mm"]
        cone_pos_deg = cone_pos_mm * vs.deg_per_mm
        q, r = vs._vspace_to_pixspace(cone_pos_deg[:, 0], cone_pos_deg[:, 1])
        q_idx = np.floor(q).astype(int)
        r_idx = np.floor(r).astype(int)

        # Ensure r_indices and q_indices are repeated for each unit
        r_indices = r_idx.reshape(-1, 1, 1, 1)
        q_indices = q_idx.reshape(-1, 1, 1, 1)
        time_points_indices = np.arange(video_copy.shape[-1])

        # Use advanced indexing for selecting pixels
        cone_input_cropped = video_copy[0, r_indices, q_indices, time_points_indices]
        cone_input = np.squeeze(cone_input_cropped)

        # Photoisomerization units need more bits to represent the signal
        cone_input = np.squeeze(cone_input_cropped).astype(np.float64)
        minl = np.min(cone_input)
        maxl = np.max(cone_input)

        params_dict = self.retina_parameters["cone_signal_parameters"]
        lambda_nm = params_dict["lambda_nm"]
        A_pupil = params_dict["A_pupil"]
        cone_input_R = self.get_photoisomerizations_from_luminance(
            cone_input, lambda_nm=lambda_nm, A_pupil=A_pupil
        )
        vs.lambda_nm = lambda_nm
        vs.A_pupil = A_pupil

        # Neutral Density filtering factor (ff) to reduce or increase luminance
        ff = np.power(10.0, -self.ND_filter)
        cone_input_R = cone_input_R * ff
        minp = np.round(np.min(cone_input_R)).astype(int)
        maxp = np.round(np.max(cone_input_R)).astype(int)

        print(f"\nLuminance range: {minl * ff:.3f} to {maxl * ff:.3f} cd/m²")
        print(f"\nR* range: {minp} to {maxp} photoisomerizations/cone/s")

        # Update visual stimulus photodiode response
        vs.photodiode_response = vs.photodiode_response * ff
        vs.photodiode_Rstar_range = [minp, maxp]

        # Update mean value
        background = vs.options_from_videofile["background"]
        background_R = self.get_photoisomerizations_from_luminance(
            background, lambda_nm=lambda_nm, A_pupil=A_pupil
        )

        background_R = background_R * ff

        print(f"\nbackground_R* {background_R:.0f} photoisomerizations/cone/s")

        tvec = vs.tvec
        dt = vs.video_dt
        duration = vs.duration

        cone_signal, cone_signal_u = self._create_cone_signal_clark(
            cone_input_R, params_dict, dt, duration, tvec, pad_value=background_R
        )

        print(f"\nCone signal min:{cone_signal_u.min():.1f} {params_dict['unit']}")
        print(f"Cone signal max:{cone_signal_u.max():.1f} {params_dict['unit']}")

        vs.cone_signal = cone_signal
        vs.cone_signal_u = cone_signal_u

        return vs

    def create_noise(self, vs: "VisualSignal", n_sweeps: int) -> "VisualSignal":
        """
        Generates cone noise.

        Parameters
        ----------
        vs : VisualSignal
            The simulation product object containing the transduction cascade from stimulus video
            to RGC unit spike response.
        n_sweeps : int
            The number of trials to simulate.

        Returns
        -------
        VisualSignal
            The updated visual signal object with cone noise.
        """
        params = self.cone_noise_parameters
        n_cones = self.n_units

        cone_frequency_data = self.ret_npz["cone_frequency_data"]
        cone_power_data = self.ret_npz["cone_power_data"]
        self.cone_interp_function = self.interpolate_data(
            cone_frequency_data, cone_power_data
        )
        self.cone_noise_wc = self.cone_general_parameters["cone_noise_wc"]

        # Make independent cone noise for multiple trials
        # The variables cone_noise_XXX may become a memory issue for long trials or big retinas.
        trials_noise = np.empty((vs.tvec.size, n_cones, 0))
        for _trial in range(n_sweeps):
            cone_noise = self._create_cone_noise(vs.tvec, n_cones, *params)
            cone_noise_expanded = np.expand_dims(cone_noise, axis=2)
            trials_noise = np.concatenate((trials_noise, cone_noise_expanded), axis=2)

        cone_noise = trials_noise

        # Normalize noise to have one mean and unit sd at the noise data frequencies
        cone_noise_norm = (cone_noise - cone_noise.mean()) / np.std(cone_noise, axis=0)

        # Transpose cone_noise_norm for shape (n_cones, n_timepoints)
        cone_noise_norm_T = np.moveaxis(cone_noise_norm, 0, 1)
        vs.cone_noise = cone_noise_norm_T

        return vs

    def connect_cone_noise_to_gcs(
        self, vs: "VisualSignal", n_sweeps: int
    ) -> "VisualSignal":
        # Calculate the synaptic noise for ganglion cells
        print("\nUsing PyTorch for connecting cone noise to ganglion cells...")

        # if model is dynamic or fixed, connect cone noise directly to ganglion cells
        cones_to_gcs_weights = self.cones_to_gcs_weights
        magn = self.retina_parameters["noise_gain"]
        cone_noise_norm_T = vs.cone_noise * magn
        cone_noise_norm = np.moveaxis(cone_noise_norm_T, 0, 1)

        if np.any(np.sum(cones_to_gcs_weights, axis=0) == 0):
            raise ValueError("Zero value in cones_to_gcs_weights, aborting...")
        weights_norm = cones_to_gcs_weights / np.sum(cones_to_gcs_weights, axis=0)
        n_gcs = weights_norm.shape[1]

        device = self.device
        cone_noise_norm_tensor = torch.tensor(
            cone_noise_norm, device=device, dtype=torch.float32
        )
        weights_norm_tensor = torch.tensor(
            weights_norm, device=device, dtype=torch.float32
        )
        magn_tensor = torch.tensor(magn, device=device, dtype=torch.float32)
        tvec_size = vs.tvec.size

        gc_synaptic_noise_tensor = torch.zeros(
            (n_gcs, tvec_size, n_sweeps), device=device
        )

        for trial in range(n_sweeps):
            noise_this_trial = magn_tensor * torch.matmul(
                cone_noise_norm_tensor[:, :, trial], weights_norm_tensor
            )
            gc_synaptic_noise_tensor[..., trial] = noise_this_trial.T

        gc_synaptic_noise = gc_synaptic_noise_tensor.cpu().numpy()

        noise_type = self.retina_parameters["noise_type"]

        # If noise type is independent, assume Gaussian statistics, independent for each gc unit
        match noise_type:
            case "shared":
                noise_out = gc_synaptic_noise
            case "independent":
                mu = np.mean(gc_synaptic_noise.flatten())
                sigma = np.std(gc_synaptic_noise.flatten())
                dims = gc_synaptic_noise.shape
                noise_out = np.random.normal(loc=mu, scale=sigma, size=dims)

        vs.gc_synaptic_noise = noise_out

        return vs


class BipolarProduct(ReceptiveFieldsBase):
    """
    A class representing bipolar cells in the retina.

    This class inherits from ReceptiveFieldsBase and handles the processing
    of visual signals through bipolar cells.

    Parameters
    ----------
    retina_parameters : Dict[str, Any]
        Dictionary containing retina parameters.
    ret_npz : Dict[str, np.ndarray]
        Dictionary containing retina data from NPZ file.
    target_gc_for_multiple_trials : Optional[int]
        Index of the target ganglion cell for multiple trials, by default None.

    Attributes
    ----------
    retina_parameters : Dict[str, Any]
        Dictionary containing retina parameters.
    ret_npz : Dict[str, np.ndarray]
        Dictionary containing retina data from NPZ file.
    n_units : int
        Number of bipolar cell units.
    target_gc_for_multiple_trials : Optional[int]
        Index of the target ganglion cell for multiple trials.
    """

    def __init__(
        self,
        retina_parameters: Dict[str, Any],
        ret_npz: Dict[str, np.ndarray],
        target_gc_for_multiple_trials: Optional[int] = None,
    ) -> None:
        super().__init__(retina_parameters)
        self.retina_parameters = retina_parameters
        self.ret_npz = ret_npz
        self.n_units = self.ret_npz["bipolar_to_gcs_cen_weights"].shape[0]
        self.target_gc_for_multiple_trials = target_gc_for_multiple_trials

    def create_signal(self, vs: "VisualSignal") -> "VisualSignal":
        """
        Apply a bipolar nonlinearity function to create ganglion cell synaptic input.

        This method processes the cone signals through bipolar cells, applying
        necessary transformations and nonlinearities to generate the synaptic
        input for ganglion cells.

        Parameters
        ----------
        vs : VisualSignal
            An object containing properties and methods related to the visual stimulus.
            Expected to have attributes `svecs_sur` (stimulus vectors for surround),
            `cone_signal`, and `cone_noise`.

        Returns
        -------
        VisualSignal
            The modified visual stimulus object with updated ganglion cell synaptic input.

        Notes
        -----
        This method performs several steps:
        1. Applies response type-specific transformation to cone output.
        2. Calculates bipolar input contrast.
        3. Computes bipolar cell center and surround responses.
        4. Applies nonlinearity (rectification) to bipolar to gc synapses, timepointwise.
        5. Generates ganglion cell activation.

        The specific nonlinearity applied is based on the Turner_2018_eLife model.
        """

        popt = self.ret_npz["bipolar_nonlinearity_parameters"]
        bipolar_to_gcs_cen_weights = self.ret_npz["bipolar_to_gcs_cen_weights"]
        bipolar_to_gcs_sur_weights = self.ret_npz["bipolar_to_gcs_sur_weights"]

        # Get constructed weights [n_cones, n_bipolars]
        cones_to_bipolars_cen_w = self.ret_npz["cones_to_bipolars_center_weights"]
        cones_to_bipolars_sur_w = self.ret_npz["cones_to_bipolars_surround_weights"]

        # [n_cones, n_timepoints]
        # Currently vs.cone_noise added at spike generation
        cone_output = vs.cone_signal

        # Sign inversion for cones' glutamate release => ON bipolars
        if self.retina_parameters["response_type"] == "on":
            cone_output = 1 - cone_output
        elif self.retina_parameters["response_type"] == "off":
            cone_output = cone_output

        # Turn cone signal to bipolar contrast signal, cmp Weber contrast. Turner 2018 eq 7
        # Turner uses whole natural image to Weber contrast, here only cone signal is available.
        # If baseline exists, use it, if not, take the mean of the first 10 timepoints
        # For natural images, consider mean over image, separately for each timepoint.
        if vs.baseline_len_tp > 0:
            baseline_len_tp = vs.baseline_len_tp
        else:
            baseline_len_tp = 10
        baseline = np.mean(cone_output[:, :baseline_len_tp], axis=1)[:, np.newaxis]
        bipolar_input_contrast = (cone_output - baseline) / baseline

        # [n_bipolars, n_timepoints], subunit center and surround
        bipolar_cen_sum = cones_to_bipolars_cen_w.T @ bipolar_input_contrast
        bipolar_sur_sum = cones_to_bipolars_sur_w.T @ bipolar_input_contrast
        subunit_sum = bipolar_cen_sum - bipolar_sur_sum

        vs.bipolar_signal = subunit_sum

        # Calculate the nonlinear neg_scaler with dimensions [n_timepoints, n_gcs]
        ##########################################################################
        # invert polarity for surround
        gc_surround_linear_input = bipolar_to_gcs_sur_weights.T @ (-1 * subunit_sum)

        # RI is Rectification Index. The abscissa values in Turner 2018 Fig 5C
        # reflect surround conductances and are scaled to match our max surround
        # linear input values for parasol on unit[-0.15, 0.15].
        # See _fit_bipolar_rectification_index for implementation.

        RI = self.parabola(gc_surround_linear_input, *popt)

        # [n_timepoints, n_gcs].  Inverts RI: neg_scaler zero is strong rectifier, whereas neg_scaler one is linear
        neg_scaler = 1 - RI.T
        ##########################################################################

        signal_input = subunit_sum.T

        ############## Ganglion cell activation computation ######################
        # Loop over time points. Expand NegScaler(tp) to [NB,NGC]
        # Multiply neg_scaler_expanded and bipolar_to_gcs_weights
        # Take dot product of signal input [timepoint] and neg_scaler_expanded[timepoint]

        def _compute_gc_input(bipo_to_gc_weights):
            n_bipo = bipo_to_gc_weights.shape[0]
            n_gc = bipo_to_gc_weights.shape[1]
            n_tp = signal_input.shape[0]

            gc_input_negative = np.zeros((n_tp, n_gc))

            for tp in range(n_tp):
                neg_scaler_expanded = np.tile(neg_scaler[tp, :], (n_bipo, 1))
                # Scale connection weights with nonlinearity scaling
                bipo_to_gc_weights_adjusted = bipo_to_gc_weights * neg_scaler_expanded

                # Nonlinear summation to negative signal input
                gc_input_negative[tp, :] = (
                    np.where(signal_input[tp, :] < 0, signal_input[tp, :], 0)
                    @ bipo_to_gc_weights_adjusted
                )

            # Linear summation for positive signal input
            gc_input_positive = (
                np.where(signal_input > 0, signal_input, 0) @ bipo_to_gc_weights
            )
            return gc_input_positive + gc_input_negative

        gc_center_input = _compute_gc_input(bipolar_to_gcs_cen_weights)
        gc_surround_input = _compute_gc_input(bipolar_to_gcs_sur_weights)

        gc_activation = gc_center_input - gc_surround_input

        if self.target_gc_for_multiple_trials is not None:
            gc_index = self.target_gc_for_multiple_trials
            gc_activation = gc_activation[gc_index, :]

        vs.generator_potentials = gc_activation.T

        return vs


class GanglionCellProduct(ReceptiveFieldsBase):
    """
    A class representing ganglion cell data, inheriting from ReceptiveFieldsBase.

    This class processes and stores information about ganglion cells, including
    their receptive fields, spatial and temporal properties, and relationships
    to visual stimuli.

    Parameters
    ----------
    retina_parameters : Dict[str, Any]
        Dictionary containing retina parameters.
    experimental_metadata : Dict[str, Any]
        Metadata for the experimental dataset.
    rfs_npz : Dict[str, Any]
        Dictionary containing receptive field data.
    gc_dataframe : pd.DataFrame
        DataFrame containing ganglion cell data.
    spike_generator_model : Any
        Model for generating spikes.
    pol2cart_df : callable
        Function to convert polar to Cartesian coordinates.

    Attributes
    ----------
    spike_generator_model : Any
        Model for generating spikes.
    mask_threshold : float
        Threshold for masking.
    refractory_parameters : Any
        Parameters for refractory period.
    experimental_metadata : Dict[str, Any]
        Metadata for the experimental dataset.
    data_microm_per_pixel : float
        Micrometers per pixel in the data.
    data_filter_fps : float
        Frames per second for the data filter.
    data_filter_timesteps : int
        Number of timesteps in the data filter.
    data_filter_duration : float
        Duration of the data filter in milliseconds.
    visual2cortical_params : Any
        Parameters for visual to cortical transformation.
    df : pd.DataFrame
        Processed DataFrame containing ganglion cell data.
    spat_rf : np.ndarray
        Spatial receptive field data.
    um_per_pix : float
        Micrometers per pixel.
    sidelen_pix : int
        Side length in pixels.
    n_units : int
        Number of units being processed.
    unit_indices : np.ndarray
        Array of unit indices being processed.
    """

    def __init__(
        self,
        retina_parameters: Dict[str, Any],
        experimental_metadata: Dict[str, Any],
        rfs_npz: Dict[str, Any],
        gc_dataframe: pd.DataFrame,
        spike_generator_model: Any,
        pol2cart_df: callable,
    ):
        super().__init__(retina_parameters)

        self.spike_generator_model = spike_generator_model
        self.mask_threshold = retina_parameters["center_mask_threshold"]
        self.fixed_mask_threshold = retina_parameters["fixed_mask_threshold"]
        self.refractory_parameters = retina_parameters["refractory_parameters"]

        assert isinstance(
            self.mask_threshold, float
        ), "mask_threshold must be float, aborting..."
        assert (
            self.mask_threshold >= 0 and self.mask_threshold <= 1
        ), "mask_threshold must be between 0 and 1, aborting..."

        self.experimental_metadata = experimental_metadata
        self.data_microm_per_pixel = self.experimental_metadata["data_microm_per_pix"]
        self.data_filter_fps = self.experimental_metadata["data_fps"]
        self.data_filter_timesteps = self.experimental_metadata[
            "data_temporalfilter_samples"
        ]
        self.data_filter_duration = self.data_filter_timesteps * (
            1000 / self.data_filter_fps
        )
        self.visual2cortical_params = retina_parameters["visual2cortical_params"]

        rspace_pos_mm = pol2cart_df(gc_dataframe)
        vspace_pos = rspace_pos_mm * self.deg_per_mm
        vspace_coords_deg = pd.DataFrame(
            {"x_deg": vspace_pos[:, 0], "y_deg": vspace_pos[:, 1]}
        )
        df = pd.concat([gc_dataframe, vspace_coords_deg], axis=1)

        if self.dog_model_type in ["ellipse_fixed"]:
            # Convert RF center radii to degrees as well
            df["semi_xc_deg"] = df.semi_xc_mm * self.deg_per_mm
            df["semi_yc_deg"] = df.semi_yc_mm * self.deg_per_mm
            # Drop rows (units) where semi_xc_deg and semi_yc_deg is zero.
            # These have bad (>3SD deviation in any ellipse parameter) fits
            df = df[(df.semi_xc_deg != 0) & (df.semi_yc_deg != 0)].reset_index(
                drop=True
            )
        if self.dog_model_type in ["ellipse_independent"]:
            # Convert RF center radii to degrees as well
            df["semi_xc_deg"] = df.semi_xc_mm * self.deg_per_mm
            df["semi_yc_deg"] = df.semi_yc_mm * self.deg_per_mm
            df["semi_xs_deg"] = df.semi_xs_mm * self.deg_per_mm
            df["semi_ys_deg"] = df.semi_ys_mm * self.deg_per_mm
            df = df[(df.semi_xc_deg != 0) & (df.semi_yc_deg != 0)].reset_index(
                drop=True
            )
        elif self.dog_model_type == "circular":
            df["rad_c_deg"] = df.rad_c_mm * self.deg_per_mm
            df["rad_s_deg"] = df.rad_s_mm * self.deg_per_mm
            df = df[(df.rad_c_deg != 0) & (df.rad_s_deg != 0)].reset_index(drop=True)

        # Drop retinal positions from the df (so that they are not used by accident)
        df = df.drop(["pos_ecc_mm", "pos_polar_deg"], axis=1)

        self.df = df

        self.spat_rf = rfs_npz["gc_img"]
        self.um_per_pix = rfs_npz["um_per_pix"]
        self.sidelen_pix = rfs_npz["pix_per_side"]

        # Run all units
        self.n_units = len(df.index)  # all units
        unit_indices = np.arange(self.n_units)
        self.unit_indices = np.atleast_1d(unit_indices)

        self._set_gc_gain_adjustment()

    def _set_gc_gain_adjustment(self) -> None:
        """
        Set gain adjustment for ganglion cells.

        """
        self.gc_gain_adjustment = self.retina_parameters["signal_gain"]

    def link_gcs_to_vs(self, vs: Any) -> None:
        """
        Links ganglion cells to visual space coordinates.

        This method creates a new DataFrame where everything is in pixels,
        endowing ganglion cells with stimulus/pixel space coordinates.

        Parameters
        ----------
        vs : Any
            Visual signal object containing methods for coordinate transformation.
        """

        df_stimpix = pd.DataFrame()
        df = self.df
        # Endow RGCs with pixel coordinates.
        pixspace_pos = np.array(
            [vs._vspace_to_pixspace(gc.x_deg, gc.y_deg) for index, gc in df.iterrows()]
        )
        # Assert that the pixel coordinates are within the stimulus space

        if self.dog_model_type in ["ellipse_fixed", "circular"]:
            pixspace_coords = pd.DataFrame(
                {"q_pix": pixspace_pos[:, 0], "r_pix": pixspace_pos[:, 1]}
            )
        elif self.dog_model_type == "ellipse_independent":
            # We need to here compute the pixel coordinates of the surround as well.
            # It would be an overkill to make pos_ecc_mm, pos_polar_deg forthe surround as well,
            # so we'll just compute the surround's pixel coordinates relative to the center's pixel coordinates.
            # 1) Get the experimental pixel coordinates of the center
            xoc = df.xoc_pix.values
            yoc = df.yoc_pix.values
            # 2) Get the experimental pixel coordinates of the surround
            xos = df.xos_pix.values
            yos = df.yos_pix.values
            # 3) Compute the experimental pixel coordinates of the surround relative to the center
            x_diff = xos - xoc
            y_diff = yos - yoc
            # 4) Tranform the experimental pixel coordinate difference to mm
            mm_per_exp_pix = self.data_microm_per_pixel / 1000
            x_diff_mm = x_diff * mm_per_exp_pix
            y_diff_mm = y_diff * mm_per_exp_pix
            # 5) Transform the mm difference to degrees difference
            x_diff_deg = x_diff_mm * self.deg_per_mm
            y_diff_deg = y_diff_mm * self.deg_per_mm
            # 6) Scale the degrees difference with eccentricity scaling factor and
            # add to the center's degrees coordinates
            x_deg_s = x_diff_deg * df.gc_scaling_factors + df.x_deg
            y_deg_s = y_diff_deg * df.gc_scaling_factors + df.y_deg
            # 7) Transform the degrees coordinates to pixel coordinates in stimulus space
            pixspace_pos_s = np.array(
                [vs._vspace_to_pixspace(x, y) for x, y in zip(x_deg_s, y_deg_s)]
            )

            pixspace_coords = pd.DataFrame(
                {
                    "q_pix": pixspace_pos[:, 0],
                    "r_pix": pixspace_pos[:, 1],
                    "q_pix_s": pixspace_pos_s[:, 0],
                    "r_pix_s": pixspace_pos_s[:, 1],
                }
            )

        # Scale RF to stimulus pixel space.
        if self.dog_model_type == "ellipse_fixed":
            df_stimpix["semi_xc"] = df.semi_xc_deg * vs.pix_per_deg
            df_stimpix["semi_yc"] = df.semi_yc_deg * vs.pix_per_deg
            df_stimpix["orient_cen_rad"] = df.orient_cen_rad
            df_stimpix["relat_sur_diam"] = df.relat_sur_diam
        elif self.dog_model_type == "ellipse_independent":
            df_stimpix["semi_xc"] = df.semi_xc_deg * vs.pix_per_deg
            df_stimpix["semi_yc"] = df.semi_yc_deg * vs.pix_per_deg
            df_stimpix["semi_xs"] = df.semi_xs_deg * vs.pix_per_deg
            df_stimpix["semi_ys"] = df.semi_ys_deg * vs.pix_per_deg
            df_stimpix["orient_cen_rad"] = df.orient_cen_rad
            df_stimpix["orient_sur_rad"] = df.orient_sur_rad
        elif self.dog_model_type == "circular":
            df_stimpix["rad_c"] = df.rad_c_deg * vs.pix_per_deg
            df_stimpix["rad_s"] = df.rad_s_deg * vs.pix_per_deg
            df_stimpix["orient_cen_rad"] = 0.0

        df_stimpix = pd.concat([df_stimpix, pixspace_coords], axis=1)

        # Fixed spatial filter sidelength according to RF pixel resolution
        # at given eccentricity (calculated at construction)
        stim_um_per_pix = 1000 / (vs.pix_per_deg * self.deg_per_mm)
        self.spatial_filter_sidelen = np.round(
            (self.um_per_pix / stim_um_per_pix) * self.sidelen_pix
        ).astype(int)

        # # Set center and surround midpoints in new pixel space
        pix_scale = self.spatial_filter_sidelen / self.sidelen_pix
        if self.dog_model_type in ["ellipse_fixed", "circular"]:
            xoc = xos = df.xoc_pix.values * pix_scale
            yoc = yos = df.yoc_pix.values * pix_scale
        elif self.dog_model_type == "ellipse_independent":
            xoc = df.xoc_pix.values * pix_scale
            yoc = df.yoc_pix.values * pix_scale
            xos = df.xos_pix.values * pix_scale
            yos = df.yos_pix.values * pix_scale
        df_stimpix["xoc_pix"] = xoc
        df_stimpix["yoc_pix"] = yoc
        df_stimpix["xos_pix"] = xos
        df_stimpix["yos_pix"] = yos

        df_stimpix["ampl_c"] = df.ampl_c_norm
        df_stimpix["ampl_s"] = df.ampl_s_norm

        self.df_stimpix = df_stimpix

        self.microm_per_pix = (1 / self.deg_per_mm) / vs.pix_per_deg * 1000
        self.temporal_filter_len = int(self.data_filter_duration / (1000 / vs.fps))


class VisualSignal(PrintableMixin):
    """
    Class containing simulation product, i.e. the information associated
    with visual signal passing through the retina.

    This includes the stimulus video, its transformations, the generator potential and spikes.

    Parameters
    ----------
    visual_stimulus_parameters : Dict[str, Any]
        Dictionary containing stimulus options.
    retina_center : complex
        Center of the stimulus in visual space.
    load_stimulus_from_videofile : callable
        Function to load stimulus from a video file.
    simulation_dt : float
        Simulation time step in seconds.
    deg_per_mm : float
        Degrees per millimeter conversion factor.
    stimulus_video : Optional[Any], optional
        Preloaded stimulus video, by default None.

    Attributes
    ----------
    video_file_name : str
        Name of the video file.
    options_from_videofile : Dict[str, Any]
        Options loaded from the video file.
    stimulus_width_pix : int
        Width of the stimulus in pixels.
    stimulus_height_pix : int
        Height of the stimulus in pixels.
    pix_per_deg : float
        Pixels per degree.
    fps : float
        Frames per second.
    stimulus_width_deg : float
        Width of the stimulus in degrees.
    stimulus_height_deg : float
        Height of the stimulus in degrees.
    photodiode_response : ndarray
        Photodiode response from the stimulus video.
    video_dt : brian2.units.second
        Time step of the video.
    stim_len_tp : int
        Length of the stimulus in time points.
    duration : brian2.units.second
        Duration of the stimulus.
    simulation_dt : brian2.units.second
        Simulation time step.
    tvec : range
        Time vector.
    baseline_len_tp : int
        Length of the baseline in time points.
    """

    def __init__(
        self,
        visual_stimulus_parameters: Dict[str, Any],
        retina_center: complex,
        load_stimulus_from_videofile: callable,
        simulation_dt: float,
        deg_per_mm: float,
        optical_aberration: float,
        pix_per_deg: float,
        stimulus_video: Optional[Any] = None,
    ):
        # Parameters directly passed to the constructor
        self.visual_stimulus_parameters = visual_stimulus_parameters
        self.retina_center = retina_center
        self.load_stimulus_from_videofile = load_stimulus_from_videofile
        self.deg_per_mm = deg_per_mm
        self.optical_aberration = optical_aberration
        self.pix_per_deg = pix_per_deg

        # Default value for computed variable
        self.stimulus_video = stimulus_video

        # Load stimulus video if not already loaded
        if self.stimulus_video is None:
            self.video_file_name = self.visual_stimulus_parameters[
                "stimulus_video_name"
            ]
            self.stimulus_video = self.load_stimulus_from_videofile(
                self.video_file_name
            )

        self.options_from_videofile = self.stimulus_video.options
        self.stimulus_width_pix = self.options_from_videofile["image_width"]
        self.stimulus_height_pix = self.options_from_videofile["image_height"]
        self.pix_per_deg = self.options_from_videofile["pix_per_deg"]
        self.fps = self.options_from_videofile["fps"]

        self.stimulus_width_deg = self.stimulus_width_pix / self.pix_per_deg
        self.stimulus_height_deg = self.stimulus_height_pix / self.pix_per_deg

        cen_x = self.options_from_videofile["center_pix"][0]
        cen_y = self.options_from_videofile["center_pix"][1]
        self.photodiode_response = self.stimulus_video.frames[:, cen_y, cen_x]

        # Assertions to ensure stimulus video properties match expected parameters
        assert (
            self.stimulus_video.video_width == self.stimulus_width_pix
            and self.stimulus_video.video_height == self.stimulus_height_pix
        ), "Check that stimulus dimensions match those of the mosaic"
        assert (
            self.stimulus_video.fps == self.fps
        ), "Check that stimulus frame rate matches that of the mosaic"
        assert (
            self.stimulus_video.pix_per_deg == self.pix_per_deg
        ), "Check that stimulus resolution matches that of the mosaic"

        self.video_dt = (1 / self.stimulus_video.fps) * b2u.second  # input
        self.stim_len_tp = self.stimulus_video.video_n_frames
        self.duration = self.stim_len_tp * self.video_dt
        self.simulation_dt = simulation_dt * b2u.second  # output
        self.tvec = range(self.stim_len_tp) * self.video_dt
        self.baseline_len_tp = self.stimulus_video.baseline_len_tp
        self.mean_luminance = self.options_from_videofile["mean"]

    def _vspace_to_pixspace(self, x: float, y: float) -> tuple[float, float]:
        """
        Converts visual space coordinates to pixel space coordinates.

        In pixel space, coordinates (q,r) correspond to matrix locations,
        i.e. (0,0) is top-left.

        Parameters
        ----------
        x : float
            Eccentricity (deg).
        y : float
            Elevation (deg).

        Returns
        -------
        q : float
            Pixel space x-coordinate.
        r : float
            Pixel space y-coordinate.
        """

        video_width_px = self.stimulus_width_pix  # self.stimulus_video.video_width
        video_height_px = self.stimulus_height_pix  # self.stimulus_video.video_height
        pix_per_deg = self.pix_per_deg  # self.stimulus_video.pix_per_deg

        # 1) Set the video center in visual coordinates as origin
        # 2) Scale to pixel space. Mirror+scale in y axis due to y-coordinate running top-to-bottom in pixel space
        # 3) Move the origin to video center in pixel coordinates
        q = pix_per_deg * (x - self.retina_center.real) + (video_width_px / 2)
        r = -pix_per_deg * (y - self.retina_center.imag) + (video_height_px / 2)

        return q, r


class SimulateRetina(RetinaMath):
    """
    Simulates retinal activity in response to visual stimuli.

    This class encapsulates the logic for simulating various aspects of retinal function,
    including cone responses, ganglion cell activity, and visual signal processing.

    Parameters
    ----------
    config : Configuration
        Configuration parameters object.
    data_io : Any
        Data input/output handler.
    viz : Any
        Visualization utilities.
    project_data : Any
        Container for project-specific data.
    retina_math : RetinaMath
        Mathematical utilities for retinal calculations.
    device : str or torch.device
        Computation device (CPU or GPU).

    Attributes
    ----------
    config : Configuration
        Configuration parameters object.
    data_io : Any
        Data input/output handler.
    viz : Any
        Visualization utilities.
    project_data : Any
        Container for project-specific data.
    retina_math : RetinaMath
        Mathematical utilities for retinal calculations.
    device : str or torch.device
        Computation device (CPU or GPU).
    """

    def __init__(
        self,
        config: Any,
        data_io: Any,
        project_data: Any,
        retina_math: RetinaMath,
        device: str,
        stimulate: Any,
    ) -> None:
        self._config: Any = config
        self._data_io: Any = data_io
        self._project_data: Any = project_data
        self._retina_math: RetinaMath = retina_math
        self._device: str = device
        self._stimulate: Any = stimulate

    @property
    def config(self) -> Any:
        return self._config

    @property
    def data_io(self) -> Any:
        return self._data_io

    @property
    def project_data(self) -> Any:
        return self._project_data

    @property
    def retina_math(self) -> RetinaMath:
        return self._retina_math

    @property
    def device(self) -> str:
        return self._device

    @property
    def stimulate(self) -> Any:
        return self._stimulate

    def get_w_z_coords(self, gcs: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Create w_coord, z_coord for cortical and visual coordinates, respectively.

        Parameters
        ----------
        gcs : Any
            Ganglion cell data.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Cortical coordinates (w_coord) and visual coordinates (z_coord).
        """

        # Create w_coord, z_coord for cortical and visual coordinates, respectively
        z_coord = gcs.df["x_deg"].values + 1j * gcs.df["y_deg"].values

        visual2cortical_params = self.config.retina_parameters["visual2cortical_params"]
        a = visual2cortical_params["a"]
        k = visual2cortical_params["k"]
        w_coord = k * np.log(z_coord + a)

        return w_coord, z_coord

    def _get_project_data_for_viz(self, vs: Any, gcs: Any, n_sweeps: int) -> None:
        """
        Bind data to project_data container for visualization.

        Parameters
        ----------
        vs : Any
            Visual signal data.
        gcs : Any
            Ganglion cell data.
        n_sweeps : int
            Number of simulation trials.
        """

        stim_to_show = {
            "stimulus_video": vs.stimulus_video,
            "df_stimpix": gcs.df_stimpix,
            "stimulus_height_pix": vs.stimulus_height_pix,
            "pix_per_deg": vs.pix_per_deg,
            "deg_per_mm": gcs.deg_per_mm,
            "retina_center": vs.retina_center,
            "qr_min_max": vs.qr_min_max,
            "spatial_filter_sidelen": gcs.spatial_filter_sidelen,
            "stimulus_cropped": vs.stimulus_cropped_adapted,
        }

        cone_responses_to_show = {
            "cone_noise": vs.cone_noise,
        }

        photodiode_to_show = {
            "photodiode_response": vs.photodiode_response,
        }

        gc_responses_to_show = {
            "n_sweeps": n_sweeps,
            "n_units": gcs.n_units,
            "all_spiketrains": vs.all_spiketrains,
            "duration": vs.duration,
            "firing_rates": vs.firing_rates,
            "generator_potentials": vs.generator_potentials,
            "video_dt": vs.video_dt,
            "tvec_new": vs.tvec_new,
            "gc_synaptic_noise": vs.gc_synaptic_noise,
        }

        # Attach data requested by other classes to project_data
        self.project_data.simulate_retina["stim_to_show"] = stim_to_show
        self.project_data.simulate_retina["gc_responses_to_show"] = gc_responses_to_show
        self.project_data.simulate_retina["photodiode_to_show"] = photodiode_to_show

        if gcs.temporal_model_type == "fixed":
            spat_temp_filter_to_show = {
                "spatial_filters": gcs.spatial_filters_flat,
                "temporal_filters": gcs.temporal_filters,
                "data_filter_duration": gcs.data_filter_duration,
                "temporal_filter_len": gcs.temporal_filter_len,
                "gc_type": gcs.gc_type,
                "response_type": gcs.response_type,
                "temporal_model_type": gcs.temporal_model_type,
                "spatial_filter_sidelen": gcs.spatial_filter_sidelen,
            }
            self.project_data.simulate_retina["spat_temp_filter_to_show"] = (
                spat_temp_filter_to_show
            )

        if gcs.temporal_model_type == "subunit":
            cone_responses_to_show["cone_signal"] = vs.cone_signal
            cone_responses_to_show["cone_signal_u"] = vs.cone_signal_u
            cone_responses_to_show["unit"] = self.config.retina_parameters[
                "cone_signal_parameters"
            ]["unit"]
            cone_responses_to_show["photodiode_Rstar_range"] = (
                vs.photodiode_Rstar_range,
            )

        self.project_data.simulate_retina["cone_responses_to_show"] = (
            cone_responses_to_show
        )

    def _initialize_cones(self) -> ConeProduct:
        """
        Initialize the cone photoreceptors for the simulation.

        Returns
        -------
        ConeProduct
            Initialized cone photoreceptor object.
        """
        ret_npz_file = self.config.retina_parameters["ret_file"]
        ret_npz = self.data_io.load_data(filename=ret_npz_file)
        target_gc_for_multiple_trials = None  # Option to use only one gc unit

        cones = ConeProduct(
            self.config.retina_parameters,
            ret_npz,
            self.config.device,
            self.config.visual_stimulus_parameters["ND_filter"],
            # RetinaMath methods:
            self.interpolate_data,
            self.lin_interp_and_double_lorenzian,
            target_gc_for_multiple_trials,
        )

        return cones

    def _get_cone_noise_from_file_if_exists(self, vs: Any, gcs: Any) -> Any:
        """
        Load cone noise from file if it exists.

        Parameters
        ----------
        vs : Any
            Visual signal object.

        Returns
        -------
        Any
            Updated visual signal object.
        """

        cone_noise_hash = self.config.retina_parameters["cone_noise_hash"]
        filename_stem_cone_noise = f"cone_noise_{cone_noise_hash}"
        cone_noise_filename_full = self.data_io.parse_path(
            "", substring=filename_stem_cone_noise
        )

        if cone_noise_filename_full is not None:
            cone_noise_npz = self.data_io.load_data(full_path=cone_noise_filename_full)
            vs.cone_noise = cone_noise_npz["cone_noise"]

        gc_type = self.config.retina_parameters["gc_type"]
        response_type = self.config.retina_parameters["response_type"]

        filename_stem_gc_noise = f"{gc_type}_{response_type}_noise_{cone_noise_hash}"
        gc_noise_filename_full = self.data_io.parse_path(
            "", substring=filename_stem_gc_noise
        )

        if gc_noise_filename_full is not None:
            gc_noise_npz = self.data_io.load_data(full_path=gc_noise_filename_full)

            vs.gc_synaptic_noise = gc_noise_npz["gc_synaptic_noise"]

        return vs

    def _get_products(self, stimulus: np.ndarray | None) -> tuple[Any, Any, Any, Any]:
        """
        Initialize and return the main components needed for the simulation.

        Parameters
        ----------
        stimulus : np.ndarray or None
            Input stimulus array or None.

        Returns
        -------
        tuple[Any, Any, Any, Any]
            Visual signal, ganglion cells, cones, and bipolar cells.
        """
        # This is needed also independently of the pipeline
        cones = self._initialize_cones()

        # Abstraction for clarity
        rfs_npz_file = self.config.retina_parameters["spatial_rfs_file"]
        rfs_npz = self.data_io.load_data(filename=rfs_npz_file)
        mosaic_file = self.config.retina_parameters["mosaic_file"]
        gc_dataframe = self.data_io.load_data(filename=mosaic_file)
        spike_generator_model = self.config.simulation_parameters[
            "spike_generator_model"
        ]
        simulation_dt = self.config.simulation_parameters["simulation_dt"]

        gcs = GanglionCellProduct(
            self.config.retina_parameters,
            self.config.experimental_metadata,
            rfs_npz,
            gc_dataframe,
            spike_generator_model,
            self.pol2cart_df,
        )

        ret_npz_file = self.config.retina_parameters["ret_file"]
        ret_npz = self.data_io.load_data(filename=ret_npz_file)

        if gcs.temporal_model_type == "subunit":
            bipolars = BipolarProduct(
                self.config.retina_parameters,
                ret_npz,
                target_gc_for_multiple_trials=None,  # Option to target one gc unit
            )
        else:
            bipolars = None

        vs = VisualSignal(
            self.config.visual_stimulus_parameters,
            self.config.retina_parameters["retina_center"],
            self.data_io.load_stimulus_from_videofile,
            simulation_dt,
            self.config.retina_parameters["deg_per_mm"],
            self.config.retina_parameters["optical_aberration"],
            self.config.visual_stimulus_parameters["pix_per_deg"],
            stimulus_video=stimulus,
        )

        vs = self._get_cone_noise_from_file_if_exists(vs, gcs)

        # Link ganglion cell receptive fields to visual signal. Eg applies rotation
        gcs.link_gcs_to_vs(vs)

        return vs, gcs, cones, bipolars

    def _get_construct_metadata_if_missing(self) -> None:
        """
        When running without constructing first, get retina parameters from output folder.
        This populates a subset of config.retina_parameters attributes with values from files
        in the output folder."""

        if not hasattr(self.config.retina_parameters, "retina_parameters_hash"):
            print(
                """
            No retina_parameters_hash found, assuming running without construct.
            Getting parameters from output folder...
            """
            )

            files_in_output_folder = self.data_io.listdir_loop(
                self.config.output_folder
            )
            for file in files_in_output_folder:
                if "mosaic.csv" in str(file):
                    self.config.retina_parameters.mosaic_file = file
                if "spatial_rfs.npz" in str(file):
                    self.config.retina_parameters.spatial_rfs_file = file
                if "ret.npz" in str(file):
                    self.config.retina_parameters.ret_file = file
                if "metadata.yaml" in str(file):
                    self.config.retina_parameters.retina_metadata_file = file
                if "cone_noise_" in str(file):
                    # Extract the hash from the filename
                    hash_part = str(file).split("cone_noise_")[-1].split(".npz")[0]
                    self.config.retina_parameters.cone_noise_hash = hash_part

            hash_part = (
                str(self.config.retina_parameters.retina_metadata_file)
                .split("parasol_on_")[-1]
                .split("_metadata.yaml")[0]
            )
            self.config.retina_parameters.retina_parameters_hash = hash_part

    def client(
        self,
        stimulus: np.ndarray | None = None,
        filename: str | None = None,
        impulse: bool = False,
        unity: bool = False,
    ) -> None:
        """
        Build and run simulation using the builder pattern.

        Parameters
        ----------
        stimulus : np.ndarray or None, optional
            Input stimulus array. If None, loads from file.
        impulse : bool, optional
            If True, runs impulse response simulation.
        unity : bool, optional
            If True, runs uniformity index simulation.
        """
        print(f"KUKKUU {self.config.device=}; {filename=}")

        self._get_construct_metadata_if_missing()
        vs, gcs, cones, bipolars = self._get_products(stimulus)
        n_sweeps = self.config.simulation_parameters["n_sweeps"]
        print(f"KUKKUU2 {self.config.device=}; {filename=}")
        builder = ConcreteSimulationBuilder(
            vs,
            gcs,
            cones,
            bipolars,
            self.retina_math,
            self.device,
            n_sweeps,
            self.stimulate,
        )

        director = SimulationDirector(builder)
        if impulse:
            contrasts = self.config.simulation_parameters["contrasts_for_impulse"]
            director.run_impulse_response(contrasts)
        elif unity:
            director.run_uniformity_index()
        else:
            if filename is not None:
                filenames = [filename]
            else:
                gc_type = self.config.retina_parameters["gc_type"]
                response_type = self.config.retina_parameters["response_type"]
                hashstr = self.config.retina_parameters["retina_parameters_hash"]
                # Generate multiple filenames if n_files > 1
                filenames = [
                    f"{gc_type}_{response_type}_{hashstr}_response_{x:02}"
                    for x in range(self.config.simulation_parameters.n_files)
                ]

            for filename in filenames:
                director.run_simulation()
                vs, gcs = director.get_simulation_result()
                if self.config.simulation_parameters["save_data"]:
                    save_variables = self.config.simulation_parameters["save_variables"]
                    self.data_io.save_retina_output(vs, gcs, filename, save_variables)

            self._get_project_data_for_viz(vs, gcs, n_sweeps)
            if len(filenames) > 1:
                WarningMsg = "Multiple files were processed. Direct visualization shows the last file."
                print(WarningMsg)

        self.project_data.simulate_retina.update(builder.project_data)
