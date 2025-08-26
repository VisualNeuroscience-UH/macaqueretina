# Third-party
import matplotlib.pyplot as plt
import numpy as np
from brian2 import units as b2u
from scipy.interpolate import interp1d
from scipy.special import gamma


class RetinaMath:
    """
    Constructor fit functions to read in data and provide continuous functions
    """

    # Need object instance of this class at ProjectManager
    def __init__(self) -> None:
        pass

    # RetinaConstruction methods
    def single_exponential_func(self, x: float, a: float, b: float, c: float) -> float:
        return a * np.exp(b * x) + c

    def double_exponential_func(
        self, x: float, a: float, b: float, c: float, d: float
    ) -> float:
        return a * np.exp(b * x) + c * np.exp(d * x)

    def double_exponential_func_log(
        self, log_x: float, a: float, b: float, c: float, d: float
    ) -> float:
        """
        Compute double exponential function value in logarithmic space.
        The function computes: log(f(x)) = log(a * exp(b * exp(log_x)) + c * exp(d * exp(log_x)))
        Parameters
        ----------
        log_x : float
            Independent variable in log space (log_x = log(x)).
        a : float
            Amplitude of first exponential term.
        b : float
            Decay rate of first exponential term.
        c : float
            Amplitude of second exponential term.
        d : float
            Decay rate of second exponential term.
        Returns
        -------
        float
            Natural logarithm of the double exponential function value.
        """
        # Convert back to linear space for the exponential calculation
        x = np.exp(log_x)
        result = self.double_exponential_func(x, a, b, c, d)
        # Replace negative values with -inf
        log_result = np.full_like(result, float("-inf"))
        positive_mask = result > 0
        log_result[positive_mask] = np.log(result[positive_mask])
        return log_result

    def triple_exponential_func(
        self, x: float, a: float, b: float, c: float, d: float, e: float, f: float
    ) -> float:
        return np.maximum(a * np.exp(b * x) + c * np.exp(d * x) + e * np.exp(f * x), 0)

    def triple_exponential_func_log(
        self, log_x: float, a: float, b: float, c: float, d: float, e: float, f: float
    ) -> float:
        """
        Compute triple exponential function value in logarithmic space with non-negative constraint.
        The function computes:
        log(f(x)) = log(max(0, a * exp(b * exp(log_x)) + c * exp(d * exp(log_x)) + e * exp(f * exp(log_x))))

        Parameters
        ----------
        log_x : float
            Independent variable in log space (log_x = log(x)).
        a : float
            Amplitude of first exponential term.
        b : float
            Decay rate of first exponential term.
        c : float
            Amplitude of second exponential term.
        d : float
            Decay rate of second exponential term.
        e : float
            Amplitude of third exponential term.
        f : float
            Decay rate of third exponential term.

        Returns
        -------
        float
            Natural logarithm of the maximum between zero and the sum of three exponential terms.
        """
        # Convert back to linear space for the exponential calculation
        x = np.exp(log_x)
        result = self.triple_exponential_func(x, a, b, c, d, e, f)

        # Replace negative values with -inf
        log_result = np.full_like(result, float("-inf"))
        positive_mask = result > 0
        log_result[positive_mask] = np.log(result[positive_mask])

        return log_result

    def generalized_gauss_func(self, x, a, x0, alpha, beta):
        """
        Generalized Gaussian distribution function with variable kurtosis.
        """
        coeff = beta / (2 * alpha * gamma(1 / beta))
        return a * coeff * np.exp(-np.abs((x - x0) / alpha) ** beta)

    def sector2area_mm2(self, radius, angle):
        """
        Calculate sector area.

        Parameters
        ----------
        radius : float
            The radius of the sector in mm.
        angle : float
            The angle of the sector in degrees.

        Returns
        -------
        sector_surface_area : float
            The area of the sector in mm2.
        """
        assert angle < 360, "Angle not possible, should be <360"

        # Calculating area of the sector
        sector_surface_area = (np.pi * (radius**2)) * (angle / 360)  # in mm2
        return sector_surface_area

    def area2circle_diameter(self, area_of_rf):
        diameter = np.sqrt(area_of_rf / np.pi) * 2

        return diameter

    def ellipse2diam(self, semi_xc, semi_yc):
        """
        Compute the spherical diameter of an ellipse given its semi-major and semi-minor axes.

        Parameters
        ----------
        semi_xc : array-like
            The lengths of the semi-major axes of the ellipses.
        semi_yc : array-like
            The lengths of the semi-minor axes of the ellipses.

        Returns
        -------
        diameters : numpy array
            The spherical diameters of the ellipses.

        Notes
        -----
        The spherical diameter is calculated as the diameter of a circle with the same area as the ellipse.
        """
        # Calculate the area of each ellipse
        areas = np.pi * semi_xc * semi_yc

        # Calculate the diameter of a circle with the same area
        diameters = 2 * np.sqrt(areas / np.pi)

        return diameters

    def get_sample_from_range_and_average(
        self, min_range, max_range, average, sample_size
    ):
        """Function to generate a sample with a target average, given range."""
        sample = np.random.randint(min_range, max_range + 1, sample_size)
        current_mean = np.mean(sample)
        adjustment_needed = average - current_mean
        total_adjustment = int(round(adjustment_needed * sample_size))

        for i in range(abs(total_adjustment)):
            if total_adjustment > 0:
                idx = np.argmin(sample)
                if sample[idx] < max_range:
                    sample[idx] += 1
            elif total_adjustment < 0:
                idx = np.argmax(sample)
                if sample[idx] > min_range:
                    sample[idx] -= 1

        # Suffle sample to adjust for the order of the adjustments
        np.random.shuffle(sample)

        return sample

    def weighted_average(self, means, sizes):
        """
        Calculate the weighted average of a set of values.

        Parameters
        ----------
        means : array-like
            The values to be averaged.
        sizes : array-like
            The weights for each value.

        Returns
        -------
        weighted_mean : float
            The weighted average of the input values.
        """

        total_weight = sum(sizes)
        weighted_mean = (
            sum(mean * size for mean, size in zip(means, sizes)) / total_weight
        )
        return weighted_mean

    # RetinaConstruction & SimulateRetina methods
    def pol2cart_df(self, df):
        """
        Convert retinal positions (eccentricity, polar angle) to visual space positions in degrees (x, y).

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing retinal positions with columns 'pos_ecc_mm' and 'pos_polar_deg'.

        Returns
        -------
        numpy.ndarray
            Numpy array of visual space positions in degrees, with shape (n, 2), where n is the number of rows in the DataFrame.
            Each row represents the Cartesian coordinates (x, y) in visual space.
        """
        rspace_pos_mm = np.array(
            [
                self.pol2cart(gc.pos_ecc_mm, gc.pos_polar_deg, deg=True)
                for index, gc in df.iterrows()
            ]
        )

        return rspace_pos_mm

    def create_ellipse_mask(self, xo, yo, semi_x, semi_y, ori, s):
        """
        Create an ellipse mask.

        This function creates an ellipse mask with the specified parameters.

        Parameters
        ----------
        xo : float
            The x-coordinate of the ellipse center.
        yo : float
            The y-coordinate of the ellipse center.
        semi_x : float
            The semi-major axis of the ellipse.
        semi_y : float
            The semi-minor axis of the ellipse.
        ori : float
            The orientation of the ellipse in radians.
        s : int
            The side length of the mask.

        Returns
        -------
        mask : ndarray
            The ellipse mask.
        """

        # Create a grid of x and y coordinates
        x = np.arange(0, s)
        y = np.arange(0, s)
        # x, y = np.meshgrid(x, y)
        Y, X = np.meshgrid(
            np.arange(0, s),
            np.arange(0, s),
            indexing="ij",
        )  # y, x
        # Rotate the coordinates
        x_rot = (X - xo) * np.cos(ori) + (Y - yo) * np.sin(ori)
        y_rot = -(X - xo) * np.sin(ori) + (Y - yo) * np.cos(ori)

        # Create the mask
        mask = ((x_rot / semi_x) ** 2 + (y_rot / semi_y) ** 2 <= 1).astype(np.uint8)

        return mask

    # SimulateRetina methods
    def pol2cart(self, radius, phi, deg=True):
        """
        Converts polar coordinates to Cartesian coordinates

        Parameters
        ----------
        radius : float
            The radius value in real distance such as mm.
        phi : float
            The polar angle value.
        deg : bool, optional
            Whether the polar angle is given in degrees or radians.
            If True, the angle is given in degrees; if False, the angle is given in radians.
            Default is True.

        Returns
        -------
        tuple
            A tuple containing the Cartesian coordinates (x, y) in same units as the radius.
        """
        # Check that radius and phi are floats or numpy arrays
        assert type(radius) in [float, np.float64, np.ndarray], "Radius must be a float"
        assert type(phi) in [float, np.float64, np.ndarray], "Phi must be a float"

        if deg is True:
            theta = phi * np.pi / 180
        else:
            theta = phi

        x = radius * np.cos(theta)  # radians fed here
        y = radius * np.sin(theta)

        return (x, y)

    def cart2pol(self, x, y, deg=True):
        """
        Converts Cartesian coordinates to polar coordinates.

        Parameters
        ----------
        x : float
            The x-coordinate in real distance such as mm.
        y : float
            The y-coordinate in real distance such as mm.
        deg : bool, optional
            Whether to return the polar angle in degrees or radians.
            If True, the angle is returned in degrees; if False, the angle is returned in radians.
            Default is True.

        Returns
        -------
        tuple
            A tuple containing the polar coordinates (radius, phi).
        """
        # Check that x and y are floats or numpy arrays
        assert type(x) in [float, np.float64, np.ndarray], "x must be a float"
        assert type(y) in [float, np.float64, np.ndarray], "y must be a float"

        radius = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        if deg:
            phi = theta * 180 / np.pi
        else:
            phi = theta

        return (radius, phi)

    def deg2_to_steradian(self, deg2):
        """
        Convert square degrees to steradians.

        Parameters
        ----------
        deg2 : float
            Area in square degrees.

        Returns
        -------
        float
            Area in steradians.
        """
        return deg2 * (np.pi / 180) ** 2

    def get_luminance_from_photoisomerizations(
        self,
        I_cone,
        A_pupil=9.3,
        a_c_end_on=0.6,
        tau_media=0.87,
        lambda_nm=560,
        V_lambda=0.995,
    ):
        """
        Calculate the luminance from the rate of photoisomerizations per cone per second.

        Parameters
        ----------
        I_cone : float
            The rate of photoisomerizations per cone per second (R* cone^-1 s^-1).
        A_pupil : float
            The area of the pupil in mm².
        a_c_end_on : float
            The end-on collecting area for the cones in um², 0.6 according to Schneeweis_1999_JNeurosci.
        tau_media : float
            The transmittance of the ocular media at wavelength λ.
        lambda_nm : int, optional
            Wavelength in nm, default is 560 nm.
        V_lambda : float
            The luminosity function value at given wavelength, default is 0.995 at 560 nm.

        Returns
        -------
        float
            Luminance in cd/m².
        """

        # Calculate the rate of photoisomerizations per um^2 per second at 1 td
        I_per_td = 2.649e-2 * lambda_nm * tau_media / V_lambda

        # Add units to the calculation. The lambda_nm above drops the nm**-1 unit
        I_per_td = I_per_td * b2u.umeter**-2 * b2u.second**-1
        a_c_end_on = a_c_end_on * b2u.umeter**2

        # Calculate the retinal illuminance (L_td) in Trolands
        L_td = I_cone / (I_per_td * a_c_end_on)

        # Calculate the luminance (L) in cd/m²
        L = L_td / A_pupil

        return L

    def get_photoisomerizations_from_luminance(
        self,
        L,
        A_pupil=9.3,
        a_c_end_on=0.6,
        tau_media=0.87,
        lambda_nm=560,
        V_lambda=0.995,
    ):
        """
        Calculate the rate of photoisomerizations per cone per second from luminance.

        Parameters
        ----------
        L : float
            Luminance in cd/m².
        A_pupil : float
            The area of the pupil in mm².
        a_c_end_on : float
            The end-on collecting area for the cones in um², 0.6 according to Schneeweis_1999_JNeurosci.
        tau_media : float
            The transmittance of the ocular media at wavelength λ.
        lambda_nm : int, optional
            Wavelength in nm, default is 560 nm.
        V : float
            The luminocity function value at given wavelength, default is 0.995 at 560 nm.

        Returns
        -------
        float
            The rate of photoisomerizations per cone per second (R* cone^-1 s^-1).

        Notes
        -----
        The retinal illuminance (L_td) in Trolands is calculated by multiplying the luminance (L) with the pupil area (A_pupil).
        Factors that affect the absorption of light by the retina include:
        - Optical point spread by diffraction
        - Scatter, which extends the point spread function
        - Transmission by ocular media, influenced by varying macular pigment absorption (60%) and wavelength-dependent filtering (50% at 450 nm and 80% at 650 nm).

        These factors combined can result in a Strehl ratio as low as 0.02 in aged eyes with significant scatter, though in optimal cases it can be about 0.2 (Westheimer_2006_ProgRetEyeRes).

        Example calculation for the rate of photoisomerizations (I_cone):
        - According to Schnapf_1990_JPhysiol, 1 troland contributes approximately 2.649 x 10^-2 photons µm^-2 s^-1 nm^-1, adjusted for wavelength and transmittance.
        - For λ = 560 nm with transmittance τ(λ) = 0.87 and V(λ) = 0.995: 2.649e-2 * 560 * (0.87/0.995) results in about 12.9 R/um^2/s.

        Additionally, Shapley_1984_ProgRetRes_chapter9 estimates about 1.2e6 quanta/deg²/s for a similar calculation.
        """
        # Calculate the retinal illuminance (L_td) in Trolands
        L_td = L * A_pupil

        # Calculate the rate of photoisomerizations per um^2 per second at 1 td
        I_per_td = 2.649e-2 * lambda_nm * tau_media / V_lambda

        # Add units to the calculation. The lambda_nm above drops the nm**-1 unit
        I_per_td = I_per_td * b2u.umeter**-2 * b2u.second**-1
        a_c_end_on = a_c_end_on * b2u.umeter**2

        # Calcualte the rate of photoisomerizations per cone per second
        I_cone = L_td * I_per_td * a_c_end_on

        # Drop units for the return value
        I_cone = I_cone / b2u.hertz

        return I_cone

    def rotate_image_grids(self, X_grid, Y_grid, angle, n_units, sidelen_x, sidelen_y):
        """
        Rotate the X and Y meshgrids by the specified angle.

        Parameters
        ----------
        X_grid : numpy.ndarray
            The X grid.
        Y_grid : numpy.ndarray
            The Y grid.
        angle : float
            The angle to rotate the grids by in degrees.
        n_units : int
            The number of units (cells).
        sidelen_x : int
            The side length of the x-axis.
        sidelen_y : int
            The side length of the y-axis.

        Returns
        -------
        X_rotated : numpy.ndarray
            The rotated X grid.
        Y_rotated : numpy.ndarray
            The rotated Y grid.

        Notes
        -----
        Note that both x and y axes apply to both X and Y grids.
        """

        # Convert degrees to radians
        rot_rad = np.deg2rad(angle)

        # Create the 2D rotation matrix
        cos_theta = np.cos(rot_rad)
        sin_theta = np.sin(rot_rad)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # Flatten the grids for matrix multiplication
        X_grid_flat = X_grid.reshape(n_units, -1)
        Y_grid_flat = Y_grid.reshape(n_units, -1)

        # Stack the X and Y grids for rotation
        coords = np.stack(
            [X_grid_flat, Y_grid_flat], axis=-1
        )  # Shape becomes (n_units, num_pixels, 2)

        # Apply the rotation matrix to each coordinate
        rotated_coords = np.dot(
            coords, rotation_matrix.T
        )  # Shape (n_units, num_pixels, 2)

        # Split the rotated coordinates into X and Y
        X_grid_rot_mm = rotated_coords[..., 0].reshape(n_units, sidelen_y, sidelen_x)
        Y_grid_rot_mm = rotated_coords[..., 1].reshape(n_units, sidelen_y, sidelen_x)

        return X_grid_rot_mm, Y_grid_rot_mm

    # General function fitting methods
    def hyperbolic_function(self, x, y_max, x_half):
        # Define the generalized hyperbolic function
        return y_max / (1 + x / x_half)

    def log_hyperbolic_function(self, x_log, log_y_max, x_half_log):
        # Define the hyperbolic function in log space
        return log_y_max - np.log(1 + np.exp(x_log - x_half_log))

    def victor_model_frequency_domain(self, f, NL, TL, HS, TS, A0, M0, D):
        """
        The model by Victor 1987 JPhysiol
        """
        # Linearized low-pass filter in frequency domain
        x_hat = (1 + 1j * f * TL) ** (-NL)

        # Adaptive high-pass filter (linearized representation)
        y_hat = (1 - HS / (1 + 1j * f * TS)) * x_hat

        # Impulse generation stage
        r_hat = A0 * np.exp(-1j * f * D) * y_hat + M0
        # Power spectrum is the square of the magnitude of the impulse generation stage output
        power_spectrum = np.abs(r_hat) ** 2

        return power_spectrum

    def lorenzian_function(self, f, a, wc):
        """
        Define a Lorentzian function for curve fitting.

        Parameters
        ----------
        f : numpy.ndarray
            Frequency axis.
        a : float
            Amplitude of the Lorentzian function.
        wc : float
            Cutoff frequency of the Lorentzian function.

        Returns
        -------
        numpy.ndarray
            The Lorentzian function evaluated at each frequency in `f`.
        """
        return a / (1 + ((f / wc) ** 2))

    def interpolate_data(self, x, y, kind="linear"):
        """Interpolate empirical data to get a continuous function."""

        fill_value = (y[0], y[-1])

        # assert that x values are sorted
        assert np.all(np.diff(x) > 0), "x values must be sorted"

        interp1d_function = interp1d(
            x,
            y,
            kind=kind,
            fill_value=fill_value,
            bounds_error=False,
        )

        return interp1d_function

    def set_metaparameters_for_log_interp_and_double_lorenzian(
        self, cone_interp_function, cone_noise_wc
    ):
        """
        Set metaparameters for fitting interpolated cone response and
        two lorenzian functions. These are necessary building blocks for
        the fitting function.

        Parameters
        ----------
        cone_interp_function : scipy.interpolate.interp1d
            Interpolated cone response function.
        cone_noise_wc : list
            List of corner frequencies for the two lorenzian functions.

        Returns
        -------
        None
        """
        self.cone_interp_function = cone_interp_function
        self.cone_noise_wc = cone_noise_wc

    def lin_interp_and_double_lorenzian(
        self, f, a0, L1_params, L2_params, cone_interp_function
    ):
        """
        Calculate the power spectrum in linear space as a combination of
        interpolated cone response and two lorenzian functions.
        """
        L1 = self.lorenzian_function(f, *L1_params)
        L2 = self.lorenzian_function(f, *L2_params)

        fitted_interpolated_data = a0 * cone_interp_function(f)

        return L1 + L2 + fitted_interpolated_data

    def fit_log_interp_and_double_lorenzian(self, log_f, *log_params):
        """
        Wrapper function for fitting interpolated cone response and
        two lorenzian functions.

        Incoming and returned data are in log space to equalize fitting
        errors across the power scale.

        Parameters
        ----------
        log_f : numpy.ndarray
            Logarithm of frequency axis.
        log_params : list
            List of parameters to be fitted in log space.

        Returns
        -------
        log of power_spectrum : numpy.ndarray
        """

        # Parameters are scalars and frequency (x_axis) is an array.
        f = np.exp(log_f)

        # Combine the fitted amplitudes with fixed corner frequencies
        L1_params = np.array([np.exp(log_params[1]), self.cone_noise_wc[0]])
        L2_params = np.array([np.exp(log_params[2]), self.cone_noise_wc[1]])
        a0 = np.exp(log_params[0])

        # Calculate the power spectrum in linear space
        power_spectrum = self.lin_interp_and_double_lorenzian(
            f, a0, L1_params, L2_params, self.cone_interp_function
        )

        log_power_spectrum = np.log(power_spectrum)

        return log_power_spectrum

    def basic_logistic_function(self, x):
        return 1 / (1 + np.exp(-x))

    def generalized_logistic_function(self, x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def parabola(self, x, a, b, c):
        return a * x**2 + b * x + c

    # Fit method
    def lowpass(self, t, n, p, tau):
        """
        Returns a lowpass filter kernel with a given time constant and order.

        Parameters
        ----------
        - t (numpy.ndarray): Time points at which to evaluate the kernel.
        - n (float): Order of the filter.
        - p (float): Normalization factor for the kernel.
        - tau (float): Time constant of the filter.

        Returns
        -------
        - y (numpy.ndarray): Lowpass filter kernel evaluated at each time point in `t`.
        """

        y = p * (t / tau) ** (n) * np.exp(-n * (t / tau - 1))
        return y

    def get_triangular_parameters(self, minimum, maximum, median, mean, sd, sem):
        """
        Estimate the parameters for a triangular distribution based on the provided
        statistics: minimum, maximum, median, mean, and standard deviation.

        Parameters
        ----------
        minimum : float
            The smallest value of the data.
        maximum : float
            The largest value of the data.
        median : float
            The median of the data.
        mean : float
            The mean of the data.
        sd : float
            The standard deviation of the data.

        Returns
        -------
        c : float
            The shape parameter of the triangular distribution, representing the mode.
        loc : float
            The location parameter, equivalent to the minimum.
        scale : float
            The scale parameter, equivalent to the difference between the maximum and minimum.

        Raises
        ------
        ValueError:
            If the provided mean and standard deviation don't closely match the expected
            values for the triangular distribution.

        Notes
        -----
        The returned parameters can be used with scipy's triang function to represent
        a triangular distribution and perform further sampling or analysis.
        """
        # The location is simply the minimum.
        loc = minimum

        # The scale is the difference between maximum and minimum.
        scale = maximum - minimum

        # Estimating c (shape parameter) based on the position of median within the range.
        c = (median - minimum) / scale

        # Validate the given mean and SD against expected values for triangular distribution
        expected_mean = (minimum + maximum + median) / 3
        expected_sd = np.sqrt(
            (
                minimum**2
                + maximum**2
                + median**2
                - minimum * maximum
                - minimum * median
                - maximum * median
            )
            / 18
        )

        tolerance = 3 * sem
        if not (
            np.abs(expected_mean - mean) < tolerance
            and np.abs(expected_sd - sd) < tolerance
        ):
            raise ValueError(
                f"The provided mean ({mean}) and SD ({sd}) don't match the expected values for a triangular distribution with the given min, max, and median. Expected mean: {expected_mean}, Expected SD: {expected_sd}. Aborting..."
            )

        return c, loc, scale

    # Fit & RetinaVAE method

    def flip_negative_spatial_rf(self, spatial_rf_unflipped, mask_noise=0):
        """
        Flips negative values of a spatial RF to positive values.

        Parameters
        ----------
        spatial_rf_unflipped: numpy.ndarray of shape (N, H, W)
            Spatial receptive field.

        Returns
        -------
        spatial_rf: numpy.ndarray of shape (N, H, W)
            Spatial receptive field with negative values flipped to positive values.
        """

        # Number of pixels to define maximum value of RF
        max_pixels = 5

        # Copy spatial_rf_unflipped to spatial_rf
        spatial_rf = np.copy(spatial_rf_unflipped)

        for i in range(spatial_rf.shape[0]):

            # Find indices for the max_pixels number of absolute strongest pixels
            max_pixels_indices = np.argsort(np.abs(spatial_rf[i].ravel()))[-max_pixels:]

            # Calculate mean value of the original max_pixels_values
            mean_max_pixels_values = np.mean(spatial_rf[i].ravel()[max_pixels_indices])

            # If mean value of the original max_pixels_values is negative,
            # flip the RF

            if mean_max_pixels_values < 0:
                spatial_rf[i] = spatial_rf[i] * -1

            if mask_noise > 0:
                threshold = mask_noise
                if mean_max_pixels_values < 0:
                    threshold *= -1  # invert multiplier, too

                mask_threshold = threshold * mean_max_pixels_values
                noise_mask = np.squeeze(
                    (spatial_rf[i] > -mask_threshold) & (spatial_rf[i] < mask_threshold)
                )

                spatial_rf[i][noise_mask] = 0

        return spatial_rf

    # Fit & SimulateRetina method
    def DoG2D_fixed_surround(
        self,
        xy_tuple,
        ampl_c,
        xoc,
        yoc,
        semi_xc,
        semi_yc,
        orient_cen_rad,
        ampl_s,
        relat_sur_diam,
        offset,
    ):
        """
        DoG model with xo, yo, theta for surround coming from center.
        Note that semi_xc and semi_yc correspond to radii while matplotlib Ellipse assumes diameters.
        """

        (x_fit, y_fit) = xy_tuple
        acen = (np.cos(orient_cen_rad) ** 2) / (2 * semi_xc**2) + (
            np.sin(orient_cen_rad) ** 2
        ) / (2 * semi_yc**2)
        bcen = -(np.sin(2 * orient_cen_rad)) / (4 * semi_xc**2) + (
            np.sin(2 * orient_cen_rad)
        ) / (4 * semi_yc**2)
        ccen = (np.sin(orient_cen_rad) ** 2) / (2 * semi_xc**2) + (
            np.cos(orient_cen_rad) ** 2
        ) / (2 * semi_yc**2)

        asur = (np.cos(orient_cen_rad) ** 2) / (2 * (relat_sur_diam * semi_xc) ** 2) + (
            np.sin(orient_cen_rad) ** 2
        ) / (2 * (relat_sur_diam * semi_yc) ** 2)
        bsur = -(np.sin(2 * orient_cen_rad)) / (4 * (relat_sur_diam * semi_xc) ** 2) + (
            np.sin(2 * orient_cen_rad)
        ) / (4 * (relat_sur_diam * semi_yc) ** 2)
        csur = (np.sin(orient_cen_rad) ** 2) / (2 * (relat_sur_diam * semi_xc) ** 2) + (
            np.cos(orient_cen_rad) ** 2
        ) / (2 * (relat_sur_diam * semi_yc) ** 2)

        ## Difference of gaussians
        model_fit = (
            offset
            + ampl_c
            * np.exp(
                -(
                    acen * ((x_fit - xoc) ** 2)
                    + 2 * bcen * (x_fit - xoc) * (y_fit - yoc)
                    + ccen * ((y_fit - yoc) ** 2)
                )
            )
            - ampl_s
            * np.exp(
                -(
                    asur * ((x_fit - xoc) ** 2)
                    + 2 * bsur * (x_fit - xoc) * (y_fit - yoc)
                    + csur * ((y_fit - yoc) ** 2)
                )
            )
        )

        return model_fit.ravel()

    def DoG2D_independent_surround(
        self,
        xy_tuple,
        ampl_c,
        xoc,
        yoc,
        semi_xc,
        semi_yc,
        orient_cen_rad,
        ampl_s,
        xos,
        yos,
        semi_xs,
        semi_ys,
        orient_sur_rad,
        offset,
    ):
        """
        DoG model with xo, yo, theta for surround independent from center.
        """

        (x_fit, y_fit) = xy_tuple
        acen = (np.cos(orient_cen_rad) ** 2) / (2 * semi_xc**2) + (
            np.sin(orient_cen_rad) ** 2
        ) / (2 * semi_yc**2)
        bcen = -(np.sin(2 * orient_cen_rad)) / (4 * semi_xc**2) + (
            np.sin(2 * orient_cen_rad)
        ) / (4 * semi_yc**2)
        ccen = (np.sin(orient_cen_rad) ** 2) / (2 * semi_xc**2) + (
            np.cos(orient_cen_rad) ** 2
        ) / (2 * semi_yc**2)

        asur = (np.cos(orient_sur_rad) ** 2) / (2 * semi_xs**2) + (
            np.sin(orient_sur_rad) ** 2
        ) / (2 * semi_ys**2)
        bsur = -(np.sin(2 * orient_sur_rad)) / (4 * semi_xs**2) + (
            np.sin(2 * orient_sur_rad)
        ) / (4 * semi_ys**2)
        csur = (np.sin(orient_sur_rad) ** 2) / (2 * semi_xs**2) + (
            np.cos(orient_sur_rad) ** 2
        ) / (2 * semi_ys**2)

        ## Difference of gaussians
        model_fit = (
            offset
            + ampl_c
            * np.exp(
                -(
                    acen * ((x_fit - xoc) ** 2)
                    + 2 * bcen * (x_fit - xoc) * (y_fit - yoc)
                    + ccen * ((y_fit - yoc) ** 2)
                )
            )
            - ampl_s
            * np.exp(
                -(
                    asur * ((x_fit - xos) ** 2)
                    + 2 * bsur * (x_fit - xos) * (y_fit - yos)
                    + csur * ((y_fit - yos) ** 2)
                )
            )
        )

        return model_fit.ravel()

    def DoG2D_circular(self, xy_tuple, ampl_c, x0, y0, rad_c, ampl_s, rad_s, offset):
        """
        DoG model with the center and surround as concentric circles and a shared center (x0, y0).
        """

        (x_fit, y_fit) = xy_tuple

        # Distance squared from the center for the given (x_fit, y_fit) points
        distance_sq = (x_fit - x0) ** 2 + (y_fit - y0) ** 2

        # Gaussian for the center
        center_gaussian = ampl_c * np.exp(-distance_sq / (2 * rad_c**2))

        # Gaussian for the surround
        surround_gaussian = ampl_s * np.exp(-distance_sq / (2 * rad_s**2))

        # Difference of gaussians
        model_fit = offset + center_gaussian - surround_gaussian

        return model_fit.ravel()

    def diff_of_lowpass_filters(self, t, n, p1, p2, tau1, tau2):
        """
        Returns the difference between two lowpass filters with different time constants and orders.
        From Chichilnisky & Kalmar JNeurosci 2002

        Parameters
        ----------
        - t (numpy.ndarray): Time points at which to evaluate the filters.
        - n (float): Order of the filters.
        - p1 (float): Normalization factor for the first filter.
        - p2 (float): Normalization factor for the second filter.
        - tau1 (float): Time constant of the first filter.
        - tau2 (float): Time constant of the second filter.

        Returns
        -------
        - y (numpy.ndarray): Difference between the two lowpass filters evaluated at each time point in `t`.
        """

        #
        y = self.lowpass(t, n, p1, tau1) - self.lowpass(t, n, p2, tau2)
        return y

    def calculate_gaussian_volumes(
        self, ampl_c, ampl_s, semi_xc, semi_yc, relat_sur_diam
    ):
        # Central Gaussian Volume
        sigma_xc = semi_xc
        sigma_yc = semi_yc
        volume_central = ampl_c * 2 * np.pi * sigma_xc * sigma_yc

        # Surround Gaussian Volume
        sigma_xs = relat_sur_diam * semi_xc
        sigma_ys = relat_sur_diam * semi_yc
        volume_surround = ampl_s * 2 * np.pi * sigma_xs * sigma_ys

        return volume_central, volume_surround
