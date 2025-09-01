# Built-in
import ast
from pathlib import Path

# Third-party
import brian2.units as b2u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.stats import pearsonr


class Analysis:
    def __init__(self, context, data_io, **kwargs) -> None:
        self._context = context
        self._data_io = data_io

        for attr, value in kwargs.items():
            setattr(self, attr, value)

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    def _min_max_scaler(self, X, feature_range=(0, 1)):
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_std = (X - X_min) / (X_max - X_min)
        X_scaled = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
        return X_scaled

    def _scale(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std

    def scaler(self, data, scale_type="standard", feature_range=[-1, 1]):
        # Data is assumed to be [samples or time, features or regressors]
        if scale_type == "standard":
            # Standardize data by removing the mean and scaling to unit variance
            data_scaled = self._scale(data)
        elif scale_type == "minmax":
            # Transform features by scaling each feature to a given range.
            # If you put in matrix, note that scales each column (feature) independently
            if data.ndim > 1:
                minmaxscaler = self._min_max_scaler(data, feature_range=feature_range)
                minmaxscaler.fit(data)
                data_scaled = minmaxscaler.transform(data)
            elif data.ndim == 1:  # Manual implementation for 1-D data
                feat_min, feat_max = feature_range
                data_std = (data - data.min()) / (data.max() - data.min())
                data_scaled = data_std * (feat_max - feat_min) + feat_min
        return data_scaled

    def _get_spikes_by_interval(self, data, sweep, t_start, t_end):
        key_name = f"spikes_{sweep}"
        data_by_trial = data[key_name]

        idx_mask = np.logical_and(
            data_by_trial[1] > t_start * b2u.second,
            data_by_trial[1] < t_end * b2u.second,
        )

        spike_units = data_by_trial[0][idx_mask]
        spike_times = data_by_trial[1][idx_mask]

        return spike_units, spike_times

    def _analyze_meanfr(self, data, sweep, t_start, t_end):
        units, times = self._get_spikes_by_interval(data, sweep, t_start, t_end)
        N_neurons = data["n_units"]
        mean_fr = times.size / (N_neurons * (t_end - t_start))

        return mean_fr

    def _analyze_sd_fr(self, data, sweep, t_start, t_end):
        units, times = self._get_spikes_by_interval(data, sweep, t_start, t_end)
        N_neurons = data["n_units"]
        delta_time = t_end - t_start
        unique, counts = np.unique(units, return_counts=True)
        unitwise_fr = counts / delta_time
        sd_fr = np.std(unitwise_fr)

        return sd_fr

    def _analyze_unit_fr(self, data, sweep, t_start, t_end):
        units, times = self._get_spikes_by_interval(data, sweep, t_start, t_end)
        N_neurons = data["n_units"]

        # Get firing rate for each neuron
        fr = np.zeros(N_neurons)
        for this_unit in range(N_neurons):
            unit_mask = units == this_unit
            times_unit = times[unit_mask]
            fr[this_unit] = times_unit.size / (t_end - t_start)

        return fr, N_neurons

    def _analyze_peak2peak_fr(
        self, data, sweep, t_start, t_end, temp_freq, bins_per_cycle=32
    ):
        # Analyze the peak-to-peak firing rate across units.
        units, times = self._get_spikes_by_interval(data, sweep, t_start, t_end)
        times = times / b2u.second

        N_neurons = data["n_units"]

        cycle_length = 1 / temp_freq  # in seconds
        # Calculate N full cycles in the interval
        n_cycles = int(np.floor((t_end - t_start) / cycle_length))

        # Corrected time interval
        t_epoch = n_cycles * cycle_length

        # Change t_end to be the end of the last full cycle
        t_end_full = t_start + t_epoch

        # Remove spikes before t_start
        times = times[times > t_start]

        # Remove spikes after t_end_full
        times = times[times < t_end_full]

        # Calculate bins matching t_end_full - t_start
        bins = np.linspace(
            t_start, t_end_full, (n_cycles * bins_per_cycle) + 1, endpoint=True
        )

        bin_width = bins[1] - bins[0]
        # add one bin to the end
        spike_counts, _ = np.histogram(times, bins=bins)

        # Compute average cycle. Average across cycles.
        spike_counts_reshaped = np.reshape(
            spike_counts, (int(len(spike_counts) / bins_per_cycle), bins_per_cycle)
        )

        spike_counts_mean_across_cycles = np.mean(spike_counts_reshaped, axis=0)
        spike_count__unit_fr = spike_counts_mean_across_cycles / (N_neurons * bin_width)

        peak2peak_counts_all = np.max(spike_counts_mean_across_cycles) - np.min(
            spike_counts_mean_across_cycles
        )

        peak2peak_counts_unit_mean = peak2peak_counts_all / N_neurons
        # Convert to Hz: ptp mean across units / time for one bin
        peak2peak_fr = peak2peak_counts_unit_mean / bin_width

        return peak2peak_fr

    def _fourier_amplitude_and_phase(
        self, data, sweep, t_start, t_end, temp_freq, phase_shift=0, bins_per_cycle=8
    ):
        """
        Calculate the F1 and F2 amplitude (amplitude at the stimulus frequency and twice the stimulus frequency) of spike rates.

        Parameters
        ----------
        data : dict
            The data dictionary containing the spike information.
        sweep : int
            The sweep number to analyze.
        t_start : float
            The start time of the interval (in seconds) to analyze.
        t_end : float
            The end time of the interval (in seconds) to analyze.
        temp_freq : float
            The frequency (in Hz) of the stimulus.
        phase_shift : float, optional
            The phase shift (in radians) to be applied. Default is 0.
        bins_per_cycle : int, optional
            The number of bins per cycle to use for the spike rate. The default is 16.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, int)
            The F1 and F2 amplitudes, F1 and F2 phases for each neuron, and the total number of neurons.

        """

        units, times = self._get_spikes_by_interval(data, sweep, t_start, t_end)
        N_neurons = data["n_units"]

        # Prepare for saving all spectra
        spectra = []

        # Get firing rate for each neuron
        amplitudes_F1 = np.zeros(N_neurons)  # Added to store F1
        amplitudes_F2 = np.zeros(N_neurons)  # Added to store F2

        phases_F1 = np.zeros(N_neurons)  # Added to store F1 phase
        phases_F2 = np.zeros(N_neurons)  # Added to store F2 phase

        cycle_length = 1 / temp_freq  # in seconds
        bins = np.arange(t_start, t_end, cycle_length / bins_per_cycle)

        # Check for non-oscillatory response
        n_cycles = int(np.floor((t_end - t_start) / cycle_length))
        if n_cycles < 2:
            print(f"Cannot analyze oscillatory response from {n_cycles} cycles")
            return np.nan, np.nan, np.nan, np.nan, np.nan

        for this_unit in range(N_neurons):
            unit_mask = units == this_unit
            times_unit = times[unit_mask]

            if len(times_unit) > 0:  # check if there are spikes for this unit
                # Bin spike rates
                spike_counts, _ = np.histogram(times_unit, bins=bins)
                # Convert spike counts to spike rates
                spike_rate = spike_counts / (cycle_length / bins_per_cycle)
                spike_rate = spike_rate - np.mean(spike_rate)

                # Analyze Fourier amplitude
                # Compute the one-dimensional n-point discrete Fourier Transform for real input
                sp = np.fft.rfft(spike_rate)
                # Compute the frequencies corresponding to the coefficients
                freq = np.fft.rfftfreq(
                    len(spike_rate), d=(cycle_length / bins_per_cycle)
                )

                # Save the spectrum for plotting
                normalized_spectrum = np.abs(sp) / len(spike_rate) * 2
                spectra.append(normalized_spectrum)

                # Get F1 amplitude and phase
                closest_freq_index = np.abs(freq - temp_freq).argmin()
                amplitudes_F1[this_unit] = (
                    np.abs(sp[closest_freq_index]) / len(spike_rate) * 2
                )
                phases_F1[this_unit] = np.angle(sp[closest_freq_index]) + phase_shift

                # Ensure phase remains in [-π, π]
                if phases_F1[this_unit] > np.pi:
                    phases_F1[this_unit] -= 2 * np.pi
                elif phases_F1[this_unit] < -np.pi:
                    phases_F1[this_unit] += 2 * np.pi

                # Get F2 amplitude and phase
                closest_freq_index = np.abs(freq - (2 * temp_freq)).argmin()
                amplitudes_F2[this_unit] = (
                    np.abs(sp[closest_freq_index]) / len(spike_rate) * 2
                )
                phases_F2[this_unit] = np.angle(sp[closest_freq_index]) + phase_shift

                # Ensure phase remains in [-π, π]
                if phases_F2[this_unit] > np.pi:
                    phases_F2[this_unit] -= 2 * np.pi
                elif phases_F2[this_unit] < -np.pi:
                    phases_F2[this_unit] += 2 * np.pi

        return (
            amplitudes_F1,
            amplitudes_F2,
            phases_F1,  # Return F1 phase
            phases_F2,  # Return F2 phase
            N_neurons,
        )

    def _generate_spikes(
        self,
        N_neurons,
        temp_freq,
        t_start,
        t_end,
        baseline_rate=10,
        modulation_depth=5,
    ):
        """
        A helper function to generate random spikes for N_neurons with sinusoidal modulation.

        Args:
        - N_neurons (int): Number of neurons.
        - temp_freq (float): Temporal frequency for sinusoidal modulation.
        - total_time (float): Total simulation time in seconds.
        - baseline_rate (float): Baseline firing rate in Hz.
        - modulation_depth (float): Depth of sinusoidal modulation in Hz.

        Returns:
        - spikes (list of np.ndarray): A list of spike times for each neuron.
        """

        sampling_rate = 10000  # Hz

        t = np.linspace(
            t_start, t_end, int((t_end - t_start) * sampling_rate)
        )  # 1 ms resolution
        modulating_signal_raw = (modulation_depth / 2) * np.sin(
            2 * np.pi * temp_freq * t
        )
        modulating_signal = baseline_rate + (
            modulating_signal_raw - np.min(modulating_signal_raw)
        )

        spikes = np.array([])

        for _ in range(N_neurons):
            neuron_spikes = []
            for i, rate in enumerate(modulating_signal):
                # For each time bin, decide whether to emit a spike based on rate
                if np.random.random() < (
                    rate / sampling_rate
                ):  # Convert Hz to rate per ms
                    neuron_spikes.append(t[i])
            spikes = np.concatenate((spikes, np.array(neuron_spikes)), axis=0)

        return np.sort(spikes)

    def _fourier_amplitude_pooled(
        self,
        data,
        sweep,
        t_start,
        t_end,
        temp_freq,
        bins_per_cycle=8,
    ):
        units, times = self._get_spikes_by_interval(data, sweep, t_start, t_end)
        times = times / b2u.second
        N_neurons = data["n_units"]

        cycle_length = 1 / temp_freq

        # Due to scalloping loss artefact causing spectral leakage, we need to
        # match the sampling points in frequency space to stimulation frequency.

        # Find the integer number of full cycles matching t_end - t_start
        n_cycles = int(np.floor((t_end - t_start) / cycle_length))

        if n_cycles < 2:
            print(f"Cannot analyze oscillatory response from {n_cycles} cycles")
            return np.nan, np.nan

        # Corrected time interval
        t_epoch = n_cycles * cycle_length

        # Change t_end to be the end of the last full cycle
        t_end_full = t_start + t_epoch

        # Remove spikes before t_start
        times = times[times > t_start]

        # Remove spikes after t_end_full
        times = times[times < t_end_full]

        # Calculate bins matching t_end_full - t_start
        bins = np.linspace(
            t_start, t_end_full, (n_cycles * bins_per_cycle) + 1, endpoint=True
        )

        bin_width = bins[1] - bins[0]

        # Bin spike rates
        spike_counts, _ = np.histogram(times, bins=bins)
        spike_rate = spike_counts / (bin_width * N_neurons)
        spike_rate = spike_rate - np.mean(spike_rate)

        # Compute Fourier Transform and associated frequencies
        sp = np.fft.rfft(spike_rate)
        freq = np.fft.rfftfreq(len(spike_rate), d=bin_width)

        # the np.fft.rfft function gives the positive frequency components
        # for real-valued inputs. This is half the total amplitude.
        # To adjust for this, we multiply the amplitude by 2:
        normalized_spectrum = 2 * np.abs(sp) / len(spike_rate)
        normalized_spectrum_per_unit = normalized_spectrum

        # Extract the F1 and F2 amplitudes
        closest_freq_index = np.abs(freq - temp_freq).argmin()
        ampl_F1 = normalized_spectrum_per_unit[closest_freq_index]

        closest_freq_index = np.abs(freq - (2 * temp_freq)).argmin()
        ampl_F2 = normalized_spectrum_per_unit[closest_freq_index]

        return ampl_F1, ampl_F2

    def _normalize_phase(self, phase_np, experiment_df, exp_variables):
        """
        Reset the phase so that the slowest temporal frequency is 0
        """
        assert (
            phase_np.shape[1] == experiment_df.shape[0]
        ), "Number of conditions do not match, aborting..."
        assert len(exp_variables) < 3, "More than 2 variables, aborting..."

        # Make df with index = conditions and columns = levels
        cond_value_df = pd.DataFrame(
            index=experiment_df.index.values, columns=exp_variables
        )

        # Make new columns with conditions' levels
        for cond_idx, cond in enumerate(exp_variables):
            levels_s = experiment_df.loc[:, cond]
            levels_s = pd.to_numeric(levels_s)
            levels_s = levels_s.round(decimals=2)
            cond_value_df[cond] = levels_s

        # Find row indeces of the slowest temporal frequency
        slowest_temp_freq = cond_value_df.loc[:, "temporal_frequency"].min()
        slowest_temp_freq_idx = cond_value_df[
            cond_value_df["temporal_frequency"] == slowest_temp_freq
        ].index.values

        len_second_dim = 0
        if len(exp_variables) == 2:
            # get number of levels for the second variable
            for cond in exp_variables:
                if cond != "temporal_frequency":
                    second_cond_levels = cond_value_df.loc[:, cond].unique()
                    len_second_dim = int(phase_np.shape[1] / len(second_cond_levels))

        phase_np_reset = np.zeros_like(phase_np)
        for this_cond in slowest_temp_freq_idx:
            slowest_idx = np.where(experiment_df.index.values == this_cond)[0][0]
            all_idx = np.arange(slowest_idx, slowest_idx + len_second_dim)
            phase_to_subtract = np.expand_dims(phase_np[:, slowest_idx], axis=1)
            phase_np_reset[:, all_idx] = phase_np[:, all_idx] - phase_to_subtract

        # If phase is over pi, subtract 2pi
        phase_np_reset[phase_np_reset > np.pi] -= 2 * np.pi

        return phase_np_reset

    def _correlation_lags(self, in1_len, in2_len, mode="full"):
        # Copied from scipy.signal correlation_lags, mode full or same

        if mode == "full":
            lags = np.arange(-in2_len + 1, in1_len)
        elif mode == "same":
            # the output is the same size as `in1`, centered
            # with respect to the 'full' output.
            # calculate the full output
            lags = np.arange(-in2_len + 1, in1_len)
            # determine the mean point in the full output
            mean_point = lags.size // 2
            # determine lag_bound to be used with respect
            # to the mean point
            lag_bound = in1_len // 2
            # calculate lag ranges for even and odd scenarios
            if in1_len % 2 == 0:
                lags = lags[(mean_point - lag_bound) : (mean_point + lag_bound)]
            else:
                lags = lags[(mean_point - lag_bound) : (mean_point + lag_bound) + 1]

        return lags

    def _get_cross_correlation_trial(
        self, data, sweep, t_start, t_end, bins, lags, unit_vec
    ):
        """
        Calculate cross correlation between units for a single sweep. For each unit, calculate
        cross correlation with all other units. The cross correlation is normalized to provide
        correlation coefficients.

        Parameters
        ----------
        data : dict
            The data dictionary containing the spike information.
        sweep : int
            The sweep number to analyze.
        t_start : float
            The start time of the interval (in seconds) to analyze.
        t_end : float
            The end time of the interval (in seconds) to analyze.
        bins : numpy.ndarray
            The bins to use for binning the spike times.
        lags : numpy.ndarray
            The lags to use for cross correlation.
        unit_vec : numpy.ndarray of ints
            The units to analyze.
        """
        # Get spike data limited to requested time interval, single sweep
        units, times = self._get_spikes_by_interval(data, sweep, t_start, t_end)

        n_units = len(unit_vec)
        spike_events = np.zeros((n_units, len(bins) - 1))
        for this_idx, this_unit in enumerate(unit_vec):
            unit_mask = units == this_unit
            times_unit = times[unit_mask]

            # Bin spike rates
            spike_counts, _ = np.histogram(times_unit, bins=bins)

            # Convert spike count to boolean. More than one in one bin becomes one.
            spike_events[this_idx] = spike_counts > 0

        ccf = np.zeros((n_units, n_units, len(lags)))
        ccoef = np.zeros((n_units, n_units))

        # Calculate cross correlation between all pairs of units
        for y_idx in range(n_units):
            for x_idx in range(n_units):
                y = spike_events[y_idx]
                y_scaled = self.scaler(y)
                x = spike_events[x_idx]
                x_scaled = self.scaler(x)

                # Standard deviations of the signals
                std_y = np.std(y_scaled)
                std_x = np.std(x_scaled)

                # Cross-correlation function
                raw_ccf = correlate(y_scaled, x_scaled, mode="full", method="direct")

                # Normalization factor for each lag
                lag_counts = np.correlate(
                    np.ones_like(y_scaled), np.ones_like(x_scaled), mode="full"
                )
                normalization = std_y * std_x * lag_counts

                # Normalized correlation coefficient for each lag
                ccf[y_idx, x_idx, :] = raw_ccf / normalization
                ccoef[y_idx, x_idx] = pearsonr(y_scaled, x_scaled)[0]

        return ccf, ccoef

    def _calc_dist_mtx(self, x_vec, y_vec, unit_vec):
        n_units = unit_vec.shape[0]
        dist_mtx = np.zeros((n_units, n_units))
        for i in range(n_units):
            for j in range(n_units):
                dist_mtx[i, j] = np.sqrt(
                    (x_vec[i] - x_vec[j]) ** 2 + (y_vec[i] - y_vec[j]) ** 2
                )
        return dist_mtx

    def _create_dist_ccoef_df(self, dist_mtx, ccoef_mtx_mean, unit_vec):
        # Initialize an empty list to store the tuples
        data_list = []

        # Iterate over the upper triangle of dist_mtx to avoid duplicates
        for i in range(dist_mtx.shape[0]):
            for j in range(i + 1, dist_mtx.shape[1]):
                yx_name = f"unit pair {unit_vec[i]}-{unit_vec[j]}"
                yx_idx = np.array([i, j])
                distance = dist_mtx[i, j]
                ccoef = ccoef_mtx_mean[i, j]
                data_list.append((yx_name, yx_idx, distance, ccoef))

        # Sort the list by distance
        sorted_data_list = sorted(data_list, key=lambda x: x[2])

        # Convert the list to a DataFrame
        distance_df = pd.DataFrame(
            sorted_data_list, columns=["yx_name", "yx_idx", "distance_mm", "ccoef"]
        )

        return distance_df

    def _create_neighbors_df(self, distance_df, unit_vec):
        # Initialize an empty list to store the nearest neighbor tuples
        nearest_neighbors = []

        # Iterate over each unit in unit_vec.
        for idx, unit in enumerate(unit_vec):
            # Filter distance_df to find the rows where the unit index is either yx_idx[0] or yx_idx[1]
            # Note that we are looking at index, not the unit number
            filtered_df = distance_df[distance_df["yx_idx"].apply(lambda x: idx in x)]

            # If the filtered_df is not empty, find the nearest neighbor
            if not filtered_df.empty:
                nearest_neighbor = filtered_df.iloc[
                    0
                ]  # the first row is the nearest neighbor
                nearest_neighbors.append(nearest_neighbor)

        # Convert the list to a DataFrame
        neighbors_df = pd.DataFrame(nearest_neighbors).reset_index(drop=True)

        return neighbors_df

    def unit_correlation(
        self, filename, my_analysis_options, gc_type, response_type, gc_units
    ):
        """
        Analyze noise correlation in neural responses based on experimental variables.

        This method computes cross-correlation, normalized to correlation coefficient
        as a function of time lag, and correlation coefficients for neural responses.
        It uses experimental conditions and ganglion cell (GC) data to analyze correlations and
        saves the results to files.

        Parameters
        ----------
        filename : str
            The name of the file containing the experimental metadata.
        my_analysis_options : dict
            A dictionary containing experimental variables and analysis parameters.
        gc_type : str, midget or parasol
            The type of ganglion cell under analysis.
        response_type : str, on or off
            The type of response being analyzed.
        gc_units : list or None
            A list of ganglion cell unit indexes for analysis. If None, all units are analyzed.

        Raises
        ------
        ValueError
            If `gc_units` is neither None nor a list.
        AssertionError
            If the number of trials is not equal across conditions.
        """

        exp_variables = my_analysis_options["exp_variables"]
        cond_names_string = "_".join(exp_variables)
        data_folder = self.context.output_folder
        experiment_df = self.data_io.get_data(filename=filename)
        cond_names = experiment_df.index.values
        t_start = my_analysis_options["t_start_ana"]
        t_end = my_analysis_options["t_end_ana"]
        n_sweeps_vec = pd.to_numeric(experiment_df.loc[:, "n_sweeps"].values)

        # Assert for equal number of trials
        assert np.all(
            n_sweeps_vec == n_sweeps_vec[0]
        ), "Not equal number of trials, aborting..."
        n_sweeps = n_sweeps_vec[0]

        bin_width = 0.01  # seconds

        bins = np.linspace(t_start, t_end, int((t_end - t_start) / bin_width))
        lags = self._correlation_lags(len(bins) - 1, len(bins) - 1)
        # Convert lags to seconds
        lags = lags * bin_width

        cond_name = cond_names[0]
        filename_prefix = f"Response_{gc_type}_{response_type}_"
        filename = Path(data_folder) / (filename_prefix + cond_name + ".gz")
        data_dict = self.data_io.get_data(filename)

        if gc_units is None:
            n_units = data_dict["n_units"]
            unit_vec = np.arange(n_units)
        elif isinstance(gc_units, list):
            n_units = len(gc_units)
            unit_vec = np.array(gc_units)
        else:
            raise ValueError("gc_units must be None or a list")

        # Cross-correlation matrix [sweep, unit_y, unit_x, lags]
        ccf_mtx = np.zeros((n_sweeps, n_units, n_units, len(lags)))
        # Correlation coefficient matrix [sweep, unit_y, unit_x]
        ccoef_mtx = np.zeros((n_sweeps, n_units, n_units))

        # Loop conditions
        for this_sweep in range(n_sweeps):
            # Cross correlation, normalized to correlation coefficient
            ccf, ccoef = self._get_cross_correlation_trial(
                data_dict, this_sweep, t_start, t_end, bins, lags, unit_vec
            )
            ccf_mtx[this_sweep, ...] = ccf
            ccoef_mtx[this_sweep, ...] = ccoef

        ccf_mtx_mean = np.mean(ccf_mtx, axis=0)
        ccf_mtx_SEM = np.std(ccf_mtx, axis=0) / np.sqrt(ccf_mtx.shape[0])
        ccoef_mtx_mean = np.mean(ccoef_mtx, axis=0)

        # Load mosaic
        gc_dataframe = self.data_io.get_data(
            filename=self.context.retina_parameters["mosaic_file"]
        )
        # Get xy coords from dataframe
        pos_ecc_mm = gc_dataframe["pos_ecc_mm"].values
        pos_polar_deg = gc_dataframe["pos_polar_deg"].values
        x_vec, y_vec = self.pol2cart(pos_ecc_mm, pos_polar_deg)

        # Calculate distances between unit_vec units
        dist_mtx = self._calc_dist_mtx(x_vec, y_vec, unit_vec)
        dist_df = self._create_dist_ccoef_df(dist_mtx, ccoef_mtx_mean, unit_vec)
        neigbor_df = self._create_neighbors_df(dist_df, unit_vec)
        neighbor_unique_df = neigbor_df.drop_duplicates(subset=["yx_name"])

        # Save results
        filename_out = f"{cond_names_string}_correlation.npz"
        npy_save_path = data_folder / filename_out
        np.savez(
            npy_save_path,
            ccf_mtx_mean=ccf_mtx_mean,
            ccf_mtx_SEM=ccf_mtx_SEM,
            lags=lags,
            unit_vec=unit_vec,
        )

        filename_out = f"{cond_names_string}_correlation_neighbors.csv"
        csv_save_path = data_folder / filename_out
        neighbor_unique_df.to_csv(csv_save_path, index=False)

        filename_out = f"{cond_names_string}_correlation_distances.csv"
        csv_save_path = data_folder / filename_out
        dist_df.to_csv(csv_save_path, index=False)

    def analyze_experiment(self, filename, my_analysis_options):
        """
        Analyze the experiment data and save the results to CSV files.

        Parameters
        ----------
        filename : str
            The name of the file containing the experimental metadata.
        my_analysis_options : dict
            A dictionary containing experimental variables and analysis parameters.

        """

        exp_variables = my_analysis_options["exp_variables"]
        cond_names_string = "_".join(exp_variables)
        experiment_df = self.data_io.get_data(filename=filename)
        data_folder = self.context.output_folder
        cond_names = experiment_df.index.values
        t_start = my_analysis_options["t_start_ana"]
        t_end = my_analysis_options["t_end_ana"]
        n_sweeps_vec = pd.to_numeric(experiment_df.loc[:, "n_sweeps"].values)
        gc_type = self.context.retina_parameters["gc_type"]
        response_type = self.context.retina_parameters["response_type"]

        if "temporal_frequency" not in exp_variables:
            temp_freq = self.context.visual_stimulus_parameters["temporal_frequency"]
            experiment_df["temporal_frequency"] = temp_freq

        # Assert for equal number of trials
        assert np.all(
            n_sweeps_vec == n_sweeps_vec[0]
        ), "Not equal number of trials, aborting..."
        n_sweeps = n_sweeps_vec[0]

        # Make dataframe with columns = conditions and index = trials
        R_popul_df = pd.DataFrame(index=range(n_sweeps_vec[0]), columns=cond_names)

        columns = cond_names.tolist()
        columns.extend(["sweep", "F_peak"])
        # Make F1 and F2 dataframe
        F_popul_df = pd.DataFrame(index=range(n_sweeps_vec[0] * 2), columns=columns)

        # Loop conditions
        for idx, cond_name in enumerate(cond_names):
            filename = Path(data_folder) / (
                f"Response_{gc_type}_{response_type}_{cond_name}.gz"
            )
            data_dict = self.data_io.get_data(filename)

            temp_freq = pd.to_numeric(
                experiment_df.loc[cond_name, "temporal_frequency"]
            )
            phase_shift = pd.to_numeric(experiment_df.loc[cond_name, "phase_shift"])

            for this_sweep in range(n_sweeps):
                mean_fr = self._analyze_meanfr(data_dict, this_sweep, t_start, t_end)
                R_popul_df.loc[this_sweep, cond_name] = mean_fr

                fr, N_neurons = self._analyze_unit_fr(
                    data_dict, this_sweep, t_start, t_end
                )
                # If first sweep, initialize R_unit_compiled and F_unit_compiled
                if idx == 0 and this_sweep == 0:
                    R_unit_compiled = np.zeros((N_neurons, len(cond_names), n_sweeps))
                    F_unit_compiled = np.zeros(
                        (N_neurons, len(cond_names), n_sweeps, 4)
                    )
                R_unit_compiled[:, idx, this_sweep] = fr

                # Amplitude spectra for pooled neurons, mean across units
                (ampl_F1, ampl_F2) = self._fourier_amplitude_pooled(
                    data_dict, this_sweep, t_start, t_end, temp_freq
                )
                F_popul_df.loc[this_sweep, "sweep"] = this_sweep
                F_popul_df.loc[this_sweep, "F_peak"] = "F1"
                F_popul_df.loc[this_sweep, cond_name] = ampl_F1
                F_popul_df.loc[this_sweep + n_sweeps_vec[0], "sweep"] = this_sweep
                F_popul_df.loc[this_sweep + n_sweeps_vec[0], "F_peak"] = "F2"
                F_popul_df.loc[this_sweep + n_sweeps_vec[0], cond_name] = ampl_F2

                # Amplitude spectra for units
                (
                    ampl_F1,
                    ampl_F2,
                    phase_F1,
                    phase_F2,
                    N_neurons,
                ) = self._fourier_amplitude_and_phase(
                    data_dict, this_sweep, t_start, t_end, temp_freq, phase_shift
                )

                F_unit_compiled[:, idx, this_sweep, 0] = ampl_F1
                F_unit_compiled[:, idx, this_sweep, 1] = ampl_F2
                F_unit_compiled[:, idx, this_sweep, 2] = phase_F1
                F_unit_compiled[:, idx, this_sweep, 3] = phase_F2

        # Set unit fr to dataframe, mean over trials
        R_unit_mean = np.mean(R_unit_compiled, axis=2)
        R_unit_df = pd.DataFrame(R_unit_mean, columns=cond_names)

        # Save results
        filename_out = f"{cond_names_string}_population_means.csv"
        csv_save_path = data_folder / filename_out
        R_popul_df.to_csv(csv_save_path)

        filename_out = f"{cond_names_string}_unit_means.csv"
        csv_save_path = data_folder / filename_out
        R_unit_df.to_csv(csv_save_path)

        all_nan_popul = F_popul_df[cond_names.tolist()].isna().all().all()
        all_nan_unit = np.isnan(F_unit_compiled).all()
        if all_nan_popul or all_nan_unit:
            print("No oscillatory response, no F1F2 responses available...")
            return

        # Set unit F1 and F2 to dataframe, mean over trials
        F_unit_mean = np.mean(F_unit_compiled, axis=2)
        F_unit_mean_ampl_reshaped = np.concatenate(
            (F_unit_mean[:, :, 0], F_unit_mean[:, :, 1]), axis=0
        )

        F_peak = ["F1"] * N_neurons + ["F2"] * N_neurons
        unit = np.tile(np.arange(N_neurons), 2)
        F_unit_ampl_df = pd.DataFrame(
            F_unit_mean_ampl_reshaped, columns=cond_names.tolist()
        )
        F_unit_ampl_df["unit"] = unit
        F_unit_ampl_df["F_peak"] = F_peak

        F_unit_mean_phase_reshaped = np.concatenate(
            (F_unit_mean[:, :, 2], F_unit_mean[:, :, 3]), axis=0
        )

        # Save results
        filename_out = f"{cond_names_string}_F1F2_population_means.csv"
        csv_save_path = data_folder / filename_out
        F_popul_df.to_csv(csv_save_path)

        filename_out = f"{cond_names_string}_F1F2_unit_ampl_means.csv"
        csv_save_path = data_folder / filename_out
        F_unit_ampl_df.to_csv(csv_save_path)

        if "temporal_frequency" in exp_variables:
            # Normalize phase -- resets phase to slowest temporal frequency
            F_unit_mean_phase_reshaped_norm = self._normalize_phase(
                F_unit_mean_phase_reshaped, experiment_df, exp_variables
            )

            F_unit_phase_df = pd.DataFrame(
                F_unit_mean_phase_reshaped_norm, columns=cond_names.tolist()
            )
            F_unit_phase_df["unit"] = unit
            F_unit_phase_df["F_peak"] = F_peak

            filename_out = f"{cond_names_string}_F1F2_unit_phase_means.csv"
            csv_save_path = data_folder / filename_out
            F_unit_phase_df.to_csv(csv_save_path)

    def get_gain_calibration_df(self, threshold, folder_pattern, signal_gain=1.0):

        matching_files_or_folders = self.data_io.all_patterns(
            self.context.path, folder_pattern
        )
        matching_folders = [f for f in matching_files_or_folders if f.is_dir()]
        matching_folders.sort()

        df = pd.DataFrame()
        for this_folder in matching_folders:
            filename = self.data_io.most_recent_pattern(
                this_folder, "*_F1F2_population_means.csv"
            )
            this_df = pd.read_csv(filename, index_col=0).drop(axis=0, index=1)

            # Get gain value from folder name
            folder_components = folder_pattern.split("_")
            name_components = this_folder.name.split("_")

            # Step 1: Find the index of the element containing the asterisk in folder_pattern
            asterisk_index = next(
                i for i, s in enumerate(folder_components) if "*" in s
            )

            # Step 2: Get the corresponding element from name_components
            corresponding_element = name_components[asterisk_index]

            # Step 3: Extract the numeric part from the corresponding element
            gain = "".join(filter(str.isdigit, corresponding_element))
            this_df["gain"] = float(gain) * signal_gain
            df = pd.concat([df, this_df], axis=0, ignore_index=True)

        # select dataframe columns whose names include substring "tf"
        tf_columns = [col for col in df.columns if "tf" in col]
        if len(tf_columns) == 0:
            print(
                "No temporal frequency columns found, cannot analyze gain at threshold."
            )
            return

        peak_response_idx_for_each_gain = np.argmax(df[tf_columns].values, axis=1)
        values, counts = np.unique(peak_response_idx_for_each_gain, return_counts=True)
        # If there are multiple equal highest counts, take the corresponding highest value as the peak
        if len(np.where(counts == counts.max())[0]) > 1:
            most_frequent_peak_idx = values[np.where(counts == counts.max())[0]][-1]
        else:
            most_frequent_peak_idx = values[np.argmax(counts)]
        peak_column = df.columns[most_frequent_peak_idx]
        df = df.sort_values(by="gain").reset_index(drop=True)

        return df, peak_column, most_frequent_peak_idx

    def relative_gain(self, filename, my_analysis_options):
        """
        Analyze the relative gain of the cone, bipolar, and ganglion cell responses.

        The cone and bipolar responses are available with the subunit temporal model.
        """

        exp_variables = my_analysis_options["exp_variables"]
        cond_names_string = "_".join(exp_variables)
        data_folder = self.context.output_folder
        experiment_df = self.data_io.get_data(filename=filename)
        cond_names = experiment_df.index.values
        n_sweeps_vec = pd.to_numeric(experiment_df.loc[:, "n_sweeps"].values)
        gc_type = self.context.retina_parameters["gc_type"]
        response_type = self.context.retina_parameters["response_type"]

        t_start = my_analysis_options["t_start_ana"]
        t_end = my_analysis_options["t_end_ana"]
        fps_vec = pd.to_numeric(experiment_df.loc[:, "fps"].values)
        assert np.all(fps_vec == fps_vec[0]), "Not equal fps, aborting..."
        tp_idx = np.arange(t_start * fps_vec[0], t_end * fps_vec[0], dtype=int)
        n_tp = len(tp_idx)

        # Assert for equal number of trials
        assert np.all(
            n_sweeps_vec == n_sweeps_vec[0]
        ), "Not equal number of trials, aborting..."
        n_sweeps = n_sweeps_vec[0]

        columns = cond_names.tolist()
        scalers = {}
        # Loop conditions
        for idx, cond_name in enumerate(cond_names):
            # Define the pattern for the filename
            pattern = f"Response_{gc_type}_{response_type}_{cond_name}_*.npz"
            data_fullpath = self.data_io.most_recent_pattern(data_folder, pattern)

            df_index_start = idx * n_tp
            df_index_end = (idx + 1) * n_tp

            data_npz = self.data_io.get_data(data_fullpath)
            available_data = [f for f in data_npz.files if "allow_pickle" not in f]
            if idx == 0:
                df = pd.DataFrame(
                    index=range(n_tp * len(cond_names)),
                    columns=["time", cond_names_string, cond_names_string + "_R"]
                    + available_data,
                )

            df.iloc[df_index_start:df_index_end, 0] = np.arange(
                t_start, t_end, 1 / fps_vec[0]
            )

            bg_lum = float(experiment_df.loc[cond_name, "background"])

            bg_R = self.get_photoisomerizations_from_luminance(bg_lum)

            df.iloc[df_index_start:df_index_end, 1] = bg_lum
            df.iloc[df_index_start:df_index_end, 2] = bg_R

            for data_idx, this_data in enumerate(available_data):
                raw_values = data_npz[this_data].mean(axis=0)[tp_idx]
                zeroed_values = raw_values - raw_values[0]
                per_R_values = zeroed_values / bg_R
                if idx == 0:
                    max_response = per_R_values.max()
                    min_response = per_R_values.min()
                    scalers[this_data] = (max_response, min_response)
                norm_values = (per_R_values - scalers[this_data][1]) / (
                    scalers[this_data][0] - scalers[this_data][1]
                )
                df.iloc[df_index_start:df_index_end, 3 + data_idx] = norm_values

        # Save results
        suffix = "_".join(data[:3] for data in available_data)
        filename_out = (
            f"exp_results_{gc_type}_{response_type}_{cond_names_string}_{suffix}.csv"
        )
        csv_save_path = data_folder / filename_out
        df.to_csv(csv_save_path)

    def response_vs_background(self, filename, my_analysis_options):
        """ """
        exp_variables = my_analysis_options["exp_variables"]
        cond_names_string = "_".join(exp_variables)
        data_folder = self.context.output_folder
        experiment_df = self.data_io.get_data(filename=filename)
        cond_names = experiment_df.index.values
        gc_type = self.context.retina_parameters["gc_type"]
        response_type = self.context.retina_parameters["response_type"]

        t_start = my_analysis_options["t_start_ana"]
        t_end = my_analysis_options["t_end_ana"]
        fps_vec = pd.to_numeric(experiment_df.loc[:, "fps"].values)
        assert np.all(fps_vec == fps_vec[0]), "Not equal fps, aborting..."
        tp_idx = np.arange(t_start * fps_vec[0], t_end * fps_vec[0], dtype=int)

        # Prepare columns for the dataframe
        background_str = np.unique(
            experiment_df.loc[:, ["background"]].values.flatten()
        ).tolist()
        unique_background = [float(s) for s in background_str]

        # Prepare rows for the dataframe
        intensities_str = experiment_df.loc[:, ["intensity"]].values.flatten()
        intensities = [ast.literal_eval(s) for s in intensities_str]
        # Assuming that the index 1 in each tuple is the flash intensity
        unique_intensity = np.unique([x[1] for x in intensities]).tolist()

        # Define the pattern for the filename
        pattern = f"Response_{gc_type}_{response_type}_{cond_names[0]}_*.npz"
        data_fullpath = self.data_io.most_recent_pattern(data_folder, pattern)

        data_npz = self.data_io.get_data(data_fullpath)
        A_pupil = data_npz["A_pupil"]
        lambda_nm = data_npz["lambda_nm"]
        available_data = [
            f
            for f in data_npz.files
            if f not in ["A_pupil", "lambda_nm", "allow_pickle"]
        ]
        df = pd.DataFrame(
            index=range(len(unique_intensity) * len(unique_background)),
            columns=[
                "background",
                "background_R",
                "flash",
                "flash_R",
            ]
            + available_data,
        )

        # Loop conditions
        for idx, cond_name in enumerate(cond_names):
            # Define the pattern for the filename
            pattern = f"Response_{gc_type}_{response_type}_{cond_name}_*.npz"

            # Independent variables
            background = float(experiment_df.loc[cond_name, "background"])
            flash = ast.literal_eval(experiment_df.loc[cond_name, "intensity"])[1]
            baseline_start_seconds = float(
                experiment_df.loc[cond_name, "baseline_start_seconds"]
            )
            baseline_start_tp = int(baseline_start_seconds * fps_vec[0])
            baseline_ixd = np.arange(baseline_start_tp)

            # Dependent variables
            data_fullpath = self.data_io.most_recent_pattern(data_folder, pattern)
            data_npz = self.data_io.get_data(data_fullpath)

            df.iloc[idx, 0] = background
            df.iloc[idx, 1] = int(
                self.get_photoisomerizations_from_luminance(
                    background, A_pupil=A_pupil, lambda_nm=lambda_nm
                )
            )
            df.iloc[idx, 2] = flash
            df.iloc[idx, 3] = int(
                self.get_photoisomerizations_from_luminance(
                    flash, A_pupil=A_pupil, lambda_nm=lambda_nm
                )
            )

            for this_data in available_data:
                response = data_npz[this_data]
                try:
                    r = response[:, tp_idx]
                except IndexError:
                    raise IndexError(
                        "Response data does not match time points, did you forget to redo the stimuli? Aborting..."
                    )
                bl_mean = response[:, baseline_ixd].mean(axis=1)[:, np.newaxis]
                r_abs = np.abs(r - bl_mean)
                r_argmax = r_abs.argmax(axis=1)
                r_max = r[:, r_argmax]
                df[this_data][idx] = r_max.mean()

        # Save results
        filename_out = (
            f"exp_results_{gc_type}_{response_type}_response_vs_background.csv"
        )
        csv_save_path = data_folder / filename_out
        df.to_csv(csv_save_path)
