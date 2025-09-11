# Built-in
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import yaml

# Local
from macaqueretina.stimuli.visual_stimulus_module import VideoBaseClass


class RelevantStimulusParameters:
    """
    This class prepares relevant stimulus parameter subsets for each stimulus type.
    This info will be used in the experiment module.
    """

    relevant_stimulus_parameters = {}

    relevant_stimulus_parameters["common_parameters"] = [
        "image_width",
        "image_height",
        "pix_per_deg",
        "fps",
        "duration_seconds",
        "baseline_start_seconds",
        "baseline_end_seconds",
        "pattern",
        "stimulus_form",
        "stimulus_position",
        "stimulus_size",
        "contrast",
        "mean",
        "background",
        "stimulus_video_name",
        "n_sweeps",
        "logarithmic",
        "retina_center",
    ]
    relevant_stimulus_parameters["sine_grating"] = [
        "temporal_frequency",
        "spatial_frequency",
        "orientation",
        "phase_shift",
    ]
    relevant_stimulus_parameters["square_grating"] = [
        "temporal_frequency",
        "spatial_frequency",
        "orientation",
        "phase_shift",
    ]
    relevant_stimulus_parameters["temporal_sine_pattern"] = [
        "phase_shift",
    ]
    relevant_stimulus_parameters["temporal_square_pattern"] = [
        "phase_shift",
    ]
    relevant_stimulus_parameters["temporal_chirp_pattern"] = [
        "temporal_frequency_range",
        "phase_shift",
    ]
    relevant_stimulus_parameters["spatially_uniform_binary_noise"] = [
        "on_proportion",
        "on_time",
        "direction",
    ]
    relevant_stimulus_parameters["annulus"] = [
        "size_inner",
        "size_outer",
    ]

    @staticmethod
    def get_relations(options):
        """
        Only a subset of stimulus parameters are relevant for the current stimulus type.
        Select the relevant stimulus parameters as a key: value dictionary.

        Parameters
        ----------
        options : dict
            A dictionary containing all stimulus options.

        Returns
        -------
        dict
            A dictionary containing relevant stimulus options.
        """
        relevant_metadata = RelevantStimulusParameters.relevant_stimulus_parameters[
            "common_parameters"
        ]

        # Get relevant parameter keys according to values of pattern and form
        for param in options.values():
            if param in RelevantStimulusParameters.relevant_stimulus_parameters.keys():
                relevant_metadata += (
                    RelevantStimulusParameters.relevant_stimulus_parameters[param]
                )

        return sorted(relevant_metadata)


class Experiment(VideoBaseClass):
    """
    Build your experiment here
    """

    def __init__(
        self, context, data_io, stimulate, simulate_retina, get_cone_noise_hash
    ):
        super().__init__()

        self._context = context
        self._data_io = data_io
        self._stimulate = stimulate
        self._simulate_retina = simulate_retina
        self.get_cone_noise_hash = get_cone_noise_hash

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def stimulate(self):
        return self._stimulate

    @property
    def simulate_retina(self):
        return self._simulate_retina

    def _replace_options(self, input_options):
        # Replace with input options
        for this_key in input_options.keys():
            self.options[this_key] = input_options[this_key]

    def _meshgrid_conditions(self, options):
        # Get all conditions and their values
        conditions_to_meshgrid = list(options.keys())
        values_to_meshgrid = [
            options[condition] for condition in conditions_to_meshgrid
        ]

        # Get meshgrid of all values
        values = np.meshgrid(*values_to_meshgrid)

        # Flatten arrays for easier indexing later
        values_flat = [v.flatten() for v in values]

        # Get cond_names
        conditions_metadata_idx = np.meshgrid(
            *[np.arange(len(v)) for v in values_to_meshgrid]
        )
        cond_array_list = [v.flatten() for v in conditions_metadata_idx]

        # Get conditions to replace the corresponding options in the stimulus
        # list with N dicts, N = N experiments to run. Each of the N dicts contains all
        # condition:value pairs
        cond_options = []
        cond_names = []
        for dict_idx in range(len(values_flat[0])):
            this_dict = {}
            this_str = ""
            for condition_idx, this_condition in enumerate(conditions_to_meshgrid):
                this_dict[this_condition] = values_flat[condition_idx][dict_idx]
                str_idx = this_condition.find("_")
                if str_idx > 0:
                    other_letters = this_condition[str_idx + 1]
                else:
                    other_letters = ""
                this_str += (
                    this_condition[0]
                    + other_letters
                    + str(cond_array_list[condition_idx][dict_idx])
                )
            cond_options.append(this_dict)
            cond_names.append(this_str)

        return cond_options, cond_names

    def _get_cond_metadata_values(self, logarithmic, min_max_values, n_steps):
        """
        The values include n_steps between the corresponding min_max_values. The steps
        can be linear or logarithmic
        """
        if logarithmic:
            values = np.logspace(
                np.log10(min_max_values[0]),
                np.log10(min_max_values[1]),
                n_steps,
            )
        else:
            values = np.linspace(min_max_values[0], min_max_values[1], n_steps)

        return values

    def _generate_gaussian_distributions(self, cond_values, stats):
        # Unpack the provided stats dictionary
        samples = np.ceil(stats["sweeps"] / 2).astype(int)
        mean1, mean2 = stats["mean"]
        sd1, sd2 = stats["sd"]

        # Ensure that the means are distinct and within the range of cond_values
        if mean1 == mean2:
            raise ValueError("Means must be distinct.")
        if any(
            mean < min(cond_values) or mean > max(cond_values)
            for mean in (mean1, mean2)
        ):
            raise ValueError("Mean values must be within the range of cond_values.")

        # Generate samples from each Gaussian distribution with the specified number of sweeps
        rng = np.random.default_rng()
        gaussian_dist1 = mean1 + sd1 * rng.standard_normal(samples)
        gaussian_dist2 = mean2 + sd2 * rng.standard_normal(samples)

        #  Sample the real numbers to closest values in the array
        closest_values1 = cond_values[
            np.argmin(np.abs(gaussian_dist1[:, None] - cond_values), axis=1)
        ]
        closest_values2 = cond_values[
            np.argmin(np.abs(gaussian_dist2[:, None] - cond_values), axis=1)
        ]

        # Combine the two numpy arrays
        combined_distributions = np.hstack((closest_values1, closest_values2))

        return combined_distributions

    def _show_histogram(
        self, cond_metadata_key, exp_variables, min_max_values, n_steps
    ):
        # Third-party
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(len(exp_variables), 1, figsize=(10, 10))
        if len(exp_variables) == 1:
            ax = [ax]
        for idx, (exp_variable, this_value_pair, this_n_steps) in enumerate(
            zip(exp_variables, min_max_values, n_steps)
        ):
            vals = cond_metadata_key[exp_variable]
            bin_edges = np.linspace(
                this_value_pair[0], this_value_pair[1], this_n_steps
            )
            ax[idx].hist(vals, bin_edges, color="blue", alpha=0.7)
            ax[idx].set_title(exp_variable)
            ax[idx].set_xlabel("Values")
            ax[idx].set_ylabel("Frequency")

        plt.show()

    def _build(self, experiment_parameters, show_histogram=False):
        """
        Setup
        """

        exp_variables = experiment_parameters["exp_variables"]
        min_max_values = experiment_parameters["min_max_values"]
        n_steps = experiment_parameters["n_steps"]
        logarithmic = experiment_parameters["logarithmic"]  # True or False
        distributions = experiment_parameters["distributions"]

        # Create a dictionary with all options to vary. The keys are the options to vary,
        # the values include n_steps between the corresponding min_max_values. The steps
        # can be linear or logarithmic
        cond_metadata_key = {}
        for idx, option in enumerate(exp_variables):
            if isinstance(self.context.visual_stimulus_parameters[option], tuple):
                assert all(
                    isinstance(x, tuple)
                    for x in [logarithmic[idx], min_max_values[idx], n_steps[idx]]
                ), "If exp_variable in visual_stimulus_parameters is tuple, min_max_values, n_steps, and logarithmic must be tuples, aborting..."
                assert (
                    len(
                        set(
                            map(
                                len,
                                [logarithmic[idx], min_max_values[idx], n_steps[idx]],
                            )
                        )
                    )
                    == 1
                ), "If exp_variable in visual_stimulus_parameters is tuple, min_max_values, n_steps, and logarithmic must have the same length, aborting..."

                n_values = len(self.context.visual_stimulus_parameters[option])
                for value_idx in range(n_values):
                    value = self._get_cond_metadata_values(
                        logarithmic[idx][value_idx],
                        min_max_values[idx][value_idx],
                        n_steps[idx][value_idx],
                    )
                    # Expand options to cover each tuple value separately
                    cond_metadata_key[f"{option}_{value_idx}"] = value

            else:
                cond_metadata_key[option] = self._get_cond_metadata_values(
                    logarithmic[idx], min_max_values[idx], n_steps[idx]
                )

        # Return cond_options -- a dict with all keywords matching visual_stimulus_module.VisualStimulus
        # and values being a list of values to replace the corresponding keyword in the stimulus
        for idx, (distribution, stats) in enumerate(distributions.items()):
            match distribution:
                case "uniform":
                    pass
                case "gaussian":
                    cond_metadata_key[exp_variables[idx]] = (
                        self._generate_gaussian_distributions(
                            cond_metadata_key[exp_variables[idx]], stats
                        )
                    )
        if show_histogram:
            self._show_histogram(
                cond_metadata_key, exp_variables, min_max_values, n_steps
            )

        (
            cond_options,
            cond_names,
        ) = self._meshgrid_conditions(cond_metadata_key)

        # Some options contain a tuple of values
        # Collate expanded tuples back for each tuple option and condition
        cond_options_collated = [{} for _ in range(len(cond_options))]
        for idx, option in enumerate(exp_variables):
            assert (
                option in self.options.keys()
            ), f"Missing {option} in visual_stimulus_parameters, check exp_variables name..."

            for cond_idx, this_cond in enumerate(cond_options):
                if isinstance(self.context.visual_stimulus_parameters[option], tuple):
                    names, values = zip(
                        *[
                            [name, value]
                            for name, value in this_cond.items()
                            if option in name
                        ]
                    )

                    cond_options_collated[cond_idx][option] = values
                else:
                    cond_options_collated[cond_idx][option] = this_cond[option]

        return cond_options_collated, cond_names

    def _create_dataframe(self, cond_options, cond_names, options):
        """
        Create a DataFrame with the varying independent stimulus parameters.

        Parameters
        ----------
        cond_options : list of dict
            A list of dictionaries, where each dictionary contains key-value pairs for the varying stimulus parameter.
        cond_names : list of str
            A list of condition names to be used as columns in the DataFrame.
        options : dict
            A dictionary containing all stimulus option settings. Keys are option names, and values are option values.
            Special handling for the "background" key if it is a string.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with the index set to `cond_names` and columns set to `options.keys()`.
            Values are populated based on `options` and `cond_options`.
        """
        df = pd.DataFrame(index=cond_names, columns=options.keys())
        n_columns = len(cond_names)

        # Capture string background and convert to appropriate numerical value
        bg = options["background"]
        if isinstance(bg, str):
            match bg:
                case "mean":
                    options["background"] = options["mean"]
                case "intensity_max":
                    options["background"] = int(options["intensity"][1])
                case "intensity_min":
                    options["background"] = int(options["intensity"][0])

        # Set all values equal to options.values()
        for key, value in options.items():
            if isinstance(value, tuple):
                repeated_tuple = tuple([value] * n_columns)
                df.loc[:, key] = repeated_tuple
            else:
                df.loc[:, key] = value

        # Set independent variable values
        for idx, this_dict in enumerate(cond_options):
            for key, value in this_dict.items():
                df.loc[cond_names[idx], key] = value

        # In case background is a string, and the corresponding target value is the independent variable,
        # replace the background with the updated numerical value
        if isinstance(bg, str):
            for idx, this_dict in enumerate(cond_options):
                for key, value in this_dict.items():
                    if key == bg:
                        df.loc[cond_names[idx], "background"] = value

        return df

    def _invert_dataframe(self, df, exp_variables):
        """
        Invert the DataFrame to get the varying independent stimulus parameters.

        Parameters
        ----------
        df : pd.DataFrame
            A pandas DataFrame with the index set to the keys of `options` and columns set to `cond_names`.
            Values are populated based on `options` and `cond_options`.
        exp_variables : list of str
            A list of varying parameters in stimulus options.

        Returns
        -------
        cond_options : list of dict
            A list of dictionaries, where each dictionary contains key-value pairs for the varying stimulus parameter.
        cond_names : list of str
            A list of condition names to be used as columns in the DataFrame.
        options : dict
            A dictionary containing all stimulus option settings. Keys are option names, and values are option values.
            Special handling for the "background" key if it is a string.
        """
        cond_names = df.index.tolist()
        cond_options = []

        for col in cond_names:
            cond_options.append({})

        for row_idx, row in enumerate(cond_names):
            for col_idx, key in enumerate(exp_variables):
                val = df.loc[row, key]
                cond_options[row_idx][key] = val

        return cond_options, cond_names

    def _relevant_stimulus_options(self, visual_stimulus_parameters):
        """
        This method reads relevant stimulus parameter relations from path, and returns
        a new dictionary relevant_metadata.

        Parameters
        ----------
        visual_stimulus_parameters : dict
            A dictionary containing stimulus options.

        Returns
        -------
        relevant_metadata : dict
            A new dictionary containing relevant stimulus options.

        """
        relevant_stimulus_parameters = RelevantStimulusParameters.get_relations(
            visual_stimulus_parameters
        )
        relevant_metadata = {}
        for this_option in relevant_stimulus_parameters:
            relevant_metadata[this_option] = visual_stimulus_parameters[this_option]

        # Handle intensity, mean and contrast parameters. Intensity is a tuple of min and max values.
        # If intensity is found active, then mean and contrast are not needed.
        if visual_stimulus_parameters["intensity"] is not None:
            relevant_metadata.pop("contrast", None)
            relevant_metadata["intensity"] = visual_stimulus_parameters["intensity"]

        return relevant_metadata

    def build_and_run(
        self,
        experiment_parameters,
        build_without_run=False,
        show_histogram=False,
    ):

        exp_variables = experiment_parameters["exp_variables"]
        cond_names_string = "_".join(exp_variables)
        experiment_hash = self.context.generate_hash(experiment_parameters)
        filename_df = f"exp_metadata_{cond_names_string}_{experiment_hash}.csv"
        output_folder = self.context.output_folder
        save_exp_metadata_path = output_folder / filename_df
        if not save_exp_metadata_path.is_file():
            cond_options, cond_names = self._build(
                experiment_parameters, show_histogram=show_histogram
            )
        else:
            print(
                f"Experiment metadata hash already exists in {save_exp_metadata_path.name}, reusing..."
            )
            exp_metadata_df = pd.read_csv(save_exp_metadata_path, index_col=0)
            cond_options, cond_names = self._invert_dataframe(
                exp_metadata_df, exp_variables
            )

        """
        Unpack and run all conditions
        """

        # Update options to match visual_stimulus_parameters in conf file
        self._replace_options(self.context.visual_stimulus_parameters)

        # Replace filename with None. If don't want to save the stimulus, None is valid,
        # but if want to save, then filename will be generated in the loop below
        run_parameters = self.context.run_parameters
        self.options["n_sweeps"] = experiment_parameters["n_sweeps"]
        run_parameters["n_sweeps"] = experiment_parameters["n_sweeps"]

        # Replace with input options
        for idx, input_options in enumerate(cond_options):
            # Create stimulus video name. Note, this updates the cond_options dict
            stimulus_video_name = "Stim_" + cond_names[idx]
            input_options["stimulus_video_name"] = stimulus_video_name

            if not build_without_run:
                # Replace options with input_options
                self._replace_options(input_options)

                # Try loading existing file, if not found, create stimulus
                try:
                    stim = self.data_io.load_stimulus_from_videofile(
                        stimulus_video_name
                    )
                except FileNotFoundError:
                    stim = self.stimulate.make_stimulus_video(self.options)
                    self.options["raw_intensity"] = stim.options["raw_intensity"]

                gc_type = self.context.retina_parameters["gc_type"]
                response_type = self.context.retina_parameters["response_type"]
                filename_prefix = f"Response_{gc_type}_{response_type}_"
                filename = Path(output_folder) / (filename_prefix + cond_names[idx])

                # Run simulation
                self.simulate_retina.client(
                    stimulus=stim,
                    filename=filename,
                )

        if not save_exp_metadata_path.is_file():

            self.options["logarithmic"] = tuple(experiment_parameters["logarithmic"])
            self.options["retina_center"] = self.context.retina_parameters[
                "retina_center"
            ]
            relevant_metadata = self._relevant_stimulus_options(self.options)

            result_df = self._create_dataframe(
                cond_options, cond_names, relevant_metadata
            )

            # Check if path exists, create parents if not
            save_exp_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(save_exp_metadata_path)

        return save_exp_metadata_path.name
