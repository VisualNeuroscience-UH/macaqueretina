"""
Module on retina management. When this module is called directly, it runs
the core_parameters.yaml defined 'run' pipeline.

We use dependency injection to make the code more modular and easier to test.
It means that during construction here at the manager level, we can inject
an object instance to constructor of a "client", which becomes an attribute
of the instance.
"""

from __future__ import annotations

# Built-in
import runpy
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

# Third-party
import numpy as np

# Local
from macaqueretina.analysis.analysis_module import Analysis
from macaqueretina.data_io.config_io import load_yaml_as_dict
from macaqueretina.data_io.data_io_module import DataIO
from macaqueretina.parameters.param_reorganizer import ParamReorganizer
from macaqueretina.project.project_utilities_module import (
    DataSampler,
    ProjectUtilitiesMixin,
)
from macaqueretina.retina.construct_retina_module import ConstructRetina
from macaqueretina.retina.fit_module import Fit
from macaqueretina.retina.retina_math_module import RetinaMath
from macaqueretina.retina.simulate_retina_module import (
    ReceptiveFieldsBase,
    SimulateRetina,
    VisualSignal,
)
from macaqueretina.retina.vae_module import RetinaVAE
from macaqueretina.stimuli.experiment_module import Experiment
from macaqueretina.stimuli.visual_stimulus_module import AnalogInput, VisualStimulus
from macaqueretina.viz.viz_module import Viz, VizResponse

if TYPE_CHECKING:
    from macaqueretina.data_io.config_io import Configuration


warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _get_validation_params_method(parameters_folder: Path) -> Callable | None:
    """
    Get parameter validation method if a .py file with 'validation' in its name
    is found in the parameters/ subfolder.

    Returns:
        Callable or None: validation function () if found, None otherwise
    """
    validation_files = list(parameters_folder.glob("*validation*.py"))
    match len(validation_files):
        case 0:
            print(
                f"No validation file provided in {parameters_folder}. "
                f"Proceeding without parameter validation."
            )
            return None
        case 1:
            try:
                # Local
                from macaqueretina.parameters.param_validation import validate_params

                return validate_params
            except ImportError as e:
                print(f"Could not import validation file: {e}")
                return None
        case n:
            raise ValueError(
                f"Expected at most 1 validation file in {parameters_folder}, but found {n} files"
                f" with 'validation' in their name:"
                f"{[file.name for file in validation_files]}"
            )


def run_core_parameter_pipeline(PM: ProjectManager) -> None:
    """Runs the pipeline(s) chosen in the core_parameters.yaml file."""
    run = PM.config.run
    if run.build_retina:
        PM.construct_retina.build_retina_client()
    if run.make_stimulus:
        PM.stimulate.make_stimulus_video()
    if run.simulate_retina:
        PM.simulate_retina.client()
    if run.visualize_DoG_img_grid.show:
        options = run.visualize_DoG_img_grid
        PM.viz.show_DoG_img_grid(
            gc_list=options.gc_list,
            n_samples=options.n_samples,
            savefigname=options.savefigname,
        )
    if run.visualize_all_gc_responses.show:
        options = run.visualize_all_gc_responses
        PM.viz.show_all_gc_responses()
    if run.experiment.run_experiment:
        experiment_script_path = run.experiment.script_path
        runpy.run_path(experiment_script_path, run_name="__main__")


class ProjectData:
    """
    This is a container for project piping data for internal use, such as visualizations.
    """

    def __init__(self) -> None:
        self.construct_retina = {}
        self.simulate_retina = {}
        self.fit = {}


class ProjectManager(ProjectUtilitiesMixin):
    def __init__(self, yaml_path=None):
        """
        Main project manager.
        In init we construct other classes and inject necessary dependencies.
        This class is allowed to house project-dependent data and methods.
        """
        self.project_manager_module_file_path = Path(__file__).resolve()
        self.git_repo_root_path = self.project_manager_module_file_path.parent.parent

        if yaml_path is None:
            self.parameters_folder = self.git_repo_root_path.joinpath("parameters/")
        else:
            self.parameters_folder = yaml_path

        self.original_config: dict = self.load_parameters()

        self._retina_extend_keys = set(
            (self.original_config.get("retina_parameters_extend") or {}).keys()
        )
        self.config: Configuration = self.validate_parameters(self.original_config)

        data_io = DataIO(self.config)
        self.data_io = data_io

        self.project_data = ProjectData()

        self.retina_math = RetinaMath()

        ana = Analysis(
            # Dependencies
            self.config,
            data_io,
            # Methods which are needed also elsewhere
            pol2cart=self.retina_math.pol2cart,
            get_photoisomerizations_from_luminance=self.retina_math.get_photoisomerizations_from_luminance,
        )

        self.ana = ana

        viz = Viz(
            # Dependencies
            self.config,
            data_io,
            self.project_data,
            ana,
            # Methods which are needed also elsewhere
            DoG2D_fixed_surround=self.retina_math.DoG2D_fixed_surround,
            DoG2D_independent_surround=self.retina_math.DoG2D_independent_surround,
            DoG2D_circular=self.retina_math.DoG2D_circular,
            pol2cart=self.retina_math.pol2cart,
            sector2area_mm2=self.retina_math.sector2area_mm2,
            interpolate_data=self.retina_math.interpolate_data,
            lorenzian_function=self.retina_math.lorenzian_function,
        )
        self.viz = viz

        self.viz_spikes_with_stimulus = VizResponse(
            self.config,
            data_io,
            self.project_data,
            VisualSignal,
        )

        self.apply_changed_config()
        # Register: any future config mutation triggers re-validation + rebuild
        self.config.set_on_change(self._on_config_mutated)

    def apply_changed_config(self):
        fit = Fit(self.project_data, self.config.experimental_metadata)

        retina_vae = RetinaVAE(self.config)

        self.construct_retina = ConstructRetina(
            self.config,
            self.data_io,
            self.viz,
            fit,
            retina_vae,
            self.retina_math,
            self.project_data,
            self.get_xy_from_npz,
        )

        self.stimulate = VisualStimulus(self.config, self.data_io, self.get_xy_from_npz)
        self.simulate_retina = SimulateRetina(
            self.config,
            self.data_io,
            self.project_data,
            self.retina_math,
            self.stimulate,
        )

        self.experiment = Experiment(
            self.config, self.data_io, self.stimulate, self.simulate_retina
        )

        self.viz.construct_retina = self.construct_retina

        analog_input = AnalogInput(
            self.config,
            self.data_io,
            self.viz,
            ReceptiveFields=ReceptiveFieldsBase,
            pol2cart_df=self.simulate_retina.pol2cart_df,
            get_w_z_coords=self.simulate_retina.get_w_z_coords,
        )
        self.analog_input = analog_input

        self.data_sampler = DataSampler

        # Set numpy random seed
        np.random.seed(self.config.numpy_seed)

    def _set_in_dict(self, d: dict, path: tuple[str, ...], value: Any) -> None:
        cur = d
        for k in path[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[path[-1]] = value

    def _map_literature_view_key_to_raw_key(self, view_key: str, gc_type: str) -> str:
        """
        Runtime uses literature_data_files.<name>_path (and some non-_path keys).
        Raw YAML schema uses <name>_datafile[_<gc_type>] for file paths, plus some
        non-gc-type-specific keys like dendr_diam_units and gc_density_1_scaling_data_and_function.
        """
        # Convert *_path -> *_datafile
        if view_key.endswith("_path"):
            base = view_key[: -len("_path")] + "_datafile"
        else:
            base = view_key  # e.g. dendr_diam_units, gc_density_1_scaling_data_and_function

        # Prefer gc-type-specific key if it exists in the raw dict
        candidate = f"{base}_{gc_type}"
        if candidate in self.original_config:
            return candidate

        # Otherwise fall back to non-suffixed key if present (or create it)
        return base

    def _view_path_to_raw_path(
        self, root_cfg: Configuration, path: tuple[str, ...]
    ) -> tuple[str, ...]:
        """
        Convert a path in the reorganized runtime config back to the raw YAML schema path.
        """
        if not path:
            return path

        # 1) retina_parameters.* may actually belong to retina_parameters_extend.*
        if path[0] == "retina_parameters" and len(path) >= 2:
            k = path[1]
            if k in self._retina_extend_keys:
                return ("retina_parameters_extend",) + path[1:]
            return path  # stays under retina_parameters in raw schema too

        # 2) literature_data_files.* lives at top-level in raw schema (often gc-type-specific)
        if path[0] == "literature_data_files" and len(path) >= 2:
            view_key = path[1]
            gc_type = root_cfg.retina_parameters.gc_type  # current runtime value
            raw_key = self._map_literature_view_key_to_raw_key(view_key, gc_type)
            # literature_data_files is a "view-only" container; raw key is top-level
            return (raw_key,) + path[2:]

        # 3) everything else: assume same path in raw schema
        return path

    def _on_config_mutated(
        self, root_cfg: Configuration, path: tuple[str, ...], value: Any
    ) -> None:
        if not path:
            return

        raw_path = self._view_path_to_raw_path(root_cfg, path)

        # Write into RAW dict (YAML-shaped)
        self._set_in_dict(self.original_config, raw_path, value)

        # Re-validate from RAW dict (so required top-level keys exist)
        validated = self.validate_parameters(self.original_config)

        # Apply validated/reorganized results back into the live config in-place
        with root_cfg.mute_notifications():
            root_cfg.replace_from(validated)

        self.apply_changed_config()

    @property
    def data_io(self):
        return self._data_io

    @data_io.setter
    def data_io(self, value):
        if isinstance(value, DataIO):
            self._data_io = value
        else:
            raise AttributeError(
                "Trying to set improper data_io. Data_io must be a DataIO object."
            )

    @property
    def simulate_retina(self):
        return self._working_retina

    @simulate_retina.setter
    def simulate_retina(self, value):
        if isinstance(value, SimulateRetina):
            self._working_retina = value
        else:
            raise AttributeError(
                "Trying to set improper simulate_retina. simulate_retina must be a SimulateRetina instance."
            )

    @property
    def stimulate(self):
        return self._stimulate

    @stimulate.setter
    def stimulate(self, value):
        if isinstance(value, VisualStimulus):
            self._stimulate = value
        else:
            raise AttributeError(
                "Trying to set improper stimulate. stimulate must be a VisualStimulus instance."
            )

    @property
    def analog_input(self):
        return self._analog_input

    @analog_input.setter
    def analog_input(self, value):
        if isinstance(value, AnalogInput):
            self._analog_input = value
        else:
            raise AttributeError(
                "Trying to set improper analog_input. analog_input must be a AnalogInput instance."
            )

    def load_parameters(self) -> dict:
        """Load configuration parameters."""
        yaml_files = list(self.parameters_folder.glob("*.yaml"))
        original_config: dict = load_yaml_as_dict(yaml_files)
        return original_config

    def validate_parameters(
        self,
        original_config: dict,
    ) -> Configuration:
        validate_params: Callable | None = _get_validation_params_method(
            self.parameters_folder
        )
        original_config["git_repo_root_path"] = self.git_repo_root_path
        original_config["project_manager_module_file_path"] = (
            self.project_manager_module_file_path
        )

        if validate_params:
            validated_config = validate_params(
                original_config,
            )
            reorganizer = ParamReorganizer()
            validated_config: Configuration = reorganizer.reorganize(validated_config)

        return validated_config
