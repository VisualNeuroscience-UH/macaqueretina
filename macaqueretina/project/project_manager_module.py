"""
Module on retina management

We use dependency injection to make the code more modular and easier to test.
It means that during construction here at the manager level, we can inject
an object instance to constructor of a "client", which becomes an attribute
of the instance.
"""

from __future__ import annotations

# Built-in
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# Local
from macaqueretina.analysis.analysis_module import Analysis
from macaqueretina.data_io.config_io import load_yaml
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


warnings.simplefilter("ignore")


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


def _dispatcher(PM: ProjectManager, config: Configuration) -> None:
    """Runs the pipeline(s) chosen in the main yaml file."""
    run = config.run
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
    if run.visualize_all_gc_responses:
        options = run.visualize_all_gc_responses
        PM.viz.show_all_gc_responses()


def load_parameters() -> Configuration:
    """Load configuration parameters."""
    project_manager_module_file_path = Path(__file__).resolve()
    git_repo_root_path = project_manager_module_file_path.parent.parent

    parameters_folder: Path = git_repo_root_path.joinpath("parameters/")
    yaml_files = list(parameters_folder.glob("*.yaml"))
    validate_params: Callable | None = _get_validation_params_method(parameters_folder)

    config: Configuration = load_yaml(yaml_files)

    if validate_params:
        validated_config = validate_params(
            config, project_manager_module_file_path, git_repo_root_path
        )
        reorganizer = ParamReorganizer()
        config = reorganizer.reorganize(validated_config)

    return config


class ProjectData:
    """
    This is a container for project piping data for internal use, such as visualizations.
    """

    def __init__(self) -> None:
        self.construct_retina = {}
        self.simulate_retina = {}
        self.fit = {}


class ProjectManager(ProjectUtilitiesMixin):
    def __init__(self, config):
        """
        Main project manager.
        In init we construct other classes and inject necessary dependencies.
        This class is allowed to house project-dependent data and methods.
        """

        self.config = config

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

        self.construct_retina = self.build_retina_instance()

        self.viz.construct_retina = self.construct_retina

        experiment = Experiment(
            self.config, self.data_io, self.stimulate, self.simulate_retina
        )

        self.experiment = experiment

        analog_input = AnalogInput(
            self.config,
            self.data_io,
            viz,
            ReceptiveFields=ReceptiveFieldsBase,
            pol2cart_df=self.simulate_retina.pol2cart_df,
            get_w_z_coords=self.simulate_retina.get_w_z_coords,
        )
        self.analog_input = analog_input

        self.data_sampler = DataSampler

        # Set numpy random seed
        np.random.seed(self.config.numpy_seed)

    def build_retina_instance(self):
        project_data = ProjectData()

        fit = Fit(project_data, self.config.experimental_metadata)

        retina_vae = RetinaVAE(self.config)

        construct_retina = ConstructRetina(
            self.config,
            self.data_io,
            self.viz,
            fit,
            retina_vae,
            self.retina_math,
            project_data,
            self.get_xy_from_npz,
        )
        return construct_retina

        stimulate = VisualStimulus(self.config, self.data_io, self.get_xy_from_npz)
        self.stimulate = stimulate

        simulate_retina = SimulateRetina(
            self.config,
            self.data_io,
            self.project_data,
            self.retina_math,
            self.config.device,
            stimulate,
        )
        self.simulate_retina = simulate_retina

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


def main():
    start_time = time.time()
    config = load_parameters()

    if config.profile is True:
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        end_time = time.time()

    PM = ProjectManager(config)

    _dispatcher(PM, config)

    end_time = time.time()
    print(
        "Total time taken: ",
        time.strftime(
            "%H hours %M minutes %S seconds", time.gmtime(end_time - start_time)
        ),
    )

    plt.show()

    if config.profile is True:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("tottime")
        stats.print_stats(20)


if __name__ == "__main__":
    main()
