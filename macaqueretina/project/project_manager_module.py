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
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from macaqueretina.analysis.analysis_module import Analysis
    from macaqueretina.data_io.config_io import Configuration
    from macaqueretina.data_io.data_io_module import DataIO
    from macaqueretina.retina.construct_retina_module import ConstructRetina
    from macaqueretina.retina.retina_math_module import RetinaMath
    from macaqueretina.retina.vae_module import RetinaVAE
    from macaqueretina.stimuli.visual_stimulus_module import VisualStimulus
    from macaqueretina.viz.viz_module import Viz, VizResponse


warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)

## Internal


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


def create_data_io_instance(config: Configuration) -> DataIO:
    """Instantiates DataIO with config."""
    from macaqueretina.data_io.data_io_module import DataIO

    return DataIO(config)


def create_retina_math_instance() -> RetinaMath:
    """Instantiates RetinaMath."""
    from macaqueretina.retina.retina_math_module import RetinaMath

    return RetinaMath()


def _create_get_xy_from_npz() -> Callable:
    """Provides get_xy_from_npz from the ProjectUtilitiesMixin."""
    from macaqueretina.project.project_utilities_module import ProjectUtilitiesMixin

    utilities = ProjectUtilitiesMixin()
    return utilities.get_xy_from_npz


def create_analysis_instance(
    config: Configuration,
    data_io: DataIO | None = None,
    retina_math: RetinaMath | None = None,
) -> Analysis:
    """
    Instantiates Analysis.
    """
    from macaqueretina.analysis.analysis_module import Analysis

    if data_io is None:
        data_io = create_data_io_instance(config)

    if retina_math is None:
        retina_math = create_retina_math_instance()

    return Analysis(
        config,
        data_io,
        pol2cart=retina_math.pol2cart,
        get_photoisomerizations_from_luminance=retina_math.get_photoisomerizations_from_luminance,
    )


def create_viz_instance(
    config: Configuration,
    data_io: DataIO | None = None,
    project_data: ProjectData | None = None,
) -> Viz:
    """Instantiates Viz."""
    from macaqueretina.viz.viz_module import Viz

    if data_io is None:
        data_io = create_data_io_instance(config)
    if project_data is None:
        project_data = ProjectData()

    retina_math = create_retina_math_instance()

    analysis = create_analysis_instance(config, data_io, retina_math)

    return Viz(
        config,
        data_io,
        project_data,
        analysis,
        DoG2D_fixed_surround=retina_math.DoG2D_fixed_surround,
        DoG2D_independent_surround=retina_math.DoG2D_independent_surround,
        DoG2D_circular=retina_math.DoG2D_circular,
        pol2cart=retina_math.pol2cart,
        sector2area_mm2=retina_math.sector2area_mm2,
        interpolate_data=retina_math.interpolate_data,
        lorenzian_function=retina_math.lorenzian_function,
    )


def create_retina_vae_instance(config: Configuration) -> RetinaVAE:
    from macaqueretina.retina.vae_module import RetinaVAE

    return RetinaVAE(config)


def create_construct_retina_instance(config: Configuration) -> ConstructRetina:
    from macaqueretina.retina.construct_retina_module import ConstructRetina
    from macaqueretina.retina.fit_module import Fit

    data_io = create_data_io_instance(config)
    project_data = ProjectData()
    viz = create_viz_instance(config, data_io, project_data)
    fit = Fit(project_data, config.experimental_metadata)
    retina_vae_instance = create_retina_vae_instance(config)
    retina_math = create_retina_math_instance()
    get_xy_from_npz = _create_get_xy_from_npz()

    return ConstructRetina(
        config,
        data_io,
        viz,
        fit,
        retina_vae_instance,
        retina_math,
        project_data,
        get_xy_from_npz,
    )


def create_viz_response_instance(
    config: Configuration,
    data_io: DataIO | None = None,
    project_data: ProjectData | None = None,
) -> VizResponse:
    from macaqueretina.retina.simulate_retina_module import VisualSignal
    from macaqueretina.viz.viz_module import VizResponse

    data_io = create_data_io_instance(config)
    project_data = ProjectData()

    return VizResponse(
        config,
        data_io,
        project_data,
        VisualSignal,
    )


def create_data_sampler_instance(
    filename, min_X, max_X, min_Y, max_Y, logX=False, logY=False
):
    """DataSampler alias."""
    from macaqueretina.project.project_utilities_module import DataSampler

    return DataSampler(filename, min_X, max_X, min_Y, max_Y, logX, logY)


def create_visual_stimulus_instance(
    config: Configuration, data_io: DataIO | None = None
):
    from macaqueretina.stimuli.visual_stimulus_module import VisualStimulus

    if data_io is None:
        data_io = create_data_io_instance(config)
    get_xy_from_npz = _create_get_xy_from_npz()

    return VisualStimulus(config, data_io, get_xy_from_npz)


def create_simulate_retina_instance(
    config: Configuration,
    data_io: DataIO | None = None,
    visual_stimulus_instance: VisualStimulus | None = None,
):
    from macaqueretina.retina.simulate_retina_module import SimulateRetina

    if data_io is None:
        data_io = create_data_io_instance(config)
    project_data = ProjectData()
    retina_math = create_retina_math_instance()
    if visual_stimulus_instance is None:
        visual_stimulus_instance = create_visual_stimulus_instance(config, data_io)

    return SimulateRetina(
        config,
        data_io,
        project_data,
        retina_math,
        visual_stimulus_instance,
    )


def create_experiment_instance(config):
    from macaqueretina.stimuli.experiment_module import Experiment

    data_io = create_data_io_instance(config)
    visual_stimulus_instance = create_visual_stimulus_instance(config, data_io)
    simulate_retina_instance = create_simulate_retina_instance(
        config, data_io, visual_stimulus_instance
    )
    return Experiment(
        config, data_io, visual_stimulus_instance, simulate_retina_instance
    )


## Exposed


def load_parameters() -> Configuration:
    """Load configuration parameters."""
    from macaqueretina.data_io.config_io import load_yaml

    project_manager_module_file_path = Path(__file__).resolve()
    git_repo_root_path = project_manager_module_file_path.parent.parent

    parameters_folder: Path = git_repo_root_path.joinpath("parameters/")
    yaml_files = list(parameters_folder.glob("*.yaml"))
    validate_params: Callable | None = _get_validation_params_method(parameters_folder)

    config: Configuration = load_yaml(yaml_files)

    config.project_manager_module_file_path = project_manager_module_file_path
    config.git_repo_root_path = git_repo_root_path

    if validate_params:
        config = validate_params(config)

    config["retina_parameters"].update(config["retina_parameters_extend"])

    import numpy as np

    np.random.seed(config.numpy_seed)

    return config


class ProjectData:
    """
    This is a singleton container for project piping data for internal use, such as
    visualizations.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.construct_retina = {}
            cls._instance.simulate_retina = {}
            cls._instance.fit = {}
        return cls._instance
