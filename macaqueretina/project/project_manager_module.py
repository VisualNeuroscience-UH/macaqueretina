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
    from macaqueretina.data_io.config_io import Configuration
    from macaqueretina.data_io.data_io_module import DataIO
    from macaqueretina.retina.retina_math_module import RetinaMath
    from macaqueretina.analysis.analysis_module import Analysis
    from macaqueretina.retina.construct_retina_module import ConstructRetina
    from macaqueretina.retina.vae_module import RetinaVAE
    from macaqueretina.viz.viz_module import Viz
    from macaqueretina.retina.fit_module import Fit


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


def create_data_io(config: Configuration) -> DataIO:
    """Instantiates DataIO with config."""
    from macaqueretina.data_io.data_io_module import DataIO

    return DataIO(config)


def create_retina_math() -> RetinaMath:
    """Instantiates RetinaMath."""
    from macaqueretina.retina.retina_math_module import RetinaMath

    return RetinaMath()


def _create_get_xy_from_npz() -> Callable:
    """Provides get_xy_from_npz from the ProjectUtilitiesMixin."""
    from macaqueretina.project.project_utilities_module import ProjectUtilitiesMixin

    utilities = ProjectUtilitiesMixin()
    return utilities.get_xy_from_npz


def create_analysis(
    config: Configuration,
    data_io: DataIO | None = None,
    retina_math: RetinaMath | None = None,
) -> Analysis:
    """
    Instantiates Analysis.
    """
    from macaqueretina.analysis.analysis_module import Analysis

    if data_io is None:
        data_io = create_data_io(config)

    if retina_math is None:
        retina_math = create_retina_math()

    return Analysis(
        config,
        data_io,
        pol2cart=retina_math.pol2cart,
        get_photoisomerizations_from_luminance=retina_math.get_photoisomerizations_from_luminance,
    )


def create_viz(
    config, data_io=None, project_data=None, analysis=None, retina_math=None
):
    """Instantiates Viz."""
    from macaqueretina.viz.viz_module import Viz

    if data_io is None:
        data_io = create_data_io(config)
    if project_data is None:
        project_data = ProjectData()
    if analysis is None:
        analysis = create_analysis(config, data_io, retina_math)
    if retina_math is None:
        retina_math = create_retina_math()

    return Viz(
        config,
        data_io,
        project_data,
        analysis,
        # TODO: these should be unpacked inside Viz, not here
        DoG2D_fixed_surround=retina_math.DoG2D_fixed_surround,
        DoG2D_independent_surround=retina_math.DoG2D_independent_surround,
        DoG2D_circular=retina_math.DoG2D_circular,
        pol2cart=retina_math.pol2cart,
        sector2area_mm2=retina_math.sector2area_mm2,
        interpolate_data=retina_math.interpolate_data,
        lorenzian_function=retina_math.lorenzian_function,
    )


def retina_vae(config):
    return RetinaVAE(config)


def _construct_retina(
    config: Configuration,
    data_io: DataIO | None = None,
    viz: Viz | None = None,
    fit: Fit | None = None,
    retina_vae: RetinaVAE | None = None,
    retina_math: RetinaMath | None = None,
    project_data: ProjectData | None = None,
    get_xy_from_npz: Callable | None = None,
) -> ConstructRetina:
    from macaqueretina.retina.construct_retina_module import ConstructRetina
    from macaqueretina.retina.fit_module import Fit
    from macaqueretina.retina.vae_module import RetinaVAE

    if data_io is None:
        data_io = create_data_io(config)

    if project_data is None:
        project_data = ProjectData()

    if viz is None:
        viz = create_viz(config, data_io, project_data)

    if fit is None:
        fit = Fit(project_data, config.experimental_metadata)

    if retina_vae is None:
        retina_vae = RetinaVAE(config)

    if retina_math is None:
        retina_math = create_retina_math()

    if get_xy_from_npz is None:
        get_xy_from_npz = _create_get_xy_from_npz()

    return ConstructRetina(
        config,
        data_io,
        viz,
        fit,
        retina_vae,
        retina_math,
        project_data,
        get_xy_from_npz,
    )


def viz_spikes_with_stimulus(config, data_io, project_data):
    from macaqueretina.viz.viz_module import VizResponse
    from macaqueretina.retina.simulate_retina_module import VisualSignal

    if data_io is None:
        data_io = create_data_io(config)

    if project_data is None:
        project_data = ProjectData()

    return VizResponse(
        config,
        data_io,
        project_data,
        VisualSignal,
    )


def data_sampler():
    """DataSampler alias."""
    from macaqueretina.project.project_utilities_module import DataSampler

    return DataSampler


def analog_input(config, data_io, viz, simulate_retina):
    from macaqueretina.stimuli.visual_stimulus_module import AnalogInput
    from macaqueretina.retina.simulate_retina_module import ReceptiveFieldsBase

    if data_io is None:
        data_io = create_data_io(config)

    if viz is None:
        viz = create_viz(config, data_io)

    return AnalogInput(
        config,
        data_io,
        viz,
        ReceptiveFields=ReceptiveFieldsBase,
        pol2cart_df=simulate_retina.pol2cart_df,
        get_w_z_coords=simulate_retina.get_w_z_coords,
    )


def create_stimulus(config, data_io, get_xy_from_npz=None):
    from macaqueretina.stimuli.visual_stimulus_module import VisualStimulus

    if data_io is None:
        data_io = create_data_io(config)

    if get_xy_from_npz is None:
        return _create_get_xy_from_npz

    return VisualStimulus(config, data_io, get_xy_from_npz)


def create_simulate_retina(config, data_io, project_data, retina_math, stimulate):
    from macaqueretina.retina.simulate_retina_module import SimulateRetina

    if data_io is None:
        data_io = create_data_io(config)
    if project_data is None:
        project_data = ProjectData()
    if retina_math is None:
        retina_math = create_retina_math()
    if stimulate is None:
        stimulate = create_stimulus(config, data_io)

    return SimulateRetina(
        config,
        data_io,
        project_data,
        retina_math,
        stimulate,
    )


def experiment(config, data_io, stimulate, simulate_retina):
    from macaqueretina.stimuli.experiment_module import Experiment

    if data_io is None:
        data_io = create_data_io(config)
    if stimulate is None:
        stimulate = create_stimulus(config, data_io)
    if simulate_retina is None:
        simulate_retina = create_simulate_retina(config, data_io, None, None, stimulate)
    return Experiment(config, data_io, stimulate, simulate_retina)


## Exposed


def load_parameters() -> Configuration:
    """Load configuration parameters."""
    from macaqueretina.data_io.config_io import load_yaml
    from macaqueretina.parameters.param_reorganizer import ParamReorganizer

    project_manager_module_file_path = Path(__file__).resolve()
    git_repo_root_path = project_manager_module_file_path.parent.parent

    parameters_folder: Path = git_repo_root_path.joinpath("parameters/")
    yaml_files = list(parameters_folder.glob("*.yaml"))
    validate_params: Callable | None = _get_validation_params_method(parameters_folder)

    config: Configuration = load_yaml(yaml_files)

    config.project_manager_module_file_path = project_manager_module_file_path
    config.git_repo_root_path = git_repo_root_path

    if validate_params:
        validated_config = validate_params(config)
        reorganizer = ParamReorganizer()
        config = reorganizer.reorganize(validated_config)

    import numpy as np

    np.random.seed(config.numpy_seed)

    return config


def build_retina(config, construct_retina=None):
    if construct_retina is None:
        builder = _construct_retina(config)
        return builder.build_retina_client()
    return construct_retina.build_retina_client


def viz(config, data_io=None, project_data=None, analysis=None, retina_math=None):
    from macaqueretina.viz.viz_module import Viz

    if data_io is None:
        data_io = create_data_io(config)
    if project_data is None:
        project_data = ProjectData()
    if retina_math is None:
        retina_math = create_retina_math()
    if analysis is None:
        analysis = create_analysis(config, data_io, retina_math)

    return Viz(config, data_io, project_data, analysis)


class ProjectData:
    # TODO: this might need thread safety for cluster runs
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


def main():
    pass
    # start_time = time.time()
    # config = load_parameters()

    # if config.profile is True:
    #     import cProfile
    #     import pstats

    #     profiler = cProfile.Profile()
    #     profiler.enable()
    #     end_time = time.time()

    # PM = ProjectManager(config)

    # run_core_parameter_pipeline(PM, config)

    # end_time = time.time()
    # print(
    #     "Total time taken: ",
    #     time.strftime(
    #         "%H hours %M minutes %S seconds", time.gmtime(end_time - start_time)
    #     ),
    # )

    # plt.show()

    # if config.profile is True:
    #     profiler.disable()
    #     stats = pstats.Stats(profiler).sort_stats("tottime")
    #     stats.print_stats(20)


if __name__ == "__main__":
    """Run the core_parameters.yaml run pipeline items."""
    main()
