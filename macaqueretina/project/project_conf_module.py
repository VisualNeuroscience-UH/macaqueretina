from __future__ import annotations

# Built-in
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable

# Third-party
import matplotlib.pyplot as plt

# Local
from macaqueretina.data_io.config_io import load_yaml
from macaqueretina.parameters.param_reorganizer import ParamReorganizer
from macaqueretina.project.project_manager_module import ProjectManager

if TYPE_CHECKING:
    # Local
    from macaqueretina.data_io.config_io import Configuration

start_time = time.time()
warnings.simplefilter("ignore")


def _get_validation_params_method(base: Path) -> Callable | None:
    """
    Get parameter validation method if a .py file with 'validation' in its name
    is found in the parameters/ subfolder.

    Returns:
        Callable or None: validation function () if found, None otherwise
    """
    validation_files = list(base.glob("*validation*.py"))
    match len(validation_files):
        case 0:
            print(
                f"No validation file provided in {base}. "
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
                f"Expected at most 1 validation file in {base}, but found {n} files"
                f" with 'validation' in their name:"
                f"{[file.name for file in validation_files]}"
            )


def load_parameters() -> Configuration:
    """Load configuration parameters. TODO from where?"""
    project_conf_module_file_path = Path(__file__).resolve()
    git_repo_root_path = project_conf_module_file_path.parent.parent

    base: Path = git_repo_root_path.joinpath("parameters/")
    yaml_files = list(base.glob("*.yaml"))

    validate_params: Callable | None = _get_validation_params_method(base)

    config: Configuration = load_yaml(yaml_files)

    if validate_params:
        validated_config = validate_params(
            config, project_conf_module_file_path, git_repo_root_path
        )
        reorganizer = ParamReorganizer()
        config = reorganizer.reorganize(validated_config)

    return config


def dispatcher(PM: ProjectManager, config: Configuration):
    # TODO what does this do?
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


def main():
    config = load_parameters()

    if config.profile is True:
        # Built-in
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        end_time = time.time()

    PM = ProjectManager(config)

    dispatcher(PM, config)

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
