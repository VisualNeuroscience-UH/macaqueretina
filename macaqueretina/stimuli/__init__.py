from macaqueretina.project.project_manager_module import (
    create_experiment,
    create_stimulus,
)

_cached_stimulus_instance = None


def make_stimulus(options=None):
    """Wrapper for VisualStimulus.make_stimulus_video()"""
    global _cached_stimulus_instance
    from macaqueretina import config as current_config

    if current_config is None:
        print(
            "Configuration not found. Run mr.load_parameters() before accessing mr.make_stimulus_video."
        )
        return []

    current_hash = current_config.hash()
    if (
        _cached_stimulus_instance is None
        or _cached_stimulus_instance[0] != current_hash
    ):
        stimulus_instance = create_stimulus(current_config)
        _cached_stimulus_instance = (current_hash, stimulus_instance)
    return _cached_stimulus_instance[1].make_stimulus_video(options)


_cached_experiment_instance = None


def run_experiment(build_without_run=False, show_histogram=False):
    """Wrapper for Experiment.build_and_run()"""
    global _cached_experiment_instance
    from macaqueretina import config as current_config

    if current_config is None:
        print(
            "Configuration not found. Run mr.load_parameters() before accessing mr.run_experiment."
        )
        return []

    current_hash = current_config.hash()
    if (
        _cached_experiment_instance is None
        or _cached_experiment_instance[0] != current_hash
    ):
        experiment_instance = create_experiment(current_config)
        _cached_experiment_instance = (current_hash, experiment_instance)
    return _cached_experiment_instance[1].build_and_run(
        build_without_run, show_histogram
    )
