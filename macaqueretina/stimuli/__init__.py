from macaqueretina.project.project_manager_module import (
    create_experiment_instance,
    create_visual_stimulus_instance,
)

_cached_experiment_instance = None


class _ExperimentWrapper:
    """Wrapper class that routes attribute access to the Experiment instance."""

    def __getattr__(self, name):
        global _cached_experiment_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.experiment."
            )

        current_hash = current_config.hash()
        if (
            _cached_experiment_instance is None
            or _cached_experiment_instance[0] != current_hash
        ):
            experiment_instance = create_experiment_instance(current_config)
            _cached_experiment_instance = (current_hash, experiment_instance)

        return getattr(_cached_experiment_instance[1], name)

    def __dir__(self):
        global _cached_experiment_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.experiment."
            )

        current_hash = current_config.hash()
        if (
            _cached_experiment_instance is None
            or _cached_experiment_instance[0] != current_hash
        ):
            experiment_instance = create_experiment_instance(current_config)
            _cached_experiment_instance = (current_hash, experiment_instance)

        return dir(_cached_experiment_instance[1])


experiment = _ExperimentWrapper()

_cached_visual_stimulus_instance = None


class _StimulusGeneratorWrapper:
    """Wrapper class that routes attribute access to the StimulusFactory instance."""

    def __getattr__(self, name):
        global _cached_visual_stimulus_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.stimulus_factory."
            )

        current_hash = current_config.hash()
        if (
            _cached_visual_stimulus_instance is None
            or _cached_visual_stimulus_instance[0] != current_hash
        ):
            visual_stimulus_instance = create_visual_stimulus_instance(current_config)
            _cached_visual_stimulus_instance = (current_hash, visual_stimulus_instance)

        return getattr(_cached_visual_stimulus_instance[1], name)

    def __dir__(self):
        global _cached_visual_stimulus_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.stimulus_factory."
            )

        current_hash = current_config.hash()
        if (
            _cached_visual_stimulus_instance is None
            or _cached_visual_stimulus_instance[0] != current_hash
        ):
            visual_stimulus_instance = create_visual_stimulus_instance(current_config)
            _cached_visual_stimulus_instance = (current_hash, visual_stimulus_instance)

        return dir(_cached_visual_stimulus_instance[1])


stimulus_factory = _StimulusGeneratorWrapper()
