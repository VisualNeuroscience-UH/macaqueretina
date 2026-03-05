from macaqueretina.project.project_manager_module import (
    create_construct_retina_instance,
    create_retina_math_instance,
    create_simulate_retina_instance,
)

_cached_construct_retina_instance = None


class _ConstructRetinaWrapper:
    """Wrapper class that routes attribute access to the ConstructRetina instance."""

    def __getattr__(self, name):
        global _cached_construct_retina_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.construct_retina."
            )

        current_hash = current_config.hash()
        if (
            _cached_construct_retina_instance is None
            or _cached_construct_retina_instance[0] != current_hash
        ):
            construct_retina_instance = create_construct_retina_instance(current_config)
            _cached_construct_retina_instance = (
                current_hash,
                construct_retina_instance,
            )

        return getattr(_cached_construct_retina_instance[1], name)

    def __dir__(self):
        global _cached_construct_retina_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.construct_retina."
            )

        current_hash = current_config.hash()
        if (
            _cached_construct_retina_instance is None
            or _cached_construct_retina_instance[0] != current_hash
        ):
            construct_retina_instance = create_construct_retina_instance(current_config)
            _cached_construct_retina_instance = (
                current_hash,
                construct_retina_instance,
            )

        return dir(_cached_construct_retina_instance[1])


construct_retina = _ConstructRetinaWrapper()


_cached_simulate_retina_instance = None


class _SimulateRetinaWrapper:
    """Wrapper class that routes attribute access to the ConstructRetina instance."""

    def __getattr__(self, name):
        global _cached_simulate_retina_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.simulate_retina."
            )

        current_hash = current_config.hash()
        if (
            _cached_simulate_retina_instance is None
            or _cached_simulate_retina_instance[0] != current_hash
        ):
            simulate_retina_instance = create_simulate_retina_instance(current_config)
            _cached_simulate_retina_instance = (current_hash, simulate_retina_instance)

        return getattr(_cached_simulate_retina_instance[1], name)

    def __dir__(self):
        global _cached_simulate_retina_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.simulate_retina."
            )

        current_hash = current_config.hash()
        if (
            _cached_simulate_retina_instance is None
            or _cached_simulate_retina_instance[0] != current_hash
        ):
            simulate_retina_instance = create_simulate_retina_instance(current_config)
            _cached_simulate_retina_instance = (current_hash, simulate_retina_instance)

        return dir(_cached_simulate_retina_instance[1])


simulate_retina = _SimulateRetinaWrapper()

_cached_retina_math_instance = None


class _RetinaMathWrapper:
    """Wrapper class that routes attribute access to the RetinaMath instance."""

    def __getattr__(self, name):
        global _cached_retina_math_instance
        if _cached_retina_math_instance is None:
            _cached_retina_math_instance = create_retina_math_instance()
        return getattr(_cached_retina_math_instance, name)

    def __dir__(self):
        global _cached_retina_math_instance
        if _cached_retina_math_instance is None:
            _cached_retina_math_instance = create_retina_math_instance()
        return dir(_cached_retina_math_instance)


retina_math = _RetinaMathWrapper()
