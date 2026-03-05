from macaqueretina.project.project_manager_module import (
    create_retina_constructor_instance,
    create_retina_math_instance,
    create_retina_simulator_instance,
)

_cached_retina_constructor_instance = None


class _RetinaConstructorWrapper:
    """Wrapper class that routes attribute access to the RetinaConstructor instance."""

    def __getattr__(self, name):
        global _cached_retina_constructor_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.retina_constructor."
            )

        current_hash = current_config.hash()
        if (
            _cached_retina_constructor_instance is None
            or _cached_retina_constructor_instance[0] != current_hash
        ):
            retina_constructor_instance = create_retina_constructor_instance(
                current_config
            )
            _cached_retina_constructor_instance = (
                current_hash,
                retina_constructor_instance,
            )

        return getattr(_cached_retina_constructor_instance[1], name)

    def __dir__(self):
        global _cached_retina_constructor_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.retina_constructor."
            )

        current_hash = current_config.hash()
        if (
            _cached_retina_constructor_instance is None
            or _cached_retina_constructor_instance[0] != current_hash
        ):
            retina_constructor_instance = create_retina_constructor_instance(
                current_config
            )
            _cached_retina_constructor_instance = (
                current_hash,
                retina_constructor_instance,
            )

        return dir(_cached_retina_constructor_instance[1])


retina_constructor = _RetinaConstructorWrapper()


_cached_retina_simulator_instance = None


class _RetinaSimulatorWrapper:
    """Wrapper class that routes attribute access to the RetinaConstructor instance."""

    def __getattr__(self, name):
        global _cached_retina_simulator_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.retina_simulator."
            )

        current_hash = current_config.hash()
        if (
            _cached_retina_simulator_instance is None
            or _cached_retina_simulator_instance[0] != current_hash
        ):
            retina_simulator_instance = create_retina_simulator_instance(current_config)
            _cached_retina_simulator_instance = (
                current_hash,
                retina_simulator_instance,
            )

        return getattr(_cached_retina_simulator_instance[1], name)

    def __dir__(self):
        global _cached_retina_simulator_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.retina_simulator."
            )

        current_hash = current_config.hash()
        if (
            _cached_retina_simulator_instance is None
            or _cached_retina_simulator_instance[0] != current_hash
        ):
            retina_simulator_instance = create_retina_simulator_instance(current_config)
            _cached_retina_simulator_instance = (
                current_hash,
                retina_simulator_instance,
            )

        return dir(_cached_retina_simulator_instance[1])


retina_simulator = _RetinaSimulatorWrapper()

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
