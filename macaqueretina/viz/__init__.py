from macaqueretina.project.project_manager_module import (
    create_viz_instance,
    create_viz_response_instance,
)

_cached_viz_instance = None


class _VizWrapper:
    """Wrapper class that routes attribute access to the Viz instance."""

    def __getattr__(self, name):
        global _cached_viz_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.viz."
            )

        current_hash = current_config.hash()
        if _cached_viz_instance is None or _cached_viz_instance[0] != current_hash:
            viz_instance = create_viz_instance(current_config)
            _cached_viz_instance = (current_hash, viz_instance)

        return getattr(_cached_viz_instance[1], name)

    def __dir__(self):
        global _cached_viz_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.viz."
            )

        current_hash = current_config.hash()
        if _cached_viz_instance is None or _cached_viz_instance[0] != current_hash:
            viz_instance = create_viz_instance(current_config)
            _cached_viz_instance = (current_hash, viz_instance)

        return dir(_cached_viz_instance[1])


viz = _VizWrapper()

_cached_viz_response_instance = None


class _VizResponseWrapper:
    """Wrapper class that routes attribute access to the VizResponse instance."""

    def __getattr__(self, name):
        global _cached_viz_response_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.viz_response."
            )

        current_hash = current_config.hash()
        if (
            _cached_viz_response_instance is None
            or _cached_viz_response_instance[0] != current_hash
        ):
            viz_response_instance = create_viz_response_instance(current_config)
            _cached_viz_response_instance = (current_hash, viz_response_instance)

        return getattr(_cached_viz_response_instance[1], name)

    def __dir__(self):
        global _cached_viz_response_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.viz_response."
            )

        current_hash = current_config.hash()
        if (
            _cached_viz_response_instance is None
            or _cached_viz_response_instance[0] != current_hash
        ):
            viz_response_instance = create_viz_response_instance(current_config)
            _cached_viz_response_instance = (current_hash, viz_response_instance)

        return dir(_cached_viz_response_instance[1])


viz_response = _VizResponseWrapper()
