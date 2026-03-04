from macaqueretina.project.project_manager_module import create_analysis_instance

_cached_analysis_instance = None


class _AnalysisWrapper:
    """Wrapper class that routes attribute access to the Analysis instance."""

    def __getattr__(self, name):
        global _cached_analysis_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.analysis."
            )

        current_hash = current_config.hash()
        if (
            _cached_analysis_instance is None
            or _cached_analysis_instance[0] != current_hash
        ):
            analysis_instance = create_analysis_instance(current_config)
            _cached_analysis_instance = (current_hash, analysis_instance)

        return getattr(_cached_analysis_instance[1], name)

    def __dir__(self):
        global _cached_analysis_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.analysis."
            )

        current_hash = current_config.hash()
        if (
            _cached_analysis_instance is None
            or _cached_analysis_instance[0] != current_hash
        ):
            analysis_instance = create_analysis_instance(current_config)
            _cached_analysis_instance = (current_hash, analysis_instance)

        return dir(_cached_analysis_instance[1])


analysis = _AnalysisWrapper()
