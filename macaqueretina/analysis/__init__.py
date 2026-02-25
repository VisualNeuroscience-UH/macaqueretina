from macaqueretina.project.project_manager_module import create_analysis

_cached_analysis_instance = None


def __getattr__(name):
    global _cached_analysis_instance

    from macaqueretina import config as current_config

    if current_config is None:
        raise AttributeError(
            f"Cannot access attribute '{name}' without configuration. Run mr.load_parameters() first."
        )

    current_hash = current_config.hash()

    if (
        _cached_analysis_instance is None
        or _cached_analysis_instance[0] != current_hash
    ):
        analysis_instance = create_analysis(current_config)
        _cached_analysis_instance = (current_hash, analysis_instance)

    return getattr(_cached_analysis_instance[1], name)


def __dir__():
    global _cached_analysis_instance

    from macaqueretina import config as current_config

    if current_config is None:
        print(
            "Configuration not found. Run mr.load_parameters() before accessing mr.analysis."
        )
        return []

    current_hash = current_config.hash()

    if (
        _cached_analysis_instance is None
        or _cached_analysis_instance[0] != current_hash
    ):
        analysis_instance = create_analysis(current_config)
        _cached_analysis_instance = (current_hash, analysis_instance)

    return dir(_cached_analysis_instance[1])
