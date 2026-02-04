from macaqueretina.project.project_manager_module import create_viz

_cached_viz_instance = None


def __getattr__(name):
    global _cached_viz_instance
    
    from macaqueretina import config as current_config
    
    current_hash = current_config.hash()

    if _cached_viz_instance is None or _cached_viz_instance[0] != current_hash:
        viz_instance = create_viz(current_config)
        _cached_viz_instance = (current_hash, viz_instance)
    
    return getattr(_cached_viz_instance[1], name)


def __dir__():
    from macaqueretina import config

    return dir(create_viz(config))
