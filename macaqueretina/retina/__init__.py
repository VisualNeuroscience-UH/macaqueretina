from macaqueretina.project.project_manager_module import _construct_retina_instance

_cached_retina_instance = None

def build_retina(return_objects_do_not_save=False):
    """Wrapper for ConstructRetina.build_retina_client()."""
    global _cached_retina_instance
    from macaqueretina import config as current_config

    if current_config is None:
        print("Configuration not found. Run mr.load_parameters() before accessing mr.build_retina().")
        return []
    
    current_hash = current_config.hash()
    if _cached_retina_instance is None or _cached_retina_instance[0] != current_hash:
        retina_instance = _construct_retina_instance(current_config)
        _cached_retina_instance = (current_hash, retina_instance)
    return _cached_retina_instance[1].build_retina_client(return_objects_do_not_save)

def save_retina(ret, gc):
    """Wrapper for ConstructRetina.save_retina()."""
    global _cached_retina_instance
    from macaqueretina import config as current_config

    if current_config is None:
        print("Configuration not found. Run mr.load_parameters() before accessing mr.save_retina().")
        return []
    
    current_hash = current_config.hash()
    if _cached_retina_instance is None or _cached_retina_instance[0] != current_hash:
        retina_instance = _construct_retina_instance(current_config)
        _cached_retina_instance = (current_hash, retina_instance)
    return _cached_retina_instance[1].save_retina(ret, gc)
