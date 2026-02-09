from macaqueretina.project.project_manager_module import _construct_retina_instance, _simulate_retina_instance

_cached_construct_retina = None

def build_retina(return_objects_do_not_save=False):
    """Wrapper for ConstructRetina.build_retina_client()."""
    global _cached_construct_retina
    from macaqueretina import config as current_config

    if current_config is None:
        print("Configuration not found. Run mr.load_parameters() before accessing mr.build_retina().")
        return []
    
    current_hash = current_config.hash()
    if _cached_construct_retina is None or _cached_construct_retina[0] != current_hash:
        retina_instance = _construct_retina_instance(current_config)
        _cached_construct_retina = (current_hash, retina_instance)
    return _cached_construct_retina[1].build_retina_client(return_objects_do_not_save)

def save_retina(ret, gc):
    """Wrapper for ConstructRetina.save_retina()."""
    global _cached_construct_retina
    from macaqueretina import config as current_config

    if current_config is None:
        print("Configuration not found. Run mr.load_parameters() before accessing mr.save_retina().")
        return []
    
    current_hash = current_config.hash()
    if _cached_construct_retina is None or _cached_construct_retina[0] != current_hash:
        retina_instance = _construct_retina_instance(current_config)
        _cached_construct_retina = (current_hash, retina_instance)
    return _cached_construct_retina[1].save_retina(ret, gc)

_cached_simulate_retina = None

def simulate_retina(stimulus=None, filename=None, impulse=None, unity=None):
    """Wrapper for SimulateRetina.client()."""
    # TODO: change "client" name. Not informative and not a client
    global _cached_simulate_retina
    from macaqueretina import config as current_config

    if current_config is None:
        print("Configuration not found. Run mr.load_parameters() before accessing mr.simulate_retina().")
        return []
    
    current_hash = current_config.hash()
    if _cached_simulate_retina is None or _cached_simulate_retina[0] != current_hash:
        retina_instance = _simulate_retina_instance(current_config)
        _cached_simulate_retina = (current_hash, retina_instance)
    return _cached_simulate_retina[1].client(stimulus, filename, impulse, unity)