from macaqueretina.project.project_manager_module import create_data_io_instance

_cached_data_io = None


def load_data(
    filename=None,
    substring=None,
    exclude_substring=None,
    return_filename=False,
    full_path=None,
):
    """Wrapper for DataIO.load_data()."""
    global _cached_data_io
    from macaqueretina import config as current_config

    if current_config is None:
        print(
            "Configuration not found. Run mr.load_parameters() before accessing mr.load_data()."
        )
        return []

    current_hash = current_config.hash()
    if _cached_data_io is None or _cached_data_io[0] != current_hash:
        data_io_instance = create_data_io_instance(current_config)
        _cached_data_io = (current_hash, data_io_instance)
    return _cached_data_io[1].load_data(
        filename,
        substring,
        exclude_substring,
        return_filename,
        full_path,
    )
