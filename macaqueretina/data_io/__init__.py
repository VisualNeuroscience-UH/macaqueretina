from macaqueretina.project.project_manager_module import create_data_io_instance

_cached_data_io_instance = None


class _DataIOWrapper:
    """Wrapper class that routes attribute access to the DataIO instance."""

    def __getattr__(self, name):
        global _cached_data_io_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.data_io."
            )

        current_hash = current_config.hash()
        if (
            _cached_data_io_instance is None
            or _cached_data_io_instance[0] != current_hash
        ):
            data_io_instance = create_data_io_instance(current_config)
            _cached_data_io_instance = (current_hash, data_io_instance)

        return getattr(_cached_data_io_instance[1], name)

    def __dir__(self):
        global _cached_data_io_instance
        from macaqueretina import config as current_config

        if current_config is None:
            raise AttributeError(
                "Configuration not found. Run mr.load_parameters() before accessing mr.data_io."
            )

        current_hash = current_config.hash()
        if (
            _cached_data_io_instance is None
            or _cached_data_io_instance[0] != current_hash
        ):
            data_io_instance = create_data_io_instance(current_config)
            _cached_data_io_instance = (current_hash, data_io_instance)

        return dir(_cached_data_io_instance[1])


data_io = _DataIOWrapper()
