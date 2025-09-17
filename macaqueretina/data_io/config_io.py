"""
Load configuration parameters from YAML files into a ConfigManager object.

Notes
-----
Only the load_yaml façade is exposed; call:

>>> from .config.config_manager import load_yaml

to access all the module's functionalities.
"""

# Built-in
from pathlib import Path
from typing import Any

# Third-party
from yaml import YAMLError, safe_load


class YamlLoader:
    """
    Load and merge configuration data from multiple YAML files.

    Handles YAML file loading and merging while checking for
    duplicate keys across files.

    Parameters
    ----------
    yaml_paths : tuple of str
        Paths to YAML configuration files to load and merge.

    Attributes
    ----------
    yaml_paths : tuple of str
        Paths to YAML files to be loaded and merged.

    Examples
    --------
    >>> loader = YamlLoader(('config.yaml', 'override.yaml'))
    >>> config = loader.load_config()
    """

    def __init__(self, yaml_paths: tuple[str, ...]) -> None:
        self.yaml_paths = yaml_paths

    def load_config(self) -> dict[str, Any]:
        """
        Core method to load the configuration YAML file(s).

        Returns
        -------
        combined_config: dict
            Dictionary containing the configuration values

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist
        ValueError
            If the YAML is invalid or empty
        RuntimeError
            For other errors during loading
        """

        combined_config: dict[str, Any] = {}
        for path in self.yaml_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"YAML file not found: {path}")

            with open(path, "r") as file:
                try:
                    config = safe_load(file)
                    if not config:
                        raise ValueError(f"Configuration file is empty: {path}")
                    combined_config = self._merge_configs(combined_config, config)

                except YAMLError as e:
                    raise ValueError(f"Invalid YAML in configuration file {path}: {e}")
                except Exception as e:
                    if isinstance(e, ValueError):
                        raise
                    raise RuntimeError(
                        f"Failed to load required config from {path}: {e}"
                    )
        return combined_config

    def _merge_configs(
        self, combined_config: dict[str, Any], current_config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Merges multiple configuration YAML files into a single dictionary.
        Checks for duplicate keys and raises an error if found.

        Parameters
        ----------
        combined_config : dict
            The main configuration dictionary to merge into
        current_config : dict
            The currently-loaded configuration dictionary to merge

        Returns
        -------
        combined_config : dict
            The merged configuration dictionary

        Raises
        ------
        ValueError
            If duplicate keys are found in the configuration files
        """

        for key, _ in current_config.items():
            if key in combined_config:
                raise ValueError(f"Duplicate key '{key}' found in configuration files.")

        combined_config.update(current_config)

        return combined_config


class NestedConfig:
    """
    Allow optional attribute-style access to nested dictionary values.

    Notes
    -----
    Allows to access parameters throughout the codebase as, e.g.:
    self.context.unit_pos.TH.shape
    Instead of:
    self.context.unit_pos["TH"]["shape"]
    or:
    self.context.unit_pos.get("TH").get("shape").
    """

    def __init__(self, config_dict: dict[str, Any]) -> None:
        self._config = config_dict

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return NestedConfig(value)
            return value
        raise AttributeError(f"No attribute '{name}' found")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_config":
            super().__setattr__(name, value)
        else:
            self._config[name] = value

    def __getitem__(self, key: str) -> Any:
        if key in self._config:
            value = self._config[key]
            if isinstance(value, dict):
                return NestedConfig(value)
            return value
        raise KeyError(f"Configuration key not found: {key}")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style item assignment."""
        self._config[key] = value

    def __dir__(self) -> list[str]:
        """Return list of default attributes plus available keys for autocompletion."""
        default_attrs = list(object.__dir__(self))
        config_keys = list(self._config.keys())
        return sorted(set(default_attrs + config_keys))

    def __repr__(self):
        """Show string representation of the parameters in the ConfigManager object"""
        return f"NestedConfig({repr(self._config)})"

    def get(self, key: str, default=None) -> Any:
        """Backwards compatibility with dictionary get() method."""
        try:
            return self[key]
        except KeyError:
            return default


class ConfigManager:
    """
    Packages the parameters loaded in YamlLoader into a ConfigManager object.
    Provides both attribute-style and dictionary access to configuration values.

    Parameters
    ----------
    *args: str
        Path(s) to YAML configuration file(s) to load and merge.

    Attributes
    ----------
    config_file_path : str
        Path to the configuration file(s) to load
    _config : dict
        Internal dictionary storing parsed configuration values from YamlLoader
    """

    def __init__(self, *args: tuple[str, ...]) -> None:

        self.config_file_paths: tuple = args
        self._config = YamlLoader(self.config_file_paths).load_config()

    def __getattr__(self, name: str) -> Any:
        """
        Provides attribute-style access to configuration values.

        Parameters
        ----------
        name : str
            The configuration key to access

        Returns
        -------
        Any
            The configuration value when the key exists

        Raises
        ------
        AttributeError
            Whenever the requested key doesn't exist in the configuration
        """

        # Get the configuration value (if it exists)
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return NestedConfig(value)
            return value

        # Raise an error when requesting a non-existent attribute
        raise AttributeError(f"Configuration has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Provide attribute-style assignment to configuration values."""
        if name in ("config_file_paths", "_config"):
            super().__setattr__(name, value)
        else:

            self._config[name] = value

    def __getitem__(self, key: str) -> Any:
        """
        Provides dictionary-style access to configuration values.

        Parameters
        ----------
        key : str
            The configuration key to retrieve

        Returns
        -------
        Any
            The configuration value when the key exists
        """
        if key in self._config:
            return self.__getattr__(key)

        # Raise an error when requesting a non-existent key
        raise KeyError(f"Configuration key not found: {key}")

    def __dir__(self) -> list[str]:
        """Return list of default attributes plus available keys for autocompletion."""
        default_attrs = list(object.__dir__(self))
        config_keys = list(self._config.keys())
        return sorted(set(default_attrs + config_keys))

    def __repr__(self):
        """Show string representation of the parameters in the ConfigManager object"""
        return f"ConfigManager({repr(self._config)})"

    @property
    def config(self) -> dict[str, Any]:
        """
        Provides access to the whole configuration dictionary.

        Returns
        -------
        dict
            The configuration dictionary

        Notes
        -----
            Access as property (configuration.config).
        """

        return self._config

    def as_dict(self) -> dict[str, Any]:
        """
        Provides access to the whole configuration dictionary.

        Returns
        -------
        dict
            The configuration dictionary

        Notes
        -----
            Access as method (configuration.as_dict()).
        """

        return self._config


# Façade
def load_yaml(args: list) -> ConfigManager:
    """
    Load project configuration from one or more YAML files.

    Parameters
    ----------

    *args: list
        Paths to YAML configuration files to load and merge.

    Returns
    -------
    ConfigManager
        Configured ConfigManager instance providing access
        to the configuration parameters set in YAML files.

    Raises
    ------
    FileNotFoundError
        If any of the args (paths) does not exist


    Examples
    --------
    Import as:
    >>> from .config.config_manager import load_yaml

    Use as:
    load_yaml(path_to_yaml, path_to_another_yaml)
    For as many YAML files as needed.
    """

    for path in args:
        if not path.exists():
            raise FileNotFoundError(f"Found no YAML configuration file in {path}")

    config = ConfigManager(*args)

    return config
