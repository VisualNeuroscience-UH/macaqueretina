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
from typing import Any, Iterable, Mapping

# Third-party
from yaml import YAMLError, safe_load


class YamlLoader:
    """Loads one or multiple YAML files.
    Merges multiple YAML files into a configuration dict."""

    def __init__(self, yaml_paths: Iterable[Path | str]) -> None:
        self.yaml_paths: list[Path] = [Path(p) for p in yaml_paths]
        self._key_sources: dict[str, Path] = {}

    def load_config(self) -> dict[str, Any]:
        """
        Core method to load the configuration YAML file(s).

        Returns
        -------
        combined_config: dict[str, Any]
            Dictionary combining parameters from multiple YAML files.

        Raises
        -------
        FileNotFoundError
            If (any of the) YAML file(s) are not found
        ValueError
            If the YAML file is empty, if its structure is non-standard, or if the
            file is invalid
        """

        combined_config: dict[str, Any] = {}
        for path in self.yaml_paths:
            if not path.exists():
                raise FileNotFoundError(f"YAML file not found: {path!s}")

            with open(path, "r", encoding="utf-8") as file:
                try:
                    config = safe_load(file)
                    if config is None:
                        raise ValueError(f"Configuration file is empty: {path!s}")
                    if not isinstance(config, dict):
                        raise ValueError(
                            f"Top-level of YAML must be a mapping in {path!s}"
                        )
                    combined_config = self._merge_configs(combined_config, config, path)
                except YAMLError as e:
                    raise ValueError(
                        f"Invalid YAML in configuration file {path!s}: {e!s}"
                    )
                except Exception:
                    raise
        return combined_config

    def _merge_configs(
        self,
        combined_config: dict[str, Any],
        current_config: dict[str, Any],
        path: Path,
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
            If duplicate keys are found in the configuration files.
            Shows which files have duplicate keys.
        """

        for key in current_config.keys():
            if key in combined_config:
                prev_path = self._key_sources.get(key, "<unknown>")
                raise ValueError(
                    f"Duplicate top-level key '{key}' found in {path!s}; "
                    f"first defined in {prev_path!s}. Rename the parameter or remove "
                    f"duplicate."
                )
            # Which file introduced the current key? Record it in _key_sources
            self._key_sources[key] = path

        combined_config.update(current_config)

        return combined_config


class NestedConfig:
    """
    Attribute-style access to nested dictionary values.

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
        self._config = dict(config_dict)

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, dict):
            return NestedConfig(value)
        return value

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            return self._wrap(self._config[name])
        raise AttributeError(f"No attribute '{name}' found")

    def __getitem__(self, key: str) -> Any:
        if key in self._config:
            return self._wrap(self._config[key])
        raise KeyError(f"Configuration key not found: {key}")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style item assignment."""
        self._config[key] = value

    def get(self, key: str, default=None) -> Any:
        """Backwards compatibility with dictionary get() method."""
        return self._config.get(key, default)

    def keys(self):
        return self._config.keys()

    def items(self):
        return self._config.items()

    def values(self):
        return self._config.values()

    def __iter__(self):
        return iter(self._config)

    def __len__(self):
        return len(self._config)

    def __contains__(self, item):
        return item in self._config

    def __repr__(self):
        return f"NestedConfig({self._config!r})"

    def to_dict(self) -> dict[str, Any]:
        def _unwrap(v):
            if isinstance(v, NestedConfig):
                return v.to_dict()
            if isinstance(v, dict):
                return {k: _unwrap(vv) for k, vv in v.items()}
            return v

        return {k: _unwrap(v) for k, v in self._config.items()}


class ConfigManager:
    """
    Handles loading and accessing values from the YAML configuration file.
    Provides both attribute-style and dictionary access to configuration values.
    """

    def __init__(self, *args: Path | str) -> None:

        self.config_file_paths: list[Path] = [Path(p) for p in args]
        self._config: dict[str, Any] = YamlLoader(self.config_file_paths).load_config()

    def update_config(self, new_config: Mapping[str, Any]) -> None:
        """Public method to replace internal configuration in a controlled way."""
        self._config = dict(new_config)

    def __getattr__(self, name: str) -> Any:
        """Attribute-style access to configuration values."""
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return NestedConfig(value)
            return value
        raise AttributeError(f"Configuration has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to configuration values."""
        if key in self._config:
            value = self._config[key]
            if isinstance(value, dict):
                return NestedConfig(value)
            return value
        raise KeyError(f"Configuration key not found: {key}")

    def keys(self):
        return self._config.keys()

    def items(self):
        return self._config.items()

    def values(self):
        return self._config.values()

    def __iter__(self):
        return iter(self._config)

    def __len__(self):
        return len(self._config)

    def __contains__(self, item):
        return item in self._config

    def __repr__(self):
        return f"ConfigManager({self._config!r})"

    @property
    def config(self) -> dict[str, Any]:
        """Access the whole configuration dictionary."""
        return dict(self._config)

    def as_dict(self) -> dict[str, Any]:
        """Access the whole configuration dictionary."""
        return dict(self._config)


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
