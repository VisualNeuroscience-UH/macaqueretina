# Built-in
import os
from pathlib import Path
from typing import Any

# Third-party
from yaml import YAMLError, safe_load

# Local
from macaqueretina.config.param_validation import validate_params
from macaqueretina.config.param_reorganizer import ParamReorganizer


class YamlLoader:

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

    def get(self, key: str, default=None) -> Any:
        """Backwards compatibility with dictionary get() method."""
        try:
            return self[key]
        except KeyError:
            return default


class ConfigManager:
    """
    Handles loading and accessing values from the YAML configuration file.
    Provides both attribute-style and dictionary access to configuration values.

    Attributes
    ----------
    config_file_path : str
        Path to the loaded configuration file
    _config : dict
        Internal dictionary storing parsed configuration values
    """

    def __init__(self, *args) -> None:

        self.config_file_paths: tuple = args
        self._config: dict[str, Any] = YamlLoader(self.config_file_paths).load_config()

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


class ConfigValidator:
    """
    Validates parameters defined in yaml files. Calls param_validation.py with
    pydantic validation scheme and implicit default values.

    This class performs comprehensive validation of configuration files,
    including mandatory parameter existence, template existence, and
    type validation.
    """

    def __init__(self) -> None:

        self.errors = []
        self.reorganizer = ParamReorganizer()

    def validate(self, config: "ConfigManager") -> "ConfigManager":
        """
        Validate configuration data.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration to validate

        Returns
        -------
        ConfigManager
            The validated configuration object
        """
        # Validate the parameters
        validated_config = validate_params(config.as_dict())

        # Update the configuration with validated parameters
        validated_config = validated_config.model_dump()

        # Reorganize the configuration as it was before the introduction of YAML files
        reorganized_config = self.reorganizer.reorganize(validated_config)

        # Validate the properties
        config._config = reorganized_config
        self.config = config
        return self.config


class ConfigurationService:
    """
    Orchestrates loading and validating configuration from a YAML file.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "ConfigurationService":
        """
        Singleton pattern to ensure one configuration service.

        Returns:
            ConfigurationService: The singleton instance
        """
        if cls._instance is None:
            cls._instance = ConfigurationService()
        return cls._instance

    def __init__(self) -> None:

        self.validator = ConfigValidator()

    def process_configuration(self, *args) -> ConfigManager:
        """
        Load and validate the configuration from a YAML file.

        Parameters
        ----------
        config_file_path : str
            Path to configuration file

        Returns
        -------
        dict
            Dictionary containing the validated configuration

        Raises
        ------
        ValueError
            If the validation fails
        """

        # Load configuration file(s)
        config_object = ConfigManager(*args)

        # Return the configuration object
        return self.validator.validate(config_object)


# Façade for project_conf_module.py
def load_project_config_client(*args: tuple | None) -> ConfigManager:
    """
    Load project configuration from a YAML file.

    This is the primary entry point for code needing access to the
    project configuration. It handles locating, loading, and validating
    the configuration file through the ConfigurationService.

    Returns
    -------
    ConfigManager
        Configured and validated ConfigManager instance providing
        access to the configuration parameters
    """

    if not args:
        # No file paths provided, fall to the default YAML path
        default_yaml_path: Path = Path(os.getcwd()) / Path(
            "config/project_config.yaml"
        )  # TODO: change

        if not default_yaml_path.exists():
            raise FileNotFoundError(
                f"No path to YAML file provided."
                f"Default YAML configuration file not found in {default_yaml_path}"
            )

        service = ConfigurationService.get_instance()
        return service.process_configuration(default_yaml_path)

    # Unpack args and check if they exist
    for path in args:
        if not Path(path).exists():
            raise FileNotFoundError(f"Found no YAML configuration file in {path}")

    service = ConfigurationService.get_instance()
    return service.process_configuration(*args)
