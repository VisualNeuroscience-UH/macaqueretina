"""
Load and merge YAML configuration files into a mutable Configuration object that
supports nested read/write dict-like and attribute-like access to the parameters.

Notes
-----
    Duplicate top-level keys across files raise ValueError showing YAML source
    locations.
    load_yaml(paths: Iterable[Path | str] | Path | str) -> Configuration: façade that
    accepts either an iterable of paths or a single Path/str.

Examples
--------
    >>> yaml_files = [...] # List of your YAML files
    >>> config = load_yaml(yaml_files)
    >>> config.parameter # Prints "parameter"
    >>> config.as_dict()  # Returns the configuration as a dict
"""

# Built-in
import datetime
import hashlib
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

# Third-party
import numpy as np
from brian2.units.fundamentalunits import Quantity
from yaml import YAMLError, safe_load

_SENTINEL = object()


class _YamlLoader:
    """
    Loads one or multiple YAML files.
    Merges multiple YAML files into a Configuration object.
    Top-level parameters must have unique names.
    """

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
            If (any of) the YAML file(s) are not found
        ValueError
            If any YAML file is empty, if its structure is non-standard, or if the
            file is invalid
        """

        combined_config: dict[str, Any] = {}
        for path in self.yaml_paths:
            if not path.exists():
                raise FileNotFoundError(f"YAML file not found: {path!s}")

            with open(path, "r", encoding="utf-8") as file:
                try:
                    yaml_contents = safe_load(file)
                    if yaml_contents is None:
                        raise ValueError(f"Configuration file is empty: {path!s}")
                    if not isinstance(yaml_contents, dict):
                        raise ValueError(
                            f"Top-level of YAML must be a mapping in {path!s}"
                        )
                    combined_config = self._merge_configs(
                        combined_config, yaml_contents, path
                    )
                except YAMLError as e:
                    raise ValueError(
                        f"Invalid YAML in configuration file {path}: {e}"
                    ) from e
                except Exception as e:
                    if isinstance(e, ValueError):
                        raise
                    raise RuntimeError(
                        f"Failed to load required Configuration from {path}: {e}"
                    ) from e
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


class Configuration(MutableMapping):
    """
    Configuration object.

    Notes
    -----

    - Nested dictionaries are converted into Configuration objects recursively.
    - Most mutating operations modify the internal state in-place; no copying is
      performed unless explicitly requested.
    - Keys that start with "_" are reserved for internal use and cannot be set via
      attribute assignment.
    """

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        """Initialize the configuration, optionally with a mapping to populate config."""
        super().__setattr__("_data", {})
        if initial:
            for k, v in dict(initial).items():
                self._data[k] = Configuration(v) if isinstance(v, dict) else v

    # MutableMapping core methods
    def __getitem__(self, key: str) -> Any:
        """Supports dict-like access (config["param"])."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Supports dict-like assignment (config["param"] = my_value)."""
        self._data[key] = Configuration(value) if isinstance(value, dict) else value

    def __delitem__(self, key: str) -> None:
        """Supports dict-like deletion: del config["param"]."""
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over top-level keys in config (for k in config)."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of top-level parameters in the config."""
        return len(self._data)

    # Dict-like access
    def __contains__(self, item):
        """Return True if item is a top-level key in the config."""
        return item in self._data

    def get(self, key: str, default=None) -> Any:
        """Allows dict-like access with get: config.get(my_param)."""
        return self._data.get(key, default)

    def keys(self):
        """Returns the top-level params as a dict-like view (supports config.keys())."""
        return self._data.keys()

    def items(self):
        """Supports config.items() (behaves like built-in dict)."""
        return self._data.items()

    def values(self):
        """Supports config.values() (behaves like built-in dict)."""
        return self._data.values()

    def pop(self, key: str, default=_SENTINEL):
        """Supports config.pop("param") (behaves like built-in dict)."""
        if default is _SENTINEL:
            return self._data.pop(key)
        return self._data.pop(key, default)

    def popitem(self):
        """Supports config.popitem() (behaves like built-in dict)."""
        return self._data.popitem()

    def clear(self):
        """Supports config.clear() to remove all params (behaves like built-in dict)."""
        self._data.clear()

    def update(self, other: Mapping[str, Any] | None = None, **kwargs):
        """Supports config.update() (behaves like built-in dict)."""
        if other:
            for k, v in dict(other).items():
                self._data[k] = type(self)(v) if isinstance(v, dict) else v
        for k, v in kwargs.items():
            self._data[k] = type(self)(v) if isinstance(v, dict) else v

    def setdefault(self, key: str, default=None):
        """Supports config.setdefault() (behaves like built-in dict)."""
        if key in self._data:
            return self._data[key]
        value = type(self)(default) if isinstance(default, dict) else default
        self._data[key] = value
        return value

    # Convert Configuration object to dict
    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation of the configuration."""

        def _unwrap(v):
            if isinstance(v, Configuration):
                return v.to_dict()
            if isinstance(v, dict):
                return {kk: _unwrap(vv) for kk, vv in v.items()}
            return v

        return {k: _unwrap(v) for k, v in self._data.items()}

    def as_dict(self) -> dict[str, Any]:
        """Alias for to_dict()."""
        return self.to_dict()

    # Attribute-like access and assignment
    def __getattr__(self, name: str) -> Any:
        """Supports attribute-like access (config.param)"""
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"No attribute '{name}' found.")

    def __setattr__(self, name: str, value: Any) -> None:
        """Supports attribute-like assignment (config.param = my_value)"""
        if name.startswith("_") or name in type(self).__dict__:
            raise AttributeError(
                f"Cannot set attribute '{name}' because it conflicts with a built-in member. "
            )
        else:
            self.__setitem__(name, value)

    def __delattr__(self, name: str) -> None:
        """Supports: del config.param"""
        if name.startswith("_") or name in type(self).__dict__:
            raise AttributeError(
                f"Cannot delete attribute '{name}'. '{name}' is a built-in member, not a configuration parameter. "
            )
        else:
            try:
                del self._data[name]
            except KeyError as e:
                raise AttributeError(f"No attribute '{name}' found.") from e

    # Representation and comparison
    def __dir__(self) -> list[str]:
        """Returns attributes and top-level keys (for tab-completion and dir())."""
        return list(super().__dir__()) + list(self._data.keys())

    def __repr__(self):
        """Supports repr(config), repr of the internal mapping."""
        return f"{self._data!r}"

    def __str__(self):
        """Supports user-friendly view: called by print(config)"""
        return str(self.as_dict())

    def __eq__(self, other: object) -> bool:
        """Equality comparison by value."""
        if isinstance(other, type(self)):
            return self.to_dict() == other.to_dict()
        if isinstance(other, dict):
            return self.to_dict() == other
        return False

    def __ne__(self, other: object) -> bool:
        """Negated equality."""
        return not self.__eq__(other)

    def __bool__(self) -> bool:
        """Truthiness: True when config has at least one param."""
        return bool(self._data)

    # Copying
    def __copy__(self):
        """Supports shallow copy with the copy built-in library."""
        return type(self)(self.to_dict())

    def __deepcopy__(self, memo):
        """Supports deep copy with the copy built-in library."""
        import copy as _copy

        return type(self)(_copy.deepcopy(self.to_dict(), memo))

    # Pickling
    def __getstate__(self):
        """Return a picklable state (plain dict) for pickling support."""
        return self.to_dict()

    def __setstate__(self, state):
        """Restore state from pickled representation (expects a plain dict)."""
        super().__setattr__("_data", {})
        for k, v in dict(state).items():
            self._data[k] = type(self)(v) if isinstance(v, dict) else v

    def __reduce__(self):
        """Helper for pickle to reconstruct the object from its dict form."""
        return (type(self), (self.to_dict(),))

    @classmethod
    def from_yaml(cls, *paths: Path | str) -> "Configuration":
        """Get parameters from the YAML files. Delegates to _YamlLoader."""
        loader = _YamlLoader(paths)
        raw_config = loader.load_config()
        return cls(raw_config)

    # Hashing
    def hash(self, length: int = 10) -> str:
        """
        Generate a hash based on a frozen snapshot of the configuration.
        It returns the first (length) characters of the sha256 hash digest.
        Intended for reproducible identification of a configuration snapshot;
        not a general-purpose cryptographic MAC.
        """

        def _immutable(obj: Any) -> Any:
            """Recursively convert to immutable types for hashing."""
            if isinstance(obj, dict):
                return tuple(sorted((k, _immutable(v)) for k, v in obj.items()))
            if isinstance(obj, list):
                return tuple(_immutable(item) for item in obj)
            if isinstance(obj, set):
                return frozenset(_immutable(item) for item in obj)
            if isinstance(obj, tuple):
                return tuple(_immutable(item) for item in obj)
            if isinstance(obj, bytearray):
                return bytes(obj)
            if isinstance(obj, Quantity):  # Something with Brian2 units
                return obj.tostring()
            if isinstance(obj, complex):
                return (obj.real, obj.imag)
            if isinstance(obj, (datetime.datetime)):
                return obj.isoformat()
            if isinstance(obj, np.ndarray):
                return (tuple(obj.shape), tuple(obj.flatten().tolist()))
            return obj

        # Make Configuration hashable
        frozen = _immutable(self.to_dict())

        # Calculate SHA256 hash digest
        frozen_bytes = str(frozen).encode("utf-8")
        hash_digest = hashlib.sha256(frozen_bytes).hexdigest()

        # Return truncated or full hash digest
        if length < 0:
            return hash_digest
        return hash_digest[:length]

    def __hash__(self) -> int:
        """
        Configuration is mutable and therefore unhashable with hash(config).
        It prevents use of built-in hash() on mutable Configuration objects.

        Raises
        ------
        TypeError
            Always raises with instructions to use .hash() method instead
        """
        raise TypeError(
            f"{type(self).__name__} object is not directly hashable. "
            f"Use .hash() method to get a hash string of the current configuration."
        )


# Façade
def load_yaml(paths: Iterable[Path | str] | Path | str) -> Configuration:
    """
    Load project configuration from one or more YAML files.

    Parameters
    ----------

    paths: Iterable[Path | str] | Path | str
        Iterable of paths (ora a single Path / str) to YAML configuration files to load
        and merge.

    Returns
    -------
    Configuration
        Configuration object providing read/write access to the configuration
        parameters set in the YAML files.

    Raises
    ------
    FileNotFoundError
        If any of the args (paths) does not exist

    Examples
    --------
        Import as:
        >>> from .data_io.config_io import load_yaml

        Use as:
        >>> load_yaml(path_to_yaml, path_to_another_yaml)
        For as many YAML files as needed.
    """

    if isinstance(paths, (str, Path)):
        paths_list = [paths]
    else:
        paths_list = list(paths)

    return Configuration.from_yaml(*paths_list)
