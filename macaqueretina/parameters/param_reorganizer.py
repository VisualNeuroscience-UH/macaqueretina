"""
Readjust parameter structure in the configuration.
Since YAML files were introduced, the configuration object carries a slight
mismatch between the dict structure coming from the YAML files and the
previous structure that was achieved with the project_conf_module. This is
temporarily addressed with this file. Solutions might be:
- Refactor the codebase to accept the new Configuration object (recommended,
  can also refactor to use attribute-like access)
- Refactor the YAML files to match the previous structure (not recommended
  as it would make the YAML files more complex)
"""

from __future__ import annotations

# Built-in
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Local
    from macaqueretina.data_io.config_io import Configuration


class ParamReorganizer:
    """Here parameters are reorganized after validation, to reflect the orignal
    dict nesting structure before the introduction of the YAML files."""

    def __init__(self) -> None:
        pass

    def reorganize(self, config: Configuration) -> dict[str, Any]:
        self.config = config.as_dict().copy()
        self._create_literature_data_files()
        self.config["retina_parameters"].update(self.config["retina_parameters_extend"])

        config.clear()
        config.update(self.config)

        return config

    def _create_literature_data_files(self) -> None:
        target = "literature_data_files"
        key_list = [
            "dendr_diam_units",
        ]

        for key in key_list:
            self._move_and_remove_key(key, target)

        keys_to_rename = [key for key in key_list if key.endswith("_datafile")]
        for old_key in keys_to_rename:
            new_key = old_key.replace("_datafile", "_path")
            self.config[target][new_key] = self.config[target].pop(old_key)

        literature_folder = self.config.get("literature_data_folder")
        for key, value in self.config[target].items():
            if key.endswith("_path") or key.endswith("_datafile") or "_datafile" in key:
                self.config[target][key] = f"{literature_folder}/{value}"

    def _move_and_remove_key(self, key: str, target: str) -> None:
        if target not in self.config.keys():
            self.config.update({target: {}})
        self.config[target].update({key: self.config[key]})
        self.config.pop(key)
