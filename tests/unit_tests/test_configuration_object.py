# Built-in
import copy
import os
import tempfile
from pathlib import Path

# Third-party
import brian2.units as b2u
import pytest
import yaml

# Local
from macaqueretina.data_io.config_io import Configuration, _YamlLoader, load_yaml


@pytest.fixture
def create_temp_yaml():
    """
    Fixture that returns a function to create a temporary YAML file.
    """

    def _create_temp_yaml(content: dict, suffix: str = ".yaml") -> str:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w") as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)
            return f.name

    yield _create_temp_yaml


@pytest.fixture
def valid_yaml_filepath():
    """
    Fixture to create a valid YAML file for testing.
    The file will be created in a temporary directory and deleted after the test.
    """
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        global yaml_content
        yaml_content = {
            "root_path": "/root/path",
            "project": "test",
            "experiment": "testing",
            "input_folder": "../in",
            "output_folder": "out",
            "config_template_files": {
                "anatomy": "/path/anatomy_test.csv",
                "physiology": "/path/physiology_test.csv",
                "connections": "/path/physiology_test.gz",
                "config_template_path": "/opt2/git_repos/macaqueretinaa/config",
            },
            "execute": {
                "simulation": False,
                "analysis": True,
                "visualization": False,
            },
            "analysis_parameters": {
                "substring": "*results*.gz",
                "neurongroups": ["NG1", "NG2"],
            },
            "visualization_parameters": {
                "substring": "*basic_analysis*.gz",
                "neurongroups": ["NG1", "NG2"],
                "savefigs": False,
                "myformat": "eps",
            },
            "run_n_sweeps": 2,
            "numpy_seed": 42,
            "spike_threshold": 5,
            "visual2cortical_params": {
                "a": 0.077 / 0.082,
                "k": 1 / 0.082,
            },
            "profile": False,
        }

        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        file_path = f.name

    # Yield the file path after closing the file
    yield file_path

    # Cleanup the file after the test
    os.unlink(file_path)


@pytest.fixture
def empty_yaml_filepath():
    """
    Fixture to create an empty YAML file for testing.
    The file will be created in a temporary directory and deleted after the test.
    """
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        yaml_content = None

        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        file_path = f.name

    # Yield the file path after closing the file
    yield file_path

    # Cleanup the file after the test
    os.unlink(file_path)


# Test loading a valid YAML file
def test_load_valid_yaml(valid_yaml_filepath):
    """
    Test loading a valid YAML file.
    """

    # Load the YAML file (positive test)
    configuration = _YamlLoader((valid_yaml_filepath,)).load_config()

    assert isinstance(configuration, dict)


# Test loading an empty YAML file
def test_load_empty_yaml(empty_yaml_filepath):
    """
    Test loading an empty YAML file.
    """

    # Load the YAML file
    with pytest.raises(ValueError):
        _YamlLoader((empty_yaml_filepath,)).load_config()


# Test loading a non-existent YAML file
def test_load_non_existent_yaml():
    """
    Test loading a non-existent YAML file.
    """

    # Load the YAML file
    with pytest.raises(FileNotFoundError):
        _YamlLoader(("/non/existent/path.yaml",)).load_config()


# Test passing a file with an extension other than .yaml
def test_pass_invalid_extension():
    """
    Test passing a file with an invalid extension.
    """

    # Create a temporary file with an invalid extension
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        file_path = f.name

    # Load the YAML file
    with pytest.raises(ValueError):
        _YamlLoader((file_path,)).load_config()

    # Cleanup the file after the test
    os.unlink(file_path)


# Test merging multiple YAML files (positive case)
def test_merge_multiple_yaml(create_temp_yaml):
    first_yaml = create_temp_yaml({"key1": "value1", "key2": "value2"})
    second_yaml = create_temp_yaml({"key3": "value3", "key4": "value4"})

    try:
        loader = _YamlLoader((first_yaml, second_yaml))
        merged_config = loader.load_config()

        assert isinstance(merged_config, dict)
        assert merged_config["key1"] == "value1"
        assert merged_config["key2"] == "value2"
        assert merged_config["key3"] == "value3"
        assert merged_config["key4"] == "value4"
    finally:
        os.unlink(first_yaml)
        os.unlink(second_yaml)


# Test merging multiple YAML files with duplicate keys
def test_merge_duplicate_keys(create_temp_yaml):
    first_yaml = create_temp_yaml({"key1": "value1"})
    second_yaml = create_temp_yaml({"key1": "value2"})

    try:
        loader = _YamlLoader((first_yaml, second_yaml))
        with pytest.raises(ValueError):
            loader.load_config()
    finally:
        os.unlink(first_yaml)
        os.unlink(second_yaml)


## Test the Configuration object


@pytest.fixture
def valid_source_dict():
    return {
        "root_path": "/root/path",
        "execute": {
            "simulation": False,
            "analysis": True,
            "visualization": False,
        },
        "analysis_parameters": {
            "substring": "*results*.gz",
            "neurongroups": ["NG1", "NG2"],
        },
        "numpy_seed": 42,
        "profile": False,
        "time": 10 * b2u.second,
        "to_pop": "something",
    }


# Test creating an empty configuration
def test_empty_configuration():
    config = Configuration({})

    with pytest.raises(AttributeError):
        config.non_existent
    with pytest.raises(KeyError):
        config["non_existent"]
    assert config.get("non_existent") == None


# Test that the various data types are kept
def test_data_types(valid_source_dict):
    config = Configuration(valid_source_dict)

    assert isinstance(config.get("root_path"), str)
    assert isinstance(config.root_path, str)
    assert isinstance(config.get("execute"), Configuration)
    assert isinstance(config.execute.simulation, bool)
    assert isinstance(config["numpy_seed"], int)
    assert isinstance(config.time, b2u.Quantity)


@pytest.fixture
def deep_nested_dict():
    return {
        "test_nested_values": {
            "nesting1": {"nesting2": {"nesting3": {"nesting4": {"nesting5": 42}}}}
        }
    }


# Test nested access
def test_nested_access(deep_nested_dict):
    config = Configuration(deep_nested_dict)

    assert config.test_nested_values.nesting1.nesting2.nesting3.nesting4.nesting5 == 42
    assert (
        config["test_nested_values"]["nesting1"]["nesting2"]["nesting3"]["nesting4"][
            "nesting5"
        ]
        == 42
    )
    assert (
        config.get("test_nested_values")
        .get("nesting1")
        .get("nesting2")
        .get("nesting3")
        .get("nesting4")
        .get("nesting5")
        == 42
    )
    assert isinstance(
        config.test_nested_values.nesting1.nesting2.nesting3.nesting4, Configuration
    )


# TODO: test from_yaml class method


# Test __contains__
def test_contains(valid_source_dict):
    config = Configuration(valid_source_dict)

    truthy = "root_path" in config
    truthy_nested = "simulation" in config.execute
    falsy = "non_esisting" in config
    falsy_nested = "non_esisting" in config.execute

    assert truthy
    assert truthy_nested
    assert not falsy
    assert not falsy_nested


# Test __getitem__, __setitem__, __delitem__ from MutableMapping
def test_mutablemapping_get_set_del(valid_source_dict):
    config = Configuration(valid_source_dict)

    assert config["root_path"] == "/root/path"
    assert isinstance(config["profile"], bool)
    assert config.get("numpy_seed") == 42
    assert isinstance(config.get("time"), b2u.Quantity)

    config["new_value"] = 1
    config.other_new_value = 1

    assert "new_value" in config
    assert "other_new_value" in config

    with pytest.raises(KeyError):
        config["non_existent"]

    assert config.get("non_existent") == None

    assert "root_path" in config
    del config.root_path
    assert "root_path" not in config

    with pytest.raises(AttributeError):
        del config.non_existent


# Test __iter__, __len__ from MutableMapping
def test_mutablemapping_iter_len(valid_source_dict):
    config = Configuration(valid_source_dict)
    nested = config.execute

    assert len(config) == len(config.as_dict()) == len(config.keys())
    assert len(config) == 7
    assert len(nested) == 3

    keys_from_iter = list(config)
    expected_keys = [
        "root_path",
        "execute",
        "analysis_parameters",
        "numpy_seed",
        "profile",
        "time",
        "to_pop",
    ]
    assert keys_from_iter == expected_keys

    collected_keys = []
    for key in config:
        collected_keys.append(key)
    assert len(collected_keys) == 7

    all_keys = [k for k in config]
    assert len(all_keys) == 7


# Test get(), keys(), items(), values() # TODO: add tests for items()
def test_dict_methods(valid_source_dict):
    config = Configuration(valid_source_dict)

    assert config.get("root_path") == "/root/path"
    assert isinstance(config.get("analysis_parameters"), Configuration)

    expected_keys = [
        "root_path",
        "execute",
        "analysis_parameters",
        "numpy_seed",
        "profile",
        "time",
        "to_pop",
    ]
    assert list(config.keys()) == expected_keys
    assert 42 in config.values()


# Test pop(), popitem(), clear()
def test_pop_popitem_clear(valid_source_dict):
    config = Configuration(valid_source_dict)

    item_pop = config.popitem()
    assert item_pop == ("to_pop", "something")
    assert "to_pop" not in config

    profile = config.pop("profile")
    assert profile == False
    assert isinstance(profile, bool)
    assert profile not in config

    config.clear()
    expected_keys = [
        "root_path",
        "execute",
        "analysis_parameters",
        "numpy_seed",
        "time",
    ]
    assert all(key not in config for key in expected_keys)


# Test update()
def test_update(valid_source_dict):
    config = Configuration(valid_source_dict)

    extra_dict1 = {"a": 1, "b": 2, "c": 3}
    config.update(extra_dict1)

    assert "a" and "b" and "c" in config

    extra_dict2 = {"d": 4, "e": 5}
    config.update(extra_dict2, f=6)

    assert "d" and "e" in config
    assert "f" in config

    extra_dict3 = {"g": {"h": 7}}
    config.update(extra_dict3)

    assert isinstance(config.g, Configuration)


# Test setdefault()
def test_setdefault(valid_source_dict):
    config = Configuration(valid_source_dict)

    value = config.setdefault("root_path")

    assert value == "/root/path"
    assert value in config.values()

    value = config.setdefault("non_existing")

    assert config.non_existing == None
    assert "non_existing" in config.keys()

    value = config.setdefault("element", 15)

    assert "element" in config.keys()
    assert config.element == 15


# Test reading non-existent attributes
def test_non_existing_attribute(valid_source_dict):
    config = Configuration(valid_source_dict)

    with pytest.raises(AttributeError):
        config.non_existing


# Test conflicts with protected attributes
def test_protected_attributes(valid_source_dict):
    config = Configuration(valid_source_dict)

    assert "hash" and "from_yaml" in type(config).__dict__

    with pytest.raises(AttributeError):
        config.hash = 30
        config.from_yaml = 20
        config._internal = "string"


def test_dict_conversion(valid_source_dict):
    config = Configuration(valid_source_dict)

    config_as_dict = config.as_dict()
    config_to_dict = config.to_dict()
    assert isinstance(config_as_dict, dict)
    assert isinstance(config_to_dict, dict)

    nested_dict = config_as_dict.get("analysis_parameters")
    assert isinstance(nested_dict, dict)

    assert config_as_dict.items() == config.items()
    assert config_to_dict.items() == config.items()


# Test comparison & equality
def test_equality(valid_source_dict):
    config1 = Configuration(valid_source_dict)
    config2 = Configuration(valid_source_dict)

    assert config1 == config2
    assert config1 is not config2

    config_as_dict = config2.as_dict()

    assert config1 == config_as_dict

    valid_source_dict.popitem()
    different_config = Configuration(valid_source_dict)

    assert config1 != different_config
    assert config2 != different_config
    assert config1 != different_config.as_dict()

    assert config1
    assert config2

    config1.clear()

    assert not config1


# Test __repr__, __str__, __hash__
def test_representation(valid_source_dict):
    config = Configuration(valid_source_dict)

    repr_string = repr(config)
    str_string = str(config)
    assert isinstance(repr_string, str)
    assert isinstance(str_string, str)

    nested_repr = repr(config.analysis_parameters)
    assert isinstance(nested_repr, str)

    simple_config = Configuration({"key": "value"})
    simple_repr = repr(simple_config)
    assert "key" in simple_repr
    assert "value" in simple_repr

    dir_list = dir(config)

    assert "root_path" in dir_list
    assert "hash" in dir_list


# Test __copy__ and __deepcopy__
def test_copying(valid_source_dict):
    config = Configuration(valid_source_dict)

    shallow_copy = copy.copy(config)

    assert config == shallow_copy
    assert config is not shallow_copy

    assert config.analysis_parameters == shallow_copy.analysis_parameters
    assert (
        config.analysis_parameters.substring
        is shallow_copy.analysis_parameters.substring
    )

    deep_copy = copy.deepcopy(config)

    assert config == deep_copy
    assert config is not deep_copy

    assert config.analysis_parameters == deep_copy.analysis_parameters
    assert config.analysis_parameters is not deep_copy.analysis_parameters


# TODO: test serialization


# Test hash()
def test_hashing(valid_source_dict):
    config = Configuration(valid_source_dict)

    assert isinstance(config.hash(), str)
    assert len(config.hash()) == 10

    hash1 = config.hash()
    hash2 = config.hash()
    assert hash1 == hash2

    config.new_value = 12
    hash_new = config.hash()
    config.pop("new_value")
    original_hash = config.hash()

    assert not hash1 == hash_new
    assert hash1 == original_hash

    hash_short = config.hash(length=4)
    hash_full = config.hash(length=-1)
    assert len(hash_short) == 4
    assert len(hash_full) == 64

    with pytest.raises(TypeError):
        hash(config)


# Test load_yaml
def test_load_yaml(valid_yaml_filepath):
    config = load_yaml(valid_yaml_filepath)

    assert isinstance(config, Configuration)
    assert "root_path" in config

    config_from_path = load_yaml(Path(valid_yaml_filepath))
    assert isinstance(config_from_path, Configuration)
    assert "root_path" in config_from_path

    # TODO: test when passing multiple paths
