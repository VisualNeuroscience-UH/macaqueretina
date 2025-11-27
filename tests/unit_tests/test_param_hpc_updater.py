import os
import subprocess
import tempfile

import pytest
import yaml


@pytest.fixture
def sample_yaml():
    return {
        "retina_parameters": {
            "gc_type": "parasol",
            "response_type": "on",
            "model_density": 1.0,
        }
    }


def get_script_path():
    # Get the absolute path to the script, assuming it's in a 'parameters' subdirectory
    # of the project root (adjust as needed)
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../macaqueretina/parameters/param_hpc_updater.py",
        )
    )


def test_param_hpc_updater(sample_yaml, monkeypatch, capsys):
    script_path = get_script_path()
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(sample_yaml, f)
        yaml_path = f.name

    try:
        monkeypatch.setenv("GC_TYPE", "midget")
        monkeypatch.setenv("RESPONSE_TYPE", "off")

        subprocess.run(
            ["python", script_path, yaml_path, "retina_parameters"],
            check=True,
            capture_output=True,
            text=True,
        )

        with open(yaml_path) as f:
            updated_data = yaml.safe_load(f)

        assert updated_data["retina_parameters"]["gc_type"] == "midget"
        assert updated_data["retina_parameters"]["response_type"] == "off"
        assert updated_data["retina_parameters"]["model_density"] == 1.0

    finally:
        os.unlink(yaml_path)


def test_param_hpc_updater_missing_key(sample_yaml, monkeypatch, capsys):
    script_path = get_script_path()
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(sample_yaml, f)
        yaml_path = f.name

    try:
        result = subprocess.run(
            ["python", script_path, yaml_path, "nonexistent_key"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Error: 'nonexistent_key' not found in YAML file." in result.stdout

    finally:
        os.unlink(yaml_path)
