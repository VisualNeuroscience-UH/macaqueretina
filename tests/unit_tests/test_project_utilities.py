# Built-in
import unittest.mock as mock
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import pytest

# Local
from macaqueretina.project.project_utilities_module import (
    DataSampler,
    ProjectUtilitiesMixin,
)


@pytest.fixture
def project_utilities_instance():
    return ProjectUtilitiesMixin()


@pytest.fixture
def mock_data_path():
    return Path("tests/mock_data")


@pytest.fixture
def data_sampler_instance(tmp_path):
    dummy_image = Path("tests/mock_data/test_image.jpg")
    test_file = Path("tests/mock_data/test_image_c.npz")

    instance = DataSampler(
        filename=dummy_image,
        min_X=0.0,
        max_X=10.0,
        min_Y=0.0,
        max_Y=10.0,
        logX=False,
        logY=False,
    )
    yield instance

    if test_file.exists():
        test_file.unlink()


def _mock_collect_and_save(data_sampler_instance):
    with mock.patch("matplotlib.pyplot.ginput") as mock_ginput:
        mock_ginput.side_effect = [
            [(10, 20), (10, 10), (20, 10)],  # Calibration points
            [(15, 16), (18, 12)],  # Data points, absolute coords
        ]
        with mock.patch("matplotlib.pyplot.imshow"), mock.patch(
            "matplotlib.pyplot.close"
        ):
            data_sampler_instance.collect_and_save_points()


class TestDataSampler:
    def test_collect_and_save_points(self, data_sampler_instance, mock_data_path):
        _mock_collect_and_save(data_sampler_instance)

        assert len(data_sampler_instance.calibration_points) == 3
        assert len(data_sampler_instance.data_points) == 2

        output_file = Path(mock_data_path / "test_image_c.npz")
        assert output_file.exists()

        data = np.load(output_file)
        assert "Xdata" in data and "Ydata" in data
        assert "calib_x" in data and "calib_y" in data
        assert len(data["Xdata"]) == 2
        assert len(data["calib_x"]) == 3

    def test_get_data_arrays(self, data_sampler_instance):
        _mock_collect_and_save(data_sampler_instance)
        x_data, y_data = data_sampler_instance.get_data_arrays()

        assert len(x_data) == 2
        assert len(y_data) == 2
        assert isinstance(x_data, np.ndarray)
        assert isinstance(y_data, np.ndarray)
        assert list(x_data) == [5.0, 8.0]
        assert list(y_data) == [4.0, 8.0]  # Note flipped y-axis


class TestProjectUtilitiesMixin:
    def test_pp_df_full(self, capsys, project_utilities_instance):
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(100, 30),
            columns=[f"col_{i}" for i in range(30)],
        )

        project_utilities_instance.pp_df_full(df)

        captured = capsys.readouterr().out

        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.max_colwidth",
            None,
        ):
            print(df)

        expected_output = capsys.readouterr().out
        assert expected_output in captured

    def test_get_xy_from_npz(self, project_utilities_instance):
        npz_data = {
            "Xdata": np.array([3.0, 1.0, 2.0]).reshape(3, 1),
            "Ydata": np.array([30.0, 10.0, 20.0]).reshape(3, 1),
        }

        x_data, y_data = project_utilities_instance.get_xy_from_npz(npz_data)

        expected_x = np.array([1.0, 2.0, 3.0])
        expected_y = np.array([10.0, 20.0, 30.0])

        np.testing.assert_array_equal(x_data, expected_x)
        np.testing.assert_array_equal(y_data, expected_y)

    def test_countlines(self, tmp_path, project_utilities_instance):
        # Create a temporary directory structure with .py files
        py_file1 = tmp_path / "file1.py"
        py_file1.write_text("line1\nline2\nline3\n")

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        py_file2 = subdir / "file2.py"
        py_file2.write_text("line1\nline2\n")

        # Should not be counted
        non_py_file = tmp_path / "file.txt"
        non_py_file.write_text("line1\nline2\n")

        total_lines = project_utilities_instance.countlines(tmp_path)

        assert total_lines == 5

        assert total_lines == 5
        assert total_lines == 5
        assert total_lines == 5
