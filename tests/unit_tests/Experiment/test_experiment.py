# Built-in
import unittest.mock as mock

# Third-party
import numpy as np
import pytest

# Set numpy seed to 42
np.random.seed(42)

# Local
from macaqueretina.stimuli.experiment_module import Experiment


class TestExperiment:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.context = mock.Mock()
        self.data_io = mock.Mock()
        self.experiment = Experiment(self.context, self.data_io)

    def test_meshgrid_conditions(self):
        # Set up input options
        options = {
            "condition1": np.array([1, 2]),
            "condition2": np.array([3, 4]),
        }

        # Call the method under test
        cond_options, cond_names = self.experiment._meshgrid_conditions(options)

        # Assert the expected output
        expected_cond_options = [
            {"condition1": np.int64(1), "condition2": np.int64(3)},
            {"condition1": np.int64(2), "condition2": np.int64(3)},
            {"condition1": np.int64(1), "condition2": np.int64(4)},
            {"condition1": np.int64(2), "condition2": np.int64(4)},
        ]
        expected_cond_names = [
            "c0c0",
            "c1c0",
            "c0c1",
            "c1c1",
        ]

        assert cond_options == expected_cond_options
        assert cond_names == expected_cond_names

    def test_generate_gaussian_distributions(self):
        # Set up input options
        cond_values = np.array([0, 1, 2, 3, 4, 5, 6])
        stats = {
            "sweeps": 20,
            "mean": (2.5, 3.5),
            "sd": (0.5, 0.5),
        }

        # Call the method under test
        combined_distributions = self.experiment._generate_gaussian_distributions(
            cond_values, stats
        )

        # Assert the expected output
        expected_values = np.array([1, 2, 3, 4, 5])

        # assert np.allclose(combined_distributions, expected_combined_distributions)
        assert np.all(np.isin(combined_distributions, expected_values))
