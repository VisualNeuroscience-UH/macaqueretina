# Built-in
import unittest
from abc import ABC
from unittest.mock import MagicMock, patch

# Third-party
import numpy as np
import pandas as pd
import pytest

# Local
from macaqueretina.retina.construct_retina_module import TemporalModelBase


# Mocked dependencies
class GanglionCell:
    def get_BK_parameter_names(self):
        return ["param1", "param2"]

    df = pd.DataFrame()
    img = np.random.rand(5, 10, 10)
    img_mask = np.ones((5, 10, 10))
    X_grid_mm = np.random.rand(5, 10, 10)
    Y_grid_mm = np.random.rand(5, 10, 10)


class DoGModel:
    exp_univariate_stat = pd.DataFrame()


class DistributionSampler:
    def sample_univariate(self, shape, loc, scale, n_cells, distribution):
        return [0] * n_cells  # Return zeros for simplicity


class RetinaMath:
    def get_triangular_parameters(self, minimum, maximum, median, mean, sd, sem):
        return (0.5, minimum, maximum - minimum)

    def interpolate_data(self, frequency_data, power_data):
        return MagicMock()

    def set_metaparameters_for_log_interp_and_double_lorenzian(
        self, interp_function, noise_wc
    ):
        pass

    def fit_log_interp_and_double_lorenzian(self, log_frequency_data, *params):
        return np.zeros_like(log_frequency_data)  # Simplified for testing


class Retina:
    cone_optimized_pos_mm = np.zeros((10, 2))
    cone_general_parameters = {"cone2gc_mock_type": 1000, "cone2gc_cutoff_SD": 2}
    experimental_archive = {
        "temporal_parameters_BK": pd.DataFrame(),
        "cone_frequency_data": [1, 2, 3],
        "cone_power_data": [0.1, 0.2, 0.3],
        "cone_noise_wc": 0.5,
        "noise_frequency_data": [1, 2, 3],
        "noise_power_data": [0.01, 0.02, 0.03],
    }
    response_type = "mock_response"
    gc_type = "mock_type"


# Concrete implementation for testing abstract methods
class TemporalModelBaseConcrete(TemporalModelBase):
    def create(self, ret, gc):
        pass

    def connect_units(self, ret, gc):
        pass


class TestTemporalModelBase(unittest.TestCase):
    def setUp(self):
        self.ganglion_cell = GanglionCell()
        self.DoG_model = DoGModel()
        self.sampler = DistributionSampler()
        self.retina_math = RetinaMath()
        self.retina = Retina()
        self.gc = self.ganglion_cell

        self.temporal_model = TemporalModelBaseConcrete(
            self.ganglion_cell, self.DoG_model, self.sampler, self.retina_math
        )

    def test_init(self):
        self.assertIsInstance(self.temporal_model, TemporalModelBase)

    def test_link_cone_noise_units_to_gcs(self):
        # Set up mock data
        self.retina.cone_optimized_pos_mm = np.zeros((10, 2))
        self.gc.df = pd.DataFrame({"cell_id": range(5)})
        self.gc.img = np.random.rand(5, 10, 10)
        self.gc.img_mask = np.ones((5, 10, 10))
        self.gc.X_grid_mm = np.random.rand(5, 10, 10)
        self.gc.Y_grid_mm = np.random.rand(5, 10, 10)

        with patch("numpy.zeros", return_value=np.zeros((10, 5))) as mock_zeros:
            result = self.temporal_model._link_cone_noise_units_to_gcs(
                self.retina, self.gc
            )
            self.assertIsNotNone(result)
            mock_zeros.assert_called_once()

    def test_get_BK_statistics(self):
        # Create dummy data for temp_params_df
        data = {
            "Parameter": ["param1", "param2"],
            "Type": ["MOCK_RESPONSE", "MOCK_RESPONSE"],
            "Minimum": [1, 2],
            "Maximum": [3, 4],
            "Median": [2, 3],
            "Mean": [2, 3],
            "SD": [0.5, 0.6],
            "SEM": [0.1, 0.1],
        }
        temp_params_df = pd.DataFrame(data)
        self.retina.experimental_archive["temporal_parameters_BK"] = temp_params_df
        self.DoG_model.exp_univariate_stat = pd.DataFrame()

        result = self.temporal_model._get_BK_statistics(self.retina)
        self.assertIsNotNone(result)
        self.assertIn("shape", result.columns)
        self.assertIn("loc", result.columns)
        self.assertIn("scale", result.columns)

    def test_sample_temporal_rfs(self):
        # Create dummy stat_df DataFrame
        data = {
            "shape": [0.5, 0.6],
            "loc": [0, 0],
            "scale": [1, 1],
            "distribution": ["triang", "triang"],
            "domain": ["temporal_BK", "temporal_BK"],
        }
        index = ["param1", "param2"]
        stat_df = pd.DataFrame(data, index=index)

        self.gc.df = pd.DataFrame({"cell_id": range(5)})
        result = self.temporal_model._sample_temporal_rfs(self.gc, stat_df)
        self.assertIsNotNone(result)
        self.assertEqual(result, self.gc)
        # Check if parameters are added to gc.df
        for param in ["param1", "param2"]:
            self.assertIn(param, result.df.columns)

    @pytest.mark.filterwarnings("ignore::scipy.optimize.OptimizeWarning")
    def test_fit_cone_noise_vs_freq(self):
        # Mock methods in RetinaMath
        self.retina_math.interpolate_data = MagicMock(return_value=MagicMock())
        self.retina_math.set_metaparameters_for_log_interp_and_double_lorenzian = (
            MagicMock()
        )
        self.retina_math.fit_log_interp_and_double_lorenzian = MagicMock(
            return_value=np.zeros(3)
        )

        result = self.temporal_model._fit_cone_noise_vs_freq(self.retina)
        self.assertIsNotNone(result)
        self.assertEqual(result, self.retina)
        self.assertTrue(hasattr(result, "cone_noise_parameters"))
        self.assertTrue(hasattr(result, "cone_noise_power_fit"))


if __name__ == "__main__":
    unittest.main()
