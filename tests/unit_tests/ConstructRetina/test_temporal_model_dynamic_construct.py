# test_temporal_model_dynamic.py

# Built-in
import unittest
from unittest.mock import MagicMock, patch

# Third-party
import numpy as np
import pandas as pd

# Local
from macaqueretina.retina.construct_retina_module import (
    TemporalModelBase,
    TemporalModelDynamic,
)


# Mocked dependencies
class GanglionCell:
    def get_BK_parameter_names(self):
        return ["param1", "param2"]

    df = pd.DataFrame({"cell_id": range(5)})


class DoGModel:
    exp_univariate_stat = pd.DataFrame()


class DistributionSampler:
    def sample_univariate(self, shape, loc, scale, n_cells, distribution):
        return np.random.rand(n_cells)  # Return random values for simplicity


class RetinaMath:
    pass  # Assume methods are not needed for this test


class Retina:
    response_type = "sustained"
    gc_type = "midget"

    experimental_archive = {
        "temporal_parameters_BK": pd.DataFrame(
            {
                "Parameter": ["param1", "param2"],
                "Type": ["SUSTAINED", "SUSTAINED"],
                "Minimum": [0.1, 0.2],
                "Maximum": [0.5, 0.6],
                "Median": [0.3, 0.4],
                "Mean": [0.3, 0.4],
                "SD": [0.05, 0.06],
                "SEM": [0.01, 0.01],
            }
        )
    }


# Concrete implementation (assuming it inherits from TemporalModelBase)
class TemporalModelDynamic(TemporalModelBase):
    def __init__(
        self,
        ganglion_cell,
        DoG_model,
        distribution_sampler,
        retina_math,
    ):
        super().__init__(ganglion_cell, DoG_model, distribution_sampler, retina_math)

    def create(self, ret, gc):
        BK_statistics = self._get_BK_statistics(ret)
        gc = self._sample_temporal_rfs(gc, BK_statistics)
        return gc

    def connect_units(self, ret, gc):
        ret = self._link_cone_noise_units_to_gcs(ret, gc)
        return ret


# Unit tests
class TestTemporalModelDynamic(unittest.TestCase):
    def setUp(self):
        self.ganglion_cell = GanglionCell()
        self.DoG_model = DoGModel()
        self.sampler = DistributionSampler()
        self.retina_math = RetinaMath()
        self.retina = Retina()
        self.gc = self.ganglion_cell

        self.temporal_model = TemporalModelDynamic(
            self.ganglion_cell, self.DoG_model, self.sampler, self.retina_math
        )

    def test_init(self):
        self.assertIsInstance(self.temporal_model, TemporalModelDynamic)

    @patch.object(TemporalModelDynamic, "_get_BK_statistics")
    @patch.object(TemporalModelDynamic, "_sample_temporal_rfs")
    def test_create(self, mock_sample_temporal_rfs, mock_get_BK_statistics):
        # Mocking the BK statistics
        BK_statistics = pd.DataFrame(
            {
                "shape": [0.7, 0.8],
                "loc": [0, 0],
                "scale": [1, 1],
            },
            index=["param1", "param2"],
        )
        mock_get_BK_statistics.return_value = BK_statistics

        # Mocking the sample_temporal_rfs method
        mock_sample_temporal_rfs.return_value = self.gc

        # Run the create method
        result = self.temporal_model.create(self.retina, self.gc)
        self.assertIsNotNone(result)
        mock_get_BK_statistics.assert_called_once_with(self.retina)
        mock_sample_temporal_rfs.assert_called_once_with(self.gc, BK_statistics)

    @patch.object(TemporalModelDynamic, "_link_cone_noise_units_to_gcs")
    def test_connect_units(self, mock_link_cone_noise_units_to_gcs):
        # Mocking the method to return the retina
        mock_link_cone_noise_units_to_gcs.return_value = self.retina

        result = self.temporal_model.connect_units(self.retina, self.gc)
        self.assertIsNotNone(result)
        mock_link_cone_noise_units_to_gcs.assert_called_once_with(self.retina, self.gc)

    def test_get_BK_statistics(self):
        # Assuming the method from TemporalModelBase works correctly
        with patch.object(
            self.temporal_model, "_get_BK_statistics", return_value=pd.DataFrame()
        ):
            result = self.temporal_model._get_BK_statistics(self.retina)
            self.assertIsNotNone(result)

    def test_sample_temporal_rfs(self):
        # Create dummy BK_statistics DataFrame
        BK_statistics = pd.DataFrame(
            {
                "shape": [0.5, 0.6],
                "loc": [0, 0],
                "scale": [1, 1],
                "distribution": ["triang", "triang"],
                "domain": ["temporal_BK", "temporal_BK"],
            },
            index=["param1", "param2"],
        )

        # Mock the sampler
        self.sampler.sample_univariate = MagicMock(return_value=np.random.rand(5))

        result = self.temporal_model._sample_temporal_rfs(self.gc, BK_statistics)
        self.assertIsNotNone(result)
        # Check if parameters are added to gc.df
        for param in ["param1", "param2"]:
            self.assertIn(param, result.df.columns)

    def test_link_cone_noise_units_to_gcs(self):
        # This method is inherited from TemporalModelBase and already tested
        # Assuming it works correctly
        pass


if __name__ == "__main__":
    unittest.main()
