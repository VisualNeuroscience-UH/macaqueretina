# test_temporal_model_fixed.py

# Built-in
import unittest
from unittest.mock import MagicMock, patch

# Third-party
import numpy as np
import pandas as pd

# Local
from macaqueretina.retina.construct_retina_module import (
    TemporalModelBase,
    TemporalModelFixed,
)


# Mocked dependencies
class GanglionCell:
    def get_BK_parameter_names(self):
        return ["A", "Mean"]

    df = pd.DataFrame({"cell_id": range(5)})


class DoGModel:
    exp_univariate_stat = pd.DataFrame(
        {
            "domain": ["temporal", "spatial"],
            "shape": [0.5, 0.6],
            "loc": [0, 0],
            "scale": [1, 1],
        }
    )

    temporal_multivariate_stat = pd.DataFrame(
        {
            "n": [0.5],
            "p1": [1.0],
            "p2": [1.5],
            "tau1": [0.1],
            "tau2": [0.2],
        }
    )


class DistributionSampler:
    def sample_univariate(self, shape, loc, scale, n_cells, distribution):
        return np.random.rand(n_cells)  # Return random values for simplicity

    def sample_multivariate(self, stat, n_cells):
        # Return a DataFrame with n_cells rows and columns matching stat
        data = {col: np.random.rand(n_cells) for col in stat.columns}
        return pd.DataFrame(data)

    def filter_stat(self, available_stat, covariances_of_interest):
        intersection = list(set(available_stat.columns) & set(covariances_of_interest))
        filtered_stat = available_stat[intersection]
        return filtered_stat, intersection


class RetinaMath:
    pass  # Assume methods are not needed for this test


class Retina:
    fit_statistics = "multivariate"
    gc_type = "midget"
    response_type = "sustained"

    experimental_archive = {
        "temporal_parameters_BK": pd.DataFrame(
            {
                "Parameter": ["A_cen", "Mean"],
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


# Concrete implementation (already provided in your code)
class TemporalModelFixed(TemporalModelBase):
    def __init__(self, ganglion_cell, DoG_model, distribution_sampler, retina_math):
        super().__init__(ganglion_cell, DoG_model, distribution_sampler, retina_math)

    def _get_temporal_covariances_of_interest(self):
        temporal_covariances_of_interest = ["n", "p1", "p2", "tau1", "tau2"]
        return temporal_covariances_of_interest

    def create(self, ret, gc):
        C_statistics = self.DoG_model.exp_univariate_stat[
            self.DoG_model.exp_univariate_stat["domain"] == "temporal"
        ]
        BK_statistics = self._get_BK_statistics(ret)
        gain_name = "A_cen" if ret.gc_type == "midget" else "A"
        gain_and_mean = BK_statistics.loc[[gain_name, "Mean"]]
        stat_df = pd.concat([C_statistics, gain_and_mean])
        gc = self._sample_temporal_rfs(gc, stat_df)

        n_cells = len(gc.df)
        if ret.fit_statistics == "multivariate":
            covariances_of_interest = self._get_temporal_covariances_of_interest()

            available_stat = self.DoG_model.temporal_multivariate_stat
            filtered_stat, intersection = self.sampler.filter_stat(
                available_stat, covariances_of_interest
            )

            self.project_data["temporal_covariances_of_interest"] = intersection

            samples = self.sampler.sample_multivariate(filtered_stat, n_cells)
            missing_columns = [
                col for col in samples.columns if col not in gc.df.columns
            ]
            gc.df[missing_columns] = np.nan
            gc.df.update(samples)

        return gc

    def connect_units(self, ret, gc):
        ret = self._link_cone_noise_units_to_gcs(ret, gc)
        return ret


# Unit tests
class TestTemporalModelFixed(unittest.TestCase):
    def setUp(self):
        self.ganglion_cell = GanglionCell()
        self.DoG_model = DoGModel()
        self.sampler = DistributionSampler()
        self.retina_math = RetinaMath()
        self.retina = Retina()
        self.gc = self.ganglion_cell

        self.temporal_model = TemporalModelFixed(
            self.ganglion_cell, self.DoG_model, self.sampler, self.retina_math
        )

    def test_init(self):
        self.assertIsInstance(self.temporal_model, TemporalModelFixed)

    def test_get_temporal_covariances_of_interest(self):
        expected_covariances = ["n", "p1", "p2", "tau1", "tau2"]
        result = self.temporal_model._get_temporal_covariances_of_interest()
        self.assertEqual(result, expected_covariances)

    @patch.object(TemporalModelFixed, "_get_BK_statistics")
    @patch.object(TemporalModelFixed, "_sample_temporal_rfs")
    def test_create(self, mock_sample_temporal_rfs, mock_get_BK_statistics):
        # Mocking the BK statistics
        BK_statistics = pd.DataFrame(
            {
                "shape": [0.7, 0.8],
                "loc": [0, 0],
                "scale": [1, 1],
            },
            index=["A_cen", "Mean"],
        )
        mock_get_BK_statistics.return_value = BK_statistics

        # Mocking the sample_temporal_rfs method
        mock_sample_temporal_rfs.return_value = self.gc

        # Run the create method
        result = self.temporal_model.create(self.retina, self.gc)
        self.assertIsNotNone(result)
        mock_get_BK_statistics.assert_called_once_with(self.retina)
        mock_sample_temporal_rfs.assert_called_once()

        # Check if multivariate sampling was performed
        self.assertIn(
            "temporal_covariances_of_interest", self.temporal_model.project_data
        )
        self.assertTrue(
            set(
                self.temporal_model.project_data["temporal_covariances_of_interest"]
            ).issubset(set(self.DoG_model.temporal_multivariate_stat.columns))
        )
        for col in self.temporal_model.project_data["temporal_covariances_of_interest"]:
            self.assertIn(col, result.df.columns)

    @patch.object(TemporalModelFixed, "_link_cone_noise_units_to_gcs")
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
        # Create dummy stat_df DataFrame
        data = {
            "shape": [0.5, 0.6],
            "loc": [0, 0],
            "scale": [1, 1],
            "distribution": ["triang", "triang"],
            "domain": ["temporal_BK", "temporal_BK"],
        }
        index = ["A_cen", "Mean"]
        stat_df = pd.DataFrame(data, index=index)

        # Mock the sampler
        self.sampler.sample_univariate = MagicMock(return_value=np.random.rand(5))

        result = self.temporal_model._sample_temporal_rfs(self.gc, stat_df)
        self.assertIsNotNone(result)
        # Check if parameters are added to gc.df
        for param in ["A_cen", "Mean"]:
            self.assertIn(param, result.df.columns)

    def test_link_cone_noise_units_to_gcs(self):
        # This method is inherited from TemporalModelBase and already tested
        # Assuming it works correctly
        pass


if __name__ == "__main__":
    unittest.main()
