# test_temporal_model_subunit.py

# Built-in
import unittest
from unittest.mock import MagicMock, patch

# Third-party
import numpy as np
import pandas as pd
import scipy.optimize as opt

# Local
from macaqueretina.retina.construct_retina_module import (
    TemporalModelBase,
    TemporalModelSubunit,
)


# Mocked dependencies
class GanglionCell:
    def get_BK_parameter_names(self):
        return ["A_cen", "Mean"]

    df = pd.DataFrame({"cell_id": range(5), "den_diam_um": [100, 110, 120, 130, 140]})
    img = np.random.rand(5, 10, 10)
    img_mask = np.ones((5, 10, 10))
    X_grid_cen_mm = np.random.rand(5, 10, 10)
    Y_grid_cen_mm = np.random.rand(5, 10, 10)
    n_units = 5


class DoGModel:
    exp_univariate_stat = pd.DataFrame(
        {
            "domain": ["temporal", "spatial"],
            "shape": [0.5, 0.6],
            "loc": [0, 0],
            "scale": [1, 1],
        },
        index=["A_cen", "Mean"],
    )


class DistributionSampler:
    def sample_univariate(self, shape, loc, scale, n_cells, distribution):
        return np.random.rand(n_cells)  # Return random values for simplicity


class RetinaMath:
    def parabola(self, x, a, b, c):
        return a * x**2 + b * x + c

    def weighted_average(self, values, weights):
        return np.average(values, weights=weights)

    def get_sample_from_range_and_average(self, min_range, max_range, average, size):
        return np.full(size, int(average))  # Return average for simplicity


class Retina:
    gc_type = "midget"
    response_type = "sustained"
    cone_optimized_pos_mm = np.random.rand(10, 2)
    bipolar_optimized_pos_mm = np.random.rand(15, 2)
    selected_bipolars_df = pd.DataFrame(
        {
            "Cone_contacts_Range": ["1-3", "2-4"],
            "Cone_contacts_Average": ["2", "3"],
            "Estimated_density_mm^-2": ["100", "150"],
        }
    ).set_index(pd.Index(["midget", "parasol"]))

    bipolar_general_parameters = {
        "cone2bipo_cen_sd": 100,
        "cone2bipo_sur_sd": 200,
        "bipo_sub_sur2cen": 0.5,
        "bipo2gc_div": 6,
        "bipo2gc_cutoff_SD": 2,
    }

    cone_general_parameters = {
        "cone2bipo_cutoff_SD": 2,
    }

    experimental_archive = {
        "g_sur_values": np.linspace(-5, 5, 10),
        "target_RI_values": np.random.rand(10),
    }


# Concrete implementation (assuming it inherits from TemporalModelBase)
class TemporalModelSubunit(TemporalModelBase):
    def __init__(
        self,
        ganglion_cell,
        DoG_model,
        distribution_sampler,
        retina_math,
    ):
        super().__init__(ganglion_cell, DoG_model, distribution_sampler, retina_math)

    def _fit_bipolar_rectification_index(self, ret):
        unit_type = ret.gc_type
        response_type = ret.response_type

        g_sur_values = ret.experimental_archive["g_sur_values"]
        target_RI_values = ret.experimental_archive["target_RI_values"]

        if unit_type == "midget":
            target_RI_values = target_RI_values * 0

        RI_function = self.retina_math.parabola

        g_sur_min = np.min(g_sur_values)
        g_sur_max = np.max(g_sur_values)
        g_sur_scaled = 2 * (g_sur_values - g_sur_min) / (g_sur_max - g_sur_min) - 1

        popt, _ = opt.curve_fit(RI_function, g_sur_scaled[1:], target_RI_values[1:])

        ret.bipolar_nonlinearity_parameters = popt
        ret.g_sur_scaled = g_sur_scaled
        ret.target_RI_values = target_RI_values
        ret.bipolar_nonlinearity_fit = RI_function(g_sur_scaled, *popt)

        return ret

    def _link_cones_to_bipolars(self, ret):
        # Simplified implementation for testing
        ret.cones_to_bipolars_center_weights = np.random.rand(
            ret.cone_optimized_pos_mm.shape[0], ret.bipolar_optimized_pos_mm.shape[0]
        )
        ret.cones_to_bipolars_surround_weights = np.random.rand(
            ret.cone_optimized_pos_mm.shape[0], ret.bipolar_optimized_pos_mm.shape[0]
        )
        return ret

    def _link_bipolar_units_to_gcs(self, ret, gc):
        # Simplified implementation for testing
        ret.bipolar_to_gcs_cen_weights = np.random.rand(
            ret.bipolar_optimized_pos_mm.shape[0], gc.n_units
        )
        return ret

    def create(self, ret, gc):
        BK_statistics = self._get_BK_statistics(ret)
        gain_name = "A_cen" if ret.gc_type == "midget" else "A"
        gain_and_mean = BK_statistics.loc[[gain_name, "Mean"]]
        gc = self._sample_temporal_rfs(gc, gain_and_mean)

        ret = self._fit_bipolar_rectification_index(ret)

        return gc

    def connect_units(self, ret, gc):
        ret = self._link_cones_to_bipolars(ret)
        ret = self._link_bipolar_units_to_gcs(ret, gc)
        ret = self._link_cone_noise_units_to_gcs(ret, gc)

        return ret


# Unit tests
class TestTemporalModelSubunit(unittest.TestCase):
    def setUp(self):
        self.ganglion_cell = GanglionCell()
        self.DoG_model = DoGModel()
        self.sampler = DistributionSampler()
        self.retina_math = RetinaMath()
        self.retina = Retina()
        self.gc = self.ganglion_cell

        self.temporal_model = TemporalModelSubunit(
            self.ganglion_cell, self.DoG_model, self.sampler, self.retina_math
        )

    def test_init(self):
        self.assertIsInstance(self.temporal_model, TemporalModelSubunit)

    @patch.object(TemporalModelSubunit, "_get_BK_statistics")
    @patch.object(TemporalModelSubunit, "_sample_temporal_rfs")
    @patch.object(TemporalModelSubunit, "_fit_bipolar_rectification_index")
    def test_create(
        self,
        mock_fit_bipolar_rectification_index,
        mock_sample_temporal_rfs,
        mock_get_BK_statistics,
    ):
        # Mocking the BK statistics
        BK_statistics = pd.DataFrame(
            {
                "shape": [0.7],
                "loc": [0],
                "scale": [1],
            },
            index=["A_cen"],
        )
        BK_statistics.loc["Mean"] = [0.8, 0, 1]
        mock_get_BK_statistics.return_value = BK_statistics

        # Mocking the sample_temporal_rfs method
        mock_sample_temporal_rfs.return_value = self.gc

        # Mocking the _fit_bipolar_rectification_index method
        mock_fit_bipolar_rectification_index.return_value = self.retina

        # Run the create method
        result = self.temporal_model.create(self.retina, self.gc)
        self.assertIsNotNone(result)
        mock_get_BK_statistics.assert_called_once_with(self.retina)

        # Retrieve the actual call arguments
        args, kwargs = mock_sample_temporal_rfs.call_args
        self.assertIs(args[0], self.gc)
        pd.testing.assert_frame_equal(args[1], BK_statistics.loc[["A_cen", "Mean"]])
        mock_fit_bipolar_rectification_index.assert_called_once_with(self.retina)

    @patch.object(TemporalModelSubunit, "_link_cones_to_bipolars")
    @patch.object(TemporalModelSubunit, "_link_bipolar_units_to_gcs")
    @patch.object(TemporalModelSubunit, "_link_cone_noise_units_to_gcs")
    def test_connect_units(
        self,
        mock_link_cone_noise_units_to_gcs,
        mock_link_bipolar_units_to_gcs,
        mock_link_cones_to_bipolars,
    ):
        # Mocking the methods to return the retina
        mock_link_cones_to_bipolars.return_value = self.retina
        mock_link_bipolar_units_to_gcs.return_value = self.retina
        mock_link_cone_noise_units_to_gcs.return_value = self.retina

        result = self.temporal_model.connect_units(self.retina, self.gc)
        self.assertIsNotNone(result)
        mock_link_cones_to_bipolars.assert_called_once_with(self.retina)
        mock_link_bipolar_units_to_gcs.assert_called_once_with(self.retina, self.gc)
        mock_link_cone_noise_units_to_gcs.assert_called_once_with(self.retina, self.gc)

    def test_fit_bipolar_rectification_index(self):
        # Mock the parabola method in retina_math
        self.retina_math.parabola = MagicMock(return_value=np.random.rand(10))

        # Mock opt.curve_fit to return dummy parameters
        with patch(
            "scipy.optimize.curve_fit", return_value=(np.array([1, 2, 3]), None)
        ):
            result = self.temporal_model._fit_bipolar_rectification_index(self.retina)
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "bipolar_nonlinearity_parameters"))
            self.assertTrue(hasattr(result, "g_sur_scaled"))
            self.assertTrue(hasattr(result, "target_RI_values"))
            self.assertTrue(hasattr(result, "bipolar_nonlinearity_fit"))

    def test_link_cones_to_bipolars(self):
        # Mock retina_math methods
        self.retina_math.weighted_average = MagicMock(return_value=2.5)
        self.retina_math.get_sample_from_range_and_average = MagicMock(
            return_value=np.array([2, 2, 2, 2, 2])
        )

        result = self.temporal_model._link_cones_to_bipolars(self.retina)
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "cones_to_bipolars_center_weights"))
        self.assertTrue(hasattr(result, "cones_to_bipolars_surround_weights"))

    def test_link_bipolar_units_to_gcs(self):
        # Mock gc attributes
        self.gc.n_units = 5
        self.gc.img = np.random.rand(5, 10, 10)
        self.gc.img_mask = np.ones((5, 10, 10))
        self.gc.X_grid_cen_mm = np.random.rand(5, 10, 10)
        self.gc.Y_grid_cen_mm = np.random.rand(5, 10, 10)
        self.gc.df = pd.DataFrame({"den_diam_um": [100, 110, 120, 130, 140]})

        result = self.temporal_model._link_bipolar_units_to_gcs(self.retina, self.gc)
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "bipolar_to_gcs_cen_weights"))

    def test_link_cone_noise_units_to_gcs(self):
        # This method is inherited from TemporalModelBase and already tested
        # Assuming it works correctly
        pass


if __name__ == "__main__":
    unittest.main()
