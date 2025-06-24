# Built-in
import unittest
from unittest.mock import Mock, patch

# Third-party
import numpy as np
import pandas as pd

# Local
from macaqueretina.retina.simulate_retina_module import (
    GanglionCellProduct,
    ReceptiveFieldsBase,
)


class TestGanglionCellData(unittest.TestCase):
    def setUp(self):
        self.mock_receptive_fields_base = patch(
            "macaqueretina.retina.simulate_retina_module.ReceptiveFieldsBase"
        ).start()

        self.mock_retina = {
            "center_mask_threshold": 0.5,
            "refractory_parameters": {"param1": 1, "param2": 2},
            "visual2cortical_params": {"param3": 3, "param4": 4},
            "gc_type": "mock_type",
            "response_type": "ON",
            "dog_model_type": "ellipse_fixed",
            "spatial_model_type": "DOG",
            "temporal_model_type": "dynamic",
            "deg_per_mm": 1.0,
        }
        self.mock_apricot_metadata_parameters = {
            "data_microm_per_pix": 1.0,
            "data_fps": 30,
            "data_temporalfilter_samples": 10,
        }
        self.mock_rfs_npz = {
            "gc_img": np.random.rand(10, 10),
            "um_per_pix": 1.0,
            "pix_per_side": 10,
        }
        self.mock_gc_dataframe = pd.DataFrame(
            {
                "x_deg": [1, 2, 3],
                "y_deg": [4, 5, 6],
                "semi_xc_mm": [0.1, 0.2, 0.3],
                "semi_yc_mm": [0.1, 0.2, 0.3],
                "ampl_c_norm": [1, 1, 1],
                "ampl_s_norm": [0.5, 0.5, 0.5],
                "orient_cen_rad": [0, 0, 0],
                "relat_sur_diam": [2, 2, 2],
                "pos_ecc_mm": [1, 2, 3],
                "pos_polar_deg": [30, 60, 90],
                "xoc_pix": [10, 20, 30],
                "yoc_pix": [15, 25, 35],
                "xos_pix": [11, 21, 31],
                "yos_pix": [16, 26, 36],
            }
        )
        self.mock_spike_generator_model = Mock()
        self.mock_pol2cart_df = Mock(return_value=np.array([[1, 2], [3, 4], [5, 6]]))

    def tearDown(self):
        patch.stopall()

    def test_init(self):
        gcd = GanglionCellProduct(
            self.mock_retina,
            self.mock_apricot_metadata_parameters,
            self.mock_rfs_npz,
            self.mock_gc_dataframe,
            unit_index=None,
            spike_generator_model=self.mock_spike_generator_model,
            pol2cart_df=self.mock_pol2cart_df,
        )

        self.assertEqual(gcd.spike_generator_model, self.mock_spike_generator_model)
        self.assertEqual(gcd.mask_threshold, 0.5)
        self.assertEqual(gcd.refractory_parameters, {"param1": 1, "param2": 2})
        self.assertEqual(
            gcd.apricot_metadata_parameters, self.mock_apricot_metadata_parameters
        )
        self.assertEqual(gcd.data_microm_per_pixel, 1.0)
        self.assertEqual(gcd.data_filter_fps, 30)
        self.assertEqual(gcd.data_filter_timesteps, 10)
        self.assertAlmostEqual(gcd.data_filter_duration, 333.33333333333337)
        self.assertEqual(gcd.visual2cortical_params, {"param3": 3, "param4": 4})
        self.assertEqual(gcd.n_units, 3)
        np.testing.assert_array_equal(gcd.unit_indices, np.array([0, 1, 2]))

    def test_init_with_unit_index(self):
        gcd = GanglionCellProduct(
            self.mock_retina,
            self.mock_apricot_metadata_parameters,
            self.mock_rfs_npz,
            self.mock_gc_dataframe,
            unit_index=[0, 2],
            spike_generator_model=self.mock_spike_generator_model,
            pol2cart_df=self.mock_pol2cart_df,
        )

        self.assertEqual(gcd.n_units, 2)
        np.testing.assert_array_equal(gcd.unit_indices, np.array([0, 2]))

    def test_link_gcs_to_vs(self):
        gcd = GanglionCellProduct(
            self.mock_retina,
            self.mock_apricot_metadata_parameters,
            self.mock_rfs_npz,
            self.mock_gc_dataframe,
            unit_index=None,
            spike_generator_model=self.mock_spike_generator_model,
            pol2cart_df=self.mock_pol2cart_df,
        )

        mock_vs = Mock()
        mock_vs._vspace_to_pixspace = Mock(return_value=(10, 20))
        mock_vs.pix_per_deg = 10
        mock_vs.fps = 60

        gcd.link_gcs_to_vs(mock_vs)

        self.assertIsNotNone(gcd.df_stimpix)
        self.assertEqual(len(gcd.df_stimpix), 3)
        self.assertTrue("q_pix" in gcd.df_stimpix.columns)
        self.assertTrue("r_pix" in gcd.df_stimpix.columns)
        self.assertAlmostEqual(gcd.microm_per_pix, 100.0)
        self.assertEqual(gcd.temporal_filter_len, 20)
