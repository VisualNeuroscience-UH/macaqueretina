# Built-in
import unittest
from unittest.mock import ANY, MagicMock, patch

# Third-party
import numpy as np

# Local
from macaqueretina.retina.construct_retina_module import SpatialModelDOG


class TestSpatialModelFIT(unittest.TestCase):

    def setUp(self):
        self.mock_DoG_model = MagicMock()
        self.mock_distribution_sampler = MagicMock()
        self.mock_retina_vae = MagicMock()
        self.mock_fit = MagicMock()
        self.mock_retina_math = MagicMock()
        self.mock_viz = MagicMock()
        self.model = SpatialModelDOG(
            self.mock_DoG_model,
            self.mock_distribution_sampler,
            self.mock_retina_vae,
            self.mock_fit,
            self.mock_retina_math,
            self.mock_viz,
        )

    def test_generate_DoG_with_rf_from_literature(self):
        ret = MagicMock()
        gc = MagicMock()
        apricot_metadata_parameters = {"data_microm_per_pix": 2.0}
        self.mock_DoG_model.exp_univariate_stat = MagicMock()
        self.mock_DoG_model.scale_to_mm.return_value = MagicMock()
        n_cells = 10
        gc.df = MagicMock()
        gc.df_fixed_values = {"param1": 1.0}
        result_gc = self.model._generate_DoG_with_rf_from_literature(
            ret, gc, apricot_metadata_parameters
        )
        self.mock_DoG_model.scale_to_mm.assert_called_once()
        self.assertEqual(gc, result_gc)

    def test_get_gc_fit_img(self):
        gc = MagicMock()
        gc.n_units = 5
        gc.pix_per_side = 32
        gc.exp_pix_per_side = 32
        gc.um_per_pix = 1000.0
        gc.parameter_names = ["param1", "param2"]
        gc.df = MagicMock()
        gc.mm_scaling_params = ["param1mm"]
        gc.zoom_scaling_params = ["param2"]
        self.mock_DoG_model.get_param_names.return_value = gc
        self.mock_DoG_model.generate_fit_img.return_value = np.zeros((32, 32))
        result_gc = self.model._get_gc_fit_img(gc)
        self.assertEqual(result_gc.img.shape, (5, 32, 32))
        self.mock_DoG_model.get_param_names.assert_called_once()
        self.mock_DoG_model.generate_fit_img.assert_called()

    def test_get_spatial_covariances_of_interest(self):
        covariances = self.model._get_spatial_covariances_of_interest()
        expected_covariances = [
            "semi_xc_pix",
            "semi_yc_pix",
            "ampl_s",
            "semi_xs_pix",
            "semi_ys_pix",
            "rad_c_pix",
            "relat_sur_diam",
        ]
        self.assertEqual(covariances, expected_covariances)

    @patch("macaqueretina.retina.construct_retina_module.apply_rf_repulsion")
    @patch(
        "macaqueretina.retina.construct_retina_module.SpatialModelDOG._get_img_grid_mm"
    )
    @patch(
        "macaqueretina.retina.construct_retina_module.SpatialModelDOG._generate_center_masks"
    )
    @patch(
        "macaqueretina.retina.construct_retina_module.SpatialModelDOG._get_full_retina_with_rf_images"
    )
    @patch(
        "macaqueretina.retina.construct_retina_module.SpatialModelDOG._get_gc_fit_img"
    )
    @patch(
        "macaqueretina.retina.construct_retina_module.SpatialModelDOG._generate_DoG_with_rf_from_literature"
    )
    def test_create(
        self,
        mock_generate_DoG,
        mock_get_gc_fit_img,
        mock_get_full_retina,
        mock_generate_center_masks,
        mock_get_img_grid_mm,
        mock_apply_rf_repulsion,
    ):
        # Mocking the input objects
        ret = MagicMock()
        gc = MagicMock()
        gc.n_units = 5
        gc.um_per_pix = 1.0
        gc.img_lu_pix = np.array([[0, 0]] * gc.n_units)
        gc.img = np.zeros((gc.n_units, 32, 32))
        gc.img_mask = np.zeros((gc.n_units, 32, 32))
        ret.whole_ret_img = "whole_ret_img"

        apricot_metadata_parameters = {"data_microm_per_pix": 2.0}
        ret.experimental_archive = {
            "apricot_metadata_parameters": apricot_metadata_parameters
        }
        ret.mask_threshold = 0.5

        # Mock the behavior of the internal methods
        mock_generate_DoG.return_value = gc
        mock_get_gc_fit_img.return_value = gc
        mock_generate_center_masks.return_value = gc
        mock_get_full_retina.return_value = (ret, gc, ret.whole_ret_img)
        mock_get_img_grid_mm.return_value = gc
        mock_apply_rf_repulsion.return_value = (ret, gc)

        # Mock DoG_model._get_dd_in_um to return gc
        self.model.DoG_model._get_dd_in_um.return_value = gc

        # Call the method under test
        result_ret, result_gc, result_img = self.model.create(ret, gc)

        # Assert the method calls and returned values
        mock_generate_DoG.assert_called_once_with(ret, gc, apricot_metadata_parameters)
        mock_get_gc_fit_img.assert_called_once_with(gc)
        mock_generate_center_masks.assert_called_once_with(ret, gc)
        mock_get_full_retina.assert_any_call(ret, gc, gc.img)
        mock_get_full_retina.assert_any_call(ret, gc, ANY, apply_pix_scaler=False)
        mock_apply_rf_repulsion.assert_called_once_with(ret, gc, self.model.viz)
        mock_get_img_grid_mm.assert_called_once_with(ret, gc)
        self.assertEqual(result_ret, ret)
        self.assertEqual(result_gc, gc)
        self.assertEqual(result_img, ret.whole_ret_img)


if __name__ == "__main__":
    unittest.main()
