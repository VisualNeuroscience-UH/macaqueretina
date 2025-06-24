# test_spatial_model_vae.py

# Built-in
import unittest
from unittest.mock import MagicMock, patch

# Third-party
import numpy as np
import pandas as pd
import torch

# Local
from macaqueretina.retina.construct_retina_module import SpatialModelVAE


class TestSpatialModelVAE(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_DoG_model = MagicMock()
        self.mock_distribution_sampler = MagicMock()
        self.mock_retina_vae = MagicMock()
        self.mock_fit = MagicMock()
        self.mock_retina_math = MagicMock()
        self.mock_viz = MagicMock()

        # Set up DoG_model.fit
        self.mock_DoG_model.fit = self.mock_fit

        # Initialize the SpatialModelVAE with mocked dependencies
        self.model = SpatialModelVAE(
            self.mock_DoG_model,
            self.mock_distribution_sampler,
            self.mock_retina_vae,
            self.mock_fit,
            self.mock_retina_math,
            self.mock_viz,
        )

        # Mock retina and ganglion cell objects
        self.ret = MagicMock()
        self.gc = MagicMock()
        self.gc.n_units = 10
        self.gc.pix_per_side = 32
        self.gc.um_per_pix = 1.0
        self.gc.df = pd.DataFrame(
            {
                "zoom_factor": np.ones(10),
                "com_x_pix": np.zeros(10),
                "com_y_pix": np.zeros(10),
            }
        )

        self.ret.gc_type = "some_gc_type"
        self.ret.response_type = "some_response_type"
        self.ret.dog_model_type = "some_dog_model_type"
        self.ret.whole_ret_lu_mm = [0, 0]

        # Mock methods that will be called within the class methods
        self.mock_retina_math.cart2pol.return_value = (np.zeros(10), np.zeros(10))
        self.model.project_data = {}

        # Set up DoG_model.transform_vae_dog_to_mm
        self.mock_DoG_model.transform_vae_dog_to_mm = MagicMock()

    def test_get_resampled_scaled_gc_img(self):
        # Prepare test data
        rfs = np.random.rand(10, 16, 16)
        pix_per_side = 32
        zoom_factor = np.ones(10) * 2

        # Call the method
        img_upsampled = self.model._get_resampled_scaled_gc_img(
            rfs, pix_per_side, zoom_factor
        )

        # Assertions
        self.assertEqual(img_upsampled.shape, (10, 32, 32))
        self.assertFalse(np.any(np.isnan(img_upsampled)))

    @patch("macaqueretina.retina.construct_retina_module.stats.gaussian_kde")
    def test_get_generated_rfs(self, mock_gaussian_kde):
        # Mock the KDE and sampling
        self.mock_retina_vae.latent_dim = 5  # Assuming latent dimension is 5
        mock_kde_instance = MagicMock()
        mock_kde_instance.resample.return_value = np.random.rand(
            10, self.mock_retina_vae.latent_dim
        )
        mock_gaussian_kde.return_value = mock_kde_instance

        # Mock the VAE decoder output
        img_size = 16
        self.mock_retina_vae.device = "cpu"
        self.mock_retina_vae.vae.decoder.return_value = torch.randn(
            10, 1, img_size, img_size
        )

        # Mock the encoded samples to return DataFrames
        encoded_samples = pd.DataFrame(
            np.random.rand(10, self.mock_retina_vae.latent_dim),
            columns=[f"EncVariable{i}" for i in range(self.mock_retina_vae.latent_dim)],
        )
        self.mock_retina_vae.get_encoded_samples.return_value = encoded_samples

        # Call the method
        img_flipped, img_reshaped = self.model._get_generated_rfs(
            self.mock_retina_vae, n_samples=10
        )

        # Assertions
        self.assertEqual(img_flipped.shape, (10, img_size, img_size))
        self.assertEqual(img_reshaped.shape, (10, img_size, img_size))

    def test_update_vae_gc_df(self):
        # Prepare test data
        gc_df_in = pd.DataFrame(
            {
                "xoc_pix": np.random.rand(10),
                "yoc_pix": np.random.rand(10),
                "ampl_c": np.random.rand(10),
                "ampl_s": np.random.rand(10),
            }
        )
        self.gc.img_lu_pix = np.random.rand(10, 2)
        self.gc.df = pd.DataFrame(
            columns=[
                "pos_ecc_mm",
                "pos_polar_deg",
                "ampl_c",
                "ampl_s",
                "xoc_pix",
                "yoc_pix",
            ]
        )

        # Mock the DoG_model.transform_vae_dog_to_mm to return a DataFrame
        transformed_df = gc_df_in.copy()
        transformed_df["pos_ecc_mm"] = np.random.rand(10)
        transformed_df["pos_polar_deg"] = np.random.rand(10)
        self.mock_DoG_model.transform_vae_dog_to_mm.return_value = transformed_df

        # Call the method
        updated_gc = self.model._update_vae_gc_df(self.ret, self.gc, gc_df_in)

        # Assertions
        self.assertIsNotNone(updated_gc)
        self.assertIn("pos_ecc_mm", updated_gc.df.columns)
        self.assertIn("pos_polar_deg", updated_gc.df.columns)
        self.assertIn("ampl_c", updated_gc.df.columns)
        self.assertIn("ampl_s", updated_gc.df.columns)

    @patch.object(SpatialModelVAE, "_get_generated_rfs")
    def test_get_vae_imgs_with_good_fits(self, mock_get_generated_rfs):
        # Mock the generated RFs
        img_size = 16
        n_samples_extra = 15  # 50% extra
        mock_imgs_processed = np.random.rand(n_samples_extra, img_size, img_size)
        mock_imgs_raw = np.random.rand(n_samples_extra, img_size, img_size)
        mock_get_generated_rfs.return_value = (mock_imgs_processed, mock_imgs_raw)

        # Mock fit results
        self.mock_fit.client.return_value = None
        self.mock_fit.good_idx_generated = np.arange(self.gc.n_units)
        self.mock_fit.all_data_fits_df = pd.DataFrame(
            {
                "xoc_pix": np.random.rand(self.gc.n_units),
                "yoc_pix": np.random.rand(self.gc.n_units),
            }
        )

        # Call the method
        updated_gc = self.model._get_vae_imgs_with_good_fits(
            self.ret, self.gc, self.mock_retina_vae
        )

        # Assertions
        self.assertEqual(
            updated_gc.img.shape,
            (self.gc.n_units, self.gc.pix_per_side, self.gc.pix_per_side),
        )
        self.assertIn("xoc_pix", updated_gc.df.columns)
        self.assertIn("yoc_pix", updated_gc.df.columns)

    def test_get_data_at_latent_space(self):
        # Mock the encoded samples
        self.mock_retina_vae.latent_dim = 5  # Assuming latent dimension is 5
        encoded_samples = pd.DataFrame(
            np.random.rand(10, self.mock_retina_vae.latent_dim),
            columns=[f"EncVariable{i}" for i in range(self.mock_retina_vae.latent_dim)],
        )
        self.mock_retina_vae.get_encoded_samples.return_value = encoded_samples

        # Call the method
        latent_data = self.model._get_data_at_latent_space(self.mock_retina_vae)

        # Assertions
        self.assertEqual(latent_data.shape[1], self.mock_retina_vae.latent_dim)
        self.assertFalse(np.any(np.isnan(latent_data)))

    @patch("macaqueretina.retina.construct_retina_module.apply_rf_repulsion")
    def test_create(self, mock_apply_rf_repulsion):
        # Mock methods called within create
        self.model._get_vae_imgs_with_good_fits = MagicMock(return_value=self.gc)
        self.model._generate_center_masks = MagicMock(return_value=self.gc)
        self.model._get_full_retina_with_rf_images = MagicMock(
            return_value=(self.ret, self.gc, np.zeros((100, 100)))
        )
        self.model._update_vae_gc_df = MagicMock(return_value=self.gc)
        self.model._add_center_mask_area_to_df = MagicMock(return_value=self.gc)
        self.model._get_img_grid_mm = MagicMock(return_value=self.gc)
        mock_apply_rf_repulsion.return_value = (self.ret, self.gc)

        # Mock attributes
        self.ret.whole_ret_img = np.zeros((100, 100))
        self.ret.whole_ret_img_mask = np.zeros((100, 100))
        self.gc.img = np.random.rand(10, 32, 32)
        self.gc.img_mask = np.random.rand(10, 32, 32)

        # Configure the DoG_model.fit.get_generated_spatial_fits to return mock values
        self.model.DoG_model.fit.get_generated_spatial_fits.return_value = (
            pd.DataFrame(),  # gen_stat_df
            np.array([]),  # gen_spat_cen_sd
            np.array([]),  # gen_spat_sur_sd
            pd.DataFrame(
                {
                    "xoc_pix": np.random.rand(self.gc.n_units),
                    "yoc_pix": np.random.rand(self.gc.n_units),
                    "ampl_c": np.random.rand(self.gc.n_units),
                    "ampl_s": np.random.rand(self.gc.n_units),
                }
            ),  # _gc_vae_df
            np.array([True] * self.gc.n_units),  # final_good_idx
        )

        # Call the method
        ret_result, gc_result, viz_whole_ret_img = self.model.create(self.ret, self.gc)

        # Assertions

        # Assertions
        self.assertIsNotNone(ret_result)
        self.assertIsNotNone(gc_result)
        self.assertIsNotNone(viz_whole_ret_img)
        self.assertIn("retina_vae", self.model.project_data)
        self.assertIn("gen_rfs", self.model.project_data)


if __name__ == "__main__":
    unittest.main()
