# Built-in
import unittest
from unittest.mock import Mock, patch

# Third-party
import numpy as np

# Local
from macaqueretina.retina.simulate_retina_module import ConeProduct, ReceptiveFieldsBase


class TestCones(unittest.TestCase):
    def setUp(self):
        # Mock the ReceptiveFieldsBase class
        self.mock_receptive_fields_base = patch(
            "macaqueretina.retina.simulate_retina_module.ReceptiveFieldsBase"
        ).start()

        # Create mock objects and data
        self.mock_retina = {
            "gc_type": "mock_gc_type",
            "response_type": "ON",
            "cone_general_parameters": {
                "cone_noise_wc": [1, 2],
                "noise_gain": 0.1,
            },
            "cone_signal_parameters": {
                "lambda_nm": 555,
                "A_pupil": 1.0,
                "alpha": 1.0,
                "beta": 1.0,
                "gamma": 1.0,
                "tau_y": 1.0,
                "n_y": 1,
                "tau_z": 1.0,
                "n_z": 1,
                "tau_r": 1.0,
                "filter_limit_time": 1.0,
                "input_gain": 1.0,
                "max_response": 1.0,
                "r_dark": 1.0,
                "unit": "mV",
            },
            "dog_model_type": "mock_dog_model",  # Potentially required key
            "spatial_model_type": "mock_spatial_model",  # Potentially required key
            "temporal_model_type": "mock_temporal_model",  # Potentially required key
            "deg_per_mm": 1.0,  # Potentially required key
        }
        self.mock_ret_npz = {
            "cones_to_gcs_weights": np.random.rand(5, 10),
            "cone_noise_parameters": np.array([1, 2, 3]),
            "cone_optimized_pos_mm": np.random.rand(5, 2),
            "cone_frequency_data": np.linspace(0, 10, 100),
            "cone_power_data": np.random.rand(100),
        }
        self.mock_device = Mock()
        self.mock_interpolate_data = Mock()
        self.mock_lin_interp_and_double_lorenzian = Mock()

        # Create a ConeProduct instance
        self.cones = ConeProduct(
            self.mock_retina,
            self.mock_ret_npz,
            self.mock_device,
            ND_filter=1.0,
            interpolate_data=self.mock_interpolate_data,
            lin_interp_and_double_lorenzian=self.mock_lin_interp_and_double_lorenzian,
        )

    def tearDown(self):
        patch.stopall()

    def test_init(self):
        self.assertEqual(self.cones.retina_parameters, self.mock_retina)
        self.assertEqual(self.cones.ret_npz, self.mock_ret_npz)
        self.assertEqual(self.cones.device, self.mock_device)
        self.assertEqual(self.cones.ND_filter, 1.0)
        self.assertEqual(self.cones.interpolate_data, self.mock_interpolate_data)
        self.assertEqual(
            self.cones.lin_interp_and_double_lorenzian,
            self.mock_lin_interp_and_double_lorenzian,
        )
        self.assertEqual(self.cones.n_units, 5)
        self.assertIsNone(self.cones.target_gc_for_multiple_trials)

    @patch(
        "macaqueretina.retina.simulate_retina_module.ConeProduct._create_cone_signal_clark"
    )
    def test_create_signal(self, mock_create_cone_signal_clark):
        mock_vs = Mock()
        mock_vs.stimulus_video.frames = np.random.rand(100, 10, 10)
        mock_vs.deg_per_mm = 1.0
        mock_vs.tvec = np.linspace(0, 1, 100)
        mock_vs.video_dt = 0.01
        mock_vs.duration = 1.0
        mock_vs.options_from_videofile = {"background": 0.5}

        # Mock the _vspace_to_pixspace method to return a tuple
        mock_vs._vspace_to_pixspace.return_value = (
            np.random.rand(5),
            np.random.rand(5),
        )

        # Mock the photodiode_response attribute
        mock_vs.photodiode_response = np.random.rand(100)

        mock_create_cone_signal_clark.return_value = (
            np.random.rand(5, 100),
            np.random.rand(5, 100),
        )

        # Mock the get_photoisomerizations_from_luminance method
        self.cones.get_photoisomerizations_from_luminance = Mock(
            return_value=np.random.uniform(0, 100)
        )

        result = self.cones.create_signal(mock_vs)

        self.assertIsNotNone(result.cone_signal)
        self.assertIsNotNone(result.cone_signal_u)
        self.assertIsNotNone(result.photodiode_response)
        mock_create_cone_signal_clark.assert_called_once()
        mock_vs._vspace_to_pixspace.assert_called_once()
        self.cones.get_photoisomerizations_from_luminance.assert_called()

    @patch("macaqueretina.retina.simulate_retina_module.ConeProduct._create_cone_noise")
    def test_create_noise(self, mock_create_cone_noise):
        mock_vs = Mock()
        mock_vs.tvec = np.linspace(0, 1, 100)
        mock_create_cone_noise.return_value = np.random.rand(100, 5)

        result = self.cones.create_noise(mock_vs, n_trials=1)

        self.assertIsNotNone(result.cone_noise)
        self.assertIsNotNone(result.cone_noise_u)
        self.assertIsNotNone(result.gc_synaptic_noise)
        mock_create_cone_noise.assert_called_once()

    def test_cornea_photon_flux_density_to_luminance(self):
        F = 1e6  # 1 million photons/mm²/s
        L = self.cones._cornea_photon_flux_density_to_luminance(F)
        self.assertGreater(L, 0)

    def test_luminance_to_cornea_photon_flux_density(self):
        L = 100  # 100 cd/m²
        F = self.cones._luminance_to_cornea_photon_flux_density(L)
        self.assertGreater(F, 0)
