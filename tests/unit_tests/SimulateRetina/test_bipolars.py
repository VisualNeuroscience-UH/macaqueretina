# Built-in
import unittest
from unittest.mock import Mock, patch

# Third-party
import numpy as np

# Local
from macaqueretina.retina.simulate_retina_module import (
    BipolarProduct,
    ReceptiveFieldsBase,
)


class TestBipolars(unittest.TestCase):
    def setUp(self):
        # Mock the ReceptiveFieldsBase class
        self.mock_receptive_fields_base = patch(
            "macaqueretina.retina.simulate_retina_module.ReceptiveFieldsBase"
        ).start()

        # Create mock objects and data
        self.mock_retina = {
            "response_type": "on",
            "gc_type": "mock_gc_type",
            "dog_model_type": "mock_dog_model",
            "spatial_model_type": "mock_spatial_model",
            "temporal_model_type": "mock_temporal_model",
            "deg_per_mm": 1.0,
        }
        self.mock_ret_npz = {
            "bipolar_to_gcs_weights": np.random.rand(5, 10),
            "bipolar_nonlinearity_parameters": np.array([1, 2, 3]),
            "cones_to_bipolars_center_weights": np.random.rand(10, 5),
            "cones_to_bipolars_surround_weights": np.random.rand(10, 5),
        }

        # Create a BipolarProduct instance
        self.bipolars = BipolarProduct(self.mock_retina, self.mock_ret_npz)

    def tearDown(self):
        patch.stopall()

    def test_init(self):
        self.assertEqual(self.bipolars.retina_parameters, self.mock_retina)
        self.assertEqual(self.bipolars.ret_npz, self.mock_ret_npz)
        self.assertEqual(self.bipolars.n_units, 5)
        self.assertIsNone(self.bipolars.target_gc_for_multiple_trials)

    def test_init_with_target_gc(self):
        bipolars = BipolarProduct(
            self.mock_retina, self.mock_ret_npz, target_gc_for_multiple_trials=2
        )
        self.assertEqual(bipolars.target_gc_for_multiple_trials, 2)

    def test_create_signal_on_response(self):
        mock_vs = Mock()
        mock_vs.cone_signal = np.random.rand(10, 100)
        mock_vs.baseline_len_tp = 20

        # Mock the parabola method
        self.bipolars.parabola = Mock(return_value=np.random.rand(10, 100))

        result = self.bipolars.create_signal(mock_vs)

        self.assertIsNotNone(result.bipolar_signal)
        self.assertIsNotNone(result.generator_potentials)
        self.assertEqual(result.generator_potentials.shape, (10, 100))

    def test_create_signal_off_response(self):
        self.bipolars.retina_parameters["response_type"] = "off"

        mock_vs = Mock()
        mock_vs.cone_signal = np.random.rand(10, 100)
        mock_vs.baseline_len_tp = 20

        # Mock the parabola method
        self.bipolars.parabola = Mock(return_value=np.random.rand(10, 100))

        result = self.bipolars.create_signal(mock_vs)

        self.assertIsNotNone(result.bipolar_signal)
        self.assertIsNotNone(result.generator_potentials)
        self.assertEqual(result.generator_potentials.shape, (10, 100))

    def test_create_signal_with_target_gc(self):
        self.bipolars.target_gc_for_multiple_trials = 2

        mock_vs = Mock()
        mock_vs.cone_signal = np.random.rand(10, 100)
        mock_vs.baseline_len_tp = 20

        # Mock the parabola method
        self.bipolars.parabola = Mock(return_value=np.random.rand(10, 100))

        result = self.bipolars.create_signal(mock_vs)

        self.assertIsNotNone(result.bipolar_signal)
        self.assertIsNotNone(result.generator_potentials)
        self.assertEqual(result.generator_potentials.shape, (100,))
