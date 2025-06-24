# Built-in
import unittest
from unittest.mock import Mock, patch

# Third-party
import brian2.units as b2u
import numpy as np

# Local
from macaqueretina.retina.simulate_retina_module import ConcreteSimulationBuilder


class TestConcreteSimulationBuilder(unittest.TestCase):
    def setUp(self):
        self.mock_vs = Mock()
        self.mock_gcs = Mock()
        self.mock_cones = Mock()
        self.mock_bipolars = Mock()
        self.mock_retina_math = Mock()
        self.mock_device = Mock()

        self.retina_parameters = {
            "gc_type": "parasol",
            "response_type": "on",
            "deg_per_mm": 5.0,
            "dog_model_type": "ellipse_fixed",
            "spatial_model_type": "DOG",
            "temporal_model_type": "fixed",
        }

        self.builder = ConcreteSimulationBuilder(
            self.mock_vs,
            self.mock_gcs,
            self.mock_cones,
            self.mock_bipolars,
            self.mock_retina_math,
            self.mock_device,
            n_trials=1,
        )

    def test_initialization(self):
        self.assertEqual(self.builder.vs, self.mock_vs)
        self.assertEqual(self.builder.gcs, self.mock_gcs)
        self.assertEqual(self.builder.cones, self.mock_cones)
        self.assertEqual(self.builder.bipolars, self.mock_bipolars)
        self.assertEqual(self.builder.retina_math, self.mock_retina_math)
        self.assertEqual(self.builder.device, self.mock_device)
        self.assertEqual(self.builder.n_trials, 1)

    def test_get_concrete_components(self):
        self.builder.gcs.gc_type = "parasol"
        self.builder.gcs.response_type = "on"
        self.builder.gcs.dog_model_type = "ellipse_fixed"
        self.builder.gcs.spatial_model_type = "DOG"
        self.builder.gcs.temporal_model_type = "fixed"

        self.builder.get_concrete_components()

        self.assertIsNotNone(self.builder.ganglion_cell)
        self.assertIsNotNone(self.builder.response_type)
        self.assertIsNotNone(self.builder.DoG_model)
        self.assertIsNotNone(self.builder.spatial_model)
        self.assertIsNotNone(self.builder.temporal_model)

    @patch("macaqueretina.retina.simulate_retina_module.interp1d")
    @patch("macaqueretina.retina.simulate_retina_module.b2")
    def test_firing_rates2brian_timed_arrays(self, mock_b2, mock_interp1d):
        mock_vs = Mock()
        mock_vs.stimulus_video.video_n_frames = 100
        mock_vs.video_dt = 0.01 * b2u.second
        mock_vs.duration = 1.0 * b2u.second
        mock_vs.simulation_dt = 0.001 * b2u.second
        mock_vs.firing_rates = np.random.rand(10, 100)  # 10 units, 100 time points

        mock_interp1d.return_value = lambda x: np.random.rand(len(x), 10)

        # Mock the TimedArray creation
        mock_b2.TimedArray.return_value = Mock()

        result = self.builder._firing_rates2brian_timed_arrays(mock_vs)

        self.assertIsNotNone(result.interpolated_rates_array)
        self.assertIsNotNone(result.tvec_new)
        self.assertIsNotNone(result.inst_rates)

    def test_get_impulse_response(self):
        self.builder._temporal_model = Mock()
        self.builder._temporal_model.impulse_response.return_value = {"test": "data"}

        self.builder.gcs.unit_indices = [0, 1, 2]
        self.builder.gcs.gc_type = "parasol"
        self.builder.gcs.response_type = "on"
        self.builder.gcs.temporal_model_type = "fixed"

        self.builder.get_impulse_response([0.1, 0.5, 1.0])

        self.assertIn("impulse_to_show", self.builder.project_data)
        impulse_data = self.builder.project_data["impulse_to_show"]
        self.assertEqual(impulse_data["gc_type"], "parasol")
        self.assertEqual(impulse_data["response_type"], "on")
        self.assertEqual(impulse_data["temporal_model_type"], "fixed")

    def test_create_spatial_filters(self):
        self.builder.gcs.spatial_filter_sidelen = 10
        self.builder.gcs.n_units = 3
        self.builder.gcs.unit_indices = [0, 1, 2]
        self.builder.gcs.mask_threshold = 0.5
        self.builder.gcs.response_type = "on"

        self.builder._spatial_model = Mock()
        self.builder._spatial_model.create_spatial_filter.return_value = np.random.rand(
            10, 10
        )

        with patch.object(
            self.builder,
            "_get_center_masks",
            return_value=np.random.randint(0, 2, (3, 10, 10)),
        ):
            with patch.object(
                self.builder,
                "_get_surround_masks",
                return_value=np.random.randint(0, 2, (3, 10, 10)),
            ):
                self.builder.create_spatial_filters()

        self.assertIsNotNone(self.builder.gcs.spatial_filters_flat)
        self.assertIsNotNone(self.builder.gcs.center_masks_flat)
        self.assertIsNotNone(self.builder.gcs.surround_masks_flat)
