# Built-in
import unittest
from unittest.mock import Mock, patch

# Third-party
import brian2.units as b2u
import numpy as np
import pandas as pd

# Local
from macaqueretina.retina.simulate_retina_module import TemporalModelFixed


class TestTemporalModelFixed(unittest.TestCase):

    def setUp(self):
        self.retina_math = Mock()
        self.ganglion_cell = Mock()
        self.response_type = Mock()
        self.model = TemporalModelFixed(
            self.retina_math, self.ganglion_cell, self.response_type
        )

    def test_initialization(self):
        self.assertIsInstance(self.model, TemporalModelFixed)
        self.assertEqual(self.model.retina_math, self.retina_math)
        self.assertEqual(self.model.ganglion_cell, self.ganglion_cell)
        self.assertEqual(self.model.response_type, self.response_type)

    def test_create_temporal_filter(self):
        gcs = Mock()
        gcs.df = Mock()
        gcs.df.iloc = Mock()
        gcs.df.iloc.__getitem__ = Mock(
            return_value=pd.Series({"n": 1, "p1": 2, "p2": 3, "tau1": 4, "tau2": 5})
        )
        gcs.data_filter_duration = 100
        gcs.temporal_filter_len = 1000

        self.retina_math.diff_of_lowpass_filters.return_value = np.ones(1000)

        result = self.model._create_temporal_filter(gcs, 0)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1000,))
        np.testing.assert_almost_equal(np.sum(np.abs(result)), 1.0)

    def test_get_linear_temporal_filters(self):
        gcs = Mock()
        gcs.unit_indices = [0, 1, 2]
        gcs.temporal_filter_len = 100

        with patch.object(
            self.model, "_create_temporal_filter", return_value=np.ones(100)
        ):
            result = self.model._get_linear_temporal_filters(gcs)

        self.assertIsInstance(result.temporal_filters, np.ndarray)
        self.assertEqual(result.temporal_filters.shape, (3, 100))

    def test_get_linear_spatiotemporal_filters(self):
        gcs = Mock()
        gcs.spatial_filters_flat = np.ones((3, 10))
        gcs.temporal_filters = np.ones((3, 5))

        result = self.model._get_linear_spatiotemporal_filters(gcs)

        self.assertIsInstance(result.spatiotemporal_filters, np.ndarray)
        self.assertEqual(result.spatiotemporal_filters.shape, (3, 10, 5))

    def test_create_fixed_generator_potential_numpy(self):
        vs = Mock()
        vs.stimulus_cropped = np.ones((3, 10, 1000))
        vs.stim_len_tp = 1000
        vs.video_dt = 0.01 * b2u.second
        vs.visual_stimulus_parameters = {"baseline_start_seconds": 1}

        gcs = Mock()
        gcs.n_units = 3
        gcs.spatiotemporal_filters = np.ones((3, 10, 5))
        gcs.data_filter_duration = 50

        # Mock the torch import to simulate its absence
        with patch.dict("sys.modules", {"torch": None}):
            result = self.model._create_fixed_generator_potential(vs, gcs)

        self.assertIsInstance(result.generator_potentials, np.ndarray)
        self.assertEqual(result.generator_potentials.shape, (3, 1000))

    def test_impulse_response(self):
        vs = Mock()
        vs.video_dt = 1.0
        gcs = Mock()
        contrasts = np.array([0.1, 0.5, 1.0])

        # Mock the _initialize_impulse method
        self.model._initialize_impulse = Mock(
            return_value=(np.arange(100), np.ones((1, 100)), 10)
        )

        # Mock the _get_linear_temporal_filters method
        mock_gcs = Mock()
        mock_gcs.temporal_filters = np.ones((3, 90))
        self.model._get_linear_temporal_filters = Mock(return_value=mock_gcs)

        # Mock the response_type.get_contrast_by_response_type method
        self.model.response_type.get_contrast_by_response_type = Mock(
            return_value=np.ones((3, 90, 3))
        )

        result = self.model.impulse_response(vs, gcs, contrasts)

        self.assertIsInstance(result, dict)
        self.assertIn("tvec", result)
        self.assertIn("svec", result)
        self.assertIn("idx_start_delay", result)
        self.assertIn("contrasts", result)
        self.assertIn("impulse_responses", result)
        self.assertEqual(result["impulse_responses"].shape, (3, 90, 3))  # Changed to 90
