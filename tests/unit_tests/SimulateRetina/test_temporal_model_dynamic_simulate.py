# Built-in
import unittest
from unittest.mock import MagicMock, Mock, patch

# Third-party
import brian2.units as b2u
import numpy as np
import pandas as pd

# Local
from macaqueretina.retina.simulate_retina_module import TemporalModelDynamic


class TestTemporalModelDynamic(unittest.TestCase):

    def setUp(self):
        # Create mock objects for dependencies
        self.retina_math_mock = MagicMock()
        self.ganglion_cell_mock = MagicMock()
        self.response_type = "test_response"

        # Create an instance of TemporalModelDynamic
        self.model = TemporalModelDynamic(
            self.retina_math_mock, self.ganglion_cell_mock, self.response_type
        )

    def test_impulse_response(self):
        # Mock inputs
        vs_mock = MagicMock()
        vs_mock.video_dt = 1.0  # Example value in ms
        gcs_mock = MagicMock()
        contrasts = [0.1, 0.5, 1.0]

        # Mock methods and return values
        tvec = np.array([0, 1, 2])
        svec = np.array([0, 1, 2])
        idx_start_delay = 0
        params = np.array([[1], [2], [3]])  # Mock parameters

        self.model._initialize_impulse = MagicMock(
            return_value=(tvec, svec, idx_start_delay)
        )
        self.ganglion_cell_mock.get_BK_parameters.return_value = params
        self.ganglion_cell_mock.create_dynamic_temporal_signal.side_effect = (
            lambda *args, **kwargs: np.array([[0], [1], [2]])
        )

        # Call the method under test
        result = self.model.impulse_response(vs_mock, gcs_mock, contrasts)

        # Assertions
        self.assertIn("tvec", result)
        self.assertIn("svec", result)
        self.assertIn("idx_start_delay", result)
        self.assertIn("contrasts", result)
        self.assertIn("impulse_responses", result)

        # Check the shape of impulse_responses
        self.assertEqual(
            result["impulse_responses"].shape,
            (params.shape[0], len(tvec), len(contrasts)),
        )

    def test_create_generator_potential(self):
        # Mock input objects
        vs_mock = Mock()
        gcs_mock = Mock()

        # Mock the _create_dynamic_contrast method
        self.model._create_dynamic_contrast = Mock()
        self.model._create_dynamic_contrast.return_value = vs_mock

        # Mock the ganglion_cell.create_dynamic_temporal_signal method
        self.model.ganglion_cell.create_dynamic_temporal_signal = Mock()
        mock_generator_potential = np.random.rand(3, 1000)  # Example shape
        self.model.ganglion_cell.create_dynamic_temporal_signal.return_value = (
            mock_generator_potential
        )

        # Set up mock data
        vs_mock.svecs_cen = np.random.rand(3, 1000)
        vs_mock.svecs_sur = np.random.rand(3, 1000)
        vs_mock.stim_len_tp = 1000
        vs_mock.video_dt = 1 * b2u.second
        gcs_mock.n_units = 3

        # Create a mock DataFrame with a custom iloc accessor
        mock_df = Mock()
        mock_df.iloc = lambda x: pd.Series(
            {"tau_cen": 1.0, "tau_sur": 2.0, "k_cen": 0.5, "k_sur": 0.3}
        )
        gcs_mock.df = mock_df

        # Call the method under test
        result_vs, result_gcs = self.model.create_generator_potential(vs_mock, gcs_mock)

        # Assertions
        self.model._create_dynamic_contrast.assert_called_once_with(vs_mock, gcs_mock)
        self.model.ganglion_cell.create_dynamic_temporal_signal.assert_called()

        # Check that the result has the expected attribute
        self.assertTrue(hasattr(result_vs, "generator_potentials"))
        self.assertEqual(result_vs.generator_potentials.shape, (3, 1000))

        # Check that the input objects were returned
        self.assertEqual(result_vs, vs_mock)
        self.assertEqual(result_gcs, gcs_mock)
