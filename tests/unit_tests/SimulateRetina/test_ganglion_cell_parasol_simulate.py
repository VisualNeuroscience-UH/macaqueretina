# Built-in
import unittest
from unittest.mock import MagicMock, patch

# Third-party
import brian2 as b2
import numpy as np

# Local
from macaqueretina.retina.simulate_retina_module import GanglionCellParasol


class TestGanglionCellParasol(unittest.TestCase):

    def setUp(self):
        class MockGanglionCell(GanglionCellParasol):
            def get_BK_parameters(self, gcs) -> np.ndarray:
                return np.array([[1, 2, 3, 4, 5, 6]] * 5)

            def create_dynamic_temporal_signal(
                self, tvec, svec, _dt, params, show_impulse=False, impulse_contrast=1.0
            ) -> np.ndarray:
                return np.random.rand(5, 100)

        self.ganglion_cell = MockGanglionCell(device="cuda")

    @patch("brian2.TimedArray", return_value=MagicMock())
    def test_create_dynamic_temporal_signal_impulse_response(self, mock_timed_array):
        # Mock brian2 functions inside the test
        with patch.object(
            self.ganglion_cell, "_set_brian_standalone_device"
        ), patch.object(self.ganglion_cell, "_teardown_brian_standalone_device"):

            tvec = np.linspace(0, 1, 100)
            svec = np.random.rand(100)
            _dt = 0.1
            params = np.random.rand(5, 6)

            # Ensure mock stimulus vector
            self.ganglion_cell._validate_svec = MagicMock(
                return_value=np.tile(svec, (5, 1))
            )

            # Call the method under test
            yvecs = self.ganglion_cell.create_dynamic_temporal_signal(
                tvec, svec, _dt, params, show_impulse=True
            )

            # Check that impulse response is not None
            self.assertIsNotNone(yvecs)

    def test_validate_svec(self):
        # Test the _validate_svec method with different scenarios
        svec = np.random.rand(100)
        n_units = 5
        n_timepoints = 100
        params = np.random.rand(5, 6)

        # Test case where size matches visual stimulus
        svecs = self.ganglion_cell._validate_svec(svec, n_units, n_timepoints, params)
        self.assertEqual(svecs.shape, (5, 100))

        # Test case where size matches impulse response
        svec = np.random.rand(100)
        svecs = self.ganglion_cell._validate_svec(svec, n_units, n_timepoints, params)
        self.assertEqual(svecs.shape, (5, 100))

        # Test invalid params dimension
        with self.assertRaises(ValueError):
            self.ganglion_cell._validate_svec(
                svec, n_units, n_timepoints, np.random.rand(5)
            )

    @patch("pandas.DataFrame.loc")
    def test_get_BK_parameters(self, mock_loc):
        # Mock the data frame to return BK parameters
        mock_loc.return_value.values = np.random.rand(5, 6)

        # Mock gcs object
        gcs = MagicMock()
        gcs.unit_indices = [0, 1, 2, 3, 4]

        # Call the method and check the result
        params = self.ganglion_cell.get_BK_parameters(gcs)

        # Check the correct shape of parameters
        self.assertEqual(params.shape, (5, 6))

    @patch("scipy.signal.fftconvolve")
    def test_lowpass_response_with_valid_data(self, mock_fftconvolve):
        # Mock parameters for the lowpass filter
        tvec = np.linspace(0, 1, 100)
        NL = 3
        TL = 0.5

        # Mock convolution results
        mock_fftconvolve.return_value = np.random.rand(100)

        # Call the lowpass filter creation method
        result = self.ganglion_cell._create_lowpass_response(tvec, NL, TL)

        # Ensure valid results
        self.assertIsNotNone(result)
        self.assertFalse(np.isinf(result).any())
        self.assertFalse(np.isnan(result).any())


if __name__ == "__main__":
    unittest.main()
