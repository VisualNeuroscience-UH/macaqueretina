# Built-in
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Third-party
import brian2 as b2
import numpy as np

# Local
from macaqueretina.retina.simulate_retina_module import GanglionCellBase


class TestGanglionCellBaseConcreteMethods(unittest.TestCase):
    def setUp(self):
        # Set up the mock device and a subclass of GanglionCellBase to test concrete methods
        class MockGanglionCell(GanglionCellBase):
            def get_BK_parameters(self) -> np.ndarray:
                return np.array([1, 2, 3])

            def create_dynamic_temporal_signal(self) -> object:
                return None

        self.ganglion_cell = MockGanglionCell(device="cuda")

    @patch("tempfile.mkdtemp")
    @patch("shutil.rmtree")
    @patch("brian2.set_device")
    @patch("brian2.get_device")
    def test_set_brian_standalone_device(
        self, mock_get_device, mock_set_device, mock_rmtree, mock_mkdtemp
    ):
        # Mock the Brian2 device behavior
        mock_get_device.return_value.has_been_run = False

        # Test setting the standalone device
        self.ganglion_cell._set_brian_standalone_device(build_on_run=True)

        # Check if the temporary directory was created and set the device was called
        mock_mkdtemp.assert_called_once()
        mock_set_device.assert_called_with(
            self.ganglion_cell.device, directory=mock_mkdtemp(), build_on_run=True
        )

    @patch("shutil.rmtree")
    @patch("brian2.set_device")
    def test_teardown_brian_standalone_device(self, mock_set_device, mock_rmtree):
        # Set up a mock temporary directory
        self.ganglion_cell.standalone_tmp_dir = tempfile.mkdtemp()

        # Test tearing down the standalone device
        self.ganglion_cell._teardown_brian_standalone_device()

        # Check if the temporary directory was removed and the device was reset
        mock_rmtree.assert_called_with(self.ganglion_cell.standalone_tmp_dir)
        mock_set_device.assert_called_with("runtime")

    def test_create_lowpass_response(self):
        # Set up the parameters for the lowpass response
        tvec = np.linspace(0, 1, 100)
        NL = 3
        TL = 0.5

        # Test the lowpass response generation
        h = self.ganglion_cell._create_lowpass_response(tvec, NL, TL)

        # Check the shape of the result and that no inf/nan values are present
        self.assertEqual(h.shape[0], tvec.shape[0])
        self.assertFalse(np.isinf(h).any())
        self.assertFalse(np.isnan(h).any())

    def test_add_delay(self):
        # Set up parameters for adding delay
        yvecs = np.random.rand(10, 100)
        D = np.random.rand(10)
        _dt = 0.1
        n_units = 10
        n_timepoints = 100

        # Test adding delay
        result = self.ganglion_cell._add_delay(yvecs, D, _dt, n_units, n_timepoints)

        # Check the shape of the result and that the delay was applied correctly
        self.assertEqual(result.shape, (n_units, n_timepoints))
        delay_timepoints = np.int16(np.median(np.round(D / (_dt / b2.ms))))
        self.assertTrue(np.all(result[:, :delay_timepoints] == 0))


if __name__ == "__main__":
    unittest.main()
