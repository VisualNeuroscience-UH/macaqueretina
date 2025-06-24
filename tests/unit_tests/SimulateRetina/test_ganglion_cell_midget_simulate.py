# Built-in
import unittest
from unittest.mock import MagicMock, patch

# Third-party
import numpy as np
import pandas as pd
import pytest

# Local
from macaqueretina.retina.simulate_retina_module import GanglionCellMidget


# Custom mock classes to mimic Brian2 objects
class MockTimedArray:
    def __init__(self, *args, **kwargs):
        pass


class MockVariableView:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name

    def __repr__(self):
        return f"<MockVariableView of {self.name}>"


class MockNeuronGroup:
    def __init__(self, *args, **kwargs):
        self.hs = MockVariableView(self, "hs")
        self.ts = MockVariableView(self, "ts")


class TestGanglionCellMidget(unittest.TestCase):

    def setUp(self):
        self.ganglion_cell = GanglionCellMidget(device="cuda")
        self.ganglion_cell.standalone_tmp_dir = "/tmp"

    # @pytest.mark.xfail  # Cannot bypass unhashable type: numpy.ndarray
    # @patch.object(GanglionCellMidget, "_set_brian_standalone_device")
    # @patch.object(GanglionCellMidget, "_teardown_brian_standalone_device")
    # @patch.object(GanglionCellMidget, "_add_delay")
    # @patch("brian2.NeuronGroup", MockNeuronGroup)
    # @patch("brian2.StateMonitor")
    # @patch("brian2.TimedArray", MockTimedArray)
    # @patch("brian2.device.run")
    # @patch("brian2.device.build")
    # @patch("brian2.run")
    # @patch("brian2.ms", MagicMock())
    # @patch("scipy.signal.fftconvolve", MagicMock(return_value=np.zeros(100)))
    # def test_create_dynamic_temporal_signal_with_delay(
    #     self,
    #     mock_brian_run,
    #     mock_device_build,
    #     mock_device_run,
    #     mock_state_monitor,
    #     mock_add_delay,
    #     mock_teardown,
    #     mock_setup,
    # ):
    #     tvec = np.linspace(0, 1, 100)
    #     svec = np.random.rand(100)
    #     _dt = 0.1
    #     params = np.random.rand(5, 6, 2)

    #     n_units = params.shape[0]
    #     n_timepoints = tvec.shape[-1]

    #     svecs = np.tile(svec, (5, 1, 2))
    #     self.ganglion_cell._validate_svec = MagicMock(return_value=svecs)

    #     mock_add_delay.return_value = np.random.rand(5, 100)

    #     # Mock Brian2 components
    #     mock_brian_run.return_value = None
    #     mock_device_build.return_value = None
    #     mock_device_run.return_value = None

    #     mock_state_monitor_instance = MagicMock()
    #     mock_state_monitor_instance.y = np.random.rand(n_units, n_timepoints)
    #     mock_state_monitor.return_value = mock_state_monitor_instance

    #     # Mock the _create_lowpass_response method
    #     self.ganglion_cell._create_lowpass_response = MagicMock(
    #         return_value=np.zeros((n_units, n_timepoints))
    #     )

    #     # Mock b2.device.run to accept any arguments and store them
    #     run_args_list = []

    #     def mock_device_run_side_effect(run_args):
    #         run_args_list.append(run_args)

    #     mock_device_run.side_effect = mock_device_run_side_effect

    #     result = self.ganglion_cell.create_dynamic_temporal_signal(
    #         tvec, svec, _dt, params, show_impulse=False
    #     )

    #     # Assertions
    #     mock_add_delay.assert_called_once()
    #     self.assertIsNotNone(result)

    #     # Check that b2.device.run was called twice (once for each domain)
    #     self.assertEqual(mock_device_run.call_count, 2)

    #     # Check that the run_args were correctly passed
    #     for run_args in run_args_list:
    #         self.assertTrue(
    #             any(isinstance(key, MockTimedArray) for key in run_args.keys())
    #         )
    #         self.assertTrue(
    #             any(isinstance(key, MockVariableView) for key in run_args.keys())
    #         )

    def test_validate_svec(self):
        # Test the _validate_svec method with different scenarios
        svec = np.random.rand(100)
        n_units = 5
        n_timepoints = 100
        params = np.random.rand(5, 6, 2)  # Mocking a 3D params array

        # Test case where size matches visual stimulus
        svecs = self.ganglion_cell._validate_svec(svec, n_units, n_timepoints, params)
        self.assertEqual(svecs.shape, (5, 100, 2))

        # Test invalid params dimension
        with self.assertRaises(ValueError):
            self.ganglion_cell._validate_svec(
                svec, n_units, n_timepoints, np.random.rand(5)
            )

    def test_get_BK_parameters(self):
        # Create a DataFrame with required columns
        data = {
            "NL_cen": np.random.rand(5),
            "NLTL_cen": np.random.rand(5),
            "TS_cen": np.random.rand(5),
            "HS_cen": np.random.rand(5),
            "D_cen": np.random.rand(5),
            "A_cen": np.random.rand(5),
            "NL_sur": np.random.rand(5),
            "NLTL_sur": np.random.rand(5),
            "TS_sur": np.random.rand(5),
            "HS_sur": np.random.rand(5),
            "A_sur": np.random.rand(5),
        }
        df = pd.DataFrame(data)

        # Mock the gcs object with unit indices
        gcs = MagicMock()
        gcs.unit_indices = [0, 1, 2, 3, 4]
        gcs.df = df

        # Call the method
        params = self.ganglion_cell.get_BK_parameters(gcs)

        # Check that params has shape (5, 6, 2)
        self.assertEqual(params.shape, (5, 6, 2))

    def test_lowpass_response_with_valid_data(self):
        # Mock parameters for the lowpass filter
        tvec = np.linspace(0, 1, 100)  # Time vector of length 100
        NL = 3  # Use scalar NL
        TL = 0.5  # Use scalar TL

        # Call the lowpass filter creation method
        result = self.ganglion_cell._create_lowpass_response(tvec, NL, TL)

        # Ensure valid results
        self.assertIsNotNone(result)
        self.assertFalse(np.isinf(result).any())
        self.assertFalse(np.isnan(result).any())

        # Ensure the result has the same length as tvec
        self.assertEqual(result.shape, tvec.shape)


if __name__ == "__main__":
    unittest.main()
