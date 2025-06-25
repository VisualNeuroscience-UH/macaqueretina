# Built-in
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Third-party
import numpy as np
import pandas as pd
import torch

# Local
from macaqueretina.retina.simulate_retina_module import (
    BipolarProduct,
    ConeProduct,
    GanglionCellProduct,
    RetinaMath,
    SimulateRetina,
    VisualSignal,
)


class TestSimulateRetina(unittest.TestCase):
    def setUp(self):
        # Create mock objects for dependencies
        self.mock_context = Mock()
        self.mock_data_io = Mock()
        self.mock_project_data = Mock()
        self.mock_retina_math = Mock(spec=RetinaMath)
        self.device = "cpu"

        self.mock_project_data.simulate_retina = {}

        # Create SimulateRetina instance for testing
        self.simulate_retina = SimulateRetina(
            self.mock_context,
            self.mock_data_io,
            self.mock_project_data,
            self.mock_retina_math,
            self.device,
        )

    def test_initialization(self):
        self.assertIsInstance(self.simulate_retina, SimulateRetina)
        self.assertEqual(self.simulate_retina.context, self.mock_context)
        self.assertEqual(self.simulate_retina.data_io, self.mock_data_io)
        self.assertEqual(self.simulate_retina.project_data, self.mock_project_data)
        self.assertEqual(self.simulate_retina.retina_math, self.mock_retina_math)
        self.assertEqual(self.simulate_retina.device, self.device)

    def test_get_w_z_coords(self):
        # Mock GanglionCellProduct object
        mock_gcs = Mock()
        mock_gcs.df = pd.DataFrame(
            {"x_deg": np.array([1, 2, 3]), "y_deg": np.array([4, 5, 6])}
        )

        # Mock context data
        self.mock_context.retina_parameters = {
            "visual2cortical_params": {"a": 1, "k": 2}
        }

        w_coord, z_coord = self.simulate_retina.get_w_z_coords(mock_gcs)

        self.assertIsInstance(w_coord, np.ndarray)
        self.assertIsInstance(z_coord, np.ndarray)
        self.assertEqual(w_coord.shape, (3,))
        self.assertEqual(z_coord.shape, (3,))

    @patch("macaqueretina.retina.simulate_retina_module.ConeProduct")
    def test_initialize_cones(self, mock_cones_class):
        # Mock necessary data
        self.mock_context.retina_parameters = {"ret_file": "mock_file.npz"}
        self.mock_context.run_parameters = {"unit_index": 0}
        self.mock_context.visual_stimulus_parameters = {"ND_filter": 1.0}
        self.mock_data_io.get_data.return_value = {"mock_data": "mock_value"}

        # Call the method
        result = self.simulate_retina._initialize_cones()

        # Assertions
        self.mock_data_io.get_data.assert_called_once_with(filename="mock_file.npz")
        mock_cones_class.assert_called_once()
        self.assertIsInstance(result, Mock)  # Since we mocked ConeProduct

    @patch("macaqueretina.retina.simulate_retina_module.ConeProduct")
    @patch("macaqueretina.retina.simulate_retina_module.GanglionCellProduct")
    @patch("macaqueretina.retina.simulate_retina_module.VisualSignal")
    @patch("macaqueretina.retina.simulate_retina_module.BipolarProduct")
    def test_get_products(
        self, mock_bipolars, mock_visual_signal, mock_ganglion_cells, mock_cones
    ):
        # Mock necessary data and methods
        self.mock_context.retina_parameters = {
            "spatial_rfs_file": "mock_rfs.npz",
            "mosaic_file": "mock_mosaic.csv",
            "ret_file": "mock_ret.npz",
            "retina_center": (0, 0),
            "deg_per_mm": 1.0,
            "gc_type": "mock_gc",
            "response_type": "mock_response",
            "dog_model_type": "mock_dog",
            "spatial_model_type": "mock_spatial",
            "temporal_model_type": "mock_temporal",
            "cone_general_parameters": {"mock_param": "mock_value"},
            "optical_aberration": 0.0,
            "cone_noise_hash": None,
            "signal_gain": 1.0,
            "center_mask_threshold": 0.1,
        }
        self.mock_context.run_parameters = {
            "unit_index": 0,
            "spike_generator_model": "mock_model",
            "simulation_dt": 0.1,
        }
        self.mock_context.visual_stimulus_parameters = {
            "mock_option": "mock_value",
            "ND_filter": 1.0,
            "pix_per_deg": 10,
        }

        self.mock_data_io.get_data.return_value = {
            "mock_data": "mock_value",
            "cones_to_gcs_weights": np.array([1, 2, 3]),
            "cone_noise_parameters": np.array([4, 5, 6]),
        }

        # Mocking GanglionCellProduct attributes
        mock_gcs = mock_ganglion_cells.return_value
        mock_gcs.temporal_model_type = "fixed"

        # Call the method
        result = self.simulate_retina._get_products(None)

        # Assertions
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], Mock)  # VisualSignal
        self.assertIsInstance(result[1], Mock)  # GanglionCellProduct
        self.assertIsInstance(result[2], Mock)  # ConeProduct, now mocked
        self.assertIsNone(
            result[3]
        )  # BipolarProduct (None for "fixed" temporal_model_type)

    @patch("macaqueretina.retina.simulate_retina_module.ConcreteSimulationBuilder")
    @patch("macaqueretina.retina.simulate_retina_module.SimulationDirector")
    def test_client(self, mock_director, mock_builder):
        # Mock necessary data and methods
        self.mock_context.run_parameters = {
            "n_trials": 1,
            "gc_response_filenames": ["mock_file.npz"],
            "save_data": True,
            "unit_index": 0,
            "spike_generator_model": "mock_model",
            "simulation_dt": 0.1,
        }
        self.mock_context.retina_parameters = {
            "ret_file": "mock_ret.npz",
            "spatial_rfs_file": "mock_rfs.npz",
            "mosaic_file": "mock_mosaic.csv",
            "retina_center": 0 + 0j,
            "deg_per_mm": 1.0,
            "gc_type": "mock_gc",
            "response_type": "mock_response",
            "dog_model_type": "ellipse_fixed",
            "spatial_model_type": "mock_spatial",
            "temporal_model_type": "mock_temporal",
            "cone_general_parameters": {"mock_param": "mock_value"},
            "center_mask_threshold": 0.1,
            "refractory_parameters": {"mock_param": "mock_value"},
            "visual2cortical_params": {"mock_param": "mock_value"},
        }
        self.mock_context.dog_metadata_parameters = {
            "data_microm_per_pix": 60,
            "data_spatialfilter_height": 13,
            "data_spatialfilter_width": 13,
            "data_fps": 30,
            "data_temporalfilter_samples": 15,
            "exp_dog_data_folder": "mock_folder",
        }
        return_value1 = {
            "mock_data": "mock_value",
            "cones_to_gcs_weights": np.array([1, 2, 3]),
            "cone_noise_parameters": np.array([4, 5, 6]),
        }

        test_stimulus = Path("tests/test_data/stim.mp4")
        self.mock_context.visual_stimulus_parameters = {
            "ND_filter": 1.0,
            "stimulus_video_name": test_stimulus,
        }

        # Mock stimulus video options and frames
        mock_vs = Mock()
        mock_gcs = Mock()
        mock_gcs.dog_model_type = "ellipse_fixed"
        mock_gcs.df = pd.DataFrame(
            {
                "pos_ecc_mm": np.array([1.0, 2.0, 3.0]),
                "pos_polar_deg": np.array([4.0, 5.0, 6.0]),
                "gc_img": np.array([4.0, 5.0, 6.0]),
                "um_per_pix": 1.0,
                "pix_per_side": 3,
                # Add necessary fields for the model type
                "semi_xc_deg": np.array([1.0, 1.5, 2.0]),
                "semi_yc_deg": np.array([1.0, 1.5, 2.0]),
                "orient_cen_rad": np.array([0.0, 0.5, 1.0]),
                "relat_sur_diam": np.array([1.2, 1.3, 1.4]),
                "semi_xc_mm": np.array([1.0, 1.5, 2.0]),
                "semi_yc_mm": np.array([1.0, 1.5, 2.0]),
                "xoc_pix": np.array([1.0, 1.5, 2.0]),
                "yoc_pix": np.array([1.0, 1.5, 2.0]),
                "ampl_c_norm": np.array([1.0, 1.5, 2.0]),
                "ampl_s_norm": np.array([1.0, 1.5, 2.0]),
            }
        )
        # Set up stimulus video mock to return appropriate attributes
        mock_stimulus_video = Mock()
        mock_stimulus_video.options = {
            "image_width": 1920,
            "image_height": 1080,
            "pix_per_deg": 30,
            "fps": 60,
            "center_pix": [960, 540],
        }
        mock_stimulus_video.frames = np.random.rand(100, 1080, 1920)  # Mock frames
        mock_stimulus_video.video_width = 1920
        mock_stimulus_video.video_height = 1080
        mock_stimulus_video.fps = 60
        mock_stimulus_video.pix_per_deg = 30
        mock_stimulus_video.video_n_frames = 100
        mock_stimulus_video.baseline_len_tp = 10

        # Mock data_io.get_data to return mocked data
        self.mock_data_io.get_data.side_effect = [
            return_value1,
            mock_gcs.df,
            mock_gcs.df,
            mock_gcs.df,
        ]

        # Mock load_stimulus_from_videofile to return the mock stimulus video
        self.mock_data_io.load_stimulus_from_videofile.return_value = (
            mock_stimulus_video
        )

        # Mock the director and builder
        mock_director_instance = mock_director.return_value
        mock_director_instance.get_simulation_result.return_value = (mock_vs, mock_gcs)

        # Call the method
        self.simulate_retina.client()

        # Assertions
        mock_builder.assert_called_once()
        mock_director.assert_called_once()
        mock_director_instance.run_simulation.assert_called_once()
        self.mock_data_io.save_retina_output.assert_called_once_with(
            mock_vs, mock_gcs, "mock_file.npz"
        )
