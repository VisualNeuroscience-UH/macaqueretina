# Built-in
import unittest
from unittest.mock import Mock, patch

# Third-party
import brian2.units as b2u
import numpy as np

# Local
from macaqueretina.retina.simulate_retina_module import VisualSignal


class TestVisualSignal(unittest.TestCase):
    def setUp(self):
        self.mock_stimulus_options = {"stimulus_video_name": "test_video.mp4"}
        self.mock_stimulus_center = complex(0, 0)
        self.mock_load_stimulus = Mock()
        self.mock_simulation_dt = 0.1
        self.mock_deg_per_mm = 1.0
        self.mock_optical_aberration = 0.0
        self.mock_pix_per_deg = 10

        # Mock the stimulus video
        self.mock_stimulus_video = Mock()
        self.mock_stimulus_video.options = {
            "image_width": 100,
            "image_height": 80,
            "pix_per_deg": 10,
            "fps": 30,
            "center_pix": (50, 40),
        }
        self.mock_stimulus_video.frames = np.random.rand(100, 80, 100)
        self.mock_stimulus_video.video_width = 100
        self.mock_stimulus_video.video_height = 80
        self.mock_stimulus_video.fps = 30
        self.mock_stimulus_video.pix_per_deg = 10
        self.mock_stimulus_video.video_n_frames = 100
        self.mock_stimulus_video.baseline_len_tp = 10

        self.mock_load_stimulus.return_value = self.mock_stimulus_video

    def test_init(self):
        visual_signal = VisualSignal(
            self.mock_stimulus_options,
            self.mock_stimulus_center,
            self.mock_load_stimulus,
            self.mock_simulation_dt,
            self.mock_deg_per_mm,
            self.mock_optical_aberration,
            self.mock_pix_per_deg,
        )

        self.assertEqual(
            visual_signal.visual_stimulus_parameters, self.mock_stimulus_options
        )
        self.assertEqual(visual_signal.retina_center, self.mock_stimulus_center)
        self.assertEqual(
            visual_signal.load_stimulus_from_videofile, self.mock_load_stimulus
        )
        self.assertEqual(visual_signal.deg_per_mm, self.mock_deg_per_mm)
        self.assertEqual(visual_signal.stimulus_width_pix, 100)
        self.assertEqual(visual_signal.stimulus_height_pix, 80)
        self.assertEqual(visual_signal.pix_per_deg, 10)
        self.assertEqual(visual_signal.fps, 30)
        self.assertEqual(visual_signal.stimulus_width_deg, 10)
        self.assertEqual(visual_signal.stimulus_height_deg, 8)
        self.assertEqual(visual_signal.stim_len_tp, 100)
        self.assertEqual(visual_signal.baseline_len_tp, 10)

    def test_vspace_to_pixspace(self):
        visual_signal = VisualSignal(
            self.mock_stimulus_options,
            self.mock_stimulus_center,
            self.mock_load_stimulus,
            self.mock_simulation_dt,
            self.mock_deg_per_mm,
            self.mock_optical_aberration,
            self.mock_pix_per_deg,
        )

        # Test conversion from visual space to pixel space
        q, r = visual_signal._vspace_to_pixspace(1, 1)
        self.assertAlmostEqual(q, 60)  # 50 (center) + 10 (1 deg * 10 pix/deg)
        self.assertAlmostEqual(r, 30)  # 40 (center) - 10 (1 deg * 10 pix/deg)

        # Test with negative values
        q, r = visual_signal._vspace_to_pixspace(-1, -1)
        self.assertAlmostEqual(q, 40)  # 50 (center) - 10 (1 deg * 10 pix/deg)
        self.assertAlmostEqual(r, 50)  # 40 (center) + 10 (1 deg * 10 pix/deg)

    def test_photodiode_response(self):
        visual_signal = VisualSignal(
            self.mock_stimulus_options,
            self.mock_stimulus_center,
            self.mock_load_stimulus,
            self.mock_simulation_dt,
            self.mock_deg_per_mm,
            self.mock_optical_aberration,
            self.mock_pix_per_deg,
        )

        self.assertEqual(visual_signal.photodiode_response.shape, (100,))
        np.testing.assert_array_equal(
            visual_signal.photodiode_response,
            self.mock_stimulus_video.frames[:, 40, 50],
        )
