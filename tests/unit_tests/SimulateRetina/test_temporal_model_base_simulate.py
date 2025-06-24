# Built-in
import unittest
from unittest.mock import Mock, patch

# Third-party
import numpy as np

# Local
from macaqueretina.retina.simulate_retina_module import TemporalModelBase


class TestTemporalModelBase(unittest.TestCase):

    def setUp(self):
        self.retina_math = Mock()
        self.ganglion_cell = Mock()
        self.response_type = Mock()

        class ConcreteTemporalModel(TemporalModelBase):
            def impulse_response(self):
                pass

            def create_generator_potential(self):
                pass

        self.model = ConcreteTemporalModel(
            self.retina_math, self.ganglion_cell, self.response_type
        )

    def test_initialization(self):
        self.assertEqual(self.model.retina_math, self.retina_math)
        self.assertEqual(self.model.ganglion_cell, self.ganglion_cell)
        self.assertEqual(self.model.response_type, self.response_type)
        self.assertEqual(self.model.project_data, {})

    def test_initialize_impulse(self):
        vs = Mock()
        gcs = Mock()
        gcs.data_filter_timesteps = 100
        gcs.data_filter_fps = 60
        dt = 1

        self.response_type.get_contrast_by_response_type.return_value = np.ones(1667)

        tvec, svec, idx_100_ms = self.model._initialize_impulse(vs, gcs, dt)

        self.assertEqual(len(tvec), 1667)
        self.assertEqual(svec.shape, (1, 1667))
        self.assertEqual(idx_100_ms, 100)
        self.response_type.get_contrast_by_response_type.assert_called_once()

    @patch("numpy.einsum")
    def test_create_dynamic_contrast(self, mock_einsum):
        vs = Mock()
        gcs = Mock()

        gcs.spatial_filters_flat = np.ones((10, 20))
        gcs.surround_masks_flat = np.ones((10, 20))
        gcs.center_masks_flat = np.ones((10, 20))
        vs.stimulus_cropped = np.ones((10, 20, 30))

        mock_einsum.return_value = np.ones((10, 30))

        result = self.model._create_dynamic_contrast(vs, gcs)

        self.assertEqual(result, vs)
        self.assertEqual(mock_einsum.call_count, 2)
        np.testing.assert_array_equal(result.svecs_sur, np.ones((10, 30)))
        np.testing.assert_array_equal(result.svecs_cen, np.ones((10, 30)))
