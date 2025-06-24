# Built-in
import unittest
from unittest.mock import Mock, patch

# Third-party
import numpy as np

# Local
from macaqueretina.retina.simulate_retina_module import SpatialModelVAE


class TestSpatialModelVAE(unittest.TestCase):

    def setUp(self):
        self.dog_model = Mock()
        self.model = SpatialModelVAE(self.dog_model)

    def test_initialization(self):
        self.assertIsInstance(self.model, SpatialModelVAE)
        self.assertEqual(self.model.DoG_model, self.dog_model)

    @patch("macaqueretina.retina.simulate_retina_module.resize")
    def test_create_spatial_filter(self, mock_resize):
        gcs = Mock()
        gcs.spatial_filter_sidelen = 5
        gcs.spat_rf = np.random.rand(10, 10, 10)

        mock_resize.return_value = np.ones((5, 5))

        result = self.model.create_spatial_filter(gcs, 0)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, 5))
        np.testing.assert_array_equal(result, np.ones((5, 5)))

        mock_resize.assert_called_once()
        args, kwargs = mock_resize.call_args
        np.testing.assert_array_equal(args[0], gcs.spat_rf[0, :, :])
        self.assertEqual(args[1], (5, 5))
        self.assertEqual(kwargs, {"anti_aliasing": True})

    @patch("macaqueretina.retina.simulate_retina_module.resize")
    def test_create_spatial_filter_different_sizes(self, mock_resize):
        sizes = [3, 5, 7]
        for size in sizes:
            with self.subTest(size=size):
                gcs = Mock()
                gcs.spatial_filter_sidelen = size
                gcs.spat_rf = np.random.rand(10, 10, 10)

                mock_resize.return_value = np.ones((size, size))

                result = self.model.create_spatial_filter(gcs, 0)

                self.assertEqual(result.shape, (size, size))
                np.testing.assert_array_equal(result, np.ones((size, size)))

                mock_resize.assert_called()
                args, kwargs = mock_resize.call_args
                np.testing.assert_array_equal(args[0], gcs.spat_rf[0, :, :])
                self.assertEqual(args[1], (size, size))
                self.assertEqual(kwargs, {"anti_aliasing": True})

    @patch("macaqueretina.retina.simulate_retina_module.resize")
    def test_create_spatial_filter_different_units(self, mock_resize):
        gcs = Mock()
        gcs.spatial_filter_sidelen = 5
        gcs.spat_rf = np.random.rand(10, 10, 10)

        mock_resize.return_value = np.ones((5, 5))

        for unit in range(10):
            with self.subTest(unit=unit):
                result = self.model.create_spatial_filter(gcs, unit)

                self.assertEqual(result.shape, (5, 5))
                np.testing.assert_array_equal(result, np.ones((5, 5)))

                mock_resize.assert_called()
                args, kwargs = mock_resize.call_args
                np.testing.assert_array_equal(args[0], gcs.spat_rf[unit, :, :])
                self.assertEqual(args[1], (5, 5))
                self.assertEqual(kwargs, {"anti_aliasing": True})


if __name__ == "__main__":
    unittest.main()
