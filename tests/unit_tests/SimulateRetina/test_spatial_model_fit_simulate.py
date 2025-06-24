# Built-in
import unittest
from unittest.mock import Mock, patch

# Third-party
import numpy as np

# Local
from macaqueretina.retina.simulate_retina_module import SpatialModelDOG


class MockDataFrameIloc:
    def __init__(self, return_value):
        self.return_value = return_value

    def __getitem__(self, key):
        return self.return_value


class TestSpatialModelFIT(unittest.TestCase):

    def setUp(self):
        self.dog_model = Mock()
        self.model = SpatialModelDOG(self.dog_model)

    def test_initialization(self):
        self.assertIsInstance(self.model, SpatialModelDOG)
        self.assertEqual(self.model.DoG_model, self.dog_model)

    def test_create_spatial_filter(self):
        # Mock ganglion cell object
        gcs = Mock()
        gcs.spatial_filter_sidelen = 5
        gcs.df_stimpix = Mock()
        gcs.df_stimpix.iloc = MockDataFrameIloc(Mock())

        # Mock _get_crop_pixels method
        self.model._get_crop_pixels = Mock(return_value=(0, 4, 0, 4))

        # Mock DoG_model.get_spatial_kernel
        self.dog_model.get_spatial_kernel.return_value = np.ones(25)

        # Call the method
        result = self.model.create_spatial_filter(gcs, 0)

        # Assertions
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, 5))
        np.testing.assert_array_equal(result, np.ones((5, 5)))

        # Check if methods were called with correct arguments
        self.model._get_crop_pixels.assert_called_once_with(gcs, 0)
        self.dog_model.get_spatial_kernel.assert_called_once()

    def test_create_spatial_filter_with_different_sizes(self):
        sizes = [3, 5, 7]
        for size in sizes:
            with self.subTest(size=size):
                gcs = Mock()
                gcs.spatial_filter_sidelen = size
                gcs.df_stimpix = Mock()
                gcs.df_stimpix.iloc = MockDataFrameIloc(Mock())

                self.model._get_crop_pixels = Mock(
                    return_value=(0, size - 1, 0, size - 1)
                )
                self.dog_model.get_spatial_kernel.return_value = np.ones(size * size)

                result = self.model.create_spatial_filter(gcs, 0)

                self.assertEqual(result.shape, (size, size))
                np.testing.assert_array_equal(result, np.ones((size, size)))

    @patch("numpy.meshgrid")
    def test_meshgrid_called_correctly(self, mock_meshgrid):
        gcs = Mock()
        gcs.spatial_filter_sidelen = 5
        gcs.df_stimpix = Mock()
        gcs.df_stimpix.iloc = MockDataFrameIloc(Mock())

        self.model._get_crop_pixels = Mock(return_value=(0, 4, 0, 4))
        self.dog_model.get_spatial_kernel.return_value = np.ones(25)

        # Mock the return value of np.meshgrid
        mock_meshgrid.return_value = (np.zeros((5, 5)), np.zeros((5, 5)))

        self.model.create_spatial_filter(gcs, 0)

        mock_meshgrid.assert_called_once()
        args = mock_meshgrid.call_args[0]
        np.testing.assert_array_equal(args[0], np.arange(0, 5, 1))
        np.testing.assert_array_equal(args[1], np.arange(0, 5, 1))


if __name__ == "__main__":
    unittest.main()
