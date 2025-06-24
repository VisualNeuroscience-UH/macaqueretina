# Built-in
import unittest
from unittest.mock import Mock

# Third-party
import numpy as np

# Local
from macaqueretina.retina.simulate_retina_module import SpatialModelBase


# Create a concrete subclass for testing
class ConcreteSpatialModel(SpatialModelBase):
    def create_spatial_filter(self):
        return np.ones((10, 10))  # Dummy implementation


class MockSeries:
    def __init__(self, data):
        self.data = np.array(data)
        self.values = self.data

    def __getattr__(self, name):
        result = getattr(self.data, name)
        if callable(result):

            def wrapped(*args, **kwargs):
                output = result(*args, **kwargs)
                return MockSeries(output) if isinstance(output, np.ndarray) else output

            return wrapped
        return result

    def astype(self, dtype):
        return MockSeries(self.data.astype(dtype))


class MockDataFrameIloc:
    def __init__(self, q_pix, r_pix):
        self.q_pix = MockSeries(q_pix)
        self.r_pix = MockSeries(r_pix)

    def __getitem__(self, key):
        return self


class TestSpatialModelBase(unittest.TestCase):

    def setUp(self):
        self.dog_model = Mock()
        self.model = ConcreteSpatialModel(self.dog_model)

    def test_initialization(self):
        self.assertIsInstance(self.model, SpatialModelBase)
        self.assertEqual(self.model._DoG_model, self.dog_model)

    def test_dog_model_property(self):
        self.assertEqual(self.model.DoG_model, self.dog_model)

    def test_get_crop_pixels_single_unit(self):
        gcs = Mock()
        gcs.df_stimpix = Mock()
        gcs.df_stimpix.iloc = MockDataFrameIloc(
            q_pix=np.array([100.4]), r_pix=np.array([200.6])
        )
        gcs.spatial_filter_sidelen = 21

        qmin, qmax, rmin, rmax = self.model._get_crop_pixels(gcs, 0)

        np.testing.assert_allclose(qmin, [90], atol=1)
        np.testing.assert_allclose(qmax, [110], atol=1)
        np.testing.assert_allclose(rmin, [190], atol=1)
        np.testing.assert_allclose(rmax, [210], atol=1)

    def test_get_crop_pixels_multiple_units(self):
        gcs = Mock()
        gcs.df_stimpix = Mock()
        gcs.df_stimpix.iloc = MockDataFrameIloc(
            q_pix=np.array([100.4, 150.6]), r_pix=np.array([200.6, 250.4])
        )
        gcs.spatial_filter_sidelen = 21

        qmin, qmax, rmin, rmax = self.model._get_crop_pixels(gcs, [0, 1])

        np.testing.assert_allclose(qmin, [90, 140], atol=1)
        np.testing.assert_allclose(qmax, [110, 160], atol=1)
        np.testing.assert_allclose(rmin, [190, 240], atol=1)
        np.testing.assert_allclose(rmax, [210, 260], atol=1)

    def test_create_spatial_filter(self):
        result = self.model.create_spatial_filter()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))


if __name__ == "__main__":
    unittest.main()
