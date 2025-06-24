# Built-in
import unittest
from unittest.mock import patch

# Third-party
import numpy as np

# Local
from macaqueretina.retina.simulate_retina_module import ResponseTypeOFF, ResponseTypeON


class TestResponseTypeON(unittest.TestCase):
    def setUp(self):
        self.response_type_on = ResponseTypeON()

    def test_get_contrast_by_response_type_with_int(self):
        result = self.response_type_on.get_contrast_by_response_type(10)
        self.assertEqual(result, 10)

    def test_get_contrast_by_response_type_with_float(self):
        result = self.response_type_on.get_contrast_by_response_type(10.5)
        self.assertEqual(result, 10.5)

    def test_get_contrast_by_response_type_with_ndarray(self):
        input_array = np.array([1, 2, 3])
        result = self.response_type_on.get_contrast_by_response_type(input_array)
        np.testing.assert_array_equal(result, input_array)


class TestResponseTypeOFF(unittest.TestCase):
    def setUp(self):
        self.response_type_off = ResponseTypeOFF()

    def test_get_contrast_by_response_type_with_int(self):
        result = self.response_type_off.get_contrast_by_response_type(10)
        self.assertEqual(result, -10)

    def test_get_contrast_by_response_type_with_float(self):
        result = self.response_type_off.get_contrast_by_response_type(10.5)
        self.assertEqual(result, -10.5)

    def test_get_contrast_by_response_type_with_ndarray(self):
        input_array = np.array([1, 2, 3])
        result = self.response_type_off.get_contrast_by_response_type(input_array)
        np.testing.assert_array_equal(result, -input_array)


if __name__ == "__main__":
    unittest.main()
