# Built-in
import unittest
from unittest.mock import Mock, patch

# Third-party
import numpy as np
import pandas as pd

# Local
from macaqueretina.retina.simulate_retina_module import DoGModelCircular


class TestDoGModelCircular(unittest.TestCase):

    def setUp(self):
        self.retina_math = Mock()
        self.model = DoGModelCircular(self.retina_math)

    def test_get_spatial_kernel(self):
        # Create mock objects
        x_grid = np.array([[1, 2], [3, 4]])
        y_grid = np.array([[5, 6], [7, 8]])
        gc = Mock()
        gc.ampl_c = 1.0
        gc.q_pix = 10
        gc.r_pix = 20
        gc.rad_c = 2.0
        gc.ampl_s = 0.5
        gc.rad_s = 4.0
        offset = 0.1

        # Mock the DoG2D_circular method
        self.retina_math.DoG2D_circular.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4]]
        )

        # Call the method
        result = self.model.get_spatial_kernel(x_grid, y_grid, gc, offset)

        # Check the result
        np.testing.assert_array_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))

        # Check if DoG2D_circular was called with correct arguments
        self.retina_math.DoG2D_circular.assert_called_once_with(
            (x_grid, y_grid),
            gc.ampl_c,
            gc.q_pix,
            gc.r_pix,
            gc.rad_c,
            gc.ampl_s,
            gc.rad_s,
            offset,
        )

    def test_get_surround_params(self):
        # Create a mock DataFrame
        df = pd.DataFrame({"rad_s": [1.0, 2.0, 3.0], "orient_cen_rad": [0.1, 0.2, 0.3]})

        # Call the method
        semi_x, semi_y, ori = self.model.get_surround_params(df)

        # Check the results
        np.testing.assert_array_equal(semi_x, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(semi_y, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(ori, np.array([0.1, 0.2, 0.3]))

        # Check that semi_x and semi_y are identical
        np.testing.assert_array_equal(semi_x, semi_y)


if __name__ == "__main__":
    unittest.main()
