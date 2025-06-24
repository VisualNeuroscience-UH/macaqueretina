# Built-in
from unittest.mock import Mock

# Third-party
import numpy as np
import pandas as pd
import pytest

# Local
from macaqueretina.retina.construct_retina_module import DoGModelEllipseFixed


@pytest.fixture
def mock_dog_model_ellipse_fixed():
    """Fixture to provide a mocked instance of DoGModelEllipseFixed."""
    ret = Mock()
    fit = Mock()
    retina_math = Mock()

    # Mocking get_experimental_statistics to return multiple values as expected
    fit.get_experimental_statistics.return_value = (
        "mock_univariate_stat",
        "mock_spatial_multivariate_stat",
        "mock_temporal_multivariate_stat",
    )

    return DoGModelEllipseFixed(ret, fit, retina_math)


def test_initialization(mock_dog_model_ellipse_fixed):
    """Test the initialization and that experimental fits are fetched during instantiation."""
    # Ensure the experimental fits were retrieved with the correct model type
    mock_dog_model_ellipse_fixed.fit.client.assert_called_once_with(
        mock_dog_model_ellipse_fixed.ret.gc_type,
        mock_dog_model_ellipse_fixed.ret.response_type,
        fit_type="experimental",
        dog_model_type="ellipse_fixed",
    )
    mock_dog_model_ellipse_fixed.fit.get_experimental_statistics.assert_called_once_with(
        "ellipse_fixed"
    )


def test_scale_to_mm(mock_dog_model_ellipse_fixed):
    """Test the scale_to_mm method to ensure scaling is done correctly."""
    df = pd.DataFrame(
        {
            "gc_scaling_factors": [1.0, 1.5],
            "semi_xc_pix": [10, 20],
            "semi_yc_pix": [15, 25],
        }
    )
    um_per_pixel = 5.0

    result_df = mock_dog_model_ellipse_fixed.scale_to_mm(df, um_per_pixel)

    # Expected results after scaling
    expected_semi_xc_mm = np.array([10, 30]) * um_per_pixel / 1000
    expected_semi_yc_mm = np.array([15, 37.5]) * um_per_pixel / 1000

    np.testing.assert_array_equal(result_df["semi_xc_mm"], expected_semi_xc_mm)
    np.testing.assert_array_equal(result_df["semi_yc_mm"], expected_semi_yc_mm)


def test_generate_fit_img(mock_dog_model_ellipse_fixed):
    """Test the generate_fit_img method to ensure the fit image is generated correctly."""
    x_grid = np.array([0, 1, 2])
    y_grid = np.array([0, 1, 2])
    popt = np.array([1, 2, 3])

    # Mock the DoG2D_fixed_surround function
    mock_dog_model_ellipse_fixed.retina_math.DoG2D_fixed_surround.return_value = (
        np.array([[1, 2, 3], [4, 5, 6]])
    )

    result_img = mock_dog_model_ellipse_fixed.generate_fit_img(x_grid, y_grid, popt)

    # Ensure the correct method was called with the right arguments
    mock_dog_model_ellipse_fixed.retina_math.DoG2D_fixed_surround.assert_called_once_with(
        (x_grid, y_grid), *popt
    )

    # Assert that the returned image is correct
    np.testing.assert_array_equal(result_img, np.array([[1, 2, 3], [4, 5, 6]]))


def test_get_param_names(mock_dog_model_ellipse_fixed):
    """Test the get_param_names method to ensure the correct parameter names are set."""
    mock_gc = Mock()

    updated_gc = mock_dog_model_ellipse_fixed.get_param_names(mock_gc)

    # Check if the parameter names are set correctly
    assert updated_gc.parameter_names == [
        "ampl_c",
        "xoc_pix",
        "yoc_pix",
        "semi_xc_pix",
        "semi_yc_pix",
        "orient_cen_rad",
        "ampl_s",
        "relat_sur_diam",
        "offset",
    ]

    # Check if the scaling parameters are set correctly
    assert updated_gc.mm_scaling_params == ["semi_xc_pix", "semi_yc_pix"]
    assert updated_gc.zoom_scaling_params == ["xoc_pix", "yoc_pix"]


def test_transform_vae_dog_to_mm(mock_dog_model_ellipse_fixed):
    """Test the transform_vae_dog_to_mm method to ensure proper transformation from pixels to mm."""
    df = pd.DataFrame(
        {
            "semi_xc_mm": [0.0, 0.0],
            "semi_yc_mm": [0.0, 0.0],
            "relat_sur_diam": [0.0, 0.0],
        }
    )

    gc_df_in = pd.DataFrame(
        {
            "semi_xc_pix": [10, 20],
            "semi_yc_pix": [15, 25],
            "orient_cen_rad": [0.5, 1.0],
            "relat_sur_diam": [1.1, 1.2],
        }
    )

    mm_per_pix = 0.005

    # Mock the ellipse2diam method
    mock_dog_model_ellipse_fixed.retina_math.ellipse2diam.return_value = [100, 200]

    result_df = mock_dog_model_ellipse_fixed.transform_vae_dog_to_mm(
        df, gc_df_in, mm_per_pix
    )

    # Ensure the correct scaling and transformation was applied
    np.testing.assert_array_equal(
        result_df["semi_xc_mm"], gc_df_in["semi_xc_pix"] * mm_per_pix
    )
    np.testing.assert_array_equal(
        result_df["semi_yc_mm"], gc_df_in["semi_yc_pix"] * mm_per_pix
    )

    # Ensure the dendritic diameter was calculated correctly using np.testing.assert_array_equal
    # Retrieve the actual call arguments for the ellipse2diam call
    call_args = mock_dog_model_ellipse_fixed.retina_math.ellipse2diam.call_args[0]

    np.testing.assert_array_equal(call_args[0], result_df["semi_xc_mm"].values * 1000)
    np.testing.assert_array_equal(call_args[1], result_df["semi_yc_mm"].values * 1000)

    # Ensure the other parameters were copied correctly
    np.testing.assert_array_equal(
        result_df["relat_sur_diam"], gc_df_in["relat_sur_diam"]
    )
    np.testing.assert_array_equal(
        result_df["orient_cen_rad"], gc_df_in["orient_cen_rad"]
    )
