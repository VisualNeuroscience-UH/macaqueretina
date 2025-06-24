# Built-in
from unittest.mock import Mock

# Third-party
import numpy as np
import pandas as pd
import pytest

# Local
from macaqueretina.retina.construct_retina_module import DoGModelCircular


@pytest.fixture
def mock_dog_model_circular():
    """Fixture to provide a mocked instance of DoGModelCircular."""
    ret = Mock()
    fit = Mock()
    retina_math = Mock()

    # Mocking get_experimental_statistics to return expected values
    fit.get_experimental_statistics.return_value = (
        "mock_univariate_stat",
        "mock_spatial_multivariate_stat",
        "mock_temporal_multivariate_stat",
    )

    return DoGModelCircular(ret, fit, retina_math)


def test_initialization(mock_dog_model_circular):
    """Test the initialization and that experimental fits are fetched during instantiation."""
    mock_dog_model_circular.fit.client.assert_called_once_with(
        mock_dog_model_circular.ret.gc_type,
        mock_dog_model_circular.ret.response_type,
        fit_type="experimental",
        dog_model_type="circular",
    )
    mock_dog_model_circular.fit.get_experimental_statistics.assert_called_once_with(
        "circular"
    )


def test_scale_to_mm(mock_dog_model_circular):
    """Test the scale_to_mm method to ensure scaling is done correctly for center and surround radii."""
    df = pd.DataFrame(
        {"gc_scaling_factors": [1.0, 1.5], "rad_c_pix": [10, 20], "rad_s_pix": [15, 25]}
    )
    um_per_pixel = 5.0

    result_df = mock_dog_model_circular.scale_to_mm(df, um_per_pixel)

    # Expected results after scaling
    expected_rad_c_mm = np.array([10, 30]) * um_per_pixel / 1000
    expected_rad_s_mm = np.array([15, 37.5]) * um_per_pixel / 1000

    np.testing.assert_array_equal(result_df["rad_c_mm"], expected_rad_c_mm)
    np.testing.assert_array_equal(result_df["rad_s_mm"], expected_rad_s_mm)


def test_get_dd_in_um(mock_dog_model_circular):
    """Test the _get_dd_in_um method to ensure dendritic diameter is correctly computed."""
    mock_gc = Mock()
    mock_gc.df = pd.DataFrame({"rad_c_mm": [0.01, 0.02]})

    updated_gc = mock_dog_model_circular._get_dd_in_um(mock_gc)

    # Check if the dendritic diameter is computed correctly
    expected_den_diam_um = np.array([0.01, 0.02]) * 2 * 1000
    np.testing.assert_array_equal(updated_gc.df["den_diam_um"], expected_den_diam_um)


def test_generate_fit_img(mock_dog_model_circular):
    """Test the generate_fit_img method to ensure the fit image is generated correctly."""
    x_grid = np.array([0, 1, 2])
    y_grid = np.array([0, 1, 2])
    popt = np.array([1, 2, 3])

    # Mock the DoG2D_circular function
    mock_dog_model_circular.retina_math.DoG2D_circular.return_value = np.array(
        [[1, 2, 3], [4, 5, 6]]
    )

    result_img = mock_dog_model_circular.generate_fit_img(x_grid, y_grid, popt)

    # Ensure the correct method was called with the right arguments
    mock_dog_model_circular.retina_math.DoG2D_circular.assert_called_once_with(
        (x_grid, y_grid), *popt
    )

    # Assert that the returned image is correct
    np.testing.assert_array_equal(result_img, np.array([[1, 2, 3], [4, 5, 6]]))


def test_get_param_names(mock_dog_model_circular):
    """Test the get_param_names method to ensure the correct parameter names are set."""
    mock_gc = Mock()

    updated_gc = mock_dog_model_circular.get_param_names(mock_gc)

    # Check if the parameter names are set correctly
    assert updated_gc.parameter_names == [
        "ampl_c",
        "xoc_pix",
        "yoc_pix",
        "rad_c_pix",
        "ampl_s",
        "rad_s_pix",
        "offset",
    ]

    # Check if the scaling parameters are set correctly
    assert updated_gc.mm_scaling_params == ["rad_c_pix", "rad_s_pix"]
    assert updated_gc.zoom_scaling_params == ["xoc_pix", "yoc_pix"]


def test_add_center_fit_area_to_df(mock_dog_model_circular):
    """Test the _add_center_fit_area_to_df method to ensure the fit area is calculated correctly."""
    mock_gc = Mock()
    mock_gc.df = pd.DataFrame({"rad_c_mm": [1.0, 2.0]})

    updated_gc = mock_dog_model_circular._add_center_fit_area_to_df(mock_gc)

    expected_areas = np.pi * np.array([1.0, 2.0]) ** 2
    np.testing.assert_array_equal(updated_gc.df["center_fit_area_mm2"], expected_areas)


def test_get_center_volume(mock_dog_model_circular):
    """Test the _get_center_volume method to ensure the center volume is computed correctly."""
    mock_gc = Mock()
    mock_gc.df = pd.DataFrame({"ampl_c": [1.0, 2.0], "rad_c_mm": [1.0, 2.0]})

    volumes = mock_dog_model_circular._get_center_volume(mock_gc)

    # Calculate expected volumes
    expected_volumes = 2 * np.pi * np.array([1.0, 2.0]) * np.array([1.0, 2.0]) ** 2

    # Assert the computed volumes match the expected values
    np.testing.assert_array_equal(volumes, expected_volumes)


def test_transform_vae_dog_to_mm(mock_dog_model_circular):
    """Test the transform_vae_dog_to_mm method to ensure proper transformation from pixels to mm."""
    df = pd.DataFrame(
        {
            "rad_c_mm": [0.0, 0.0],
            "rad_s_mm": [0.0, 0.0],
        }
    )

    gc_df_in = pd.DataFrame({"rad_c_pix": [10, 20], "rad_s_pix": [15, 25]})

    mm_per_pix = 0.005

    result_df = mock_dog_model_circular.transform_vae_dog_to_mm(
        df, gc_df_in, mm_per_pix
    )

    # Ensure the correct scaling and transformation was applied
    np.testing.assert_array_equal(
        result_df["rad_c_mm"], gc_df_in["rad_c_pix"] * mm_per_pix
    )
    np.testing.assert_array_equal(
        result_df["rad_s_mm"], gc_df_in["rad_s_pix"] * mm_per_pix
    )

    # Ensure the dendritic diameter was calculated correctly
    expected_den_diam_um = result_df["rad_c_mm"] * 2 * 1000
    np.testing.assert_array_equal(result_df["den_diam_um"], expected_den_diam_um)
