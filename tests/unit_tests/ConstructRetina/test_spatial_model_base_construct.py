# Built-in
from unittest.mock import Mock, patch

# Third-party
import numpy as np
import pandas as pd
import pytest

# Local
from macaqueretina.retina.construct_retina_module import SpatialModelBase
from macaqueretina.retina.retina_math_module import RetinaMath


# Mock subclass to test the abstract class
class MockSpatialModel(SpatialModelBase):
    def create(self):
        pass  # Implement abstract method with a mock


@pytest.fixture
def mock_spatial_model():
    """Fixture to provide a mocked instance of SpatialModelBase."""
    DoG_model = Mock()
    sampler = Mock()
    retina_vae = Mock()
    fit = Mock()
    retina_math = Mock()
    viz = Mock()
    return MockSpatialModel(DoG_model, sampler, retina_vae, fit, retina_math, viz)


def pol2cart(ecc, polar, deg=True):
    x = ecc * np.cos(np.radians(polar))  # radians fed here
    y = ecc * np.sin(np.radians(polar))
    return x, y


def test_apply_local_zoom_compensation(mock_spatial_model):
    """Test the _apply_local_zoom_compensation method."""
    epps, pps, pix, zoom = 10.0, 20.0, 5.0, 1.5
    result = mock_spatial_model._apply_local_zoom_compensation(epps, pps, pix, zoom)

    expected_result = -zoom * ((epps / 2) - pix) + (pps / 2)
    assert result == expected_result


def test_get_img_grid_mm(mock_spatial_model):
    """Test the _get_img_grid_mm method to ensure the grid is computed correctly."""
    mock_ret = Mock()
    mock_gc = Mock()

    mock_gc.img_mask = np.zeros((3, 10, 10))  # Shape of the RF image
    mock_gc.um_per_pix = 5.0
    mock_gc.n_units = 3
    mock_gc.img_lu_pix = np.array([[0, 0], [1, 1], [2, 2]])

    mock_ret.whole_ret_lu_mm = [0, 0]
    mock_ret.polar_lim_deg = [30, 60]  # Set polar_lim_deg to be a list of two values

    # Mock the rotation method to return appropriate shaped arrays
    mock_spatial_model.DoG_model.retina_math.rotate_image_grids = Mock(
        return_value=(
            np.zeros((3, 10, 10)),  # Mock X_grid_rot_mm
            np.zeros((3, 10, 10)),  # Mock Y_grid_rot_mm
        )
    )

    result_gc = mock_spatial_model._get_img_grid_mm(mock_ret, mock_gc)

    # Check the shapes of the result grids
    assert result_gc.X_grid_cen_mm.shape == (3, 10, 10)
    assert result_gc.Y_grid_cen_mm.shape == (3, 10, 10)


def test_get_retina_corners(mock_spatial_model):
    """Test that the corners of the retina are correctly calculated in mm."""
    mock_ret = Mock()
    mock_ret.ecc_lim_mm = [1.0, 2.0]  # Eccentricity limits
    mock_ret.polar_lim_deg = [45.0, 90.0]  # Polar angle limits
    rot_deg = 67.5  # Average polar angle

    mock_spatial_model.DoG_model.retina_math.pol2cart = Mock(
        side_effect=lambda ecc, polar, deg=True: (ecc + 1, polar + 1)
    )

    corners_mm = mock_spatial_model._get_retina_corners(
        mock_ret, rot_deg, mock_spatial_model.DoG_model.retina_math.pol2cart
    )

    expected_corners = np.array([[2.0, 23.5], [2.0, -21.5], [3.0, -21.5], [3.0, 23.5]])
    np.testing.assert_array_almost_equal(corners_mm, expected_corners)


def test_calculate_retina_size(mock_spatial_model):
    """Test that the retina size in pixels is correctly calculated from corner coordinates."""
    corners_mm = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    mm_per_pix = 0.005  # mm per pixel
    pad_size_mm = 0.1  # Padding

    ret_pix_x, ret_pix_y, min_x_mm_im, max_y_mm_im = (
        mock_spatial_model._calculate_retina_size(corners_mm, mm_per_pix, pad_size_mm)
    )

    assert ret_pix_x == 640  # Calculated width in pixels
    assert ret_pix_y == 640  # Calculated height in pixels
    assert min_x_mm_im == -0.1  # Minimum x-coordinate in mm after padding
    assert max_y_mm_im == 3.1  # Maximum y-coordinate in mm after padding


def test_convert_to_pixel_positions(mock_spatial_model):
    """Test conversion of polar coordinates to pixel coordinates."""
    pos_ecc_mm = np.array([1.0, 2.0])  # Eccentricities
    pos_polar_deg = np.array([45.0, 90.0])  # Polar angles
    rot_deg = 67.5  # Rotation
    min_x_mm_im = 0.0  # Minimum x in mm
    max_y_mm_im = 3.1  # Maximum y in mm
    mm_per_pix = 0.005  # mm per pixel

    # Mock pol2cart to return Cartesian coordinates after applying rotation
    mock_spatial_model.DoG_model.retina_math.pol2cart = Mock(
        side_effect=lambda ecc, polar, deg=True: pol2cart(ecc, polar, deg)
    )

    x_pix_c, y_pix_c = mock_spatial_model._convert_to_pixel_positions(
        pos_ecc_mm,
        pos_polar_deg,
        rot_deg,
        min_x_mm_im,
        max_y_mm_im,
        mm_per_pix,
        mock_spatial_model.DoG_model.retina_math.pol2cart,
    )
    # Expected values based on trigonometry with rotation applied
    x_mm = np.array(
        [
            1.0 * np.cos(np.radians(45.0 - rot_deg)),
            2.0 * np.cos(np.radians(90.0 - rot_deg)),
        ]
    )
    expected_x_pix_c = np.array(
        [
            (x_mm[0] - min_x_mm_im) / mm_per_pix,
            (x_mm[1] - min_x_mm_im) / mm_per_pix,
        ]
    )

    y_mm = np.array(
        [
            1.0 * np.sin(np.radians(45.0 - rot_deg)),
            2.0 * np.sin(np.radians(90.0 - rot_deg)),
        ]
    )
    expected_y_pix_c = np.array(
        [
            (max_y_mm_im - y_mm[0]) / mm_per_pix,
            (max_y_mm_im - y_mm[1]) / mm_per_pix,
        ]
    )
    np.testing.assert_array_almost_equal(x_pix_c, expected_x_pix_c)
    np.testing.assert_array_almost_equal(y_pix_c, expected_y_pix_c)


def test_apply_pixel_scaling(mock_spatial_model):
    """Test application of pixel scaling or direct pixel return."""
    df = pd.DataFrame(
        {
            "yoc_pix": [10, 20],  # Y center positions
            "xoc_pix": [10, 20],  # X center positions
            "zoom_factor": [1.0, 1.5],  # Zoom factors
        }
    )

    gc = Mock()
    gc.exp_pix_per_side = 30  # Expanded pixel side
    gc.pix_per_side = 20  # Original pixel side

    # Test without pixel scaling (should return original pixel values)
    xoc_pix_scaled, yoc_pix_scaled = mock_spatial_model._apply_pixel_scaling(
        df, apply_pix_scaler=False, gc=gc
    )
    assert np.array_equal(xoc_pix_scaled, df["xoc_pix"].values)
    assert np.array_equal(yoc_pix_scaled, df["yoc_pix"].values)

    # Mock _apply_local_zoom_compensation to return scaled values
    mock_spatial_model._apply_local_zoom_compensation = Mock(
        side_effect=lambda epps, pps, pix, zoom: pix * zoom
    )

    # Test with pixel scaling (should apply zoom)
    xoc_pix_scaled, yoc_pix_scaled = mock_spatial_model._apply_pixel_scaling(
        df, apply_pix_scaler=True, gc=gc
    )
    expected_xoc_pix_scaled = df["xoc_pix"].values * df["zoom_factor"].values
    expected_yoc_pix_scaled = df["yoc_pix"].values * df["zoom_factor"].values

    np.testing.assert_array_almost_equal(xoc_pix_scaled, expected_xoc_pix_scaled)
    np.testing.assert_array_almost_equal(yoc_pix_scaled, expected_yoc_pix_scaled)


def test_convert_center_positions(mock_spatial_model):
    """Test the combined conversion of polar coordinates to pixel coordinates and optional scaling."""
    df = pd.DataFrame(
        {
            "pos_ecc_mm": [1.0, 2.0],  # Eccentricities
            "pos_polar_deg": [45.0, 90.0],  # Polar angles
            "yoc_pix": [10, 20],  # Y center positions
            "xoc_pix": [10, 20],  # X center positions
            "zoom_factor": [1.0, 1.5],  # Zoom factors
        }
    )

    rot_deg = 67.5  # Rotation
    min_x_mm_im = 0.0  # Minimum x in mm
    max_y_mm_im = 3.1  # Maximum y in mm
    mm_per_pix = 0.005  # mm per pixel
    apply_pix_scaler = True  # Enable pixel scaling

    # Mock pol2cart to return Cartesian coordinates
    mock_spatial_model.DoG_model.retina_math.pol2cart = Mock(
        side_effect=lambda ecc, polar, deg=True: pol2cart(ecc, polar, deg)
    )

    gc = Mock()
    gc.exp_pix_per_side = 30  # Expanded pixel side
    gc.pix_per_side = 20  # Original pixel side

    # Mock _apply_local_zoom_compensation to return scaled values
    mock_spatial_model._apply_local_zoom_compensation = Mock(
        side_effect=lambda epps, pps, pix, zoom: pix * zoom
    )

    # Call the combined method
    x_pix_c, y_pix_c, xoc_pix_scaled, yoc_pix_scaled = (
        mock_spatial_model._convert_center_positions(
            df, rot_deg, min_x_mm_im, max_y_mm_im, mm_per_pix, apply_pix_scaler, gc
        )
    )

    # Expected pixel coordinates without scaling (before zoom)
    expected_x_pix_c = np.array(
        [
            (1.0 * np.cos(np.radians(45.0 - rot_deg)) - min_x_mm_im) / mm_per_pix,
            (2.0 * np.cos(np.radians(90.0 - rot_deg)) - min_x_mm_im) / mm_per_pix,
        ]
    )

    expected_y_pix_c = np.array(
        [
            (max_y_mm_im - (1.0 * np.sin(np.radians(45.0 - rot_deg)))) / mm_per_pix,
            (max_y_mm_im - (2.0 * np.sin(np.radians(90.0 - rot_deg)))) / mm_per_pix,
        ]
    )

    # Expected pixel coordinates after zoom scaling
    expected_xoc_pix_scaled = df["xoc_pix"].values * df["zoom_factor"].values
    expected_yoc_pix_scaled = df["yoc_pix"].values * df["zoom_factor"].values

    # Validate results
    np.testing.assert_array_almost_equal(x_pix_c, expected_x_pix_c)
    np.testing.assert_array_almost_equal(y_pix_c, expected_y_pix_c)
    np.testing.assert_array_almost_equal(xoc_pix_scaled, expected_xoc_pix_scaled)
    np.testing.assert_array_almost_equal(yoc_pix_scaled, expected_yoc_pix_scaled)


def test_place_gc_images_on_retina(mock_spatial_model):
    """Test that ganglion cell images are correctly placed on the retina."""
    df = pd.DataFrame(
        {
            "yoc_pix": [10, 20],  # Y center positions
            "xoc_pix": [10, 20],  # X center positions
        }
    )

    gc_img = np.random.rand(2, 20, 20)  # Two ganglion cells with 20x20 images
    ret_img_pix = np.zeros((100, 100))  # Retina image of size 100x100 pixels
    pix_per_side = 20  # Side length of each ganglion cell image
    x_pix_c = np.array([30.0, 50.0])
    y_pix_c = np.array([30.0, 50.0])
    xoc_pix_scaled = np.array([10.0, 20.0])
    yoc_pix_scaled = np.array([10.0, 20.0])

    ret_img_pix, gc_img_lu_pix = mock_spatial_model._place_gc_images_on_retina(
        df,
        gc_img,
        ret_img_pix,
        pix_per_side,
        x_pix_c,
        y_pix_c,
        xoc_pix_scaled,
        yoc_pix_scaled,
    )

    # Ensure the ganglion cell images are placed at the correct locations
    assert gc_img_lu_pix.shape == (2, 2)  # Ensure left upper coordinates are stored
    assert np.all(
        ret_img_pix[20:40, 20:40] != 0
    )  # Check if the first GC image is placed at (20, 20)
    assert np.all(
        ret_img_pix[30:50, 30:50] != 0
    )  # Check if the second GC image is placed at (30, 30)


def test_generate_center_masks(mock_spatial_model):
    """Test the _generate_center_masks method to ensure masks are extracted correctly."""
    # img_stack = np.random.rand(3, 10, 10)  # 3 images of 10x10
    gc = Mock()
    gc.img = np.random.rand(3, 10, 10)
    ret = Mock()
    ret.mask_threshold = 0.5

    gc = mock_spatial_model._generate_center_masks(ret, gc)

    assert gc.img_mask.shape == gc.img.shape
    assert np.any(gc.img_mask)  # Ensure there is at least some mask created


def test_add_center_mask_area_to_df(mock_spatial_model):
    """Test the _add_center_mask_area_to_df method to ensure center mask areas are added to the DataFrame."""
    mock_gc = Mock()
    mock_gc.img_mask = np.ones((2, 10, 10))  # Two receptive fields, both fully masked
    mock_gc.um_per_pix = 5.0
    mock_gc.df = pd.DataFrame({"other_col": [1, 2]})

    updated_gc = mock_spatial_model._add_center_mask_area_to_df(mock_gc)

    expected_areas = (
        np.ones(2) * 10 * 10 * (mock_gc.um_per_pix**2) / 1e6
    )  # Calculate in mm^2
    np.testing.assert_array_equal(updated_gc.df["center_mask_area_mm2"], expected_areas)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
