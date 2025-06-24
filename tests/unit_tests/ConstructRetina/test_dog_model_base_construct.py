# Built-in
from abc import ABC, abstractmethod
from typing import Any
from unittest.mock import Mock, patch

# Third-party
import numpy as np
import pandas as pd
import pytest

# Local
from macaqueretina.retina.construct_retina_module import DoGModelBase


# Mock subclass of DoGModelBase for testing
class MockDoGModel(DoGModelBase):
    def scale_to_mm(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_fit_img(
        self, x_grid: np.ndarray, y_grid: np.ndarray, popt: np.ndarray
    ) -> np.ndarray:
        return np.zeros((len(x_grid), len(y_grid)))

    def get_param_names(self, gc: Any) -> list[str]:
        return ["param1", "param2"]

    def transform_vae_dog_to_mm(
        self, df: pd.DataFrame, gc_df_in: pd.DataFrame, mm_per_pix: float
    ) -> pd.DataFrame:
        return df


@pytest.fixture
def mock_dog_model():
    """Fixture to provide an instance of MockDoGModel."""
    ret = Mock()
    fit = Mock()
    retina_math = Mock()
    return MockDoGModel(ret, fit, retina_math)


def test_initialization(mock_dog_model):
    """Test that the DoGModelBase initialization works as expected."""
    assert mock_dog_model.ret is not None
    assert mock_dog_model.fit is not None
    assert mock_dog_model.retina_math is not None


def test_add_center_fit_area_to_df(mock_dog_model):
    """Test the _add_center_fit_area_to_df method to ensure it calculates the correct center area."""
    mock_gc = Mock()
    mock_gc.df = pd.DataFrame({"semi_xc_mm": [1.0, 2.0], "semi_yc_mm": [1.0, 2.0]})

    updated_gc = mock_dog_model._add_center_fit_area_to_df(mock_gc)

    # Test the correct center fit area is added to the DataFrame
    expected_areas = np.pi * np.array([1.0, 2.0]) * np.array([1.0, 2.0])
    pd.testing.assert_series_equal(
        updated_gc.df["center_fit_area_mm2"],
        pd.Series(expected_areas),
        check_names=False,
    )


def test_get_dd_in_um(mock_dog_model):
    """Test the _get_dd_in_um method to ensure it correctly computes dendritic diameters."""
    mock_gc = Mock()
    mock_gc.df = pd.DataFrame(
        {"semi_xc_mm": [0.001, 0.002], "semi_yc_mm": [0.001, 0.002]}
    )

    # Mock the ellipse2diam method
    mock_dog_model.retina_math.ellipse2diam.return_value = [10, 20]

    updated_gc = mock_dog_model._get_dd_in_um(mock_gc)

    # Ensure the correct values were added to the DataFrame
    pd.testing.assert_series_equal(
        updated_gc.df["den_diam_um"],
        pd.Series([10, 20]),
        check_names=False,
    )

    # Ensure the retina_math method was called with the correct arguments
    np.testing.assert_array_equal(
        mock_dog_model.retina_math.ellipse2diam.call_args[0][0],
        mock_gc.df["semi_xc_mm"].values * 1000,
    )
    np.testing.assert_array_equal(
        mock_dog_model.retina_math.ellipse2diam.call_args[0][1],
        mock_gc.df["semi_yc_mm"].values * 1000,
    )


def test_get_center_volume(mock_dog_model):
    """Test the _get_center_volume method to ensure it correctly computes center volume."""
    mock_gc = Mock()
    mock_gc.df = pd.DataFrame(
        {"ampl_c": [1.0, 2.0], "semi_xc_mm": [1.0, 2.0], "semi_yc_mm": [1.0, 2.0]}
    )

    volumes = mock_dog_model._get_center_volume(mock_gc)

    # Calculate expected volumes
    expected_volumes = (
        2 * np.pi * np.array([1.0, 2.0]) * np.array([1.0, 2.0]) * np.array([1.0, 2.0])
    )

    # Assert the computed volumes match the expected values
    np.testing.assert_array_equal(volumes, expected_volumes)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
