# Built-in
from abc import ABC
from unittest.mock import Mock, patch

# Third-party
import numpy as np
import pandas as pd
import pytest

# Local
from macaqueretina.retina.construct_retina_module import GanglionCellBase


class GanglionCellConcrete(GanglionCellBase):
    """Concrete implementation of GanglionCellBase for testing purposes."""

    def get_BK_parameter_names(self):
        return ["param1", "param2"]

    def get_proportion(self, ret):
        return 0.5


@pytest.fixture
def ganglion_cell():
    """Fixture to provide an instance of GanglionCellConcrete."""
    return GanglionCellConcrete()


def test_initialization(ganglion_cell):
    """Test the initialization of the GanglionCellConcrete object."""
    assert ganglion_cell.n_units is None
    assert ganglion_cell.um_per_pix is None
    assert isinstance(ganglion_cell.df, pd.DataFrame)
    assert "pos_ecc_mm" in ganglion_cell.df.columns


def test_set_initial_attributes(ganglion_cell):
    """Test setting initial attributes of the ganglion cell."""
    # Assign some attributes
    ganglion_cell.n_units = 100
    ganglion_cell.um_per_pix = 1.0
    ganglion_cell.pix_per_side = 256

    # Test if the attributes were set correctly
    assert ganglion_cell.n_units == 100
    assert ganglion_cell.um_per_pix == 1.0
    assert ganglion_cell.pix_per_side == 256


def test_dataframe_structure(ganglion_cell):
    """Test the structure of the DataFrame used in the ganglion cell."""
    expected_columns = [
        "pos_ecc_mm",
        "pos_polar_deg",
        "xoc_pix",
        "yoc_pix",
        "ecc_group_idx",
        "gc_scaling_factors",
        "zoom_factor",
        "den_diam_um",
        "center_mask_area_mm2",
        "center_fit_area_mm2",
        "ampl_c",
        "ampl_s",
        "ampl_c_norm",
        "ampl_s_norm",
        "relat_sur_ampl",
        "offset",
        "tonic_drive",
        "A",
        "Mean",
    ]
    assert list(ganglion_cell.df.columns) == expected_columns


def test_get_BK_parameter_names(ganglion_cell):
    """Test the get_BK_parameter_names method."""
    result = ganglion_cell.get_BK_parameter_names()
    assert result == ["param1", "param2"]


def test_get_proportion(ganglion_cell):
    """Test the get_proportion method with a mock retina object."""
    mock_retina = Mock()
    result = ganglion_cell.get_proportion(mock_retina)
    assert result == 0.5


def test_fixed_values_in_dataframe(ganglion_cell):
    """Test that fixed values are correctly added to the DataFrame."""
    # Check if the DataFrame contains the fixed value column 'offset'
    assert ganglion_cell.df_fixed_values["offset"] == 0.0
