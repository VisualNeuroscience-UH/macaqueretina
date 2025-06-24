# Built-in
from unittest.mock import Mock

# Third-party
import pytest

# Local
from macaqueretina.retina.construct_retina_module import (
    GanglionCellBase,
    GanglionCellParasol,
)


@pytest.fixture
def ganglion_cell_parasol():
    """Fixture to provide an instance of GanglionCellParasol."""
    return GanglionCellParasol()


def test_initialization(ganglion_cell_parasol):
    """Test the initialization of the GanglionCellParasol object."""
    assert ganglion_cell_parasol.n_units is None
    assert ganglion_cell_parasol.um_per_pix is None
    assert ganglion_cell_parasol.pix_per_side is None
    assert ganglion_cell_parasol.um_per_side is None
    assert ganglion_cell_parasol.img is None
    assert ganglion_cell_parasol.img_mask is None
    assert ganglion_cell_parasol.img_lu_pix is None
    assert ganglion_cell_parasol.X_grid_mm is None
    assert ganglion_cell_parasol.Y_grid_mm is None
    assert ganglion_cell_parasol.cones_to_gcs_weights is None


def test_get_BK_parameter_names(ganglion_cell_parasol):
    """Test the get_BK_parameter_names method of GanglionCellParasol."""
    expected_names = [
        "A",
        "NLTL",
        "NL",
        "TL",
        "HS",
        "T0",
        "Chalf",
        "D",
        "Mean",
    ]
    result = ganglion_cell_parasol.get_BK_parameter_names()
    assert result == expected_names
