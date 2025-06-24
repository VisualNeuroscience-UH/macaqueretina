# Built-in
from unittest.mock import Mock

# Third-party
import pytest

# Local
from macaqueretina.retina.construct_retina_module import GanglionCellMidget


@pytest.fixture
def ganglion_cell_midget():
    """Fixture to provide an instance of GanglionCellMidget."""
    return GanglionCellMidget()


def test_initialization(ganglion_cell_midget):
    """Test the initialization of the GanglionCellMidget object."""
    assert ganglion_cell_midget.n_units is None
    assert ganglion_cell_midget.um_per_pix is None
    assert ganglion_cell_midget.pix_per_side is None
    assert ganglion_cell_midget.um_per_side is None
    assert ganglion_cell_midget.img is None
    assert ganglion_cell_midget.img_mask is None
    assert ganglion_cell_midget.img_lu_pix is None
    assert ganglion_cell_midget.X_grid_mm is None
    assert ganglion_cell_midget.Y_grid_mm is None
    assert ganglion_cell_midget.cones_to_gcs_weights is None


def test_get_BK_parameter_names(ganglion_cell_midget):
    """Test the get_BK_parameter_names method of GanglionCellMidget."""
    expected_names = [
        "A_cen",
        "NLTL_cen",
        "NL_cen",
        "HS_cen",
        "TS_cen",
        "D_cen",
        "A_sur",
        "NLTL_sur",
        "NL_sur",
        "HS_sur",
        "TS_sur",
        "deltaNLTL_sur",
        "Mean",
    ]
    result = ganglion_cell_midget.get_BK_parameter_names()
    assert result == expected_names
