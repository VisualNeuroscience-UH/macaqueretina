# Built-in
from unittest.mock import Mock

# Third-party
import numpy as np
import pytest

# Local
from macaqueretina.retina.construct_retina_module import Retina


@pytest.fixture
def retina_parameters_params():
    """Fixture providing default parameters for initializing the Retina object."""
    return {
        "experimental_archive": "exp_archive_001",
        "gc_type": "parasol",
        "response_type": "ON",
        "spatial_model_type": "Gaussian",
        "dog_model_type": "Difference of Gaussian",
        "temporal_model_type": "Linear",
        "fit_statistics": {"R2": 0.9},
        "center_mask_threshold": 0.5,
        "gc_placement_parameters": {"gc_density": 0.8},
        "cone_placement_parameters": {"cone_density": 0.9},
        "cone_general_parameters": {"filtering": "linear"},
        "bipolar_placement_parameters": {"bipolar_density": 0.7},
        "bipolar_general_parameters": {"connectivity": "linear"},
        "dd_regr_model": Mock(),  # Mock the regression model
        "deg_per_mm": 0.25,
        "bipolar2gc_dict": {"mapping": "default"},
        "receptive_field_repulsion_parameters": {"repulsion_strength": 0.5},
        "ecc_limits_deg": [0, 90],
        "ecc_limit_for_dd_fit": 45,
        "pol_limits_deg": [0, 180],
        "model_density": 0.9,
        "proportion_of_parasol_gc_type": 0.6,
        "proportion_of_midget_gc_type": 0.4,
        "proportion_of_ON_response_type": 0.5,
        "proportion_of_OFF_response_type": 0.5,
    }


def test_retina_initialization(retina_parameters_params):
    """Test that the Retina object is initialized with correct parameters."""
    retina = Retina(retina_parameters_params)

    assert retina.experimental_archive == "exp_archive_001"
    assert retina.gc_type == "parasol"
    assert retina.response_type == "ON"
    assert retina.deg_per_mm == 0.25
    assert retina.ecc_lim_mm.shape == (2,)
    assert retina.ecc_lim_mm[0] == 0
    assert retina.ecc_limit_for_dd_fit_mm == 45 / 0.25
    assert retina.model_density == 0.9


def test_retina_assertions(retina_parameters_params):
    """Test that incorrect values in retina_parameters raise the correct assertions."""
    # Test invalid eccentricity limits (not a list or wrong length)
    retina_parameters_params["ecc_limits_deg"] = "not a list"
    with pytest.raises(
        AssertionError, match="Wrong type or length of eccentricity, aborting"
    ):
        Retina(retina_parameters_params)

    # Test invalid polar limits (not a list or wrong length)
    retina_parameters_params["ecc_limits_deg"] = [0, 90]  # correct this one
    retina_parameters_params["pol_limits_deg"] = "not a list"
    with pytest.raises(
        AssertionError, match="Wrong type or length of pol_limits_deg, aborting"
    ):
        Retina(retina_parameters_params)

    # Test model density out of bounds
    retina_parameters_params["pol_limits_deg"] = [0, 180]  # correct this one
    retina_parameters_params["model_density"] = 1.1
    with pytest.raises(AssertionError, match="Density should be <=1.0, aborting"):
        Retina(retina_parameters_params)


def test_dd_regr_model_called(retina_parameters_params):
    """Test that the dd_regr_model is correctly assigned and can be called."""
    retina = Retina(retina_parameters_params)

    # Ensure the dd_regr_model is a mock and test a call to it
    assert isinstance(retina.dd_regr_model, Mock)

    # Simulate calling the regression model and check call
    retina.dd_regr_model.predict = Mock(return_value=5)
    result = retina.dd_regr_model.predict([[1, 2], [3, 4]])
    retina.dd_regr_model.predict.assert_called_once_with([[1, 2], [3, 4]])
    assert result == 5


def test_retina_computed_attributes_are_none(retina_parameters_params):
    """Test that computed attributes are initialized as None."""
    retina = Retina(retina_parameters_params)

    assert retina.whole_ret_img is None
    assert retina.whole_ret_lu_mm is None
    assert retina.cones_to_gcs_weights is None


def test_retina_proportions(retina_parameters_params):
    """Test that the proportions of ganglion cell types and responses are set correctly."""
    retina = Retina(retina_parameters_params)

    assert retina.proportion_of_parasol_gc_type == 0.6
    assert retina.proportion_of_midget_gc_type == 0.4
    assert retina.proportion_of_ON_response_type == 0.5
    assert retina.proportion_of_OFF_response_type == 0.5
