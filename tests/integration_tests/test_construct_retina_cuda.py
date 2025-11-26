# Built-in
from pathlib import Path

# Third-party
import pytest

# Local
import macaqueretina as mr


@pytest.fixture(scope="module")
def retina_config():
    """
    Fixture to reset and provide the retina configuration.
    While returned as retina_config, this will keep the reference to mr.config.
    """
    mr.config.retina_parameters.ecc_limits_deg = [4.4, 5.6]
    mr.config.retina_parameters.pol_limits_deg = [-1.6, 1.6]
    mr.config.retina_parameters.force_retina_build = True
    mr.config.retina_parameters.gc_type = "parasol"
    mr.config.retina_parameters.response_type = "on"
    mr.config.retina_parameters.spatial_model_type = "DOG"
    mr.config.retina_parameters.temporal_model_type = "fixed"
    mr.config.retina_parameters.dog_model_type = "ellipse_fixed"
    mr.config.device = "cuda"
    mr.config.numpy_seed = 1
    return mr.config


# Define all possible parameter combinations
GC_TYPES = ["parasol", "midget"]
RESPONSE_TYPES = ["on", "off"]
SPATIAL_MODEL_TYPES = ["DOG", "VAE"]
TEMPORAL_MODEL_TYPES = ["fixed", "dynamic", "subunit"]
DOG_MODEL_TYPES = ["ellipse_fixed", "circular"]


@pytest.mark.parametrize(
    "gc_type,response_type,spatial_model_type,temporal_model_type,dog_model_type",
    [
        (gc, resp, spatial, temporal, dog)
        for gc in GC_TYPES
        for resp in RESPONSE_TYPES
        for spatial in SPATIAL_MODEL_TYPES
        for temporal in TEMPORAL_MODEL_TYPES
        for dog in DOG_MODEL_TYPES
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_retina_construction(
    retina_config,
    tmp_path,
    gc_type,
    response_type,
    spatial_model_type,
    temporal_model_type,
    dog_model_type,
):
    # Set parameters
    retina_config.retina_parameters.gc_type = gc_type
    retina_config.retina_parameters.response_type = response_type
    retina_config.retina_parameters.spatial_model_type = spatial_model_type
    retina_config.retina_parameters.temporal_model_type = temporal_model_type
    retina_config.retina_parameters.dog_model_type = dog_model_type

    mr.config.output_folder = Path(tmp_path)
    mr.config.retina_parameters.model_density = 0.8
    ret, gc = mr.construct_retina(return_objects_do_not_save=True)

    assert hasattr(ret, "gc_type")
    assert ret.gc_type == gc_type

    assert hasattr(ret, "response_type")
    assert ret.response_type == response_type

    assert hasattr(ret, "spatial_model_type")
    assert ret.spatial_model_type == spatial_model_type

    assert hasattr(ret, "temporal_model_type")
    assert ret.temporal_model_type == temporal_model_type

    assert hasattr(ret, "dog_model_type")
    assert ret.dog_model_type == dog_model_type

    output_folder = Path(retina_config.output_folder)
    assert output_folder.exists()

    # Other retina object attributes
    ret_attribute_names = [
        "bipolar2gc_dict",
        "bipolar_density_params",
        "bipolar_general_parameters",
        "bipolar_optimized_pos_mm",
        "bipolar_placement_parameters",
        "cone_density_params",
        "cone_frequency_data",
        "cone_general_parameters",
        "cone_noise_parameters",
        "cone_noise_power_fit",
        "cone_optimized_pos_mm",
        "cone_optimized_pos_pol",
        "cone_placement_parameters",
        "cone_power_data",
        "cones_to_gcs_weights",
        "dd_regr_model",
        "deg_per_mm",
        "ecc_lim_mm",
        "ecc_limit_for_dd_fit_mm",
        "experimental_archive",
        "fit_statistics",
        "gc_density_params",
        "gc_placement_parameters",
        "gc_proportion",
        "mask_threshold",
        "model_density",
        "noise_frequency_data",
        "noise_power_data",
        "polar_lim_deg",
        "proportion_of_OFF_response_type",
        "proportion_of_ON_response_type",
        "proportion_of_midget_gc_type",
        "proportion_of_parasol_gc_type",
        "receptive_field_repulsion_parameters",
        "sector_surface_areas_mm2",
        "selected_bipolars_df",
        "whole_ret_img",
        "whole_ret_img_mask",
        "whole_ret_lu_mm",
    ]

    for attr in ret_attribute_names:
        assert hasattr(ret, attr), f"Attribute {attr} does not exist in macaqueretina"
        assert getattr(ret, attr) is not None, f"Attribute {attr} is None"

    # Ganglion cell object attributes
    gc_attribute_names = [
        "X_grid_cen_mm",
        "X_grid_sur_mm",
        "Y_grid_cen_mm",
        "Y_grid_sur_mm",
        "df",
        "exp_pix_per_side",
        "img",
        "img_lu_pix",
        "img_mask",
        "img_mask_sur",
        "n_units",
        "pix_per_side",
        "um_per_pix",
        "um_per_side",
    ]

    # Test for existence and non-None
    for attr in gc_attribute_names:
        assert hasattr(gc, attr), f"Attribute {attr} does not exist in macaqueretina"
        assert getattr(gc, attr) is not None, f"Attribute {attr} is None"
