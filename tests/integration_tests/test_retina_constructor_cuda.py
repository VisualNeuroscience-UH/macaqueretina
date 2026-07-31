# Built-in
import warnings
from pathlib import Path

# Third-party
import pytest

# Local
import macaqueretina as mr

mr.load_parameters()


@pytest.fixture(scope="module")
def retina_config():
    """
    Fixture to reset and provide the retina configuration.
    While returned as retina_config, this will keep the reference to mr.config.
    """
    mr.config.retina_parameters.ecc_limits_deg = (4.4, 5.6)
    mr.config.retina_parameters.pol_limits_deg = (-1.6, 1.6)
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
def test_retina_construction(
    retina_config,
    tmp_path,
    gc_type,
    response_type,
    spatial_model_type,
    temporal_model_type,
    dog_model_type,
):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Set parameters
    retina_config.retina_parameters.gc_type = gc_type
    retina_config.retina_parameters.response_type = response_type
    retina_config.retina_parameters.spatial_model_type = spatial_model_type
    retina_config.retina_parameters.temporal_model_type = temporal_model_type
    retina_config.retina_parameters.dog_model_type = dog_model_type

    mr.config.output_folder = Path(tmp_path)
    ret, gc = mr.retina_constructor.construct(return_objects=True)

    created_files = [f.stem for f in list(tmp_path.glob("*"))]
    # Extract string after last underscore in each filename
    name_parts = [f.stem.split("_")[-1] for f in list(tmp_path.glob("*"))]
    assert set(name_parts) == set(["metadata", "mosaic", "rfs", "ret"])
    assert len(created_files) == 4

    assert "img" in gc.keys()
    assert "img_mask" in gc.keys()
    assert "X_grid_cen_mm" in gc.keys()
    assert "Y_grid_cen_mm" in gc.keys()
    assert "um_per_pix" in gc.keys()
    assert "pix_per_side" in gc.keys()
    assert "df" in gc.keys()

    assert "cone_optimized_pos_mm" in ret.keys()
    assert "cone_optimized_pos_pol" in ret.keys()
    assert "cone_noise_hash" in ret.keys()
    assert "cones_to_gcs_weights" in ret.keys()
    assert "cone_noise_parameters" in ret.keys()
    assert "noise_frequency_data" in ret.keys()
    assert "noise_power_data" in ret.keys()
    assert "cone_frequency_data" in ret.keys()
    assert "cone_power_data" in ret.keys()
    assert "cone_noise_power_fit" in ret.keys()
    assert "bipolar_optimized_pos_mm" in ret.keys()

    output_folder = Path(retina_config.output_folder)
    assert output_folder.exists()

    warnings.resetwarnings()
