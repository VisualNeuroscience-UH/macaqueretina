# Built-in

# Third-party
import pytest

# Local
import macaqueretina as mr

# Define parameters to vary
GC_TYPES = ["parasol", "midget"]
RESPONSE_TYPES = ["on", "off"]
TEMPORAL_MODEL_TYPES = ["fixed", "dynamic", "subunit"]
SPIKE_GENERATOR_MODELS = ["refractory", "poisson"]


@pytest.fixture(scope="module")
def simulation_config(tmp_path_factory):
    """
    Fixture to set up retina, stimulus, and simulation config.
    Uses separate temp dirs for retina output and stimulus.
    """
    # Temp directories
    output_dir = tmp_path_factory.mktemp("retina_output")
    stimulus_dir = tmp_path_factory.mktemp("stimulus")

    # Retina config
    mr.config.retina_parameters.force_retina_build = True
    mr.config.retina_parameters.gc_type = "parasol"
    mr.config.retina_parameters.response_type = "on"
    mr.config.retina_parameters.spatial_model_type = "DOG"
    mr.config.retina_parameters.dog_model_type = "ellipse_fixed"
    mr.config.retina_parameters.model_density = 0.8
    mr.config.numpy_seed = 1

    # Stimulus config
    mr.config.visual_stimulus_parameters.pattern = "sine_grating"
    mr.config.visual_stimulus_parameters.stimulus_form = "circular"
    mr.config.visual_stimulus_parameters.duration_seconds = 0.1
    mr.config.visual_stimulus_parameters.fps = 30
    mr.config.visual_stimulus_parameters.image_width = 200
    mr.config.visual_stimulus_parameters.image_height = 200
    mr.config.visual_stimulus_parameters.mean = 128
    mr.config.visual_stimulus_parameters.contrast = 0.5
    mr.config.visual_stimulus_parameters.stimulus_size = 1.0

    # Simulation config
    mr.config.simulation_parameters.simulation_dt = 0.001  # 1 ms for testing

    # Directories
    mr.config.output_folder = output_dir
    mr.config.stimulus_folder = stimulus_dir

    mr.config.device = "cuda"

    return mr.config


@pytest.fixture(scope="module")
def stimulus_video(simulation_config):
    """
    Fixture to generate the stimulus video once and reuse it.
    """
    return mr.make_stimulus()


@pytest.mark.parametrize(
    "gc_type,response_type,temporal_model_type,spike_generator_model",
    [
        (gc_type, response_type, temporal, spike_model)
        for gc_type in GC_TYPES
        for response_type in RESPONSE_TYPES
        for temporal in TEMPORAL_MODEL_TYPES
        for spike_model in SPIKE_GENERATOR_MODELS
    ],
)
@pytest.mark.filterwarnings("ignore::scipy.optimize.OptimizeWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_simulation_client(
    simulation_config,
    stimulus_video,
    gc_type,
    response_type,
    temporal_model_type,
    spike_generator_model,
):
    # Set parameters
    simulation_config.retina_parameters.gc_type = gc_type
    simulation_config.retina_parameters.response_type = response_type
    simulation_config.retina_parameters.temporal_model_type = temporal_model_type
    simulation_config.simulation_parameters.spike_generator_model = (
        spike_generator_model
    )

    # Construct retina
    mr.construct_retina()

    # Run simulation
    gc_type = mr.config.retina_parameters["gc_type"]
    response_type = mr.config.retina_parameters["response_type"]
    hashstr = mr.config.retina_parameters["retina_parameters_hash"]
    filename = f"{gc_type}_{response_type}_{hashstr}_response_testing"

    mr.simulate_retina(filename=filename)

    # Assertions
    output_dir = simulation_config.output_folder
    assert output_dir.exists()
    assert any(output_dir.iterdir()), "No output files generated"

    # Check existing output files
    output_files = list(output_dir.glob(f"{filename}*.gz"))
    assert len(output_files) > 0, "No output files found matching the filename pattern"
