"""
Parameter validation with Pydantic.

Project and data specific.

Each parameter in the project configuration, after being loaded, is validated
against the required type. Any parameters that need to be derived from
other parameters are computed here.
"""

from __future__ import annotations

# Built-in
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Third-party
import brian2.units as b2u
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    # Local
    from macaqueretina.data_io.config_io import Configuration


class BaseConfigModel(BaseModel):
    """
    Base class for configuration models.
    All models inherit from this class to ensure extra parameters are allowed
    and retained from everywhere in the YAML files.
    """

    def __init__(self, **data: dict):
        provided_fields = set(data.keys())
        for field_name, field_content in type(self).model_fields.items():
            if (
                field_name not in provided_fields
                and hasattr(field_content, "default")
                and field_content.default is not None
            ):
                print(
                    f"Parameter '{field_name}' not provided in the YAML file(s), "
                    f"using default value: {field_content.default} (set in param_validation.py)",
                )

        super().__init__(**data)

    model_config = ConfigDict(extra="allow")


class BaseInternalConfigModel(BaseModel):
    """
    Base class for configuration models.
    All models inherit from this class to ensure extra parameters are allowed
    and retained from everywhere in the YAML files.
    """

    def __init__(self, **data: dict):
        provided_fields = set(data.keys())
        super().__init__(**data)

    model_config = ConfigDict(extra="allow")


## From retina_parameters.yaml
class RetinaParameters(BaseConfigModel):
    gc_type: Literal["parasol", "midget"]
    response_type: Literal["on", "off"]
    spatial_model_type: Literal["DOG", "VAE"]
    temporal_model_type: Literal["fixed", "dynamic", "subunit"]
    dog_model_type: Literal["ellipse_fixed", "ellipse_independent", "circular"]
    ecc_limits_deg: list[float, float] = Field(
        default=[4.5, 5.5], description="eccentricity in degrees"
    )
    pol_limits_deg: list[float, float] = Field(
        default=[-1.5, 1.5], description="polar angle in degrees"
    )
    model_density: float = Field(
        le=1.0,
        default=1.0,
        description="1.0 for 100% \of the literature density of ganglion cells",
    )
    retina_center: complex = Field(
        default=complex(5.0 + 0j),
        description="degrees, this is stimulus_position (0, 0)",
    )
    force_retina_build: bool = Field(
        default=True, description="If True, rebuilds retina even if the hash matches"
    )
    model_file_name: Path | None = Field(
        default=None,
        description="null for most recent or 'model_[GC TYPE]_[RESPONSE TYPE]_[DEVICE]_[TIMESTAMP].pt' at input_folder. Applies to VAE only",
    )

    @field_validator("retina_center", mode="before")
    @classmethod
    def parse_complex(cls, v):
        if isinstance(v, str):
            return complex(v)


## From visual_stimulus_parameters.yaml
class VisualStimulusParameters(BaseConfigModel):
    pattern: str = Field(
        default="temporal_square_pattern",
        description="Options: 'sine_grating', 'square_grating', 'colored_temporal_noise', 'white_gaussian_noise', 'natural_images', 'natural_video', 'temporal_sine_pattern', 'temporal_square_pattern', 'temporal_chirp_pattern', 'contrast_chirp_pattern', 'spatially_uniform_binary_noise'",
    )
    image_width: int = Field(default=240, description="image (canvas) width in pixels")
    image_height: int = 240
    pix_per_deg: int = 60
    dtype_name: str = (
        "float16"  # low contrast needs "float16", for performance, use "uint8"
    )
    fps: int = 300
    duration_seconds: float = Field(
        default=0.5, description="actual frames will be floor(duration_seconds * fps)"
    )
    baseline_start_seconds: float = 0.1
    baseline_end_seconds: float = 0.5
    stimulus_form: str = "circular"
    size_inner: float = Field(default=0.1, description="deg, applies to annulus only")
    size_outer: float = Field(default=1.0, description="deg, applies to annulus only")
    stimulus_position: tuple = Field(
        default=(0.0, 0.0), description="relative to stimuls center in retina"
    )
    stimulus_size: float = Field(
        default=1.5, description="deg, radius for circle, sidelen/2 for rectangle."
    )
    temporal_frequency: float = Field(default=0.01, description="Hz")
    temporal_frequency_range: tuple = Field(
        default=(0.5, 50), description="Hz, applies to temporal chirp only"
    )
    spatial_frequency: float = Field(default=2.0, description="cpd")
    orientation: float = Field(default=90.0, description="degrees")
    phase_shift: float = Field(default=0.0, description="radians")
    stimulus_video_name: str | None
    contrast: float = Field(default=1.0)
    mean: float = Field(
        default=128.0, description="Mean luminance in  cd/m2 and adaptation level"
    )
    intensity: tuple | None = Field(
        description="Intensity overrides contrast and mean when provided"
    )
    background: Literal["mean", "intensity_min", "intensity_max"] | int | float = Field(
        default="mean"
    )
    ND_filter: float = Field(
        default=0.0,
        description="log10 neutral density filter factor, can be negative",
    )


class StimulusMetadataParameters(BaseConfigModel):
    stimulus_file: Path
    pix_per_deg: int = Field(
        default=30, description="VanHateren_1998_ProcRSocLondB 2 arcmin per pixel"
    )
    fps: int = 25


class RunParameters(BaseConfigModel):
    n_sweeps: int = Field(description="For each of the response files")
    spike_generator_model: Literal["poisson", "refractory"]
    save_data: bool
    simulation_dt: float = Field(default=0.0001, description="in sec 0.001 = 1 ms")
    save_variables: list[str]


class VaeTrainParameters(BaseInternalConfigModel):
    vae_run_mode: Literal["load_model", "train_model"] = Field(
        default="load_model", description="train_model requires experimental data"
    )
    epochs: int = Field(default=500, description="Number of training epochs")
    lr_step_size: int = Field(
        default=20, description="Learning rate decay step size (in epochs)"
    )
    lr_gamma: float | int = Field(
        default=0.9,
        description="Learning rate decay (multiplier for learning rate)",
    )
    resolution_hw: int = Field(
        default=13, description="Both x and y images will be sampled to this space."
    )
    latent_dim: int = Field(
        default=32, description="Latent dimension (powers of 2 between 2 and 128)"
    )
    channels: int = Field(default=16, description="Number of channels")
    lr: float = Field(default=0.0005, description="Learning rate")
    batch_size: int | None = Field(default=256, description="Batch size")
    test_split: float = Field(
        default=0.2,
        description="Split data for validation and testing (both will take this fraction of data)",
    )
    kernel_stride: Literal["k3s1", "k3s2", "k5s2", "k5s1", "k7s1"] = "k7s1"
    conv_layers: int = Field(default=2, description="Number of convolutional layers")
    batch_norm: bool = Field(default=True, description="Use batch normalization")
    latent_distribution: Literal["normal", "uniform"] = "uniform"

    @field_validator("latent_dim", mode="after")
    @classmethod
    def latent_dim_pow_2(cls, latent_dim: int) -> int:
        """Check whether latent_dim is a power of 2 between 2 and 128"""
        if not (2 <= latent_dim <= 128):
            raise ValueError(
                "latent_dim (in config/constants.yaml, vae_train_parameters) must be a power of 2 between 2 and 128."
            )
        if latent_dim & (latent_dim - 1) != 0:
            raise ValueError(
                "latent_dim (in config/constants.yaml, vae_train_parameters) must be a power of 2 between 2 and 128."
            )

        return latent_dim

    class AugmentationDict(BaseInternalConfigModel):
        rotation: int = Field(default=0, description="Rotation in degrees")
        translation: tuple[int, int] = Field(
            default=(0, 0), description="Fraction of image, in (x, y) -directions"
        )
        noise: float = Field(
            default=0,
            description="Noise float in [0, 1] (noise is added to the image)",
        )
        flip: float = Field(
            default=0.5,
            description="Flip probability, both horizontal and vertical",
        )
        data_multiplier: int = Field(
            default=4, description="How many times to get the data w/ augmentation"
        )

    augmentation_dict: AugmentationDict | None = AugmentationDict()


## From constants.yaml
class NoiseGainDefault(BaseConfigModel):
    class On(BaseConfigModel):
        parasol: float
        midget: float

    on: On

    class Off(BaseConfigModel):
        parasol: float
        midget: float

    off: Off


class DdRegrModel(BaseConfigModel):
    parasol: Literal["linear", "quadratic", "cubic", "powerlaw"] = Field(
        default="powerlaw",
        description="Dendritic diameter regression model for parasol",
    )
    midget: Literal["linear", "quadratic", "cubic", "powerlaw"] = Field(
        default="quadratic",
        description="Dendritic diameter regression model for midget",
    )


class RetinaParametersAppend(BaseConfigModel):
    noise_type: Literal["shared", "independent"] = Field(default="shared")
    fixed_mask_threshold: float = Field(
        default=0.1,
        description="Applies to rf volume normalization. This value is also the only surround mask threshold.",
    )
    center_mask_threshold: float = Field(
        default=0.1,
        description="Limits rf center extent to values above this proportion of the peak values after volume normalization.",
    )
    data_noise_threshold: float = Field(
        default=0.1,
        description="0.1 for +-10% /from abs max removed, 0 for no noise removal. Applies to DOG spatial model statistics.",
    )
    fit_statistics: Literal["univariate", "multivariate"] = Field(
        default="multivariate",
        description="Applies only to DOG spatial and fixed temporal model statistics.",
    )
    dd_regr_model: None = Field(
        default=None, description="Dendritic diameter regression model"
    )
    ecc_limit_for_dd_fit: float = Field(
        default=20.0, description="degrees, math.inf for no limit"
    )


class SignalGain(BaseConfigModel):
    threshold: float = Field(default=10.0, description="Threshold in Hz")
    parasol: dict[str, float] = Field(default_factory=dict)
    midget: dict[str, float] = Field(default_factory=dict)


class ExperimentalMetadata(BaseInternalConfigModel):
    data_microm_per_pix: int | None = 60
    data_spatialfilter_height: int | None = 13
    data_spatialfilter_width: int | None = 13
    data_fps: int | None = 30
    data_temporalfilter_samples: int | None = 15
    relative_data_path: Path | None = Path(".")

    @computed_field
    @property
    def experimental_data_folder(self) -> Path:
        return git_repo_path.joinpath(self.relative_data_path)

    @computed_field
    @property
    def exp_rf_stat_folder(self) -> Path:
        return git_repo_path.joinpath(r"retina/dog_statistics")


class Visual2CorticalParams(BaseConfigModel):
    a: float = 0.077 / 0.082  # ~0.94
    k: float = 1 / 0.082  # ~12.2


class ConeGeneralParameters(BaseConfigModel):
    rm: float = 25
    k: float = 2.77e-4
    sensitivity_min: float = 5e2
    sensitivity_max: float = 2e4
    cone2gc_midget: float = 9
    cone2gc_parasol: float = 27
    cone2gc_cutoff_SD: float = 1
    cone2bipo_cutoff_SD: float = 1
    cone_noise_wc: list[float] = [14, 160]


class ConeSignalParameters(BaseConfigModel):
    unit: str = "pA"
    A_pupil: float = 9.0
    lambda_nm: float = 555
    input_gain: float = 1.0
    r_dark: float = -136
    max_response: float = 116.8
    alpha: float = 19.4
    beta: float = 0.36
    gamma: float = 0.448
    tau_y: float = 4.49
    n_y: float = 4.33
    tau_z: float = 166
    n_z: float = 1.0
    tau_r: float = 4.78
    filter_limit_time: float = 3.0

    @field_validator("r_dark", "max_response", "alpha", mode="after")
    @classmethod
    def add_b2u_pa(cls, v) -> b2u.Quantity:
        return v * b2u.pA

    @field_validator("beta", "tau_y", "tau_z", "tau_r", "alpha", mode="after")
    @classmethod
    def add_b2u_ms(cls, v):
        return v * b2u.ms

    @field_validator("filter_limit_time", mode="after")
    @classmethod
    def add_b2u_second(cls, v):
        return v * b2u.second


class BipolarGeneralParameters(BaseConfigModel):
    bipo2gc_div: int = 6
    bipo2gc_cutoff_SD: int = 2
    cone2bipo_cen_sd: int = 10
    cone2bipo_sur_sd: int = 150
    bipo_sub_sur2cen: float = 0.9


class RefractoryParameters(BaseConfigModel):
    abs_refractory: int = 1
    rel_refractory: int = 3
    p_exp: int = 4
    clip_start: int = 0
    clip_end: int = 100


class GcPlacementParameters(BaseConfigModel):
    algorithm: Literal["voronoi", "force"] | None = "force"
    n_iterations: int = 30
    change_rate: float = 0.0005
    unit_repulsion_stregth: int = 10
    unit_distance_threshold: float = 0.2
    diffusion_speed: float = 0.001
    border_repulsion_stength: float = 0.2
    border_distance_threshold: float = 0.001
    show_placing_progress: bool = False
    show_skip_steps: int = 1


class ConePlacementParameters(BaseConfigModel):
    algorithm: Literal["voronoi", "force"] | None = "force"
    n_iterations: int = 15
    change_rate: float = 0.0005
    unit_repulsion_stregth: int = 2
    unit_distance_threshold: float = 0.25
    diffusion_speed: float = 0.002
    border_repulsion_stength: float = 5
    border_distance_threshold: float = 0.001
    show_placing_progress: bool = False
    show_skip_steps: int = 1


class BipolarPlacementParameters(BaseConfigModel):
    algorithm: Literal["voronoi", "force"] | None = "force"
    n_iterations: int = 15
    change_rate: float = 0.0005
    unit_repulsion_stregth: int = 2
    unit_distance_threshold: float = 0.2
    diffusion_speed: float = 0.005
    border_repulsion_stength: float = 5
    border_distance_threshold: float = 0.0001
    show_placing_progress: bool = False
    show_skip_steps: int = 1


class NIterations(BaseConfigModel):
    parasol: int
    midget: int


class ReceptiveFieldRepulsionParameters(BaseConfigModel):
    change_rate: float = 0.005
    cooling_rate: float = 0.999
    border_repulsion_stength: float = 5
    show_repulsion_progress: bool = False
    show_only_unit: int | None = None
    show_skip_steps: int = 5
    savefigname: str | None


class Bipolar2gcDict(BaseConfigModel):
    class Midget(BaseConfigModel):
        on: list = ["IMB"]
        off: list = ["FMB"]

    midget: Midget

    class Parasol(BaseConfigModel):
        on: list = ["DB4", "DB5"]
        off: list = ["DB2", "DB3"]

    parasol: Parasol


class DendrDiamUnits(BaseConfigModel):
    data1: list[str, str] = ["mm", "um"]
    data2: list[str, str] = ["mm", "um"]
    data3: list[str, str] = ["deg", "um"]


## Main validation class TODO : explain with few sentences what this does.
class ConfigParams(BaseConfigModel):

    # Experiment parameters
    model_root_path: Path = Field(description="Update this to your model root path")
    project: str = Field(description="Project name")
    experiment: str = Field(
        description="Current experiment. Use distinct folders for distinct stimuli."
    )
    input_folder: str
    output_folder: str
    numpy_seed: int = Field(
        ge=0,
        le=1000000,
        default=42,
        description="Remove random variations by setting the numpy random seed",
    )
    device: Literal["cpu", "cuda"]
    run: dict[str, Any]

    # Retina parameters
    retina_parameters: RetinaParameters

    # Visual stimulus parameters
    visual_stimulus_parameters: VisualStimulusParameters
    external_stimulus_parameters: StimulusMetadataParameters
    n_files: int = Field(
        description="Each gc response file contains n_sweeps with independent cone noise and spike generator."
    )
    run_parameters: RunParameters
    vae_train_parameters: VaeTrainParameters | None = VaeTrainParameters()

    # Constants
    noise_gain_default: NoiseGainDefault
    dd_regr_model: DdRegrModel
    retina_parameters_append: RetinaParametersAppend
    experimental_metadata: ExperimentalMetadata | None = ExperimentalMetadata()

    proportion_of_parasol_gc_type: float = 0.08
    proportion_of_midget_gc_type: float = 0.64
    proportion_of_ON_response_type: float = 0.40
    proportion_of_OFF_response_type: float = 0.60

    deg_per_mm: float
    optical_aberration: float
    visual2cortical_params: Visual2CorticalParams
    cone_general_parameters: ConeGeneralParameters
    cone_signal_parameters: ConeSignalParameters
    bipolar_general_parameters: BipolarGeneralParameters
    refractory_parameters: RefractoryParameters
    gc_placement_parameters: GcPlacementParameters
    cone_placement_parameters: ConePlacementParameters
    bipolar_placement_parameters: BipolarPlacementParameters
    n_iterations: NIterations
    receptive_field_repulsion_parameters: ReceptiveFieldRepulsionParameters
    bipolar2gc_dict: Bipolar2gcDict

    @computed_field
    @property
    def literature_data_folder(self) -> Path:
        return git_repo_path.joinpath(r"retina/literature_data")

    gc_density_1_datafile: str
    gc_density_1_scaling_data_and_function: list
    gc_density_2_datafile: str
    gc_density_control_datafile: str
    dendr_diam_units: DendrDiamUnits

    dendr_diam1_datafile_parasol: str
    dendr_diam2_datafile_parasol: str
    dendr_diam3_datafile_parasol: str
    temporal_BK_model_datafile_parasol: str
    spatial_DoG_datafile_parasol: str
    dendr_diam1_datafile_midget: str
    dendr_diam2_datafile_midget: str
    dendr_diam3_datafile_midget: str
    temporal_BK_model_datafile_midget: str
    spatial_DoG_datafile_midget: str

    cone_density1_datafile: str
    cone_density2_datafile: str
    cone_noise_datafile: str
    cone_response_datafile: str
    bipolar_table_datafile: str
    parasol_on_RI_values_datafile: str
    parasol_off_RI_values_datafile: str
    temporal_pattern_datafile: str

    profile: bool = False

    @field_validator("model_root_path", mode="after")
    @classmethod
    def convert_model_root_path_to_path(cls, model_root_path) -> Path:
        if not model_root_path.exists():
            print(
                f"\033[91m \nModel root path {model_root_path} does not exist.\n"
                f"Update core_parameters.yaml model_root_path.\n"
                f"Using current working directory. \033[0m\n"
            )
            return Path.cwd()

        return Path(model_root_path)

    @computed_field
    @property
    def stimulus_folder(self) -> str:
        """Stimulus images and videos"""
        return Path(f"stim_{self.output_folder}")

    @model_validator(mode="after")
    def set_derived_values(self):
        # Set parameters that depend on another value in a different class
        self.project_conf_module_file_path = proj_conf_mod_file_path
        self.git_repo_root_path = git_repo_path
        if self.visual_stimulus_parameters.stimulus_video_name is None:
            self.visual_stimulus_parameters.stimulus_video_name = (
                f"{self.stimulus_folder}.mp4"
            )
        self.retina_parameters_append.dd_regr_model = getattr(
            self.dd_regr_model, self.retina_parameters.gc_type
        )
        self.receptive_field_repulsion_parameters.n_iterations = getattr(
            self.n_iterations, self.retina_parameters.gc_type
        )

        # Set signal gain from a separate yaml file
        self.retina_parameters.signal_gain = (
            self.signal_gain.get(self.retina_parameters.gc_type)
            .get(self.retina_parameters.response_type)
            .get(self.retina_parameters.spatial_model_type)
            .get(self.retina_parameters.temporal_model_type)
        )

        self.experimental_metadata.mask_noise = (
            self.retina_parameters_append.data_noise_threshold
        )
        if self.retina_parameters.gc_type == "parasol":
            self.dendr_diam1_datafile = self.dendr_diam1_datafile_parasol
            self.dendr_diam2_datafile = self.dendr_diam2_datafile_parasol
            self.dendr_diam3_datafile = self.dendr_diam3_datafile_parasol
            self.temporal_BK_model_datafile = self.temporal_BK_model_datafile_parasol
            self.spatial_DoG_datafile = self.spatial_DoG_datafile_parasol
        elif self.retina_parameters.gc_type == "midget":
            self.dendr_diam1_datafile = self.dendr_diam1_datafile_midget
            self.dendr_diam2_datafile = self.dendr_diam2_datafile_midget
            self.dendr_diam3_datafile = self.dendr_diam3_datafile_midget
            self.temporal_BK_model_datafile = self.temporal_BK_model_datafile_midget
            self.spatial_DoG_datafile = self.spatial_DoG_datafile_midget

        self.path = self.model_root_path.joinpath(Path(self.project), self.experiment)
        return self


def _validate_paths(config):

    # Create new root/project/experiment path if it doesn't exist
    if not config.path.is_dir():
        config.path.mkdir(parents=True, exist_ok=True)

    # Validate main project path, must be absolute
    if not config.path.is_absolute():
        raise KeyError("The 'path' parameter is not an absolute path, aborting...")

    # Create the output, stimulus and input folders if they don't exist
    config.output_folder = config.path.joinpath(config.output_folder)
    config.stimulus_folder = config.path.joinpath(config.stimulus_folder)
    config.input_folder = config.path.joinpath(config.input_folder)

    config.input_folder.mkdir(parents=True, exist_ok=True)
    config.output_folder.mkdir(parents=True, exist_ok=True)
    config.stimulus_folder.mkdir(parents=True, exist_ok=True)


# Façade
def validate_params(
    config: Configuration,
    project_conf_module_file_path: Path,
    git_repo_root_path: Path,
) -> ConfigParams:
    """
    Validate and convert parameters to the appropriate types.

    Parameters
    ----------
    config:
        Configuration object with the loaded configuration from the YAML files

    Returns
    -------
    ConfigParams:
        ConfigParams object with the validated parameters, plus any computed
        field from ConfigParams.
    """
    global proj_conf_mod_file_path
    proj_conf_mod_file_path = project_conf_module_file_path

    global git_repo_path
    git_repo_path = git_repo_root_path

    # Store retina_parameter.yaml keys as core parameters for downstream hashing
    config.retina_core_parameter_keys = config.retina_parameters.keys()

    validated_config: ConfigParams = ConfigParams(**config.as_dict())
    validated_config: dict = validated_config.model_dump()

    config.clear()
    config.update(validated_config)

    _validate_paths(config)

    return config
