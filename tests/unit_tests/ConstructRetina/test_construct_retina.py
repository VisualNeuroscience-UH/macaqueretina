# test_construct_retina.py

# Built-in
from unittest.mock import MagicMock, PropertyMock, patch

# Third-party
import numpy as np
import pandas as pd

# Local
from macaqueretina.retina.construct_retina_module import ConstructRetina

# Initialize project_data_mock correctly

# # Initialize the ConstructRetina instance with the mocks
# self.construct_retina = ConstructRetina(
#     context=self.context_mock,
#     data_io=self.data_io_mock,
#     viz=self.viz_mock,
#     fit=self.fit_mock,
#     retina_vae=self.retina_vae_mock,
#     retina_math=self.retina_math_mock,
#     project_data=self.project_data_mock,
#     get_xy_from_npz=self.get_xy_from_npz_mock,
# )


class ProjectData:
    def __init__(self):
        self.construct_retina = {}


class TestConstructRetina:
    def setup_method(self):
        # Create mock objects for the dependencies
        self.context_mock = MagicMock()
        self.data_io_mock = MagicMock()
        self.viz_mock = MagicMock()
        self.fit_mock = MagicMock()
        self.retina_vae_mock = MagicMock()
        self.retina_math_mock = MagicMock()
        # self.project_data_mock = {}
        self.get_xy_from_npz_mock = MagicMock()

        self.project_data_mock = ProjectData()
        # self.project_data_mock = MagicMock()
        # self.project_data_mock.construct_retina = {}

        # Set up context mock
        self.context_mock.device = MagicMock()
        self.context_mock.retina_parameters = {
            "spatial_model_type": "VAE",
            "training_mode": "some_mode",
            "spatial_rfs_file": "spatial_rfs.npz",
            "ret_file": "retina.npz",
            "mosaic_file": "mosaic.csv",
            "retina_parameters_hash": "abc123",
            "force_retina_build": False,
        }
        self.context_mock.literature_data_files = {
            "gc_density_path": "path/to/gc_density.npz",
            "cone_density1_path": "path/to/cone_density1.npz",
            "cone_density2_path": "path/to/cone_density2.npz",
            "bipolar_table_path": "path/to/bipolar_table.csv",
            "dendr_diam1_path": "path/to/dendr_diam1.npz",
            "dendr_diam2_path": "path/to/dendr_diam2.npz",
            "dendr_diam3_path": "path/to/dendr_diam3.npz",
            "dendr_diam_units": "um",
            "temporal_BK_model_path": "path/to/temporal_BK_model.npz",
            "cone_response_path": "path/to/cone_response.npz",
            "cone_noise_path": "path/to/cone_noise.npz",
            "parasol_some_response_RI_values_fullpath": "path/to/RI_values.npz",
        }
        self.context_mock.apricot_metadata_parameters = {
            "metadata_key": "metadata_value"
        }
        self.context_mock.output_folder = MagicMock()

        # Initialize the ConstructRetina instance with the mocks
        self.construct_retina = ConstructRetina(
            context=self.context_mock,
            data_io=self.data_io_mock,
            viz=self.viz_mock,
            fit=self.fit_mock,
            retina_vae=self.retina_vae_mock,
            retina_math=self.retina_math_mock,
            project_data=self.project_data_mock,
            get_xy_from_npz=self.get_xy_from_npz_mock,
        )

    def test_build_exists_force_build(self):
        # Test when force_retina_build is True
        retina_parameters = {"force_retina_build": True}
        with patch("builtins.print") as mock_print:
            result = self.construct_retina._build_exists(retina_parameters)
            assert result == False
            mock_print.assert_called_with("Forcing the build of the retina.")

    def test_build_exists_hash_exists(self):
        # Test when hash exists (all files are found)
        retina_parameters = {
            "force_retina_build": False,
            "retina_parameters_hash": "abc123",
            "spatial_rfs_file": "spatial_rfs_abc123.npz",
            "ret_file": "retina_abc123.npz",
            "mosaic_file": "mosaic_abc123.csv",
        }

        self.data_io_mock.parse_path.side_effect = lambda x: None  # No exception

        with patch("builtins.print") as mock_print:
            result = self.construct_retina._build_exists(retina_parameters)
            assert result == True
            mock_print.assert_called_with(
                "Hash exists. Continuing without building the retina."
            )

    def test_build_exists_hash_not_exists(self):
        # Test when hash does not exist (FileNotFoundError is raised)
        retina_parameters = {
            "force_retina_build": False,
            "retina_parameters_hash": "abc123",
            "spatial_rfs_file": "spatial_rfs_abc123.npz",
            "ret_file": "retina_abc123.npz",
            "mosaic_file": "mosaic_abc123.csv",
        }

        def parse_path_side_effect(path):
            raise FileNotFoundError

        self.data_io_mock.parse_path.side_effect = parse_path_side_effect

        with patch("builtins.print") as mock_print:
            result = self.construct_retina._build_exists(retina_parameters)
            assert result == False
            mock_print.assert_called_with("Hash does not exist. Building the retina.")

    def test_get_density_from(self):
        # Mock data_io.get_data to return density data
        density_data = {"Xdata": np.array([1, 2, 3]), "Ydata": np.array([10, 20, 30])}
        self.data_io_mock.get_data.return_value = density_data

        # Call the method
        cell_eccentricity, cell_density = self.construct_retina._get_density_from(
            ["dummy_path"]
        )

        # Verify the results
        np.testing.assert_array_equal(cell_eccentricity, [1, 2, 3])
        np.testing.assert_array_equal(
            cell_density, [10000, 20000, 30000]
        )  # Scaled by 1e3

    def test_get_literature_data(self):
        # Mock methods and data
        self.construct_retina._get_density_from = MagicMock(
            return_value=(np.array([1, 2]), np.array([100, 200]))
        )
        self.data_io_mock.get_data.return_value = pd.DataFrame({"A": [1, 2, 3]})
        self.get_xy_from_npz_mock.return_value = (
            np.array([1, 2, 3]),
            np.array([10, 20, 30]),
        )
        self.context_mock.retina_parameters = {
            "cone_general_parameters": {"cone_noise_wc": 0.5},
            "response_type": "some_response",
        }

        # Call the method
        literature = self.construct_retina._get_literature_data()

        # Verify that literature data contains expected keys
        expected_keys = [
            "gc_eccentricity",
            "gc_density",
            "cone_eccentricity",
            "cone_density",
            "bipolar_df",
            "dendr_diam1",
            "dendr_diam2",
            "dendr_diam3",
            "dendr_diam_units",
            "temporal_parameters_BK",
            "cone_frequency_data",
            "cone_power_data",
            "cone_noise_wc",
            "noise_frequency_data",
            "noise_power_data",
            "g_sur_values",
            "target_RI_values",
        ]
        for key in expected_keys:
            assert key in literature

    def test_append_apricot_metadata_parameters(self):
        data = {"some_data": 123}
        updated_data = self.construct_retina._append_apricot_metadata_parameters(data)
        assert "apricot_metadata_parameters" in updated_data
        assert updated_data["apricot_metadata_parameters"] == {
            "metadata_key": "metadata_value"
        }

    def test_builder_client_build_exists(self):
        # Test when build exists
        self.construct_retina._build_exists = MagicMock(return_value=True)
        self.construct_retina._get_literature_data = MagicMock()
        self.construct_retina._append_apricot_metadata_parameters = MagicMock()
        self.construct_retina.save_retina = MagicMock()

        self.construct_retina.build_retina_client()

        # Verify that methods after _build_exists are not called
        self.construct_retina._get_literature_data.assert_not_called()
        self.construct_retina._append_apricot_metadata_parameters.assert_not_called()
        self.construct_retina.save_retina.assert_not_called()

    def test_builder_client_build_does_not_exist(self):
        # Test when build does not exist
        self.construct_retina._build_exists = MagicMock(return_value=False)
        self.construct_retina._get_literature_data = MagicMock(
            return_value={"data": "value"}
        )
        self.construct_retina._append_apricot_metadata_parameters = MagicMock(
            return_value={"data": "value", "apricot_metadata_parameters": {}}
        )
        self.construct_retina.save_retina = MagicMock()

        # Mock Retina and builder
        with patch(
            "macaqueretina.retina.construct_retina_module.Retina"
        ) as RetinaMock, patch(
            "macaqueretina.retina.construct_retina_module.ConcreteRetinaBuilder"
        ) as BuilderMock, patch(
            "macaqueretina.retina.construct_retina_module.RetinaBuildDirector"
        ) as DirectorMock:

            retina_instance = MagicMock()
            gc_instance = MagicMock()
            RetinaMock.return_value = retina_instance

            builder_instance = MagicMock()
            BuilderMock.return_value = builder_instance
            # Set builder_instance.project_data to a known value
            builder_instance.project_data = {"key": "value"}

            director_instance = MagicMock()
            DirectorMock.return_value = director_instance
            director_instance.get_retina.return_value = (retina_instance, gc_instance)

            self.construct_retina.build_retina_client()

            # Verify that methods are called
            self.construct_retina._get_literature_data.assert_called_once()
            self.construct_retina._append_apricot_metadata_parameters.assert_called_once()
            self.construct_retina.save_retina.assert_called_once_with(
                retina_instance, gc_instance
            )

            # Verify that project_data is updated
            assert self.construct_retina.project_data.construct_retina == {
                "key": "value"
            }

    def test_save_retina(self):
        # Mock retina and ganglion cell
        ret_mock = MagicMock()
        gc_mock = MagicMock()
        gc_mock.img = np.array([1, 2, 3])
        gc_mock.img_mask = np.array([True, False, True])
        gc_mock.X_grid_mm = np.array([0.1, 0.2, 0.3])
        gc_mock.Y_grid_mm = np.array([0.4, 0.5, 0.6])
        gc_mock.um_per_pix = 0.01
        gc_mock.pix_per_side = 64
        gc_mock.df = pd.DataFrame({"A": [1, 2, 3]})

        gc_mock.df.to_csv = MagicMock()

        # Mock attributes in retina
        ret_mock_attributes = {
            "cone_optimized_pos_mm": np.array([0.1]),
            "cone_optimized_pos_pol": np.array([0.2]),
            "cones_to_gcs_weights": np.array([0.3]),
            "cone_noise_parameters": np.array([0.4]),
            "noise_frequency_data": np.array([0.5]),
            "noise_power_data": np.array([0.6]),
            "cone_frequency_data": np.array([0.7]),
            "cone_power_data": np.array([0.8]),
            "cone_noise_power_fit": np.array([0.9]),
            "cones_to_bipolars_center_weights": np.array([1.0]),
            "cones_to_bipolars_surround_weights": np.array([1.1]),
            "bipolar_to_gcs_weights": np.array([1.2]),
            "bipolar_optimized_pos_mm": np.array([1.3]),
            "bipolar_nonlinearity_parameters": np.array([1.4]),
            "bipolar_nonlinearity_fit": np.array([1.5]),
            "g_sur_scaled": np.array([1.6]),
            "target_RI_values": np.array([1.7]),
        }
        for attr, value in ret_mock_attributes.items():
            setattr(ret_mock, attr, value)

        # Mock data_io methods
        self.data_io_mock.save_np_dict_to_npz = MagicMock()
        self.context_mock.output_folder.joinpath.return_value = "path/to/mosaic.csv"

        # Call the method
        with patch("builtins.print") as mock_print:
            self.construct_retina.save_retina(ret_mock, gc_mock)

            # Verify that data_io.save_np_dict_to_npz is called twice
            assert self.data_io_mock.save_np_dict_to_npz.call_count == 2

            # Verify that gc.df.to_csv is called with the correct filepath
            gc_mock.df.to_csv.assert_called_once_with("path/to/mosaic.csv")

            # Verify print statements
            mock_print.assert_any_call("\nSaving gc and ret data...")
            mock_print.assert_any_call("Saving model mosaic to path/to/mosaic.csv")
