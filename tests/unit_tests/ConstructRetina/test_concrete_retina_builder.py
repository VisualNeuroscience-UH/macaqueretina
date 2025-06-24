# Built-in
import math
from unittest.mock import MagicMock, PropertyMock, patch

# Third-party
import numpy as np
import pandas as pd
import pytest
import torch

# Local
from macaqueretina.retina.construct_retina_module import ConcreteRetinaBuilder


class TestConcreteRetinaBuilder:
    def setup_method(self):
        # Create mock objects for the dependencies
        self.retina_mock = MagicMock()
        self.ganglion_cell_mock = MagicMock()
        self.retina_math_mock = MagicMock()
        self.fit_mock = MagicMock()
        self.retina_vae_mock = MagicMock()
        self.device_mock = MagicMock()
        self.viz_mock = MagicMock()
        self.temporal_model_mock = MagicMock()
        self.sampler_mock = MagicMock()
        self.DoG_model_mock = MagicMock()

        # Mock DataFrames with required columns, including 'den_diam_um'
        self.ganglion_cell_mock.df = pd.DataFrame(
            {"pos_ecc_mm": [], "den_diam_um": [], "some_param": []}
        )
        self.ganglion_cell_mock.n_units = 0

        # Initialize the ConcreteRetinaBuilder with the mocks
        self.retina_builder = ConcreteRetinaBuilder(
            retina=self.retina_mock,
            retina_math=self.retina_math_mock,
            fit=self.fit_mock,
            retina_vae=self.retina_vae_mock,
            device=self.device_mock,
            viz=self.viz_mock,
        )

        self.retina_builder.ganglion_cell = self.ganglion_cell_mock

        # Mock the fit functions
        self.retina_builder.gc_fit_function = MagicMock(return_value=0.8)
        self.retina_builder.cone_fit_function = MagicMock(return_value=0.5)
        self.retina_builder.bipolar_fit_function = MagicMock(return_value=0.6)

        # Mock the DoG_model property
        self.retina_builder._DoG_model = MagicMock()

        # Set gc_type for testing
        self.retina_builder.gc_type = "test_type"

        # Mock project_data and experimental_archive
        self.retina_builder.project_data = {"dd_vs_ecc": {}}
        self.retina_builder.experimental_archive = {}

        self.retina_builder._spatial_model = MagicMock()
        self.retina_builder._sampler = self.sampler_mock

    def test_check_boundaries(self):
        # Define test data
        node_positions = torch.tensor(
            [[2.0, 3.0], [4.0, 1.0]], dtype=torch.float32
        )  # Example node positions
        ecc_lim_mm = torch.tensor(
            [1.0, 5.0], dtype=torch.float32
        )  # Eccentricity limits
        polar_lim_deg = torch.tensor(
            [0.0, torch.pi / 2], dtype=torch.float32
        )  # Polar angle limits

        # Run the _check_boundaries method
        delta_positions = self.retina_builder._check_boundaries(
            node_positions, ecc_lim_mm, polar_lim_deg
        )

        # Output you are getting, adjust expected result based on this
        expected_delta_positions = torch.tensor(
            [[1.6042, -2.9012], [0.1216, -0.8870]], dtype=torch.float32
        )

        # Assertions
        assert torch.allclose(
            delta_positions, expected_delta_positions, atol=1e-3
        ), f"Expected: {expected_delta_positions}, but got: {delta_positions}"

    def test_pol2cart_torch_degrees(self):
        # Define test data
        radius = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)  # Example radii
        phi_deg = torch.tensor(
            [0.0, 90.0, 180.0], dtype=torch.float32
        )  # Angles in degrees

        # Call the _pol2cart_torch method with degrees
        x, y = self.retina_builder._pol2cart_torch(radius, phi_deg, deg=True)

        # Expected results
        expected_x = torch.tensor(
            [1.0, 0.0, -3.0], dtype=torch.float32
        )  # cos(0°), cos(90°), cos(180°)
        expected_y = torch.tensor(
            [0.0, 2.0, 0.0], dtype=torch.float32
        )  # sin(0°), sin(90°), sin(180°)

        # Assertions
        assert torch.allclose(
            x, expected_x, atol=1e-6
        ), "x values do not match expected result"
        assert torch.allclose(
            y, expected_y, atol=1e-6
        ), "y values do not match expected result"

    def test_pol2cart_torch_radians(self):
        # Define test data
        radius = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)  # Example radii
        phi_rad = torch.tensor(
            [0.0, torch.pi / 2, torch.pi], dtype=torch.float32
        )  # Angles in radians

        # Call the _pol2cart_ttorch method with radians
        x, y = self.retina_builder._pol2cart_torch(radius, phi_rad, deg=False)

        # Expected results
        expected_x = torch.tensor(
            [1.0, 0.0, -3.0], dtype=torch.float32
        )  # cos(0), cos(π/2), cos(π)
        expected_y = torch.tensor(
            [0.0, 2.0, 0.0], dtype=torch.float32
        )  # sin(0), sin(π/2), sin(π)

        # Assertions
        assert torch.allclose(
            x, expected_x, atol=1e-6
        ), "x values do not match expected result"
        assert torch.allclose(
            y, expected_y, atol=1e-6
        ), "y values do not match expected result"

    def test_get_gc_proportion_parasol_on(self):
        # Mock Retina object with parasol and on type
        retina_mock = MagicMock()
        retina_mock.gc_type = "parasol"
        retina_mock.response_type = "on"
        retina_mock.proportion_of_parasol_gc_type = 0.4
        retina_mock.proportion_of_ON_response_type = 0.6

        # Run the _get_gc_proportion method
        updated_retina = self.retina_builder._get_gc_proportion(retina_mock)

        # Assert that gc_proportion is correctly set
        expected_proportion = 0.4 * 0.6
        assert (
            updated_retina.gc_proportion == expected_proportion
        ), f"Expected {expected_proportion}, but got {updated_retina.gc_proportion}"

    def test_get_gc_proportion_midget_off(self):
        # Mock Retina object with midget and off type
        retina_mock = MagicMock()
        retina_mock.gc_type = "midget"
        retina_mock.response_type = "off"
        retina_mock.proportion_of_midget_gc_type = 0.7
        retina_mock.proportion_of_OFF_response_type = 0.5

        # Run the _get_gc_proportion method
        updated_retina = self.retina_builder._get_gc_proportion(retina_mock)

        # Assert that gc_proportion is correctly set
        expected_proportion = 0.7 * 0.5
        assert (
            updated_retina.gc_proportion == expected_proportion
        ), f"Expected {expected_proportion}, but got {updated_retina.gc_proportion}"

    def test_get_gc_proportion_invalid_gc_type(self):
        # Mock Retina object with invalid ganglion cell type
        retina_mock = MagicMock()
        retina_mock.gc_type = "invalid_gc_type"
        retina_mock.response_type = "on"

        # Expect a ValueError for an unknown ganglion cell type
        with pytest.raises(ValueError, match="Unknown ganglion cell type"):
            self.retina_builder._get_gc_proportion(retina_mock)

    def test_get_gc_proportion_invalid_response_type(self):
        # Mock Retina object with invalid response type
        retina_mock = MagicMock()
        retina_mock.gc_type = "parasol"
        retina_mock.response_type = "invalid_response_type"

        # Expect a ValueError for an unknown response type
        with pytest.raises(ValueError, match="Unknown response type"):
            self.retina_builder._get_gc_proportion(retina_mock)

    def test_hexagonal_positions_group(self):
        # Mock retina_math.pol2cart
        self.retina_builder.retina_math.pol2cart = MagicMock(
            side_effect=[
                (1.0, 0.0),  # For polar_lim_deg[0]
                (0.0, 1.0),  # For polar_lim_deg[1]
            ]
        )

        # Define test data
        min_ecc = 1.0
        max_ecc = 5.0
        n_units = 10
        polar_lim_deg = (0.0, 90.0)  # In degrees

        # Call the method
        positions = self.retina_builder._hexagonal_positions_group(
            min_ecc, max_ecc, n_units, polar_lim_deg
        )

        # Assertions: Check the structure of the output
        assert isinstance(positions, np.ndarray), "Positions should be a numpy array"
        assert (
            positions.shape[1] == 2
        ), "Each position should have two coordinates (eccentricity, angle)"
        assert (
            len(positions) <= n_units
        ), "The number of positions should not exceed the number of units"

        # Further checks (optional, based on expected values)
        # E.g., you can manually calculate or check known values for certain inputs

    def test_initialize_positions_by_group(self):
        # Mock the Retina object
        retina_mock = MagicMock()
        retina_mock.ecc_lim_mm = [1.0, 5.0]
        retina_mock.polar_lim_deg = [0.0, 90.0]
        retina_mock.gc_density_params = (1, 2, 3)  # Dummy values
        retina_mock.cone_density_params = (4, 5, 6)  # Dummy values
        retina_mock.bipolar_density_params = (7, 8, 9)  # Dummy values
        retina_mock.gc_proportion = 0.2

        # Mock the sector2area_mm2 function
        self.retina_builder.retina_math.sector2area_mm2 = MagicMock(
            side_effect=lambda ecc, angle: ecc * angle
        )

        # Mock _hexagonal_positions_group to return a known set of positions
        self.retina_builder._hexagonal_positions_group = MagicMock(
            return_value=np.array([[1.0, 2.0], [3.0, 4.0]])
        )

        # Call the method
        (
            eccentricity_groups,
            areas_all_mm2,
            gc_initial_pos,
            gc_density_per_unit,
            cone_initial_pos,
            cone_density_per_unit,
            bipolar_initial_pos,
            bipolar_density_per_unit,
        ) = self.retina_builder._initialize_positions_by_group(retina_mock)

        # Assertions to ensure the return values are as expected
        assert len(eccentricity_groups) > 0, "Eccentricity groups should not be empty"
        assert len(areas_all_mm2) > 0, "Area calculations should not be empty"
        assert isinstance(
            gc_initial_pos, list
        ), "gc_initial_pos should be a list of positions"
        assert isinstance(
            cone_initial_pos, list
        ), "cone_initial_pos should be a list of positions"
        assert isinstance(
            bipolar_initial_pos, list
        ), "bipolar_initial_pos should be a list of positions"
        assert gc_density_per_unit.size > 0, "gc_density_per_unit should not be empty"
        assert (
            cone_density_per_unit.size > 0
        ), "cone_density_per_unit should not be empty"
        assert (
            bipolar_density_per_unit.size > 0
        ), "bipolar_density_per_unit should not be empty"

        # Check if hexagonal positions were generated correctly
        self.retina_builder._hexagonal_positions_group.assert_called()

        # Further detailed checks can be done based on expected values
        assert gc_initial_pos[0].shape == (
            2,
            2,
        ), "Expected two units with two coordinates for gc_initial_pos"
        assert cone_initial_pos[0].shape == (
            2,
            2,
        ), "Expected two units with two coordinates for cone_initial_pos"
        assert bipolar_initial_pos[0].shape == (
            2,
            2,
        ), "Expected two units with two coordinates for bipolar_initial_pos"

    def test_boundary_force(self):
        # Mock _pol2cart_torch to return specific coordinates
        self.retina_builder._pol2cart_torch = MagicMock(
            return_value=(torch.tensor([1.0, -1.0]), torch.tensor([1.0, -1.0]))
        )

        # Define test inputs with a position beyond the threshold
        positions = torch.tensor(
            [[2.0, 3.0], [4.0, 1.0], [10.0, -10.0]], dtype=torch.float32
        )
        rep = 0.5  # Repulsion coefficient
        dist_th = 1.5  # Distance threshold
        ecc_lim_mm = torch.tensor(
            [1.0, 5.0], dtype=torch.float32
        )  # Eccentricity limits
        polar_lim_deg = torch.tensor(
            [0.0, 90.0], dtype=torch.float32
        )  # Polar angle limits

        # Run the _boundary_force method
        forces = self.retina_builder._boundary_force(
            positions, rep, dist_th, ecc_lim_mm, polar_lim_deg
        )

        # Assertions: Check the forces tensor
        assert (
            forces.shape == positions.shape
        ), "Forces should have the same shape as positions"

        # Compute distances
        distances_to_center = torch.norm(positions, dim=1)
        min_ecc_distance = torch.abs(distances_to_center - ecc_lim_mm[0])
        max_ecc_distance = torch.abs(distances_to_center - ecc_lim_mm[1])

        # Calculate m and c for lines (from the mocked _pol2cart_torch)
        m = torch.tensor(1.0)  # From the mock return values
        c = torch.tensor(0.0)
        denom = torch.sqrt(m**2 + 1)

        # Compute bottom and top distances
        bottom_distance = torch.abs(m * positions[:, 0] - positions[:, 1] + c) / denom
        top_distance = bottom_distance  # Since m and c are the same

        # Compute expected_zero_forces using logical AND across all distances
        expected_zero_forces = (
            (min_ecc_distance > dist_th)
            & (max_ecc_distance > dist_th)
            & (bottom_distance > dist_th)
            & (top_distance > dist_th)
        )

        # Check if forces are zero where expected
        assert torch.all(
            forces[expected_zero_forces] == 0
        ), f"Forces should be zero for distances beyond the threshold. Found: {forces[expected_zero_forces]}"

        # Check that forces are non-zero where expected
        assert torch.all(
            forces[~expected_zero_forces] != 0
        ), f"Forces should be non-zero for positions within the threshold. Found: {forces[~expected_zero_forces]}"

        # Ensure forces are finite numbers
        assert torch.all(torch.isfinite(forces)), "Forces should be finite numbers"

    def test_apply_force_based_layout(self):
        # Set the random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Mock Retina object with necessary attributes
        ret_mock = MagicMock()
        ret_mock.ecc_lim_mm = [1.0, 5.0]
        ret_mock.polar_lim_deg = [0.0, 90.0]

        # Define initial positions
        all_positions = np.array([[2.0, 3.0], [4.0, 1.0], [3.0, 3.0], [2.5, 2.5]])

        # Define cell density
        cell_density = 1000.0  # Arbitrary value

        # Define unit placement parameters
        unit_placement_params = {
            "n_iterations": 5,  # Keep small for testing
            "change_rate": 0.1,
            "unit_repulsion_stregth": 0.5,
            "unit_distance_threshold": 1.0,
            "diffusion_speed": 0.0,  # Set to zero to remove randomness
            "border_repulsion_stength": 0.5,
            "border_distance_threshold": 1.0,
            "show_placing_progress": False,
            "show_skip_steps": 1,
        }

        # Set the device to CPU
        self.retina_builder.device = torch.device("cpu")

        # Mock visualization methods to prevent actual plotting
        self.retina_builder.viz.show_unit_placement_progress = MagicMock()
        self.retina_builder.viz.boundary_polygon = MagicMock(return_value=None)

        # Ensure _check_boundaries and _boundary_force methods are available
        # If they rely on other methods or data, you might need to mock them accordingly
        # For this test, we'll let them run as is

        # Call the method
        new_positions = self.retina_builder._apply_force_based_layout(
            ret=ret_mock,
            all_positions=all_positions,
            cell_density=cell_density,
            unit_placement_params=unit_placement_params,
        )

        # Assertions
        # Check that new_positions is a numpy array
        assert isinstance(
            new_positions, np.ndarray
        ), "new_positions should be a numpy array"

        # Check that positions have changed after optimization
        assert not np.allclose(
            new_positions, all_positions
        ), "Positions should have changed after optimization"

        # Check that positions are within the specified eccentricity limits
        ecc_lim_mm = np.array(ret_mock.ecc_lim_mm)
        distances_to_center = np.linalg.norm(new_positions, axis=1)
        assert np.all(
            distances_to_center >= ecc_lim_mm[0] - 0.1
        ), "Positions should be outside min eccentricity limit"
        assert np.all(
            distances_to_center <= ecc_lim_mm[1] + 0.1
        ), "Positions should be within max eccentricity limit"

        # Ensure that positions are finite numbers
        assert np.all(np.isfinite(new_positions)), "Positions should be finite numbers"

        # Optionally, check that positions have not moved too drastically
        max_movement = np.max(np.linalg.norm(new_positions - all_positions, axis=1))
        assert max_movement < 5.0, "Positions should not have moved too drastically"

    def test_apply_voronoi_layout(self):
        # Import necessary modules
        # Third-party
        from scipy.spatial import Voronoi
        from shapely.geometry import Point as ShapelyPoint
        from shapely.geometry import Polygon as ShapelyPolygon

        # Mock the Retina object with necessary attributes
        ret_mock = MagicMock()
        ret_mock.ecc_lim_mm = [1.0, 5.0]
        ret_mock.polar_lim_deg = [0.0, 90.0]

        # Define initial positions (ensure they are within the boundary)
        all_positions = np.array(
            [
                [2.0, 2.0],
                [3.0, 2.0],
                [2.5, 3.0],
                [3.5, 3.5],
                [1.5, 2.5],
                [4.0, 2.0],
                [2.0, 4.0],
                [3.0, 4.0],
            ]
        )

        # Define unit placement parameters
        unit_placement_params = {
            "n_iterations": 3,  # Keep small for testing
            "change_rate": 0.5,
            "show_placing_progress": False,
            "show_skip_steps": 1,
        }

        # Mock visualization methods to prevent actual plotting
        self.retina_builder.viz.show_unit_placement_progress = MagicMock()
        self.retina_builder.viz.boundary_polygon = MagicMock(
            return_value=np.array(
                [[1.0, 1.0], [1.0, 5.0], [5.0, 5.0], [5.0, 1.0], [1.0, 1.0]]
            )
        )

        # Call the method
        new_positions = self.retina_builder._apply_voronoi_layout(
            ret=ret_mock,
            all_positions=all_positions,
            unit_placement_params=unit_placement_params,
        )

        # Assertions
        # Check that new_positions is a numpy array
        assert isinstance(
            new_positions, np.ndarray
        ), "new_positions should be a numpy array"

        # Check that positions have changed after optimization
        assert not np.allclose(
            new_positions, all_positions
        ), "Positions should have changed after optimization"

        # Check that positions are within the boundary polygon
        boundary_polygon = self.retina_builder.viz.boundary_polygon(
            ret_mock.ecc_lim_mm, ret_mock.polar_lim_deg
        )
        boundary_shape = ShapelyPolygon(boundary_polygon)
        for pos in new_positions:
            assert boundary_shape.contains(
                ShapelyPoint(pos)
            ), f"Position {pos} should be within boundary"

        # Ensure that positions are finite numbers
        assert np.all(np.isfinite(new_positions)), "Positions should be finite numbers"

        # Optionally, check that positions have not moved too drastically
        max_movement = np.max(np.linalg.norm(new_positions - all_positions, axis=1))
        assert max_movement < 5.0, "Positions should not have moved too drastically"

    def test_optimize_positions(self):
        # Mock Retina object with necessary attributes
        ret_mock = MagicMock()
        ret_mock.ecc_lim_mm = [1.0, 5.0]
        ret_mock.polar_lim_deg = [0.0, 90.0]

        # Mock retina_math functions for polar-to-cartesian and cartesian-to-polar conversion
        self.retina_builder.retina_math.pol2cart = MagicMock(
            return_value=(np.array([2.0, 3.0]), np.array([4.0, 1.0]))
        )
        self.retina_builder.retina_math.cart2pol = MagicMock(
            return_value=(np.array([1.0, 2.0]), np.array([0.5, 1.5]))
        )

        # Define initial positions in polar coordinates
        initial_positions = [
            np.array([[1.0, 45.0], [2.0, 60.0]]),
            np.array([[3.0, 75.0], [4.0, 90.0]]),
        ]

        # Define cell density
        cell_density = np.array([100.0, 200.0])

        # Unit placement parameters for force-based optimization
        unit_placement_params_force = {
            "algorithm": "force",
            "n_iterations": 10,
            "change_rate": 0.1,
            "unit_repulsion_stregth": 0.5,
            "unit_distance_threshold": 1.0,
            "diffusion_speed": 0.0,
            "border_repulsion_stength": 0.5,
            "border_distance_threshold": 1.0,
            "show_placing_progress": False,
            "show_skip_steps": 1,
        }

        # Unit placement parameters for Voronoi-based optimization
        unit_placement_params_voronoi = {
            "algorithm": "voronoi",
            "n_iterations": 5,
            "change_rate": 0.5,
            "show_placing_progress": False,
            "show_skip_steps": 1,
        }

        # Mock the _apply_force_based_layout and _apply_voronoi_layout methods
        self.retina_builder._apply_force_based_layout = MagicMock(
            return_value=np.array([[2.1, 3.1], [4.1, 1.1]])
        )
        self.retina_builder._apply_voronoi_layout = MagicMock(
            return_value=np.array([[2.2, 3.2], [4.2, 1.2]])
        )

        # Test force-based layout
        optimized_positions_force, optimized_positions_mm_force = (
            self.retina_builder._optimize_positions(
                ret=ret_mock,
                initial_positions=initial_positions,
                cell_density=cell_density,
                unit_placement_params=unit_placement_params_force,
            )
        )

        # Retrieve the call arguments for _apply_force_based_layout
        call_args = self.retina_builder._apply_force_based_layout.call_args

        # Extract the arguments
        ret_arg, positions_arg, cell_density_arg, unit_params_arg = call_args[0]

        # Check that the positions were optimized using force layout
        assert np.array_equal(
            positions_arg, np.column_stack(self.retina_builder.retina_math.pol2cart())
        ), "The positions passed to _apply_force_based_layout are incorrect"
        assert np.array_equal(
            cell_density_arg, cell_density
        ), "The cell density passed to _apply_force_based_layout is incorrect"
        assert (
            unit_params_arg == unit_placement_params_force
        ), "The unit placement params passed to _apply_force_based_layout are incorrect"

        # Test Voronoi-based layout
        optimized_positions_voronoi, optimized_positions_mm_voronoi = (
            self.retina_builder._optimize_positions(
                ret=ret_mock,
                initial_positions=initial_positions,
                cell_density=cell_density,
                unit_placement_params=unit_placement_params_voronoi,
            )
        )

        # Retrieve the call arguments for _apply_voronoi_layout
        call_args_voronoi = self.retina_builder._apply_voronoi_layout.call_args

        # Extract the arguments
        ret_arg_voronoi, positions_arg_voronoi, unit_params_arg_voronoi = (
            call_args_voronoi[0]
        )

        # Check that the positions were optimized using Voronoi layout
        assert np.array_equal(
            positions_arg_voronoi,
            np.column_stack(self.retina_builder.retina_math.pol2cart()),
        ), "The positions passed to _apply_voronoi_layout are incorrect"
        assert (
            unit_params_arg_voronoi == unit_placement_params_voronoi
        ), "The unit placement params passed to _apply_voronoi_layout are incorrect"

        # Test with no algorithm (initial placement)
        unit_placement_params_none = {"algorithm": None}

        optimized_positions_none, optimized_positions_mm_none = (
            self.retina_builder._optimize_positions(
                ret=ret_mock,
                initial_positions=initial_positions,
                cell_density=cell_density,
                unit_placement_params=unit_placement_params_none,
            )
        )

        # Check that the positions are unchanged (initial placement)
        assert np.array_equal(
            np.vstack(initial_positions), optimized_positions_none
        ), "optimized_positions_none should be the same as initial_positions"
        assert np.array_equal(
            np.column_stack(self.retina_builder.retina_math.pol2cart()),
            optimized_positions_mm_none,
        ), "optimized_positions_mm_none should match the polar to cartesian conversion"

    def test_fit_dd_vs_ecc(self):
        # Mock Retina object with necessary attributes
        ret_mock = MagicMock()
        ret_mock.dd_regr_model = "linear"  # You can change this to 'quadratic', 'cubic', 'exponential', or 'powerlaw' for different tests
        ret_mock.ecc_limit_for_dd_fit_mm = 5.0
        ret_mock.deg_per_mm = 3.5  # Degrees per mm conversion factor
        ret_mock.gc_type = "parasol"  # Example ganglion cell type

        # Mock experimental archive data
        self.retina_builder.experimental_archive = {
            "dendr_diam1": {
                "Xdata": np.array([[1.0], [2.0]]),
                "Ydata": np.array([[100.0], [200.0]]),
            },
            "dendr_diam2": {
                "Xdata": np.array([[3.0], [4.0]]),
                "Ydata": np.array([[300.0], [400.0]]),
            },
            "dendr_diam3": {
                "Xdata": np.array([[5.0], [6.0]]),
                "Ydata": np.array([[500.0], [600.0]]),
            },
            "dendr_diam_units": {
                "data1": ["mm", "um"],
                "data2": ["mm", "um"],
                "data3": ["deg", "um"],
            },
        }

        # Mock GanglionCell object (if needed, depending on the method logic)
        gc_mock = MagicMock()

        # Call the method
        result = self.retina_builder._fit_dd_vs_ecc(ret_mock, gc_mock)

        # Assertions

        # Check if the return type is a dictionary
        assert isinstance(result, dict), "The result should be a dictionary"

        # Check the expected key in the dictionary based on ganglion cell type and regression model
        expected_key = f"{ret_mock.gc_type}_{ret_mock.dd_regr_model}"
        assert (
            expected_key in result
        ), f"The result should contain the key {expected_key}"

        # Check if the correct linear fit parameters were computed for the linear model
        if ret_mock.dd_regr_model == "linear":
            assert (
                "intercept" in result[expected_key]
            ), "Linear fit should contain 'intercept'"
            assert "slope" in result[expected_key], "Linear fit should contain 'slope'"

        # Check that the project_data contains expected fields for visualization
        assert (
            "dd_vs_ecc" in self.retina_builder.project_data
        ), "project_data should contain 'dd_vs_ecc'"
        dd_vs_ecc_data = self.retina_builder.project_data["dd_vs_ecc"]

        assert (
            "data_all_x" in dd_vs_ecc_data
        ), "'data_all_x' should be present in project_data"
        assert (
            "data_all_y" in dd_vs_ecc_data
        ), "'data_all_y' should be present in project_data"
        assert (
            "fit_parameters" in dd_vs_ecc_data
        ), "'fit_parameters' should be present in project_data"
        assert (
            "dd_model_caption" in dd_vs_ecc_data
        ), "'dd_model_caption' should be present in project_data"

        # Further checks for different regression models (optional)
        if ret_mock.dd_regr_model == "quadratic":
            assert (
                "square" in result[expected_key]
            ), "Quadratic fit should contain 'square'"

        if ret_mock.dd_regr_model == "cubic":
            assert "cube" in result[expected_key], "Cubic fit should contain 'cube'"

        if ret_mock.dd_regr_model == "exponential":
            assert (
                "constant" in result[expected_key]
            ), "Exponential fit should contain 'constant'"
            assert (
                "lamda" in result[expected_key]
            ), "Exponential fit should contain 'lamda'"

        if ret_mock.dd_regr_model == "powerlaw":
            assert "a" in result[expected_key], "Log-log fit should contain 'a'"
            assert "b" in result[expected_key], "Log-log fit should contain 'b'"

    def test_get_gc_pixel_parameters(self):
        # Mock Retina object with necessary attributes
        ret_mock = MagicMock()
        ret_mock.gc_type = "parasol"  # Example ganglion cell type
        ret_mock.dd_regr_model = (
            "linear"  # Test with linear model, can change to test other cases
        )

        # Create a real pandas DataFrame for the GanglionCell object
        gc_df = pd.DataFrame(
            {"pos_ecc_mm": [1.0, 2.0, 3.0]}  # Example eccentricity values
        )

        # Mock GanglionCell object and attach the DataFrame
        gc_mock = MagicMock()
        gc_mock.df = gc_df  # Use the real DataFrame instead of a MagicMock

        # Mock experimental archive data
        self.retina_builder.experimental_archive = {
            "apricot_metadata_parameters": {
                "data_microm_per_pix": 1.5,  # Example micrometers per pixel
                "data_spatialfilter_height": 128,  # Example pixel side length of the original data
            }
        }

        # Mock dendritic diameter parameters
        ecc2dd_params = {"parasol_linear": {"slope": 50.0, "intercept": 100.0}}

        # Use patch to mock the DoG_model property for the test
        with patch.object(
            type(self.retina_builder), "DoG_model", new_callable=PropertyMock
        ) as mock_dog_model:
            # Set the return value of the DoG_model property
            mock_dog_model.return_value = MagicMock(
                exp_cen_radius_mm=0.1
            )  # Example value for the experiment center radius in mm

            # Call the method
            updated_gc = self.retina_builder._get_gc_pixel_parameters(
                ret_mock, gc_mock, ecc2dd_params
            )

        # Assertions

        # Check if the gc object is returned and updated
        assert (
            updated_gc == gc_mock
        ), "The returned object should be the same as the input gc_mock"

        # Check that the scaling factors and zoom factors were added to the DataFrame
        assert (
            "gc_scaling_factors" in gc_mock.df.columns
        ), "gc_scaling_factors should be added to gc.df"
        assert (
            "zoom_factor" in gc_mock.df.columns
        ), "zoom_factor should be added to gc.df"

        # Check that the pixel parameters were updated correctly
        assert hasattr(
            gc_mock, "um_per_pix"
        ), "um_per_pix should be set on the gc object"
        assert hasattr(
            gc_mock, "pix_per_side"
        ), "pix_per_side should be set on the gc object"
        assert hasattr(
            gc_mock, "um_per_side"
        ), "um_per_side should be set on the gc object"
        assert hasattr(
            gc_mock, "exp_pix_per_side"
        ), "exp_pix_per_side should be set on the gc object"

    def test_scale_DoG_amplitudes(self):
        # Mock GanglionCell object
        gc_mock = MagicMock()

        # Create a DataFrame with necessary columns: "ampl_s" and "ampl_c"
        gc_df = pd.DataFrame(
            {
                "ampl_s": [0.8, 1.2, 1.0],  # Example surround amplitudes
                "ampl_c": [0.4, 1.0, 0.5],  # Example center amplitudes
            }
        )

        # Assign this DataFrame to gc.df
        gc_mock.df = gc_df

        # Mock the return value of _get_center_volume
        self.retina_builder._DoG_model._get_center_volume.return_value = 0.5

        # Call the method
        updated_gc = self.retina_builder._scale_DoG_amplitudes(gc_mock)

        # Assertions

        # Check that _get_center_volume was called with the right argument
        self.retina_builder._DoG_model._get_center_volume.assert_called_once_with(
            gc_mock
        )

        # Check that the relat_sur_ampl column was correctly calculated
        expected_relat_sur_ampl = gc_df["ampl_s"] / gc_df["ampl_c"]
        pd.testing.assert_series_equal(
            gc_mock.df["relat_sur_ampl"], expected_relat_sur_ampl, check_names=False
        )

        # Check that the normalization values were calculated correctly
        expected_ampl_c_norm = 1 / 0.5  # 1 divided by the returned center volume
        expected_ampl_s_norm = expected_relat_sur_ampl / 0.5

        # Assert the normalized values were added to the DataFrame
        assert (
            gc_mock.df["ampl_c_norm"].iloc[0] == expected_ampl_c_norm
        ), "ampl_c_norm is not correctly calculated"
        pd.testing.assert_series_equal(
            gc_mock.df["ampl_s_norm"], expected_ampl_s_norm, check_names=False
        )

        # Check that the returned object is the updated ganglion cell model
        assert (
            updated_gc == gc_mock
        ), "The returned object should be the same as the input gc_mock"

    @pytest.mark.parametrize(
        "dd_regr_model, params, dd, expected_eccentricity",
        [
            ("linear", {"intercept": 2.0, "slope": 0.5}, 4.0, 4.0),  # (4-2)/0.5 = 4
            (
                "quadratic",
                {"square": 1.0, "slope": 0.5, "intercept": 2.0},
                7.0,
                2.0,
            ),  # x^2 + 0.5x + 2 = 7, root is x = 2
            (
                "cubic",
                {"cube": 1.0, "square": 0.5, "slope": 0.2, "intercept": 1.0},
                8.0,
                1.7281,
            ),  # update expected root for cubic
            ("powerlaw", {"a": 2.0, "b": 3.0}, 8.0, 1.5874),  # (8/2)^(1/3) = 1.5874
        ],
    )
    def test_get_ecc_from_dd(self, dd_regr_model, params, dd, expected_eccentricity):
        # Mock the dendr_diam_parameters with specific gc_type and regression model
        dendr_diam_parameters = {f"test_type_{dd_regr_model}": params}

        # Test the function
        result = self.retina_builder._get_ecc_from_dd(
            dendr_diam_parameters, dd_regr_model, dd
        )

        # Assert the result
        assert math.isclose(result, expected_eccentricity, rel_tol=1e-4)

    @pytest.mark.parametrize(
        "gc_type, dog_model_type, spatial_model_type, temporal_model_type, expected_components",
        [
            (
                "parasol",
                "ellipse_fixed",
                "DOG",
                "fixed",  # Retina model types
                {
                    "ganglion_cell": "GanglionCellParasol",
                    "DoG_model": "DoGModelEllipseFixed",
                    "spatial_model": "SpatialModelDOG",
                    "temporal_model": "TemporalModelFixed",
                },
            ),
            (
                "midget",
                "ellipse_independent",
                "VAE",
                "dynamic",  # Retina model types
                {
                    "ganglion_cell": "GanglionCellMidget",
                    "DoG_model": "DoGModelEllipseIndependent",
                    "spatial_model": "SpatialModelVAE",
                    "temporal_model": "TemporalModelDynamic",
                },
            ),
            (
                "parasol",
                "circular",
                "DOG",
                "subunit",  # Retina model types
                {
                    "ganglion_cell": "GanglionCellParasol",
                    "DoG_model": "DoGModelCircular",
                    "spatial_model": "SpatialModelDOG",
                    "temporal_model": "TemporalModelSubunit",
                },
            ),
        ],
    )
    def test_get_concrete_components(
        self,
        gc_type,
        dog_model_type,
        spatial_model_type,
        temporal_model_type,
        expected_components,
    ):
        # Mock the attributes in retina
        self.retina_mock.gc_type = gc_type
        self.retina_mock.dog_model_type = dog_model_type
        self.retina_mock.spatial_model_type = spatial_model_type
        self.retina_mock.temporal_model_type = temporal_model_type

        # Mock all the classes being used inside the method
        with patch(
            "macaqueretina.retina.construct_retina_module.GanglionCellParasol"
        ) as MockGanglionCellParasol, patch(
            "macaqueretina.retina.construct_retina_module.GanglionCellMidget"
        ) as MockGanglionCellMidget, patch(
            "macaqueretina.retina.construct_retina_module.DoGModelEllipseFixed"
        ) as MockDoGModelEllipseFixed, patch(
            "macaqueretina.retina.construct_retina_module.DoGModelEllipseIndependent"
        ) as MockDoGModelEllipseIndependent, patch(
            "macaqueretina.retina.construct_retina_module.DoGModelCircular"
        ) as MockDoGModelCircular, patch(
            "macaqueretina.retina.construct_retina_module.SpatialModelDOG"
        ) as MockSpatialModelFIT, patch(
            "macaqueretina.retina.construct_retina_module.SpatialModelVAE"
        ) as MockSpatialModelVAE, patch(
            "macaqueretina.retina.construct_retina_module.TemporalModelFixed"
        ) as MockTemporalModelFixed, patch(
            "macaqueretina.retina.construct_retina_module.TemporalModelDynamic"
        ) as MockTemporalModelDynamic, patch(
            "macaqueretina.retina.construct_retina_module.TemporalModelSubunit"
        ) as MockTemporalModelSubunit, patch(
            "macaqueretina.retina.construct_retina_module.DistributionSampler"
        ) as MockDistributionSampler:

            # Call the method
            self.retina_builder.get_concrete_components()

            # Assertions based on expected_components

            # Check that the expected ganglion cell class was instantiated
            expected_gc_class_name = expected_components["ganglion_cell"]
            expected_gc_mock = {
                "GanglionCellParasol": MockGanglionCellParasol,
                "GanglionCellMidget": MockGanglionCellMidget,
            }[expected_gc_class_name]
            expected_gc_mock.assert_called_once()
            assert self.retina_builder._ganglion_cell == expected_gc_mock.return_value

            # Check that the expected DoG model class was instantiated
            expected_dog_class_name = expected_components["DoG_model"]
            expected_dog_mock = {
                "DoGModelEllipseFixed": MockDoGModelEllipseFixed,
                "DoGModelEllipseIndependent": MockDoGModelEllipseIndependent,
                "DoGModelCircular": MockDoGModelCircular,
            }[expected_dog_class_name]
            expected_dog_mock.assert_called_once_with(
                self.retina_mock, self.fit_mock, self.retina_math_mock
            )
            assert self.retina_builder._DoG_model == expected_dog_mock.return_value

            # Check that the distribution sampler was instantiated
            MockDistributionSampler.assert_called_once()
            assert self.retina_builder._sampler == MockDistributionSampler.return_value

            # Check that the expected spatial model class was instantiated
            expected_spatial_class_name = expected_components["spatial_model"]
            expected_spatial_mock = {
                "SpatialModelDOG": MockSpatialModelFIT,
                "SpatialModelVAE": MockSpatialModelVAE,
            }[expected_spatial_class_name]
            expected_spatial_mock.assert_called_once_with(
                self.retina_builder._DoG_model,
                self.retina_builder._sampler,
                self.retina_vae_mock,
                self.fit_mock,
                self.retina_math_mock,
                self.viz_mock,
            )
            assert (
                self.retina_builder._spatial_model == expected_spatial_mock.return_value
            )

            # Check that the expected temporal model class was instantiated
            expected_temporal_class_name = expected_components["temporal_model"]
            expected_temporal_mock = {
                "TemporalModelFixed": MockTemporalModelFixed,
                "TemporalModelDynamic": MockTemporalModelDynamic,
                "TemporalModelSubunit": MockTemporalModelSubunit,
            }[expected_temporal_class_name]
            expected_temporal_mock.assert_called_once_with(
                self.retina_builder._ganglion_cell,
                self.retina_builder._DoG_model,
                self.retina_builder._sampler,
                self.retina_math_mock,
            )
            assert (
                self.retina_builder._temporal_model
                == expected_temporal_mock.return_value
            )

    @pytest.mark.parametrize(
        "gc_type, response_type, bipolar2gc_dict, cell_fits, expected_b2c_ratio",
        [
            (
                "parasol",
                "ON",
                {"parasol": {"ON": ["bipolar_type_1", "bipolar_type_2"]}},
                {
                    "gc": [0, -1, 0, 0],
                    "cone": [0, -1, 0, 0, 0, 0],
                    "bipolar": [0, -1, 0, 0, 0, 0],
                },
                2.3,  # 1.2 + 1.1
            )
        ],
    )
    def test_fit_cell_density_data(
        self, gc_type, response_type, bipolar2gc_dict, cell_fits, expected_b2c_ratio
    ):
        # Set up mock retina and experimental data
        self.retina_mock.gc_type = gc_type
        self.retina_mock.response_type = response_type
        self.retina_mock.bipolar2gc_dict = bipolar2gc_dict

        # Assign real data to experimental_archive
        self.retina_builder.experimental_archive["gc_eccentricity"] = np.array(
            [1, 2, 3]
        )
        self.retina_builder.experimental_archive["gc_density"] = np.array([5, 6, 7])
        self.retina_builder.experimental_archive["gc_control_eccentricity"] = np.array(
            [1, 2, 3]
        )
        self.retina_builder.experimental_archive["gc_control_density"] = np.array(
            [5, 6, 7]
        )
        self.retina_builder.experimental_archive["cone_eccentricity"] = np.array(
            [1, 2, 3]
        )
        self.retina_builder.experimental_archive["cone_density"] = np.array([8, 9, 10])

        # Correctly create bipolar_df to match the expected structure
        bipolar_data = pd.DataFrame(
            {
                "Name": ["Bipolar_cone_ratio"],
                "bipolar_type_1": ["1.2"],
                "bipolar_type_2": ["1.1"],
            }
        )

        self.retina_builder.experimental_archive["bipolar_df"] = bipolar_data

        # Mock retina math functions
        self.retina_math_mock.double_exponential_func = MagicMock()
        self.retina_math_mock.triple_exponential_func = MagicMock()

        # Mock curve fitting
        with patch("scipy.optimize.curve_fit") as mock_curve_fit:
            mock_curve_fit.side_effect = [
                (np.array(cell_fits["gc"]), None),  # Ganglion cell fit
                (np.array(cell_fits["cone"]), None),  # Cone cell fit
            ]

            # Call the method
            self.retina_builder.fit_cell_density_data()

            # Check that the _fit_density function was called with correct arguments
            mock_curve_fit.assert_any_call(
                self.retina_math_mock.double_exponential_func,
                self.retina_builder.experimental_archive["gc_eccentricity"],
                self.retina_builder.experimental_archive["gc_density"],
                p0=[0, -1, 0, 0],
            )

            mock_curve_fit.assert_any_call(
                self.retina_math_mock.triple_exponential_func,
                self.retina_builder.experimental_archive["cone_eccentricity"],
                self.retina_builder.experimental_archive["cone_density"],
                p0=[0, -1, 0, 0, 0, 0],
            )

            # Verify that the bipolar to cone ratio was processed correctly
            bipolar_types = bipolar2gc_dict[gc_type][response_type]
            bipolar_df = self.retina_builder.experimental_archive["bipolar_df"]
            # The method resets the index; replicate that here
            bipolar_df = bipolar_df.set_index(bipolar_df.columns[0])
            b2c_ratio_s = bipolar_df.loc["Bipolar_cone_ratio"]
            b2c_ratios = np.array([float(b2c_ratio_s[b]) for b in bipolar_types])
            print(f"\n{b2c_ratio_s=}")
            print(f"\n{bipolar_types=}")
            print(f"\n{b2c_ratios=}")
            # Sum ratios
            calculated_b2c_ratio = np.sum(b2c_ratios)

            # Add tolerance to the comparison
            assert np.isclose(calculated_b2c_ratio, expected_b2c_ratio, rtol=1e-5)

            # Verify the parameters were saved to the retina mock
            assert np.array_equal(self.retina_mock.gc_density_params, cell_fits["gc"])
            assert np.array_equal(
                self.retina_mock.cone_density_params, cell_fits["cone"]
            )
            assert np.array_equal(
                self.retina_mock.bipolar_density_params, cell_fits["bipolar"]
            )

            # Verify that the selected bipolars were correctly set
            assert np.array_equal(
                self.retina_mock.selected_bipolars_df.columns, bipolar_types
            )

    def test_place_units(self):
        # Mock the _get_gc_proportion method
        self.retina_builder._get_gc_proportion = MagicMock(
            return_value=self.retina_mock
        )

        # Mock the _initialize_positions_by_group method
        self.retina_builder._initialize_positions_by_group = MagicMock(
            return_value=(
                [np.array([0, 1])],  # eccentricity_groups (2 elements to match gc)
                np.array([1.0, 2.0, 3.0]),  # sector_surface_areas_mm2
                np.array([[1, 2], [3, 4]]),  # gc_initial_pos (2 elements)
                np.array([10, 20]),  # gc_density
                np.array([[5, 6], [7, 8]]),  # cone_initial_pos (2 elements)
                np.array([15, 25]),  # cone_density
                np.array([[9, 10], [11, 12]]),  # bipolar_initial_pos (2 elements)
                np.array([5, 10]),  # bipolar_density
            )
        )

        # Mock the _optimize_positions method
        self.retina_builder._optimize_positions = MagicMock(
            side_effect=[
                (
                    np.array([[1, 30], [2, 60]]),
                    np.array([[1, 1], [2, 2]]),
                ),  # GC optimization
                (
                    np.array([[3, 90], [4, 120]]),
                    np.array([[3, 3], [4, 4]]),
                ),  # Cone optimization
                (
                    np.array([[5, 150], [6, 180]]),
                    np.array([[5, 5], [6, 6]]),
                ),  # Bipolar optimization
            ]
        )

        # Call the method
        self.retina_builder.place_units()

        # Verify that _get_gc_proportion was called
        self.retina_builder._get_gc_proportion.assert_called_once_with(self.retina_mock)

        # Verify that _initialize_positions_by_group was called
        self.retina_builder._initialize_positions_by_group.assert_called_once_with(
            self.retina_mock
        )

        # Verify the call arguments to _optimize_positions
        calls = self.retina_builder._optimize_positions.call_args_list

        # For ganglion cell optimization
        gc_call = calls[0]
        assert gc_call[0][0] == self.retina_mock  # First argument (retina)
        assert np.array_equal(
            gc_call[0][1], np.array([[1, 2], [3, 4]])
        )  # gc_initial_pos
        assert np.array_equal(gc_call[0][2], np.array([10, 20]))  # gc_density
        assert (
            gc_call[0][3] == self.retina_mock.gc_placement_parameters
        )  # gc_placement_parameters

        # For cone optimization
        cone_call = calls[1]
        assert cone_call[0][0] == self.retina_mock  # First argument (retina)
        assert np.array_equal(
            cone_call[0][1], np.array([[5, 6], [7, 8]])
        )  # cone_initial_pos
        assert np.array_equal(cone_call[0][2], np.array([15, 25]))  # cone_density
        assert (
            cone_call[0][3] == self.retina_mock.cone_placement_parameters
        )  # cone_placement_parameters

        # For bipolar optimization
        bipolar_call = calls[2]
        assert bipolar_call[0][0] == self.retina_mock  # First argument (retina)
        assert np.array_equal(
            bipolar_call[0][1], np.array([[9, 10], [11, 12]])
        )  # bipolar_initial_pos
        assert np.array_equal(bipolar_call[0][2], np.array([5, 10]))  # bipolar_density
        assert (
            bipolar_call[0][3] == self.retina_mock.bipolar_placement_parameters
        )  # bipolar_placement_parameters

        # Check that the ganglion cell DataFrame was updated with the correct values
        assert np.array_equal(
            self.ganglion_cell_mock.df["pos_ecc_mm"], np.array([1, 2])
        )
        assert np.array_equal(
            self.ganglion_cell_mock.df["pos_polar_deg"], np.array([30, 60])
        )
        assert np.array_equal(
            self.ganglion_cell_mock.df["ecc_group_idx"], np.array([0, 1])
        )

        # Check the number of units is correct
        assert self.ganglion_cell_mock.n_units == 2

        # Check the values assigned to the retina
        assert np.array_equal(
            self.retina_mock.sector_surface_areas_mm2, np.array([1.0, 2.0, 3.0])
        )
        assert np.array_equal(
            self.retina_mock.cone_optimized_pos_mm, np.array([[3, 3], [4, 4]])
        )
        assert np.array_equal(
            self.retina_mock.cone_optimized_pos_pol, np.array([[3, 90], [4, 120]])
        )
        assert np.array_equal(
            self.retina_mock.bipolar_optimized_pos_mm, np.array([[5, 5], [6, 6]])
        )

    def test_create_spatial_receptive_fields(self):
        # Mock _fit_dd_vs_ecc
        self.retina_builder._fit_dd_vs_ecc = MagicMock(return_value="ecc2dd_params")

        # Mock _get_gc_pixel_parameters
        self.retina_builder._get_gc_pixel_parameters = MagicMock(
            return_value=self.ganglion_cell_mock
        )

        # Mock spatial model create method
        self.retina_builder._spatial_model.create = MagicMock(
            return_value=(
                self.retina_mock,
                self.ganglion_cell_mock,
                "viz_whole_ret_img",
            )
        )

        # Mock _add_center_fit_area_to_df
        self.retina_builder.DoG_model._add_center_fit_area_to_df = MagicMock(
            return_value=self.ganglion_cell_mock
        )

        # Mock _scale_DoG_amplitudes
        self.retina_builder._scale_DoG_amplitudes = MagicMock(
            return_value=self.ganglion_cell_mock
        )

        # Call the method
        self.retina_builder.create_spatial_receptive_fields()

        # Assert _fit_dd_vs_ecc was called with correct arguments
        self.retina_builder._fit_dd_vs_ecc.assert_called_once_with(
            self.retina_mock, self.ganglion_cell_mock
        )

        # Assert _get_gc_pixel_parameters was called
        self.retina_builder._get_gc_pixel_parameters.assert_called_once_with(
            self.retina_mock, self.ganglion_cell_mock, "ecc2dd_params"
        )

        # Assert spatial_model.create was called
        self.retina_builder.spatial_model.create.assert_called_once_with(
            self.retina_mock, self.ganglion_cell_mock
        )

        # Assert DoG_model._add_center_fit_area_to_df was called
        self.retina_builder.DoG_model._add_center_fit_area_to_df.assert_called_once_with(
            self.ganglion_cell_mock
        )

        # Assert _scale_DoG_amplitudes was called
        self.retina_builder._scale_DoG_amplitudes.assert_called_once_with(
            self.ganglion_cell_mock
        )

        # Check project_data was updated correctly
        assert "img_ret" in self.retina_builder.project_data["ret"]
        assert "img_ret_masked" in self.retina_builder.project_data["ret"]
        assert "img_ret_adjusted" in self.retina_builder.project_data["ret"]

        # Verify project_data for dd_vs_ecc
        assert "dd_DoG_x" in self.retina_builder.project_data["dd_vs_ecc"]
        assert "dd_DoG_y" in self.retina_builder.project_data["dd_vs_ecc"]

        # Check that the ganglion cell and retina were updated
        assert self.retina_builder.ganglion_cell == self.ganglion_cell_mock
        assert self.retina_builder.retina == self.retina_mock

    @pytest.mark.parametrize(
        "temporal_model_type, expected_methods",
        [
            ("fixed", ["_link_cone_noise_units_to_gcs"]),
            ("dynamic", ["_link_cone_noise_units_to_gcs"]),
            (
                "subunit",
                [
                    "_link_cones_to_bipolars",
                    "_link_bipolar_units_to_gcs",
                    "_link_cone_noise_units_to_gcs",
                ],
            ),
        ],
    )
    def test_connect_units(self, temporal_model_type, expected_methods):
        # Create a mock temporal model
        temporal_model_mock = MagicMock()

        # Mock specific methods that might be called in the different temporal model types
        for method_name in expected_methods:
            setattr(
                temporal_model_mock,
                method_name,
                MagicMock(return_value=self.retina_mock),
            )

        # Define a side effect function for connect_units based on temporal_model_type
        if temporal_model_type in ["fixed", "dynamic"]:

            def connect_units_side_effect(ret, gc):
                return getattr(temporal_model_mock, "_link_cone_noise_units_to_gcs")(
                    ret, gc
                )

        elif temporal_model_type == "subunit":

            def connect_units_side_effect(ret, gc):
                ret = getattr(temporal_model_mock, "_link_cones_to_bipolars")(ret)
                ret = getattr(temporal_model_mock, "_link_bipolar_units_to_gcs")(
                    ret, gc
                )
                ret = getattr(temporal_model_mock, "_link_cone_noise_units_to_gcs")(
                    ret, gc
                )
                return ret

        temporal_model_mock.connect_units = MagicMock(
            side_effect=connect_units_side_effect
        )

        # Assign the mock temporal model to the builder's _temporal_model attribute
        self.retina_builder._temporal_model = temporal_model_mock

        # Set the temporal model type if needed in your logic
        self.retina_mock.temporal_model_type = temporal_model_type

        # Call the connect_units method
        self.retina_builder.connect_units()

        # Verify that temporal_model.connect_units was called once with the correct arguments
        temporal_model_mock.connect_units.assert_called_once_with(
            self.retina_mock, self.ganglion_cell_mock
        )

        # Now verify that the expected internal methods were called with the correct arguments
        for method_name in expected_methods:
            method = getattr(temporal_model_mock, method_name)
            if method_name == "_link_cones_to_bipolars":
                # _link_cones_to_bipolars is called with only (ret)
                method.assert_called_once_with(self.retina_mock)
            else:
                # _link_bipolar_units_to_gcs and _link_cone_noise_units_to_gcs are called with (ret, gc)
                method.assert_called_once_with(
                    self.retina_mock, self.ganglion_cell_mock
                )

        # Ensure retina is updated correctly
        assert self.retina_builder.retina == self.retina_mock

    def test_create_temporal_receptive_fields(self):
        self.retina_builder._temporal_model = self.temporal_model_mock

        # Mock the return values of temporal model methods
        self.temporal_model_mock.create.return_value = self.ganglion_cell_mock
        self.temporal_model_mock._fit_cone_noise_vs_freq.return_value = self.retina_mock

        # Call the method
        self.retina_builder.create_temporal_receptive_fields()

        # Verify that temporal_model.create was called with the correct arguments
        self.temporal_model_mock.create.assert_called_once_with(
            self.retina_mock, self.ganglion_cell_mock
        )

        # Verify that temporal_model._fit_cone_noise_vs_freq was called with the correct arguments
        self.temporal_model_mock._fit_cone_noise_vs_freq.assert_called_once_with(
            self.retina_mock
        )

        # Verify that the ganglion_cell and retina were updated correctly
        assert self.retina_builder.ganglion_cell == self.ganglion_cell_mock
        assert self.retina_builder.retina == self.retina_mock

    def test_create_tonic_drive(self):
        # Mock exp_univariate_stat DataFrame with sample data for "tonic" domain
        tonic_data = pd.DataFrame(
            {
                "shape": [1.0],
                "loc": [0.0],
                "scale": [1.0],
                "distribution": ["normal"],
                "domain": ["tonic"],
            },
            index=["some_param"],
        )
        self.DoG_model_mock.exp_univariate_stat = tonic_data

        # Mock DataFrame for ganglion cells with one row
        self.ganglion_cell_mock.df = pd.DataFrame({"some_param": [0]})
        self.ganglion_cell_mock.n_units = 1

        # Mock the sampler to return a fixed array
        self.sampler_mock.sample_univariate.return_value = [0.5]

        # Patch the DoG_model property to return your mock
        with patch.object(
            type(self.retina_builder), "DoG_model", new_callable=PropertyMock
        ) as mock_DoG_model_prop:
            mock_DoG_model_prop.return_value = self.DoG_model_mock

            # Patch the sampler property to return your mock
            with patch.object(
                type(self.retina_builder), "sampler", new_callable=PropertyMock
            ) as mock_sampler_prop:
                mock_sampler_prop.return_value = self.sampler_mock

                # Call the method
                self.retina_builder.create_tonic_drive()

                # Verify that sample_univariate was called with the correct parameters
                self.sampler_mock.sample_univariate.assert_called_once_with(
                    1.0, 0.0, 1.0, len(self.ganglion_cell_mock.df), "normal"
                )

                # Check that the result was assigned to the correct parameter in the ganglion cell DataFrame
                assert "some_param" in self.ganglion_cell_mock.df.columns
                assert list(self.ganglion_cell_mock.df["some_param"]) == [0.5]

                # Verify that the ganglion_cell attribute was updated correctly
                assert self.retina_builder.ganglion_cell == self.ganglion_cell_mock
