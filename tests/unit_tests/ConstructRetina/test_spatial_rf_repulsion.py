# test_rf_repulsion.py

# Built-in
import unittest
from unittest.mock import MagicMock, patch

# Third-party
import numpy as np

# Local
from macaqueretina.retina.rf_repulsion_utils import (
    _compute_boundary_effect,
    _compute_forces,
    _final_visualization,
    _initialize_parameters,
    _initialize_rigid_body_matrices,
    _initialize_visualization,
    _resample_rfs,
    _update_positions_and_rotations,
    _update_retina,
    _update_rf_coordinates,
    _visualize_progress,
    apply_rf_repulsion,
)


class TestRFRepulsionFunctions(unittest.TestCase):
    def setUp(self):
        # Mock the ret object
        self.ret = MagicMock()
        self.ret.whole_ret_img = np.zeros((100, 100))
        self.ret.receptive_field_repulsion_parameters = {
            "show_repulsion_progress": False,
            "change_rate": 0.01,
            "n_iterations": 5,
            "show_skip_steps": 1,
            "border_repulsion_stength": 1.0,
            "cooling_rate": 0.9,
            "show_only_unit": None,
            "savefigname": None,
        }
        self.ret.ecc_lim_mm = 5.0
        self.ret.polar_lim_deg = 360

        # Mock the gc object
        self.gc = MagicMock()
        self.gc.img = np.random.rand(10, 10, 10)  # 10 units, 10x10 RFs
        self.gc.img_mask = np.ones((10, 10, 10), dtype=bool)
        self.gc.img_lu_pix = np.random.randint(0, 90, size=(10, 2))
        self.gc.um_per_pix = 1.0
        self.gc.df = {"com_x_pix": [], "com_y_pix": []}

        # Mock the viz object
        self.viz = MagicMock()
        self.viz.show_repulsion_progress = MagicMock()
        self.viz.boundary_polygon = MagicMock(
            return_value=np.array([[10, 10], [10, 90], [90, 90], [90, 10]])
        )

    @patch("macaqueretina.retina.rf_repulsion_utils._initialize_parameters")
    @patch("macaqueretina.retina.rf_repulsion_utils._initialize_visualization")
    @patch("macaqueretina.retina.rf_repulsion_utils._initialize_rigid_body_matrices")
    @patch("macaqueretina.retina.rf_repulsion_utils._update_rf_coordinates")
    @patch("macaqueretina.retina.rf_repulsion_utils._update_retina")
    @patch("macaqueretina.retina.rf_repulsion_utils._compute_forces")
    @patch("macaqueretina.retina.rf_repulsion_utils._update_positions_and_rotations")
    @patch("macaqueretina.retina.rf_repulsion_utils._visualize_progress")
    @patch("macaqueretina.retina.rf_repulsion_utils._resample_rfs")
    @patch("macaqueretina.retina.rf_repulsion_utils._final_visualization")
    def test_apply_rf_repulsion(
        self,
        mock_final_visualization,
        mock_resample_rfs,
        mock_visualize_progress,
        mock_update_positions_and_rotations,
        mock_compute_forces,
        mock_update_retina,
        mock_update_rf_coordinates,
        mock_initialize_rigid_body_matrices,
        mock_initialize_visualization,
        mock_initialize_parameters,
    ):
        # Set up mock return values
        mock_initialize_parameters.return_value = (
            self.ret.whole_ret_img.shape,
            self.ret.receptive_field_repulsion_parameters,
            10,  # n_units
            10,  # H
            10,  # W
            np.zeros((10, 2)),
            np.zeros((10, 10, 10)),
            np.ones((10, 10, 10), dtype=bool),
            np.zeros((10, 10, 10)),
            np.ones(10),
            MagicMock(),
            np.zeros(self.ret.whole_ret_img.shape),
            np.zeros((3, 100)),
        )
        mock_initialize_rigid_body_matrices.return_value = (
            np.zeros((10, 3, 3)),
            np.zeros((10, 3, 100)),
        )
        mock_update_rf_coordinates.return_value = (
            np.zeros((10, 10, 10), dtype=int),
            np.zeros((10, 10, 10), dtype=int),
        )
        mock_update_retina.return_value = np.zeros(self.ret.whole_ret_img.shape)
        mock_compute_forces.return_value = (
            np.zeros(10),
            np.zeros(10),
            np.zeros(10),
            np.zeros(10),
            np.zeros(10),
        )
        mock_update_positions_and_rotations.return_value = (
            np.zeros((10, 3, 3)),
            np.zeros((10, 3, 100)),
            0.01,
        )
        mock_resample_rfs.return_value = (
            np.zeros((10, 10, 10)),
            np.zeros((10, 2), dtype=int),
            np.zeros(10),
            np.zeros(10),
            np.zeros(self.ret.whole_ret_img.shape),
        )

        # Call the function under test
        ret_result, gc_result = apply_rf_repulsion(self.ret, self.gc, self.viz)

        # Assertions to ensure that the functions were called
        mock_initialize_parameters.assert_called_once_with(self.ret, self.gc, self.viz)
        mock_initialize_rigid_body_matrices.assert_called_once()
        mock_update_rf_coordinates.assert_called()
        mock_update_retina.assert_called()
        mock_compute_forces.assert_called()
        mock_update_positions_and_rotations.assert_called()
        mock_resample_rfs.assert_called_once()
        mock_final_visualization.assert_not_called()  # Because show_repulsion_progress is False

        # Assertions on return values
        self.assertEqual(ret_result, self.ret)
        self.assertEqual(gc_result, self.gc)

    def test_initialize_parameters(self):
        # Call the function
        result = _initialize_parameters(self.ret, self.gc, self.viz)

        # Assertions on the result
        (
            img_ret_shape,
            params,
            n_units,
            H,
            W,
            rf_positions,
            rfs,
            rfs_mask,
            masked_rfs,
            sum_masked_rfs,
            boundary_polygon_path,
            retina_boundary_effect,
            homogeneous_coords,
        ) = result

        self.assertEqual(img_ret_shape, self.ret.whole_ret_img.shape)
        self.assertEqual(params, self.ret.receptive_field_repulsion_parameters)
        self.assertEqual(n_units, 10)
        self.assertEqual(H, 10)
        self.assertEqual(W, 10)
        self.assertEqual(rf_positions.shape, (10, 2))
        self.assertEqual(rfs.shape, (10, 10, 10))
        self.assertEqual(rfs_mask.shape, (10, 10, 10))
        self.assertEqual(masked_rfs.shape, (10, 10, 10))
        self.assertEqual(sum_masked_rfs.shape, (10,))
        self.assertIsNotNone(boundary_polygon_path)
        self.assertEqual(retina_boundary_effect.shape, self.ret.whole_ret_img.shape)
        self.assertEqual(homogeneous_coords.shape, (3, 100))

    def test_initialize_visualization(self):
        # Mock parameters
        img_ret_shape = self.ret.whole_ret_img.shape
        H = 10
        # Call the function
        fig_args = _initialize_visualization(
            self.ret, self.gc, img_ret_shape, H, self.viz
        )

        # Assertions
        self.viz.show_repulsion_progress.assert_called_once()
        self.assertIsNotNone(fig_args)

    def test_compute_boundary_effect(self):
        # Call the function
        boundary_polygon_path, retina_boundary_effect = _compute_boundary_effect(
            self.ret, self.gc, self.ret.whole_ret_img.shape, 10, 1.0, self.viz
        )

        # Assertions
        self.viz.boundary_polygon.assert_called_once()
        self.assertIsNotNone(boundary_polygon_path)
        self.assertEqual(retina_boundary_effect.shape, self.ret.whole_ret_img.shape)

    def test_initialize_rigid_body_matrices(self):
        # Mock parameters
        n_units = 10
        rf_positions = np.zeros((10, 2))
        homogeneous_coords = np.zeros((3, 100))
        # Call the function
        Mrb_pre, new_coords = _initialize_rigid_body_matrices(
            n_units, rf_positions, homogeneous_coords
        )

        # Assertions
        self.assertEqual(Mrb_pre.shape, (10, 3, 3))
        self.assertEqual(new_coords.shape, (10, 3, 100))

    def test_update_rf_coordinates(self):
        # Mock parameters
        new_coords = np.zeros((10, 3, 100))
        n_units = 10
        H = 10
        W = 10
        # Call the function
        Xt, Yt = _update_rf_coordinates(new_coords, n_units, H, W)

        # Assertions
        self.assertEqual(Xt.shape, (10, 10, 10))
        self.assertEqual(Yt.shape, (10, 10, 10))

    def test_update_retina(self):
        # Mock parameters
        iteration = 0
        Xt = np.zeros((10, 10, 10), dtype=int)
        Yt = np.zeros((10, 10, 10), dtype=int)
        rfs = np.random.rand(10, 10, 10)
        boundary_polygon_path = MagicMock()
        boundary_polygon_path.contains_points.return_value = np.array([True])
        retina_boundary_effect = np.zeros(self.ret.whole_ret_img.shape)
        border_repulsion_strength = 1.0
        n_units = 10
        H = 10
        W = 10
        Mrb_pre = np.zeros((10, 3, 3))

        # Call the function
        retina = _update_retina(
            iteration,
            Xt,
            Yt,
            rfs,
            self.ret,
            boundary_polygon_path,
            retina_boundary_effect,
            border_repulsion_strength,
            n_units,
            H,
            W,
            Mrb_pre,
        )

        # Assertions
        self.assertEqual(retina.shape, self.ret.whole_ret_img.shape)

    def test_compute_forces(self):
        # Mock parameters
        Xt = np.zeros((10, 10, 10), dtype=int)
        Yt = np.zeros((10, 10, 10), dtype=int)
        rfs = np.random.rand(10, 10, 10)
        rfs_mask = np.ones((10, 10, 10), dtype=bool)
        masked_rfs = rfs * rfs_mask
        sum_masked_rfs = np.sum(masked_rfs, axis=(1, 2))
        grad_y = np.zeros(self.ret.whole_ret_img.shape)
        grad_x = np.zeros(self.ret.whole_ret_img.shape)
        n_units = 10
        H = 10
        W = 10

        # Call the function
        net_force_y, net_force_x, net_torque, com_y, com_x = _compute_forces(
            Xt,
            Yt,
            rfs,
            rfs_mask,
            masked_rfs,
            sum_masked_rfs,
            grad_y,
            grad_x,
            n_units,
            H,
            W,
        )

        # Assertions
        self.assertEqual(net_force_y.shape, (10,))
        self.assertEqual(net_force_x.shape, (10,))
        self.assertEqual(net_torque.shape, (10,))
        self.assertEqual(com_y.shape, (10,))
        self.assertEqual(com_x.shape, (10,))

    def test_update_positions_and_rotations(self):
        # Mock parameters
        Mrb_pre = np.zeros((10, 3, 3))
        net_force_y = np.zeros(10)
        net_force_x = np.zeros(10)
        net_torque = np.zeros(10)
        change_rate = 0.01
        cooling_rate = 0.9
        homogeneous_coords = np.zeros((3, 100))
        n_units = 10

        # Call the function
        Mrb_pre_updated, new_coords, new_change_rate = _update_positions_and_rotations(
            Mrb_pre,
            net_force_y,
            net_force_x,
            net_torque,
            change_rate,
            cooling_rate,
            homogeneous_coords,
            n_units,
        )

        # Assertions
        self.assertEqual(Mrb_pre_updated.shape, (10, 3, 3))
        self.assertEqual(new_coords.shape, (10, 3, 100))
        self.assertAlmostEqual(new_change_rate, change_rate * cooling_rate)

    def test_visualize_progress(self):
        # Mock parameters
        iteration = 0
        params = {
            "n_iterations": 5,
            "show_skip_steps": 1,
            "savefigname": None,
            "show_only_unit": None,
        }
        Xt = np.zeros((10, 10, 10), dtype=int)
        Yt = np.zeros((10, 10, 10), dtype=int)
        rfs_mask = np.ones((10, 10, 10), dtype=bool)
        reference_retina = np.zeros(self.ret.whole_ret_img.shape)
        retina_viz = np.zeros(self.ret.whole_ret_img.shape)
        fig_args = {}
        com_x = np.zeros(10)
        com_y = np.zeros(10)

        # Call the function
        _visualize_progress(
            iteration,
            params,
            self.ret,
            self.gc,
            Xt,
            Yt,
            rfs_mask,
            reference_retina,
            retina_viz,
            fig_args,
            com_x,
            com_y,
            self.viz,
        )

        # Assertions
        self.viz.show_repulsion_progress.assert_called_once()

    def test_resample_rfs(self):
        # Mock parameters
        n_units = 10
        H = 10
        W = 10
        com_x = np.zeros(10)
        com_y = np.zeros(10)
        rfs = np.random.rand(10, H, W)
        Yt = np.zeros((n_units, H, W), dtype=int)
        Xt = np.zeros((n_units, H, W), dtype=int)
        for i in range(n_units):
            Yt[i], Xt[i] = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        img_ret_shape = self.ret.whole_ret_img.shape

        # Call the function
        new_gc_img, new_gc_img_lu_pix, com_x_local, com_y_local, final_retina = (
            _resample_rfs(n_units, H, W, com_x, com_y, rfs, Yt, Xt, img_ret_shape)
        )

        # Assertions
        self.assertEqual(new_gc_img.shape, (10, 10, 10))
        self.assertEqual(new_gc_img_lu_pix.shape, (10, 2))
        self.assertEqual(com_x_local.shape, (10,))
        self.assertEqual(com_y_local.shape, (10,))
        self.assertEqual(final_retina.shape, img_ret_shape)

    def test_final_visualization(self):
        # Mock parameters
        reference_retina = np.zeros(self.ret.whole_ret_img.shape)
        final_retina = np.zeros(self.ret.whole_ret_img.shape)
        iteration = 5
        H = 10
        fig_args = {}

        # Call the function
        _final_visualization(
            reference_retina, final_retina, iteration, self.gc, H, fig_args, self.viz
        )

        # Assertions
        self.viz.show_repulsion_progress.assert_called_once()

    def test_apply_rf_repulsion_full(self):
        # This test runs the full function without mocking subfunctions
        # Ensure that the function runs without errors with mocked dependencies

        # Set show_repulsion_progress to False to avoid visualization during tests
        self.ret.receptive_field_repulsion_parameters["show_repulsion_progress"] = False

        # Call the function
        ret_result, gc_result = apply_rf_repulsion(self.ret, self.gc, self.viz)

        # Assertions
        self.assertEqual(ret_result, self.ret)
        self.assertEqual(gc_result, self.gc)

        # Check that the gc.img and gc.img_lu_pix have been updated
        self.assertIsNotNone(self.gc.img)
        self.assertIsNotNone(self.gc.img_lu_pix)

        # Check that ret.whole_ret_img has been updated
        self.assertIsNotNone(self.ret.whole_ret_img)


if __name__ == "__main__":
    unittest.main()
