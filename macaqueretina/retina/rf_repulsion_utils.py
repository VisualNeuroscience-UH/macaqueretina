# Built-in
from typing import Any, Tuple

# Third-party
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

"""
This is a utility module, where I have experimented using the o1 language model to generate the code. It used
brute force to chop a long function into subsubfunctions. The subfunctions have been tested and work as expected.
They are not very readable though, because of the semi-infinite function signatures.
"""


def apply_rf_repulsion(ret: Any, gc: Any, viz: Any) -> Tuple[Any, Any]:
    """
    Apply mutual repulsion to receptive fields (RFs) to ensure optimal coverage of a simulated retina.
    It involves multiple iterations to gradually move the RFs until they cover the retina
    with minimal overlapping, considering boundary effects and force gradients.

    Parameters:
    -----------
    ret : Retina object
        The retina object containing retina parameters and images.
    gc : GanglionCell object
        The ganglion cell object containing RFs and their properties.

    Returns:
    --------
    ret : Retina object
        Updated retina object after RF adjustments.
    gc : GanglionCell object
        Updated ganglion cell object with new RF positions.
    """
    # Extract parameters and initialize variables
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
    ) = _initialize_parameters(ret, gc, viz)
    # Initialize visualization if needed
    if params["show_repulsion_progress"]:
        fig_args = _initialize_visualization(ret, gc, img_ret_shape, H, viz)

    # Initialize rigid body matrices
    Mrb_pre, new_coords = _initialize_rigid_body_matrices(
        n_units, rf_positions, homogeneous_coords
    )

    # Main optimization loop
    for iteration in range(params["n_iterations"]):
        # Update RF coordinates
        Xt, Yt = _update_rf_coordinates(new_coords, n_units, H, W)

        # Update retina image
        retina = _update_retina(
            iteration,
            Xt,
            Yt,
            rfs,
            ret,
            boundary_polygon_path,
            retina_boundary_effect,
            params["border_repulsion_stength"],
            n_units,
            H,
            W,
            Mrb_pre,
        )
        if iteration == 0:
            reference_retina = retina.copy()

        # Compute gradients
        grad_y, grad_x = np.gradient(retina)

        # Compute forces and torques
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

        # Update positions and rotations
        Mrb_pre, new_coords, params["change_rate"] = _update_positions_and_rotations(
            Mrb_pre,
            net_force_y,
            net_force_x,
            net_torque,
            params["change_rate"],
            params["cooling_rate"],
            homogeneous_coords,
            n_units,
        )

        # Drop boundary effect
        reference_retina, retina = _drop_boundary_effect(
            reference_retina, retina, boundary_polygon_path
        )

        # Get masked retina for visualization
        center_mask = _get_center_mask(Yt, Xt, rfs_mask, ret)

        # Visualization during optimization
        if (
            params["show_repulsion_progress"]
            and iteration % params["show_skip_steps"] == 0
        ):
            _visualize_progress(
                iteration,
                params,
                ret,
                gc,
                Xt,
                Yt,
                rfs_mask,
                reference_retina,
                center_mask,
                retina,
                fig_args,
                com_x,
                com_y,
                viz,
            )

    # Resample RFs to rectangular H, W resolution
    new_gc_img, new_gc_img_lu_pix, com_x_local, com_y_local, final_retina = (
        _resample_rfs(n_units, H, W, com_x, com_y, rfs, Yt, Xt, img_ret_shape)
    )

    # Final visualization
    if params["show_repulsion_progress"]:
        _final_visualization(
            params,
            reference_retina,
            center_mask,
            final_retina,
            iteration,
            gc,
            H,
            fig_args,
            viz,
        )

    # Update gc and ret objects
    gc.img = new_gc_img
    gc.img_lu_pix = new_gc_img_lu_pix
    gc.df["com_x_pix"] = com_x_local
    gc.df["com_y_pix"] = com_y_local
    ret.whole_ret_img = final_retina

    return ret, gc


def _initialize_parameters(ret: Any, gc: Any, viz: Any) -> Tuple[
    Tuple[int, int],
    dict,
    int,
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    mplPath.Path,
    np.ndarray,
    np.ndarray,
]:
    """Extracts parameters and initializes variables."""
    img_ret_shape = ret.whole_ret_img.shape
    params = ret.receptive_field_repulsion_parameters

    n_units, H, W = gc.img.shape
    assert H == W, "RF must be square, aborting..."

    rf_positions = np.array(gc.img_lu_pix, dtype=float)
    rfs = np.array(gc.img, dtype=float)
    rfs_mask = np.array(gc.img_mask, dtype=bool)
    masked_rfs = rfs * rfs_mask
    sum_masked_rfs = np.sum(masked_rfs, axis=(1, 2))

    # Compute boundary effect
    boundary_polygon_path, retina_boundary_effect = _compute_boundary_effect(
        ret, gc, img_ret_shape, H, params["border_repulsion_stength"], viz
    )

    # Homogeneous coordinates
    Y0, X0 = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    homogeneous_coords = np.stack([X0.flatten(), Y0.flatten(), np.ones(H * W)], axis=0)

    return (
        img_ret_shape,
        params,
        n_units,
        H,
        W,
        rf_positions.astype(np.float32),
        rfs.astype(np.float32),
        rfs_mask,
        masked_rfs.astype(np.float32),
        sum_masked_rfs.astype(np.float32),
        boundary_polygon_path,
        retina_boundary_effect.astype(np.int32),
        homogeneous_coords.astype(np.float32),
    )


def _initialize_visualization(
    ret: Any,
    gc: Any,
    img_ret_shape: Tuple[int, int],
    H: int,
    viz: Any,
) -> dict:
    """Initializes visualization settings."""

    fig_args = viz.show_repulsion_progress(
        np.zeros(img_ret_shape),
        np.zeros(img_ret_shape),
        ecc_lim_mm=ret.ecc_lim_mm,
        polar_lim_deg=ret.polar_lim_deg,
        stage="init",
        um_per_pix=gc.um_per_pix,
        sidelen=H,
    )
    return fig_args


def _compute_boundary_effect(
    ret: Any,
    gc: Any,
    img_ret_shape: Tuple[int, int],
    H: int,
    border_repulsion_strength: float,
    viz: Any,
) -> Tuple[mplPath.Path, np.ndarray]:
    """Computes the boundary effect for the retina."""
    boundary_polygon = viz.boundary_polygon(
        ret.ecc_lim_mm,
        ret.polar_lim_deg,
        um_per_pix=gc.um_per_pix,
        sidelen=H,
    )
    boundary_polygon_path = mplPath.Path(boundary_polygon)
    Y, X = np.meshgrid(
        np.arange(img_ret_shape[0]),
        np.arange(img_ret_shape[1]),
        indexing="ij",
    )
    boundary_points = np.vstack((X.flatten(), Y.flatten())).T
    inside_boundary = boundary_polygon_path.contains_points(boundary_points)
    boundary_mask = inside_boundary.reshape(img_ret_shape)
    retina_boundary_effect = np.where(boundary_mask, 0, border_repulsion_strength)
    return boundary_polygon_path, retina_boundary_effect


def _initialize_rigid_body_matrices(
    n_units: int, rf_positions: np.ndarray, homogeneous_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Initializes rigid body matrices for each RF."""
    Mrb_pre = np.tile(np.eye(3), (n_units, 1, 1))
    Mrb_pre[:, :2, 2] = rf_positions
    new_coords = Mrb_pre @ homogeneous_coords
    return Mrb_pre.astype(np.float32), new_coords.astype(np.float32)


def _update_rf_coordinates(
    new_coords: np.ndarray, n_units: int, H: int, W: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Updates RF coordinates based on transformations."""
    Xt = new_coords[:, 0, ...].round().reshape(n_units, H, W).astype(int)
    Yt = new_coords[:, 1, ...].round().reshape(n_units, H, W).astype(int)
    return Xt, Yt


def _update_retina(
    iteration: int,
    Xt: np.ndarray,
    Yt: np.ndarray,
    rfs: np.ndarray,
    ret: Any,
    boundary_polygon_path: mplPath.Path,
    retina_boundary_effect: np.ndarray,
    border_repulsion_strength: float,
    n_units: int,
    H: int,
    W: int,
    Mrb_pre: np.ndarray,
) -> np.ndarray:
    """Updates the retina image with new RF positions."""
    retina = np.zeros(ret.whole_ret_img.shape)
    for i in range(n_units):
        idx = np.where(rfs[i, ...] == np.max(rfs[i], axis=(0, 1)))
        pos = np.stack((Xt[i, idx[1], idx[0]], Yt[i, idx[1], idx[0]]), axis=1)
        inside_boundary = boundary_polygon_path.contains_points(pos)
        if inside_boundary:
            retina[Yt[i], Xt[i]] += rfs[i]
        else:
            inside = np.where(retina_boundary_effect == 0)
            choice = np.random.choice(len(inside[0]))
            y_start = int(inside[0][choice] - idx[0].item())
            x_start = int(inside[1][choice] - idx[1].item())
            y_end = y_start + H
            x_end = x_start + W
            retina[y_start:y_end, x_start:x_end] += rfs[i]
            Yt[i, ...] = np.arange(y_start, y_end)[:, None]
            Xt[i, ...] = np.arange(x_start, x_end)
            Mrb_pre[i, :2, 2] = [x_start, y_start]

    retina += retina_boundary_effect
    return retina


def _compute_forces(
    Xt: np.ndarray,
    Yt: np.ndarray,
    rfs: np.ndarray,
    rfs_mask: np.ndarray,
    masked_rfs: np.ndarray,
    sum_masked_rfs: np.ndarray,
    grad_y: np.ndarray,
    grad_x: np.ndarray,
    n_units: int,
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes forces and torques acting on each RF."""
    force_y = -1 * grad_y[Yt, Xt] * rfs * rfs_mask
    force_x = -1 * grad_x[Yt, Xt] * rfs * rfs_mask

    com_y = np.sum(masked_rfs * Yt, axis=(1, 2)) / sum_masked_rfs
    com_x = np.sum(masked_rfs * Xt, axis=(1, 2)) / sum_masked_rfs

    com_y_mtx = np.tile(com_y, (H, W, 1)).transpose(2, 0, 1)
    com_x_mtx = np.tile(com_x, (H, W, 1)).transpose(2, 0, 1)
    radius_vec = np.stack([Yt - com_y_mtx, Xt - com_x_mtx], axis=-1)
    torques = force_y * radius_vec[..., 1] - force_x * radius_vec[..., 0]
    net_torque = np.sum(torques, axis=(1, 2))
    net_force_y = np.sum(force_y, axis=(1, 2))
    net_force_x = np.sum(force_x, axis=(1, 2))

    # Normalize forces and torques
    net_torque = np.pi * net_torque / (np.max(np.abs(net_torque)) + 1e-8)
    net_force_y = 50 * net_force_y / (np.max(np.abs(net_force_y)) + 1e-8)
    net_force_x = 50 * net_force_x / (np.max(np.abs(net_force_x)) + 1e-8)

    return net_force_y, net_force_x, net_torque, com_y, com_x


def _update_positions_and_rotations(
    Mrb_pre: np.ndarray,
    net_force_y: np.ndarray,
    net_force_x: np.ndarray,
    net_torque: np.ndarray,
    change_rate: float,
    cooling_rate: float,
    homogeneous_coords: np.ndarray,
    n_units: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Updates the positions and rotations of RFs."""
    rot = change_rate * net_torque
    tr_x = change_rate * net_force_x
    tr_y = change_rate * net_force_y

    rot_mtx = np.array(
        [
            [np.cos(rot), -np.sin(rot), np.zeros(n_units)],
            [np.sin(rot), np.cos(rot), np.zeros(n_units)],
            [np.zeros(n_units), np.zeros(n_units), np.ones(n_units)],
        ]
    ).transpose(2, 0, 1)
    trans_mtx = np.array(
        [
            [np.ones(n_units), np.zeros(n_units), tr_x],
            [np.zeros(n_units), np.ones(n_units), tr_y],
            [np.zeros(n_units), np.zeros(n_units), np.ones(n_units)],
        ]
    ).transpose(2, 0, 1)

    Mrb_change = trans_mtx @ rot_mtx
    Mrb = Mrb_pre @ Mrb_change
    Mrb_pre = Mrb
    new_coords = Mrb @ homogeneous_coords
    change_rate *= cooling_rate

    return Mrb_pre, new_coords, change_rate


def _drop_boundary_effect(reference_retina, retina, boundary_polygon_path):
    """Drops the boundary effect from the retina."""

    # Create mask and use it to zero areas outside the polygon
    Y, X = np.meshgrid(
        np.arange(retina.shape[0]), np.arange(retina.shape[1]), indexing="ij"
    )

    retina_points = np.vstack((X.flatten(), Y.flatten())).T
    inside_boundary = boundary_polygon_path.contains_points(retina_points)
    mask = inside_boundary.reshape(retina.shape)
    retina = np.ma.masked_array(retina, ~mask)
    retina = retina.filled(0)
    reference_retina = np.ma.masked_array(reference_retina, ~mask)
    reference_retina = reference_retina.filled(0)
    return reference_retina, retina


def _visualize_progress(
    iteration: int,
    params: dict,
    ret: Any,
    gc: Any,
    Xt: np.ndarray,
    Yt: np.ndarray,
    rfs_mask: np.ndarray,
    reference_retina: np.ndarray,
    center_mask: np.ndarray,
    retina_viz: np.ndarray,
    fig_args: dict,
    com_x: np.ndarray,
    com_y: np.ndarray,
    viz: Any,
) -> None:
    """Handles visualization during optimization."""
    if params["show_only_unit"] is not None:
        unit_retina = np.zeros(ret.whole_ret_img.shape)
        unit_idx = params["show_only_unit"]
        fig_args["additional_points"] = [com_x[unit_idx], com_y[unit_idx]]
        fig_args["unit_idx"] = unit_idx
        unit_img = np.ones(rfs_mask[unit_idx].shape) * 0.1
        unit_img += rfs_mask[unit_idx]
        unit_retina[Yt[unit_idx], Xt[unit_idx]] += unit_img
        retina_viz = unit_retina.copy()

    if iteration % params["show_skip_steps"] == 0:
        viz.show_repulsion_progress(
            reference_retina,
            center_mask,
            new_retina=retina_viz,
            stage="update",
            iteration=iteration,
            um_per_pix=gc.um_per_pix,
            sidelen=gc.img.shape[1],
            **fig_args,
        )


def _resample_rfs(
    n_units: int,
    H: int,
    W: int,
    com_x: np.ndarray,
    com_y: np.ndarray,
    rfs: np.ndarray,
    Yt: np.ndarray,
    Xt: np.ndarray,
    img_ret_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resamples RFs to a regular grid and updates positions."""
    new_gc_img = np.zeros((n_units, H, W))
    Yout = np.zeros((n_units, H, W), dtype=np.int32)
    Xout = np.zeros((n_units, H, W), dtype=np.int32)

    for i in range(n_units):
        y_top = np.round(com_y[i] - H / 2).astype(int)
        x_left = np.round(com_x[i] - W / 2).astype(int)
        y_out = np.arange(y_top, y_top + H)
        x_out = np.arange(x_left, x_left + W)
        y_out_grid, x_out_grid = np.meshgrid(y_out, x_out, indexing="ij")
        Yout[i] = y_out_grid
        Xout[i] = x_out_grid

        points = np.array([Yt[i].ravel(), Xt[i].ravel()]).T
        values = rfs[i].ravel()
        new_points = np.array([y_out_grid.ravel(), x_out_grid.ravel()]).T

        resampled_values = griddata(
            points, values, new_points, method="cubic", fill_value=0
        )

        new_gc_img[i, ...] = resampled_values.reshape(H, W)

    new_gc_img_lu_pix = np.array(
        [Xout[:, 0, 0], Yout[:, 0, 0]], dtype=np.int32
    ).T  # x, y
    com_x_local = com_x - new_gc_img_lu_pix[:, 0]
    com_y_local = com_y - new_gc_img_lu_pix[:, 1]

    final_retina = np.zeros(img_ret_shape)
    for i in range(n_units):
        final_retina[Yout[i], Xout[i]] += new_gc_img[i]

    return new_gc_img, new_gc_img_lu_pix, com_x_local, com_y_local, final_retina


def _final_visualization(
    params: dict,
    reference_retina: np.ndarray,
    center_mask: np.ndarray,
    final_retina: np.ndarray,
    iteration: int,
    gc: Any,
    H: int,
    fig_args: dict,
    viz: Any,
) -> None:
    """Displays the final visualization after optimization."""
    viz.show_repulsion_progress(
        reference_retina,
        center_mask,
        new_retina=final_retina,
        stage="final",
        iteration=iteration,
        um_per_pix=gc.um_per_pix,
        sidelen=H,
        savefigname=params["savefigname"],
        **fig_args,
    )
    plt.ioff()  # Turn off interactive mode


def _get_center_mask(
    Yt: np.ndarray, Xt: np.ndarray, rfs_mask: np.ndarray, ret: Any
) -> np.ndarray:
    """Computes the center mask for visualization."""
    center_mask = np.zeros(ret.whole_ret_img.shape)
    for i in range(len(rfs_mask)):
        center_mask[Yt[i], Xt[i]] += rfs_mask[i]
    return center_mask
