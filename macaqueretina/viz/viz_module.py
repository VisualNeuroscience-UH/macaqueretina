# Built-in
import copy
import math
import os
import time
from functools import reduce
from pathlib import Path
from typing import Optional

# Third-party
import brian2.units as b2u
import matplotlib.colors as mcolors
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats
import seaborn as sns
import torch
import torchvision.transforms.functional as TF
from matplotlib import cm
from matplotlib.patches import Circle, Ellipse, Polygon
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from sklearn.manifold import TSNE

# Local
from macaqueretina.retina.vae_module import AugmentedDataset


class Viz:
    """
    Methods to viz_module the retina

    Some methods import object instance as call parameter (ConstructRetina, SimulateRetina, etc).
    """

    cmap = "gist_earth"  # viridis or cividis would be best for color-blind

    def __init__(self, context, data_io, project_data, ana, **kwargs) -> None:
        self._context = context
        self._data_io = data_io
        self._project_data = project_data
        self._ana = ana

        for attr, value in kwargs.items():
            setattr(self, attr, value)

        # Some settings related to plotting
        self.cmap_stim = "gray"
        self.cmap_spatial_filter = "bwr"

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def project_data(self):
        return self._project_data

    @property
    def ana(self):
        return self._ana

    def data_is_valid(self, data, accept_empty=False):
        try:
            data = data / data.get_best_unit()
        except:
            pass

        if accept_empty == True:
            is_valid = isinstance(data, np.ndarray)
        else:
            is_valid = isinstance(data, np.ndarray) and data.size > 0

        return is_valid

    def _figsave(self, figurename="", myformat="png", subfolderpath="", suffix=""):
        """
        Save the current figure and its configuration to the working directory or a specified subfolder path.
        This method saves the current figure with various customization options for the
        filename, format, and location. By default, figures are saved as 'MyFigure.png'.
        The figure's font settings are configured such that fonts are preserved as they are,
        and not converted into paths.

        Parameters
        ----------
        figurename : str, optional
            The name of the figure file. If it's specified with an extension, the figure
            is saved with that name. If it's a relative path, the figure is saved to that path.
            If not provided, the figure is saved as 'MyFigure.png'. Defaults to "".
        myformat : str, optional
            The format of the figure (e.g., 'png', 'jpg', 'svg', etc.).
            If provided with a leading ".", the "." is removed. Defaults to 'png'.
        subfolderpath : str, optional
            The subfolder within the working directory to which the figure is saved.
            If figurename is a path, this value will be overridden by the parent directory
            of figurename. Defaults to "".
        suffix : str, optional
            A suffix that is added to the end of the filename, just before the file extension.
            Defaults to "".

        Notes
        -----
        - The fonts in the figure are configured to be saved as fonts, not as paths.
        - If the specified subfolder doesn't exist, it is created.
        - If both `figurename` and `subfolderpath` are paths, `figurename` takes precedence,
        and `subfolderpath` is overridden.
        - A configuration file with the same base name plus '_conf.py' suffix is saved alongside the figure.
        """
        plt.rcParams["svg.fonttype"] = "none"  # Fonts as fonts and not as paths
        plt.rcParams["ps.fonttype"] = "type3"  # Fonts as fonts and not as paths

        # Convert inputs to Path objects
        figurename = Path(figurename)
        subfolderpath = Path(subfolderpath)

        # Extract path components if figurename includes a path
        if figurename.parent != Path("."):
            subfolderpath = figurename.parent
            figurename = figurename.name

        # Clean up format specification
        myformat = myformat.lstrip(".")

        # Handle empty figurename case early
        if not figurename:
            final_filename = f"MyFigure.{myformat}"
        else:
            # Build filename components
            base_name = Path(figurename).stem + suffix

            # Use existing extension or default to myformat
            extension = Path(figurename).suffix
            if not extension:
                extension = f".{myformat}"

            final_filename = f"{base_name}{extension}"

        # Construct full save path
        path = self.context.path
        project_conf_module_file_path = self.context.project_conf_module_file_path
        full_subfolderpath = Path.joinpath(path, subfolderpath)
        save_path = Path.joinpath(full_subfolderpath, final_filename)

        # Create directory if it doesn't exist
        if not full_subfolderpath.is_dir():
            full_subfolderpath.mkdir(parents=True, exist_ok=True)

        # Save the figure
        print(f"Saving figure to {save_path}")
        plt.savefig(
            save_path,
            dpi=None,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format=extension[1:],  # Remove leading dot from extension
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )

    # Fit visualization
    def show_temporal_filter_response(
        self, gc_list=None, n_samples=None, savefigname=None
    ):
        """
        Show temporal filter response for each unit.
        """
        exp_temp_filt = self.project_data.fit["exp_temp_filt"]
        xdata = exp_temp_filt["xdata"]
        xdata_finer = exp_temp_filt["xdata_finer"]
        title = exp_temp_filt["title"]

        # if n_samples is not None:
        if isinstance(gc_list, list):
            gc_list = [f"cell_ix_{ci}" for ci in gc_list]
        elif isinstance(n_samples, int):
            gc_list = [ci for ci in exp_temp_filt.keys() if ci.startswith("cell_ix_")]
            gc_list = np.random.choice(gc_list, n_samples, replace=False)
        else:
            raise ValueError("Either gc_list or n_samples must be provided.")

        fig, ax = plt.subplots(figsize=(8, 4))

        for this_cell_ix in gc_list:
            ydata = exp_temp_filt[f"{this_cell_ix}"]["ydata"]
            y_fit = exp_temp_filt[f"{this_cell_ix}"]["y_fit"]
            ax.scatter(xdata, ydata)
            ax.plot(
                xdata_finer,
                y_fit,
                c="grey",
            )

        n_units = len(gc_list)
        # ax.title(f"{title} ({n_units} units)")
        ax.set_title(f"{title} ({n_units} units)")

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_spatial_filter_response(
        self,
        spat_filt,
        n_samples=1,
        gc_list=None,
        com_data=None,
        title="",
        savefigname=None,
    ):
        """
        Display the spatial filter response of the selected units, along with the corresponding DoG models.

        Parameters
        ----------
        spat_filt : dict
            Dictionary containing spatial filter data.
            - 'filters': numpy.ndarray
            - 'x_grid', 'y_grid': numpy.ndarray
            - 'dog_model_type': str
            - 'num_pix_x', 'num_pix_y': int
            - Other keys starting with "cell_ix_" for individual unit data.
        n_samples : int, optional
            Number of units to sample. The default is 1. np.inf will display all units.
        gc_list : list of int, optional
            Indices of specific units to display. Overrides n_samples if provided.
        com_data : dict, optional
            Dictionary containing the centre of mass data for the rf centres.
        title : str, optional
            Title for the plot. Default is an empty string.
        savefigname : str, optional
            If provided, saves the plot to a file with this name.

        """
        filters = spat_filt["filters"]
        x_grid = spat_filt["x_grid"]
        y_grid = spat_filt["y_grid"]
        dog_model_type = spat_filt["dog_model_type"]
        pixel_array_shape_x = spat_filt["num_pix_x"]
        pixel_array_shape_y = spat_filt["num_pix_y"]

        # get cell_ixs
        cell_ixs_list = [ci for ci in spat_filt.keys() if ci.startswith("cell_ix_")]
        cell_ix = [int(ci.split("_")[-1]) for ci in cell_ixs_list]
        if gc_list is not None:
            # cell_ixs_list = [cell_ixs_list[i] for i in gc_list]
            cell_ixs_list = [
                name for ix, name in enumerate(cell_ixs_list) if cell_ix[ix] in gc_list
            ]
            n_samples = len(cell_ixs_list)
            if n_samples < len(gc_list):
                # Find missing unit indices
                cell_ix_selected = [int(ci.split("_")[-1]) for ci in cell_ixs_list]
                missing_cell_ixs = np.setdiff1d(gc_list, cell_ix_selected)
                print(f"Rejected unit indices: {missing_cell_ixs}")
        elif n_samples < len(cell_ixs_list):
            cell_ixs_list = np.random.choice(cell_ixs_list, n_samples, replace=False)
        elif n_samples == np.inf:
            n_samples = len(cell_ixs_list)

        # Create a single figure for all the samples
        fig, axes = plt.subplots(figsize=(8, 2 * n_samples), nrows=n_samples, ncols=2)
        if n_samples == 1:  # Ensure axes is a 2D array for consistency
            axes = np.array([axes])

        suptitle = f"{title}"
        if com_data is not None:
            suptitle = (
                f"{suptitle}, red dot is the centre of mass for rf centre (masked)"
            )

        imshow_cmap = "viridis"
        ellipse_edgecolor = "white"
        colorscale_min_max = [None, None]  # [-0.2, 0.8]

        for idx, this_cell_ix in enumerate(cell_ixs_list):
            this_cell_ix_numerical = int(this_cell_ix.split("_")[-1])

            # Add unit index text to the left side of each row
            axes[idx, 0].text(
                x_grid.min() - 5,  # Adjust the x-coordinate as needed
                (y_grid.min() + y_grid.max()) / 2,  # Vertical centering
                f"unit Index: {this_cell_ix_numerical}",
                verticalalignment="center",
                horizontalalignment="right",
            )

            # Get DoG model fit parameters to popt
            popt = filters[this_cell_ix_numerical, :]
            spatial_data_array = spat_filt[this_cell_ix]["spatial_data_array"]

            if com_data is not None:
                com_x = com_data["centre_of_mass_x"][this_cell_ix_numerical]
                com_y = com_data["centre_of_mass_y"][this_cell_ix_numerical]
                axes[idx, 0].plot(com_x, com_y, ".r")

            cen = axes[idx, 0].imshow(
                spatial_data_array,
                cmap=imshow_cmap,
                origin="lower",
                extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
                vmin=colorscale_min_max[0],
                vmax=colorscale_min_max[1],
            )
            fig.colorbar(cen, ax=axes[idx, 0])

            # Ellipses for DoG2D_fixed_surround. Circular params are mapped to ellipse_fixed params
            if dog_model_type == "ellipse_fixed":
                gc_img_fitted = self.DoG2D_fixed_surround((x_grid, y_grid), *popt)
                e1 = Ellipse(
                    (popt[np.array([1, 2])]),
                    popt[3],
                    popt[4],
                    angle=-popt[5] * 180 / np.pi,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                )
                e2 = Ellipse(
                    (popt[np.array([1, 2])]),
                    popt[7] * popt[3],
                    popt[7] * popt[4],
                    angle=-popt[5] * 180 / np.pi,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                    linestyle="--",
                )

            elif dog_model_type == "ellipse_independent":
                gc_img_fitted = self.DoG2D_independent_surround((x_grid, y_grid), *popt)
                e1 = Ellipse(
                    (popt[np.array([1, 2])]),
                    popt[3],
                    popt[4],
                    angle=-popt[5] * 180 / np.pi,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                )
                e2 = Ellipse(
                    (popt[np.array([7, 8])]),
                    popt[9],
                    popt[10],
                    angle=-popt[11] * 180 / np.pi,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                    linestyle="--",
                )
            elif dog_model_type == "circular":
                gc_img_fitted = self.DoG2D_circular((x_grid, y_grid), *popt)
                e1 = Ellipse(
                    (popt[np.array([1, 2])]),
                    popt[3],
                    popt[3],
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                )
                e2 = Ellipse(
                    (popt[np.array([1, 2])]),
                    popt[5],
                    popt[5],
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                    linestyle="--",
                )

            axes[idx, 0].add_artist(e1)
            axes[idx, 0].add_artist(e2)

            sur = axes[idx, 1].imshow(
                gc_img_fitted.reshape(pixel_array_shape_y, pixel_array_shape_x),
                cmap=imshow_cmap,
                origin="lower",
                extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
                vmin=colorscale_min_max[0],
                vmax=colorscale_min_max[1],
            )
            fig.colorbar(sur, ax=axes[idx, 1])

        plt.tight_layout()
        # plt.suptitle(suptitle, fontsize=10)
        plt.subplots_adjust(top=0.95)

        if savefigname:
            self._figsave(figurename=title + "_" + savefigname)

    # ConstructRetina visualization
    def _get_imgs(
        self,
        df,
        nsamples,
        exp_spat_filt,
        this_trial_idx,
    ):
        log_dir = df["logdir"][this_trial_idx]

        # Get folder name starting "checkpoint"
        checkpoint_folder_name = [f for f in os.listdir(log_dir) if "checkpoint" in f][
            0
        ]
        checkpoint_path = Path(log_dir) / checkpoint_folder_name / "model.pth"

        # Load the model
        model = torch.load(checkpoint_path)

        if hasattr(model, "test_data"):
            test_data = model.test_data[:nsamples, :, :, :]
        else:
            # Make a list of dict keys starting "cell_ix_" from exp_spat_filt dictionary
            keys = exp_spat_filt.keys()
            cell_ix_names_list = [key for key in keys if "cell_ix_" in key]
            # Make a numpy array of numbers following "cell_ix_"
            cell_ix_array = np.array(
                [int(key.split("cell_ix_")[1]) for key in cell_ix_names_list]
            )

            # Take first nsamples from cell_ix_array. They have to be constant,
            # because we are showing multiple sets of images on top of each other.
            samples = cell_ix_array[:nsamples]

            test_data = np.zeros(
                [
                    nsamples,
                    1,
                    exp_spat_filt["num_pix_y"],
                    exp_spat_filt["num_pix_x"],
                ]
            )
            for idx, this_sample in enumerate(samples):
                test_data[idx, 0, :, :] = exp_spat_filt[f"cell_ix_{this_sample}"][
                    "spatial_data_array"
                ]

        # Hack to reuse the AugmentedDataset._feature_scaling method. Scales to [0,1]
        test_data = AugmentedDataset._feature_scaling("", test_data)

        test_data = torch.from_numpy(test_data).float()
        img_size = model.decoder.unflatten.unflattened_size
        test_data = TF.resize(test_data, img_size[-2:], antialias=True)

        self.device = self.context.device
        samples = range(0, nsamples)

        model.eval()
        model.to(self.device)

        img = test_data.to(self.device)

        with torch.no_grad():
            rec_img = model(img)

        img = img.cpu().squeeze().numpy()
        rec_img = rec_img.cpu().squeeze().numpy()

        return img, rec_img, samples

    def _subplot_dependent_boxplots(self, axd, kw, df, dep_vars, config_vars_changed):
        """Boxplot dependent variables for one ray tune experiment"""

        # config_vars_changed list contain the varied columns in dataframe df
        # From config_vars_changed, 'config/model_id' contain the replications of the same model
        # Other config_vars_changed contain the models of interest
        # Make an seaborn boxplot for each model of interest
        config_vars_changed.remove("config/model_id")

        # If there are more than one config_vars_changed,
        # make a new dataframe column with the values of the config_vars_changed as strings
        if len(config_vars_changed) > 1:
            df["config_vars"] = (
                df[config_vars_changed].astype(str).agg(",".join, axis=1)
            )
            # config_vars_changed = ["config_vars"]
            config_vars_for_label = [
                col.removeprefix("config/") for col in config_vars_changed
            ]
            # Combine the string listed in config_vars_changed to one string
            config_vars_label = ",".join(config_vars_for_label)

        else:
            df["config_vars"] = df[config_vars_changed[0]]
            config_vars_label = [
                col.removeprefix("config/") for col in config_vars_changed
            ][0]

        # Make one subplot for each dependent variable
        # Plot labels only after the last subplot
        for idx, dep_var in enumerate(dep_vars):
            ax = axd[f"{kw}{idx}"]

            # Create the boxplot with seaborn
            ax_sns = sns.boxplot(
                x="config_vars", y=dep_var, data=df, ax=ax, whis=[0, 100]
            )
            # If any of the df["config_vars"] has length > 4, make x-label rotated 90 degrees
            if any(df["config_vars"].astype(str).str.len() > 4):
                ax_sns.set_xticklabels(ax.get_xticklabels(), rotation=90)

            # Set the title of the subplot to be the dependent variable name
            ax.set_title(dep_var)

            # Set y-axis label
            ax.set_ylabel("")

            # Set x-axis label
            if idx == 0:  # only the first subplot gets an x-axis label
                ax.set_xlabel(config_vars_label)
            else:
                ax.set_xlabel("")

    def _get_best_trials(self, df, dep_var, best_is, num_best_trials):
        """
        Get the indices of the best trials for a dependent variable.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the results of the hyperparameter search.
        dep_var : str
            Name of the dependent variable.
        best_is : str
            Whether the best trials are the ones with the highest or lowest values.
        num_best_trials : int
            Number of best trials to return. Overrides frac_best.

        Returns
        -------
        best_trials : list
            List of indices of the best trials.
        """

        # get array of values for this dependent variable
        dep_var_vals = df[dep_var].values

        # get the indices of the num_best_trials
        if best_is == "min":
            best_trials = np.argsort(dep_var_vals)[:num_best_trials]
        elif best_is == "max":
            best_trials = np.argsort(dep_var_vals)[-num_best_trials:]

        return best_trials, dep_var_vals

    def _subplot_dependent_variables(self, axd, kw, result_grid, dep_vars, best_trials):
        """Plot dependent variables as a function of epochs."""

        df = result_grid.get_dataframe()
        # Find all columns with string "config/"
        config_cols = [x for x in df.columns if "config/" in x]

        # From the config_cols, identify columns where there is more than one unique value
        # These are the columns which were varied in the search space
        varied_cols = []
        for col in config_cols:
            if len(df[col].unique()) > 1:
                varied_cols.append(col)

        # Drop the "config/" part from the column names
        varied_cols = [x.replace("config/", "") for x in varied_cols]

        # # remove "model_id" from the varied columns
        # varied_cols.remove("model_id")

        num_colors = len(best_trials)
        colors = plt.cm.get_cmap("tab20", num_colors).colors

        total_n_epochs = 0
        # Make one subplot for each dependent variable
        for idx, dep_var in enumerate(dep_vars):
            # Create a new plot for each label
            color_idx = 0
            ax = axd[f"{kw}{idx}"]

            for i, result in enumerate(result_grid):
                if i not in best_trials:
                    continue

                if idx == 0:
                    label = f"{dep_vars[color_idx]}: " + ",".join(
                        f"{x}={result.config[x]}" for x in varied_cols
                    )
                    legend = True
                    first_ax = ax

                else:
                    label = None
                    legend = False

                result.metrics_dataframe.plot(
                    "training_iteration",
                    dep_var,
                    ax=ax,
                    label=label,
                    color=colors[color_idx],
                    legend=legend,
                )

                if len(result.metrics_dataframe) > total_n_epochs:
                    total_n_epochs = len(result.metrics_dataframe)

                # At the end (+1) of the x-axis, add mean and SD of last 50 epochs as dot and vertical line, respectively
                last_50 = result.metrics_dataframe.tail(50)
                mean = last_50[dep_var].mean()
                std = last_50[dep_var].std()
                n_epochs = result.metrics_dataframe.tail(1)["training_iteration"]
                ax.plot(
                    n_epochs + n_epochs // 5,
                    mean,
                    "o",
                    color=colors[color_idx],
                )
                ax.plot(
                    [n_epochs + n_epochs // 5] * 2,
                    [mean - std, mean + std],
                    "-",
                    color=colors[color_idx],
                )

                color_idx += 1

            if idx == 0:
                ax.set_ylabel("Metrics")

            # Add legend and bring it to the front
            leg = first_ax.legend(
                loc="center left", bbox_to_anchor=((idx + 2.0), 0.5, 1.0, 0.2)
            )
            first_ax.set_zorder(1)

            # change the line width for the legend
            for line in leg.get_lines():
                line.set_linewidth(3.0)

            # Set x and y axis tick font size 8
            ax.tick_params(axis="both", which="major", labelsize=8)

            # Change legend font size to 8
            for text in leg.get_texts():
                text.set_fontsize(8)

            ax.grid(True)

            # set x axis labels off
            ax.set_xlabel("")
            # set x ticks off
            ax.set_xticks([])

        first_ax.set_title(
            f"Evolution for best trials (ad {total_n_epochs} epochs)\nDot and vertical line indicate mean and SD of last 50 epochs",
            loc="left",
        )

    def _subplot_img_recoimg(self, axd, kw, subidx, img, samples, title):
        """
        Plot sample images
        """
        for pos_idx, sample_idx in enumerate(samples):
            if subidx is None:
                ax = axd[f"{kw}{pos_idx}"]
            else:
                ax = axd[f"{kw}{subidx}{pos_idx}"]
            ax.imshow(img[sample_idx], cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if pos_idx == 0:
                # ax.set_title(title, fontsize=8, fontdict={'verticalalignment': 'baseline', 'horizontalalignment': 'left'})
                # Print title to the left of the first image. The coordinates are in axes coordinates
                ax.text(
                    -1.0,
                    0.5,
                    title,
                    fontsize=8,
                    fontdict={
                        "verticalalignment": "baseline",
                        "horizontalalignment": "left",
                    },
                    transform=ax.transAxes,
                )

    def _show_tune_depvar_evolution(self, result_grid, dep_vars, highlight_trial=None):
        """Plot results from ray tune"""

        df = result_grid.get_dataframe()
        # Find all columns with string "config/"
        config_cols = [x for x in df.columns if "config/" in x]

        # From the config_cols, identify columns where there is more than one unique value
        # These are the columns which were varied in the search space
        varied_cols = []
        for col in config_cols:
            if len(df[col].unique()) > 1:
                varied_cols.append(col)

        # Drop the "config/" part from the column names
        varied_cols = [x.replace("config/", "") for x in varied_cols]

        num_colors = len(result_grid.get_dataframe())
        if highlight_trial is None:
            colors = plt.cm.get_cmap("tab20", num_colors).colors
            highlight_idx = None
        else:
            [highlight_idx] = [
                idx
                for idx, r in enumerate(result_grid)
                if highlight_trial in r.metrics["trial_id"]
            ]
            # set all other colors low contrast gray, and the highlight color to red
            colors = np.array(
                ["gray" if idx != highlight_idx else "red" for idx in range(num_colors)]
            )

        # Make one subplot for each dependent variable
        nrows = 2
        ncols = len(dep_vars) // 2
        plt.figure(figsize=(ncols * 5, nrows * 5))

        for idx, dep_var in enumerate(dep_vars):
            # Create a new plot for each label
            color_idx = 0
            ax = plt.subplot(nrows, ncols, idx + 1)
            label = None

            for result in result_grid:
                # Too cluttered for a legend
                # if idx == 0 and highlight_idx is None:
                #     label = ",".join(f"{x}={result.config[x]}" for x in varied_cols)
                #     legend = True
                # else:
                #     legend = False

                ax_plot = result.metrics_dataframe.plot(
                    "training_iteration",
                    dep_var,
                    ax=ax,
                    label=label,
                    color=colors[color_idx],
                    legend=False,
                )

                # At the end (+1) of the x-axis, add mean and SD of last 50 epochs as dot and vertical line, respectively
                last_50 = result.metrics_dataframe.tail(50)
                mean = last_50[dep_var].mean()
                std = last_50[dep_var].std()
                n_epochs = result.metrics_dataframe.tail(1)["training_iteration"]
                ax.plot(
                    n_epochs + n_epochs // 5,
                    mean,
                    "o",
                    color=colors[color_idx],
                )
                ax.plot(
                    [n_epochs + n_epochs // 5] * 2,
                    [mean - std, mean + std],
                    "-",
                    color=colors[color_idx],
                )

                color_idx += 1
            ax.set_title(f"{dep_var}")
            ax.set_ylabel(dep_var)
            ax.grid(True)

    def show_gc_positions(self):
        """
        Show retina unit positions and receptive fields

        ConstructRetina call.
        """

        ecc_mm = self.construct_retina.gc_df["pos_ecc_mm"].to_numpy()
        pol_deg = self.construct_retina.gc_df["pos_polar_deg"].to_numpy()
        gc_density_params = self.construct_retina.gc_density_params

        # to cartesian
        xcoord, ycoord = self.pol2cart(ecc_mm, pol_deg)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(
            xcoord.flatten(),
            ycoord.flatten(),
            "b.",
            label=self.construct_retina.gc_type,
        )
        ax.axis("equal")
        ax.legend()
        ax.set_title("Cartesian retina")
        ax.set_xlabel("Eccentricity (mm)")
        ax.set_ylabel("Elevation (mm)")

    def boundary_polygon(self, ecc_lim_mm, polar_lim_deg, um_per_pix=None, sidelen=0):
        """
        Create a boundary polygon based on given eccentricity and polar angle limits.

        Parameters
        ----------
        ecc_lim_mm : np.ndarray
            An array representing the eccentricity limits in millimeters for
            left and right boundaries (shape: [2]).
        polar_lim_deg : np.ndarray
            An array representing the polar angle limits in degrees for
            bottom and top boundaries (shape: [2]).
        um_per_pix : float, optional
            Microns per pixel. If provided, boundary in pix space, if None, boundary in mm space.
        side_len : int, optional
            Side length of the square image in pixels. If provided, boundary is padded by one rf
            in each side.

        Returns
        -------
        boundary_polygon : np.ndarray
            Array of Cartesian coordinates forming the vertices of the boundary polygon.
        """

        n_points = 100

        # # Generate points for bottom and top polar angle limits
        # bottom_x, bottom_y = self.pol2cart(
        #     np.full(n_points, ecc_lim_mm[0]),
        #     np.linspace(polar_lim_deg[0], polar_lim_deg[1], n_points),
        # )
        # top_x, top_y = self.pol2cart(
        #     np.full(n_points, ecc_lim_mm[1]),
        #     np.linspace(polar_lim_deg[0], polar_lim_deg[1], n_points),
        # )

        # Generate points along the arcs for min and max eccentricities
        theta_range = np.linspace(polar_lim_deg[0], polar_lim_deg[1], n_points)
        min_ecc_x, min_ecc_y = self.pol2cart(
            np.full_like(theta_range, ecc_lim_mm[0]), theta_range
        )
        max_ecc_x, max_ecc_y = self.pol2cart(
            np.full_like(theta_range, ecc_lim_mm[1]), theta_range
        )

        if um_per_pix is not None:
            max_x_mm = np.max(max_ecc_x)
            min_x_mm = np.min(min_ecc_x)
            max_y_mm = np.max(max_ecc_y)
            min_y_mm = np.min(min_ecc_y)

            # Pad with one full rf in each side. This prevents need to cutting the
            # rf imgs at the borders later on
            pad_size_x_mm = sidelen * um_per_pix / 1000
            pad_size_y_mm = sidelen * um_per_pix / 1000

            min_x_mm = min_x_mm - pad_size_x_mm
            max_x_mm = max_x_mm + pad_size_x_mm
            min_y_mm = min_y_mm - pad_size_y_mm
            max_y_mm = max_y_mm + pad_size_y_mm

            min_ecc_y = (max_y_mm - min_ecc_y) * 1000 / um_per_pix
            max_ecc_y = (max_y_mm - max_ecc_y) * 1000 / um_per_pix
            min_ecc_x = (min_ecc_x - min_x_mm) * 1000 / um_per_pix
            max_ecc_x = (max_ecc_x - min_x_mm) * 1000 / um_per_pix

        # Combine them to form the vertices of the bounding polygon
        boundary_polygon = []

        # Add points from bottom arc
        for bx, by in zip(min_ecc_x, min_ecc_y):
            boundary_polygon.append((bx, by))

        # Add points from top arc (in reverse order)
        for tx, ty in reversed(list(zip(max_ecc_x, max_ecc_y))):
            boundary_polygon.append((tx, ty))

        return np.array(boundary_polygon)

    def visualize_mosaic(self, savefigname=None):
        """
        Visualize the mosaic of ganglion cells in retinal mm coordinates.

        This function plots the ganglion cells as ellipses on a Cartesian plane and adds
        a boundary polygon representing sector limits.

        Parameters
        ----------
        savefigname : str, optional
            The name of the file to save the figure. If None, the figure is not saved.
        """

        mosaic_file = self.context.retina_parameters["mosaic_file"]
        gc_df = self.data_io.get_data(mosaic_file)

        ecc_mm = gc_df["pos_ecc_mm"].to_numpy()
        pol_deg = gc_df["pos_polar_deg"].to_numpy()

        ecc_lim_deg = self.context.retina_parameters["ecc_limits_deg"]
        ecc_lim_mm = (
            np.array(ecc_lim_deg) / self.context.retina_parameters["deg_per_mm"]
        )
        pol_lim_deg = self.context.retina_parameters["pol_limits_deg"]
        boundary_polygon = self.boundary_polygon(ecc_lim_mm, pol_lim_deg)

        # Obtain mm values
        if self.context.retina_parameters["dog_model_type"] == "circular":
            semi_xc = gc_df["rad_c_mm"]
            semi_yc = gc_df["rad_c_mm"]
            angle_in_deg = np.zeros(len(gc_df))
        elif self.context.retina_parameters["dog_model_type"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            semi_xc = gc_df["semi_xc_mm"]
            semi_yc = gc_df["semi_yc_mm"]
            angle_in_deg = gc_df["orient_cen_rad"] * 180 / np.pi

        # to cartesian
        xcoord, ycoord = self.pol2cart(ecc_mm, pol_deg)

        fig, ax = plt.subplots(nrows=1, ncols=1)

        polygon = Polygon(boundary_polygon, closed=True, fill=None, edgecolor="r")
        ax.add_patch(polygon)

        ax.plot(
            xcoord.flatten(),
            ycoord.flatten(),
            "b.",
            label=self.context.retina_parameters["gc_type"],
        )
        # Ellipse parameters: Ellipse(xy, width, height, angle=0, **kwargs). Only possible one at the time, unfortunately.
        for index in np.arange(len(xcoord)):
            ellipse_center_x = xcoord[index]
            ellipse_center_y = ycoord[index]
            diameter_xc = semi_xc[index] * 2
            diameter_yc = semi_yc[index] * 2
            e1 = Ellipse(
                (ellipse_center_x, ellipse_center_y),
                diameter_xc,
                diameter_yc,
                angle=angle_in_deg[index],
                edgecolor="b",
                linewidth=0.5,
                fill=False,
            )
            ax.add_artist(e1)

        ax.axis("equal")
        ax.legend()
        ax.set_title("Cartesian retina")
        ax.set_xlabel("Eccentricity (mm)")
        ax.set_ylabel("Elevation (mm)")

        if savefigname:
            self._figsave(figurename=savefigname)

    def _delete_extra_axes(self, fig, axes, n_items, n_ax_rows, n_ax_cols):
        """
        If the number of distributions is less than n_ax_rows * n_ax_cols, remove the empty axes
        """
        if n_items < n_ax_rows * n_ax_cols:
            for idx in range(n_items, n_ax_rows * n_ax_cols):
                fig.delaxes(axes[idx])

    def show_distribution_statistics(
        self,
        statistics,
        distribution="spatial",
        correlation_reference=None,
        savefigname=None,
    ):
        """
        Show histograms of receptive field parameters, and correlation between receptive field parameters.
        ConstructRetina call.

        Parameters
        ----------
        correlation_reference : str, optional
            The name of the distribution to use as a reference for correlation.
            If None, no correlation is shown.
        savefigname : str, optional
            The name of the file to save the figure. If None, the figure is not saved.
        """

        include_multivariate = (statistics == "multivariate") & (
            self.context.retina_parameters["spatial_model_type"] == "DOG"
        )

        match distribution:
            case "spatial":
                distr_of_interest = [
                    "semi_xc_pix",
                    "semi_yc_pix",
                    "ampl_s",
                    "relative_surround_volume",
                    "orient_cen_rad",
                ]
                experimental_data = self.project_data.fit["spatial_data_and_model"][
                    "experimental_data"
                ]
                univariate_statistics = self.project_data.fit["spatial_data_and_model"][
                    "univariate_statistics"
                ]

                model_parameters = univariate_statistics["model_parameters"]
                model_fit_curves = univariate_statistics["model_fit_curves"]
                if include_multivariate:
                    multivariate_statistics = self.project_data.fit[
                        "spatial_data_and_model"
                    ]["multivariate_statistics"]
                    covariances_of_interest = self.project_data.construct_retina[
                        "spatial_covariances_of_interest"
                    ]

            case "temporal":
                distr_of_interest = ["n", "p1", "p2", "tau1", "tau2"]
                experimental_data = self.project_data.fit["temporal_data_and_model"][
                    "experimental_data"
                ]
                univariate_statistics = self.project_data.fit[
                    "temporal_data_and_model"
                ]["univariate_statistics"]
                model_parameters = univariate_statistics["model_parameters"]
                model_fit_curves = univariate_statistics["model_fit_curves"]
                if include_multivariate:
                    multivariate_statistics = self.project_data.fit[
                        "temporal_data_and_model"
                    ]["multivariate_statistics"]
                    covariances_of_interest = self.project_data.construct_retina[
                        "temporal_covariances_of_interest"
                    ]

            case "tonic_drive" | "tonic":
                distr_of_interest = ["tonic_drive"]
                experimental_data = self.project_data.fit["tonic_data_and_model"][
                    "experimental_data"
                ]
                model_parameters = self.project_data.fit["tonic_data_and_model"][
                    "model_parameters"
                ]
                model_fit_curves = self.project_data.fit["tonic_data_and_model"][
                    "model_fit_curves"
                ]

        x_model_fit, y_model_fit = model_fit_curves[0], model_fit_curves[1]

        distr_id_name = [
            (idx, key)
            for idx, key in enumerate(model_parameters.keys())
            if key in distr_of_interest
        ]

        dist_idx, distrs = zip(*distr_id_name)
        n_distributions = len(distrs)
        # plot the univariate distributions and fits.
        n_ax_cols = 3
        n_ax_rows = math.ceil(n_distributions / n_ax_cols)
        nbins = 20
        fig, axes = plt.subplots(n_ax_rows, n_ax_cols, figsize=(13, 4))

        fig.suptitle(
            f"Univariate statistics for {distribution} data",
            fontsize=16,
            fontweight="bold",
        )
        axes = axes.flatten()
        for idx, this_distr in enumerate(dist_idx):
            ax = axes[idx]
            _ax = ax.twinx()

            bin_values, _, _ = ax.hist(experimental_data[:, this_distr], bins=nbins)
            ax.set_ylabel("Count")
            median = np.median(experimental_data[:, this_distr])

            if model_fit_curves != None:  # Assumes tuple of arrays
                x_this_distr = x_model_fit[:, this_distr]
                y_this_distr = y_model_fit[:, this_distr]

                # Turn into probability density
                probability_distribution = (y_this_distr / y_this_distr.sum()) * nbins
                _ax.plot(
                    x_this_distr,
                    probability_distribution,
                    "r-",
                    linewidth=6,
                    alpha=0.6,
                )

                _ax.set_ylim([0, 1.1 * probability_distribution.max()])
                _ax.set_ylabel("Probability density")

                model_parameters[distrs[idx]]
                shape = model_parameters[distrs[idx]]["shape"]
                loc = model_parameters[distrs[idx]]["loc"]
                scale = model_parameters[distrs[idx]]["scale"]
                model_function = model_parameters[distrs[idx]]["distribution"]

                ax.annotate(
                    "shape = {0:.2f}\nloc = {1:.2f}\nscale = {2:.2f}\nmedian = {3:.2f}".format(
                        shape, loc, scale, median
                    ),
                    xy=(1, 1),  # Point at the right upper corner of the axis
                    xycoords="axes fraction",
                    xytext=(-10, -10),  # Offset from the corner, adjust as needed
                    textcoords="offset points",
                    horizontalalignment="right",  # Right align text
                    verticalalignment="top",  # Top align text
                )
                ax.set_title("{0} fit for {1}".format(model_function, distrs[idx]))

                # Rescale y axis if model fit goes high. Shows histogram better
                if y_model_fit[:, this_distr].max() > 1.5 * bin_values.max():
                    ax.set_ylim([ax.get_ylim()[0], 1.1 * bin_values.max()])

        self._delete_extra_axes(fig, axes, n_distributions, n_ax_rows, n_ax_cols)

        if savefigname:
            self._figsave(figurename=savefigname)

        # Check correlations
        if correlation_reference is not None:
            assert (
                correlation_reference in distrs
            ), f"correlation_reference must be None or a distribution: {','.join(map(str,distrs))}"

            distr_tuple_idx = distrs.index(correlation_reference)
            corr_ref_idx = dist_idx[distr_tuple_idx]

            fig2, axes2 = plt.subplots(n_ax_rows, n_ax_cols, figsize=(13, 4))
            axes2 = axes2.flatten()

            fig2.suptitle(
                f"Correlation between {correlation_reference} and other data",
                fontsize=16,
                fontweight="bold",
            )

            for idx, this_distr in enumerate(dist_idx):
                ax2 = axes2[idx]
                data_all_x = experimental_data[:, corr_ref_idx]
                data_all_y = experimental_data[:, this_distr]
                r, p = stats.pearsonr(data_all_x, data_all_y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    data_all_x, data_all_y
                )
                ax2.plot(data_all_x, data_all_y, ".")
                data_all_x.sort()
                ax2.plot(data_all_x, intercept + slope * data_all_x, "b-")
                ax2.annotate(
                    "\nr={0:.2g},\np={1:.2g}".format(r, p),
                    xy=(1, 1),
                    xycoords="axes fraction",
                    xytext=(-10, -10),
                    textcoords="offset points",
                    horizontalalignment="right",
                    verticalalignment="top",
                )
                ax2.set_title(
                    "Correlation between {0} and {1}".format(
                        distrs[distr_tuple_idx], distrs[idx]
                    )
                )

            self._delete_extra_axes(fig2, axes2, n_distributions, n_ax_rows, n_ax_cols)

        if savefigname:
            self._figsave(figurename=savefigname, suffix="_corr")

        if include_multivariate:

            x_model_fits = np.linspace(
                multivariate_statistics["means"]
                - 3 * multivariate_statistics["std_devs"],
                multivariate_statistics["means"]
                + 3 * multivariate_statistics["std_devs"],
                100,
            )
            y_model_fits = stats.norm.pdf(
                x_model_fits,
                multivariate_statistics["means"],
                multivariate_statistics["std_devs"],
            )
            fig3, axes3 = plt.subplots(n_ax_rows, n_ax_cols, figsize=(13, 4))

            fig3.suptitle(
                f"Multivariate statistics with Gaussian distributions",
                fontsize=16,
                fontweight="bold",
            )

            axes3 = axes3.flatten()
            for idx, this_distr in enumerate(dist_idx):
                ax3 = axes3[idx]
                _ax3 = ax3.twinx()
                data = experimental_data[:, this_distr]
                bin_values, _, _ = ax3.hist(data, bins=nbins)
                ax3.set_ylabel("Count")

                x_this_distr, y_this_distr = (
                    x_model_fits[:, this_distr],
                    y_model_fits[:, this_distr],
                )

                _ax3.plot(
                    x_this_distr,
                    y_this_distr,
                    "r-",
                    linewidth=6,
                    alpha=0.6,
                )

                _ax3.set_ylim([0, 1.1 * y_this_distr.max()])
                _ax3.set_ylabel("Probability density")

                mean = multivariate_statistics["means"][this_distr]
                std = multivariate_statistics["std_devs"][this_distr]
                ax3.annotate(
                    "mean = {0:.2f}\nstd = {1:.2f}".format(mean, std),
                    xy=(1, 1),  # Point at the right upper corner of the axis
                    xycoords="axes fraction",
                    xytext=(-10, -10),  # Offset from the corner, adjust as needed
                    textcoords="offset points",
                    horizontalalignment="right",  # Right align text
                    verticalalignment="top",  # Top align text
                )

                ax3.set_title(
                    "Gaussian fit for {0}".format(distrs[idx]),
                )

            if savefigname:
                self._figsave(figurename=savefigname, suffix="_multivariate")

            covariance_matrix = multivariate_statistics["covariance_matrix"]
            covariance_matrix_keys = multivariate_statistics["keys"]

            # Convert dict_keys to a list for indexing
            covariance_matrix_keys_list = list(covariance_matrix_keys)

            # Get indices of interest
            indices = [
                covariance_matrix_keys_list.index(key)
                for key in covariances_of_interest
            ]

            # Convert indices list to a NumPy array
            indices_array = np.array(indices)

            n_params = len(indices_array)
            # Use NumPy array indexing
            interesting_keys_list = np.array(covariance_matrix_keys_list)[indices_array]
            interesting_covariance_matrix = covariance_matrix[indices_array, :][
                :, indices_array
            ]

            fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

            ax4.imshow(interesting_covariance_matrix)
            # plot colorbar
            cbar = ax4.figure.colorbar(
                ax4.imshow(interesting_covariance_matrix), ax=ax4
            )
            cbar.set_label("Covariance")

            # assign param_distribution_dict.keys() as x and y axis tics
            ax4.set_xticks(range(n_params))
            ax4.set_xticklabels(interesting_keys_list, rotation=45)
            ax4.set_yticks(range(n_params))
            ax4.set_yticklabels(interesting_keys_list)
            ax4.set_title("Covariance matrix")

            if savefigname:
                self._figsave(figurename=savefigname, suffix="_covariance")

    def show_dendrite_diam_vs_ecc(self, log_x=False, log_y=False, savefigname=None):
        """
        Plot dendritic diameter as a function of retinal eccentricity with linear, quadratic, or cubic fitting.
        """
        dd_vs_ecc = self.project_data.construct_retina["dd_vs_ecc"]
        data_all_x = dd_vs_ecc["data_all_x"]
        data_all_y = dd_vs_ecc["data_all_y"]
        dd_DoG_x = dd_vs_ecc["dd_DoG_x"]
        dd_DoG_y = dd_vs_ecc["dd_DoG_y"]
        fit_parameters = dd_vs_ecc["fit_parameters"]
        dd_model_caption = dd_vs_ecc["dd_model_caption"]
        title = dd_vs_ecc["title"]

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(data_all_x, data_all_y, "b.", label="Data")
        ax.plot(dd_DoG_x, dd_DoG_y, "r.", label="DoG fit")

        ax.set_xlabel("Retinal eccentricity (mm)")
        ax.set_ylabel("Dendritic diameter (um)")
        ax.legend()

        if dd_model_caption:
            if self.context.retina_parameters["dd_regr_model"] == "linear":
                intercept = fit_parameters[1]
                slope = fit_parameters[0]
                ax.plot(data_all_x, intercept + slope * data_all_x, "k--")
                ax.annotate(
                    f"{dd_model_caption} : \ny={intercept:.1f} + {slope:.1f}x",
                    xycoords="axes fraction",
                    xy=(0.5, 0.15),
                    ha="left",
                    color="k",
                )

            elif self.context.retina_parameters["dd_regr_model"] == "quadratic":
                intercept = fit_parameters[2]
                slope = fit_parameters[1]
                square = fit_parameters[0]
                ax.plot(
                    data_all_x,
                    intercept + slope * data_all_x + square * data_all_x**2,
                    "k--",
                )
                ax.annotate(
                    f"{dd_model_caption}: \ny={intercept:.1f} + {slope:.1f}x + {square:.1f}x^2",
                    xycoords="axes fraction",
                    xy=(0.5, 0.15),
                    ha="left",
                    color="k",
                )

            elif self.context.retina_parameters["dd_regr_model"] == "cubic":
                intercept = fit_parameters[3]
                slope = fit_parameters[2]
                square = fit_parameters[1]
                cube = fit_parameters[0]
                ax.plot(
                    data_all_x,
                    intercept
                    + slope * data_all_x
                    + square * data_all_x**2
                    + cube * data_all_x**3,
                    "k--",
                )
                ax.annotate(
                    f"{dd_model_caption}: \ny={intercept:.1f} + {slope:.1f}x + {square:.1f}x^2 + {cube:.1f}x^3",
                    xycoords="axes fraction",
                    xy=(0.5, 0.15),
                    ha="left",
                    color="k",
                )

            elif self.context.retina_parameters["dd_regr_model"] == "exponential":
                constant = fit_parameters[0]
                lamda = fit_parameters[1]
                ax.plot(data_all_x, constant + np.exp(data_all_x / lamda), "k--")
                ax.annotate(
                    f"{dd_model_caption}: \ny={constant:.1f} + exp(x/{lamda:.1f})",
                    xycoords="axes fraction",
                    xy=(0.5, 0.15),
                    ha="left",
                    color="k",
                )

            elif self.context.retina_parameters["dd_regr_model"] == "powerlaw":
                # a = 10 ** fit_parameters[0]
                a = fit_parameters[0]
                b = fit_parameters[1]
                # Calculate the fitted values using the power law relationship
                fitted_y = a * np.power(data_all_x, b)
                ax.plot(data_all_x, fitted_y, "k--", label="Log-log fit")
                ax.annotate(
                    f"{dd_model_caption}: \nD={a:.2f} * E^{b:.2f}",
                    xycoords="axes fraction",
                    xy=(0.5, 0.15),
                    ha="left",
                    color="k",
                )
                if log_y:
                    ax.set_ylim([1, 1000])

        # Set x and y axis logarithmic, if called for
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        plt.title(title)

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_cone_noise_vs_freq(self, savefigname=None):
        """
        Plot cone noise as a function of temporal frequency using the model by Victor 1987 JPhysiol.
        """

        ret_file_npz = self.data_io.get_data(self.context.retina_parameters["ret_file"])
        noise_frequency_data = ret_file_npz["noise_frequency_data"]
        noise_power_data = ret_file_npz["noise_power_data"]
        cone_noise_parameters = ret_file_npz["cone_noise_parameters"]
        title = "cone_noise_vs_freq"

        cone_frequency_data = ret_file_npz["cone_frequency_data"]
        cone_power_data = ret_file_npz["cone_power_data"]
        cone_noise_power_fit = ret_file_npz["cone_noise_power_fit"]

        self.cone_interp_function = self.interpolate_data(
            cone_frequency_data, cone_power_data
        )
        self.cone_noise_wc = self.context.retina_parameters["cone_general_parameters"][
            "cone_noise_wc"
        ]

        # Separate functions for the three components of the cone noise
        L1_model = self.lorenzian_function(
            noise_frequency_data, cone_noise_parameters[1], self.cone_noise_wc[0]
        )
        L2_model = self.lorenzian_function(
            noise_frequency_data, cone_noise_parameters[2], self.cone_noise_wc[1]
        )
        scaled_cone_response = cone_noise_parameters[0] * self.cone_interp_function(
            noise_frequency_data
        )

        fig, ax = plt.subplots()
        ax.plot(noise_frequency_data, noise_power_data, "b.", label="Data")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Cone noise power (pA^2/Hz)")

        ax.plot(noise_frequency_data, cone_noise_power_fit, "k--", label="Model Fit")

        # subscript = "<sub>corner</sub>"
        subscript = "$_{corner}$"
        # Plot L1 model
        ax.plot(
            noise_frequency_data,
            L1_model,
            "r--",
            label=f"lorenzian with freq{subscript}={self.cone_noise_wc[0]}",
        )

        # Plot L2 model
        ax.plot(
            noise_frequency_data,
            L2_model,
            "g--",
            label=f"lorenzian with freq{subscript}={self.cone_noise_wc[1]}",
        )

        # Plot scaled cone response
        ax.plot(
            noise_frequency_data,
            scaled_cone_response,
            "m--",
            label="Scaled cone response",
        )

        ax.set_ylim([0.001, 0.3])

        ax.set_xscale("log")
        ax.set_yscale("log")

        plt.title(title)
        ax.legend()

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_exp_build_process(self, show_all_spatial_fits=False):
        """
        Visualize the stages of retina mosaic building process.
        """

        # If show_all_spatial_fits is true, show the spatial fits
        if show_all_spatial_fits is True:
            spat_filt = self.project_data.fit["exp_spat_filt"]
            self.show_spatial_filter_response(
                spat_filt,
                n_samples=np.inf,
                # title="Experimental",
                # pause_to_show=True,
            )
            return

        self.show_temporal_filter_response(n_samples=2)
        self.visualize_mosaic()
        self.show_dendrite_diam_vs_ecc()

    def show_experimental_data_DoG_fit(
        self, gc_list=None, n_samples=2, savefigname=None
    ):
        """
        Show the experimental and generated spatial receptive fields. The type of
        spatial model in use (VAE or other) determines what exactly is displayed.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to show. Default is 2.
        savefigname : str, optional
            Name of the file to save the figure. If None, the figure won't be saved.
        gc_list : list, optional
            List of specific samples to display. Overrides n_samples if provided.
        savefigname : str, optional
            The name of the file to save the figure. If None, the figure is not saved.
        """
        if self.context.retina_parameters["spatial_model_type"] == "DOG":
            spat_filt = self.project_data.fit["exp_spat_filt"]
            self.show_spatial_filter_response(
                spat_filt,
                n_samples=n_samples,
                gc_list=gc_list,
                title="Experimental",
                savefigname=savefigname,
            )
        elif self.context.retina_parameters["spatial_model_type"] == "VAE":
            gen_rfs = self.project_data.construct_retina["gen_rfs"]
            spat_filt = self.project_data.fit["gen_spat_filt"]
            self.show_spatial_filter_response(
                spat_filt,
                n_samples=n_samples,
                gc_list=gc_list,
                com_data=gen_rfs,
                title="Generated",
                savefigname=savefigname,
            )

    def show_latent_space_and_samples(self):
        """
        Plot the latent samples on top of the estimated kde, one sublot for each successive two dimensions of latent_dim
        """
        # Third-party
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        gen_latent_space = self.project_data.construct_retina["gen_latent_space"]

        latent_samples = gen_latent_space["samples"]
        latent_stats = gen_latent_space["data"]
        latent_dim = gen_latent_space["dim"]

        # Make a grid of subplots
        n_cols = 4
        n_rows = int(np.ceil(latent_dim / n_cols))
        if n_rows == 1:
            n_cols = latent_dim
        elif n_rows > 4:
            n_rows = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2 * n_rows))
        fig_suptitle_text = "Latent space and samples"
        axes = axes.flatten()

        # Plot the latent samples on top of the estimated kde
        for ax_idx, i in enumerate(range(0, latent_dim, 2)):
            if ax_idx > 15:
                fig_suptitle_text = (
                    "Latent space and samples (plotting only the first 32 dimensions)"
                )
                break

            # Get only two dimensions at a time
            values = latent_stats[:, [i, i + 1]].T
            # Evaluate the kde using only the same two dimensions
            # Both uniform and normal distr during learning is sampled
            # using gaussian kde estimate. The kde estimate is basically smooth histogram,
            # so it is not a problem that the data is not normal.
            kernel = stats.gaussian_kde(values)

            # Construct X and Y grids using the same two dimensions
            x = np.linspace(latent_stats[:, i].min(), latent_stats[:, i].max(), 100)
            y = np.linspace(
                latent_stats[:, i + 1].min(), latent_stats[:, i + 1].max(), 100
            )
            X, Y = np.meshgrid(x, y)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions).T, X.shape)

            # Plot the estimated kde and samples on top of it
            axes[ax_idx].contour(X, Y, Z, levels=10)
            axes[ax_idx].scatter(latent_samples[:, i], latent_samples[:, i + 1])
            # axes[ax_idx].scatter(latent_stats[:, i], latent_stats[:, i + 1])

            # Make marginal plots of the contours as contours and samples as histograms.
            # Place the marginal plots on the right and top of the main plot
            ax_marg_x = inset_axes(
                axes[ax_idx],
                width="100%",  # width  of parent_bbox width
                height="30%",  # height : 1 inch
                loc="upper right",
                # bbox_to_anchor=(1.05, 1.05),
                bbox_to_anchor=(0, 0.95, 1, 0.3),
                bbox_transform=axes[ax_idx].transAxes,
                borderpad=0,
            )
            ax_marg_y = inset_axes(
                axes[ax_idx],
                width="30%",  # width of parent_bbox width
                height="100%",  # height : 1 inch
                loc="lower left",
                # bbox_to_anchor=(-0.05, -0.05),
                bbox_to_anchor=(1, 0, 0.4, 1),
                bbox_transform=axes[ax_idx].transAxes,
                borderpad=0,
            )

            # Plot the marginal plots
            nx, bins, _ = ax_marg_x.hist(latent_samples[:, i], bins=20, density=True)
            ny, bins, _ = ax_marg_y.hist(
                latent_samples[:, i + 1],
                bins=20,
                density=True,
                orientation="horizontal",
            )

            # Plot the one-dimensional marginal shapes of the kde
            x_margin_contour = nx.max() * Z.mean(axis=0) / Z.mean(axis=0).max()
            y_margin_contour = ny.max() * Z.mean(axis=1) / Z.mean(axis=1).max()
            ax_marg_x.plot(x, x_margin_contour, color="r")
            ax_marg_y.plot(y_margin_contour, y, color="r")

            # Remove the ticks from the marginal plots
            ax_marg_x.tick_params(
                axis="both", which="both", bottom=False, top=False, labelbottom=False
            )
            ax_marg_y.tick_params(
                axis="both", which="both", left=False, right=False, labelleft=False
            )

            # Set the title of the main plot
            axes[ax_idx].set_title(f"Latent dims {i}, {i+1}")

        # plt.tight_layout()
        fig.suptitle(fig_suptitle_text)

    def show_gen_spat_post_hist(self):
        """
        Show the original experimental spatial receptive fields and
        the generated spatial receptive fields before and after postprocessing.
        """

        # Get the keys for the cell_ix arrays
        cell_key_list = [
            key
            for key in self.project_data.fit["exp_spat_filt"].keys()
            if "cell_ix" in key
        ]
        img_shape = self.project_data.fit["exp_spat_filt"]["cell_ix_0"][
            "spatial_data_array"
        ].shape
        # The shape of the array is N units, y_pixels, x_pixels
        img_exp = np.zeros([len(cell_key_list), img_shape[0], img_shape[1]])
        for i, cell_key in enumerate(cell_key_list):
            img_exp[i, :, :] = self.project_data.fit["exp_spat_filt"][cell_key][
                "spatial_data_array"
            ]

        img_pre = self.project_data.construct_retina["gen_spat_img"]["img_raw"]
        img_post = self.project_data.construct_retina["gen_spat_img"]["img_processed"]

        plt.subplot(1, 3, 1)
        plt.hist(img_exp.flatten(), bins=100)
        # plot median value as a vertical line
        plt.axvline(np.median(img_exp), color="r")
        plt.title(f"Experimental, median: {np.median(img_exp):.2f}")

        plt.subplot(1, 3, 2)
        plt.hist(img_pre.flatten(), bins=100)
        plt.axvline(np.median(img_pre), color="r")
        plt.title(f"Generated raw, median: {np.median(img_pre):.2f}")

        plt.subplot(1, 3, 3)
        plt.hist(img_post.flatten(), bins=100)
        plt.axvline(np.median(img_post), color="r")
        plt.title(f"Generated processed, median: {np.median(img_post):.2f}")

    def show_gen_exp_spatial_rf(
        self, ds_name="test_ds", n_samples=10, savefigname=None
    ):
        """
        Plot the outputs of the autoencoder.
        """
        retina_vae = self.project_data.construct_retina["retina_vae"]
        assert (
            self.context.retina_parameters["spatial_model_type"] == "VAE"
        ), "Only model type VAE is supported for show_gen_exp_spatial_rf()"
        if ds_name == "train_ds":
            ds = retina_vae.train_loader.dataset
        elif ds_name == "valid_ds":
            ds = retina_vae.val_loader.dataset
        else:
            ds = retina_vae.test_loader.dataset

        plt.figure(figsize=(16, 4.5))

        vae = retina_vae.vae
        vae.eval()
        len_ds = len(ds)
        samples = np.random.choice(len_ds, n_samples, replace=False)

        for pos_idx, sample_idx in enumerate(samples):
            ax = plt.subplot(2, len(samples), pos_idx + 1)
            img = ds[sample_idx][0].unsqueeze(0).to(retina_vae.device)
            plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.text(
                0.05,
                0.85,
                retina_vae.apricot_data.data_labels2names_dict[
                    ds[sample_idx][1].item()
                ],
                fontsize=10,
                color="red",
                transform=ax.transAxes,
            )
            if pos_idx == 0:
                ax.set_title("Original images")

            ax = plt.subplot(2, len(samples), len(samples) + pos_idx + 1)
            with torch.no_grad():
                rec_img = vae(img)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if pos_idx == 0:
                ax.set_title("Reconstructed images")

        # Set the whole figure title as ds_name
        plt.suptitle(ds_name)

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_latent_tsne_space(self):
        retina_vae = self.project_data.construct_retina["retina_vae"]
        train_df = retina_vae.get_encoded_samples(
            dataset=retina_vae.train_loader.dataset
        )
        valid_df = retina_vae.get_encoded_samples(dataset=retina_vae.val_loader.dataset)
        test_df = retina_vae.get_encoded_samples(dataset=retina_vae.test_loader.dataset)

        # Add a column to each df with the dataset name
        train_df["dataset"] = "train"
        valid_df["dataset"] = "valid"
        test_df["dataset"] = "test"

        # Concatenate the dfs
        encoded_samples = pd.concat([train_df, valid_df, test_df])

        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30)

        if encoded_samples.shape[0] < tsne.perplexity:
            tsne.perplexity = encoded_samples.shape[0] - 1

        tsne_results = tsne.fit_transform(
            encoded_samples.drop(["label", "dataset"], axis=1)
        )

        ax0 = sns.relplot(
            # data=tsne_results,
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            hue=encoded_samples.dataset.astype(str),
        )
        ax0.set(xlabel="tsne-2d-one", ylabel="tsne-2d-two")
        plt.title("TSNE plot of encoded samples")

    def show_ray_experiment(self, ray_exp, this_dep_var, highlight_trial=None):
        """
        Show the results of a ray experiment. If ray_exp is None, then
        the most recent experiment is shown.

        Parameters
        ----------
        ray_exp : str
            The name of the ray experiment
        this_dep_var : str
            The dependent variable to use for selecting the best trials
        highlight_trial : int
            The trial to highlight in the evolution plot
        """

        info_columns = ["trial_id", "iteration"]
        dep_vars = ["train_loss", "val_loss", "mse", "ssim", "kid_mean", "kid_std"]
        dep_vars_best = ["min", "min", "min", "max", "min", "min"]
        config_prefix = "config/"

        if ray_exp is None:
            most_recent = True
        else:
            most_recent = False

        result_grid = self.data_io.load_ray_results_grid(
            most_recent=most_recent, ray_exp=ray_exp
        )
        df = result_grid.get_dataframe()

        # Get configuration variables
        config_vars_all = [c for c in df.columns if config_prefix in c]

        # Drop columns that are constant in the experiment
        constant_cols = []
        for col in config_vars_all:
            if len(df[col].unique()) == 1:
                constant_cols.append(col)
        config_vars_changed = [
            col for col in config_vars_all if col not in constant_cols
        ]
        config_vars = [col.removeprefix(config_prefix) for col in config_vars_changed]

        # Remove all rows containing nan values in the dependent variables
        df = df.dropna(subset=dep_vars)

        # Collect basic data from the experiment
        n_sweeps = len(df)
        n_errors = result_grid.num_errors

        # Columns to describe the experiment
        exp_info_columns = info_columns + config_vars_changed + dep_vars
        print(df[exp_info_columns].describe())

        # Find the row indeces of the n best trials
        best_trials_across_dep_vars = []
        for dep_var, dep_var_best in zip(dep_vars, dep_vars_best):
            if dep_var_best == "min":
                best_trials_across_dep_vars.append(df[dep_var].idxmin())
            elif dep_var_best == "max":
                best_trials_across_dep_vars.append(df[dep_var].idxmax())
            if this_dep_var in dep_var:
                this_dep_var_best = dep_var_best

        df_filtered = df[exp_info_columns].loc[best_trials_across_dep_vars]
        # Print the exp_info_columns for the best trials
        print(f"Best trials: in order of {dep_vars=}")
        print(df_filtered)

        self._show_tune_depvar_evolution(
            result_grid, dep_vars, highlight_trial=highlight_trial
        )

        nrows = 9
        ncols = len(dep_vars)
        nsamples = 10

        layout = [
            ["dh0", "dh1", "dh2", "dh3", "dh4", "dh5", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            # ["dv0", "dv1", "dv2", "dv3", "dv4", "dv5", ".", ".", ".", "."],
            ["im0", "im1", "im2", "im3", "im4", "im5", "im6", "im7", "im8", "im9"],
            ["re0" + str(i) for i in range(10)],
            ["re1" + str(i) for i in range(10)],
            ["re2" + str(i) for i in range(10)],
            ["re3" + str(i) for i in range(10)],
            ["re4" + str(i) for i in range(10)],
        ]
        fig, axd = plt.subplot_mosaic(layout, figsize=(ncols * 2, nrows))

        # Fraction of best = 1/4
        frac_best = 0.25
        num_best_trials = int(len(df) * frac_best)

        self._subplot_dependent_boxplots(axd, "dh", df, dep_vars, config_vars_changed)

        exp_spat_filt = self.project_data.fit["exp_spat_filt"]

        num_best_trials = 5  # Also N reco img to show
        best_trials, dep_var_vals = self._get_best_trials(
            df, this_dep_var, this_dep_var_best, num_best_trials
        )

        img, rec_img, samples = self._get_imgs(
            df, nsamples, exp_spat_filt, best_trials[0]
        )

        title = f"Original \nimages"
        self._subplot_img_recoimg(axd, "im", None, img, samples, title)

        title = f"Reco for \n{this_dep_var} = \n{dep_var_vals[best_trials[0]]:.3f}, \nidx = {best_trials[0]}"
        self._subplot_img_recoimg(axd, "re", 0, rec_img, samples, title)

        for idx, this_trial in enumerate(best_trials[1:]):
            img, rec_img, samples = self._get_imgs(
                df, nsamples, exp_spat_filt, this_trial
            )

            title = f"Reco for \n{this_dep_var} = \n{dep_var_vals[this_trial]:.3f}, \nidx = {this_trial}"
            # Position idx in layout: enumerate starts at 0, so add 1.
            self._subplot_img_recoimg(axd, "re", idx + 1, rec_img, samples, title)

    def show_unit_placement_progress(
        self,
        original_positions,
        ecc_lim_mm=None,
        polar_lim_deg=None,
        positions=None,
        init=False,
        iteration=0,
        intersected_polygons=None,
        boundary_polygon=None,
        **fig_args,
    ):
        if init is True:
            # ecc_lim_mm = self.construct_retina.ecc_lim_mm
            # polar_lim_deg = self.construct_retina.polar_lim_deg

            # Init plotting
            # Convert self.polar_lim_deg to Cartesian coordinates
            pol2cart = self.pol2cart

            bottom_x, bottom_y = pol2cart(
                np.array([ecc_lim_mm[0], ecc_lim_mm[1]]),
                np.array([polar_lim_deg[0], polar_lim_deg[0]]),
            )
            top_x, top_y = pol2cart(
                np.array([ecc_lim_mm[0], ecc_lim_mm[1]]),
                np.array([polar_lim_deg[1], polar_lim_deg[1]]),
            )

            # Concatenate to get the corner points
            corners_x = np.concatenate([bottom_x, top_x])
            corners_y = np.concatenate([bottom_y, top_y])

            # Initialize the plot before the loop
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.scatter(corners_x, corners_y, color="black", marker="x", zorder=2)
            ax2.scatter(corners_x, corners_y, color="black", marker="x", zorder=2)

            ax1.set_aspect("equal")
            ax2.set_aspect("equal")
            scatter1 = ax1.scatter([], [], color="blue", marker="o")
            scatter2 = ax2.scatter([], [], color="red", marker="o")

            # Obtain corners based on original_positions
            min_x = np.min(original_positions[:, 0]) - 0.1
            max_x = np.max(original_positions[:, 0]) + 0.1
            min_y = np.min(original_positions[:, 1]) - 0.1
            max_y = np.max(original_positions[:, 1]) + 0.1

            # Set axis limits based on min and max values of original_positions
            ax1.set_xlim(min_x, max_x)
            ax1.set_ylim(min_y, max_y)
            ax2.set_xlim(min_x, max_x)
            ax2.set_ylim(min_y, max_y)

            # set horizontal (x) and vertical (y) units as mm for both plots
            ax1.set_xlabel("horizontal (mm)")
            ax1.set_ylabel("vertical (mm)")
            ax2.set_xlabel("horizontal (mm)")
            ax2.set_ylabel("vertical (mm)")

            plt.ion()  # Turn on interactive mode
            plt.show()

            return {
                "fig": fig,
                "ax1": ax1,
                "ax2": ax2,
                "scatter1": scatter1,
                "scatter2": scatter2,
                "intersected_voronoi_polygons": [],
            }

        else:
            fig = fig_args["fig"]
            ax1 = fig_args["ax1"]
            ax2 = fig_args["ax2"]
            scatter1 = fig_args["scatter1"]
            scatter2 = fig_args["scatter2"]

            scatter1.set_offsets(original_positions)
            ax1.set_title(f"orig pos")

            scatter2.set_offsets(positions)
            ax2.set_title(f"new pos iteration {iteration}")

            # Draw boundary polygon with no fill
            if boundary_polygon is not None:
                polygon = Polygon(
                    boundary_polygon, closed=True, fill=None, edgecolor="r"
                )
                ax2.add_patch(polygon)

            if intersected_polygons is not None:
                if fig_args["intersected_voronoi_polygons"] is not None:
                    # Remove old polygons
                    for poly in fig_args["intersected_voronoi_polygons"]:
                        poly.remove()
                    fig_args["intersected_voronoi_polygons"].clear()

                # Plot intersected Voronoi polygons
                for polygon in intersected_polygons:
                    poly = ax2.fill(*zip(*polygon), alpha=0.4, edgecolor="black")
                    fig_args["intersected_voronoi_polygons"].extend(poly)
            # Update the plot
            fig.canvas.flush_events()

    def _get_masked_data(self, data, boundary_polygon):
        # Create a mask for the polygon
        height, width = data.shape
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        polygon_path = Polygon(boundary_polygon).get_path()
        mask = polygon_path.contains_points(points).reshape(height, width)

        # Apply the mask to the data
        masked_data = np.where(mask, data, np.nan)

        return masked_data

    def _get_masked_stats(self, data, boundary_polygon):
        """
        Calculate statistics for the values inside the boundary_polygon.
        """
        # Mask the data
        masked_data = self._get_masked_data(data, boundary_polygon)

        # Calculate statistics only for the values inside the polygon
        max_val = np.nanmax(masked_data)
        min_val = np.nanmin(masked_data)
        mean_val = np.nanmean(masked_data)
        sd_val = np.nanstd(masked_data)

        return max_val, min_val, mean_val, sd_val

    def show_repulsion_progress(
        self,
        reference_retina,
        center_mask,
        ecc_lim_mm=None,
        polar_lim_deg=None,
        new_retina=None,
        stage="",
        iteration=0,
        um_per_pix=None,
        sidelen=0,
        savefigname=None,
        **fig_args,
    ):
        if stage == "init":
            boundary_polygon = self.boundary_polygon(
                ecc_lim_mm, polar_lim_deg, um_per_pix=um_per_pix, sidelen=sidelen
            )

            # Initialize the plot before the loop
            fig, ax = plt.subplots(2, 3, figsize=(10, 8))

            ax[0, 0].set_aspect("equal")
            ax[0, 1].set_aspect("equal")
            ax[0, 2].set_aspect("equal")

            plt.ion()  # Turn on interactive mode
            plt.show()

            return {
                "fig": fig,
                "ax": ax,
                "boundary_polygon": boundary_polygon,
            }

        elif stage in ["update", "final"]:
            fig = fig_args["fig"]
            ax = fig_args["ax"]
            boundary_polygon = fig_args["boundary_polygon"]
            polygon1 = Polygon(boundary_polygon, closed=True, fill=None, edgecolor="r")
            polygon2 = Polygon(boundary_polygon, closed=True, fill=None, edgecolor="r")
            polygon3 = Polygon(boundary_polygon, closed=True, fill=None, edgecolor="r")

            # Set new data and redraw
            ax[0, 0].clear()
            ax[0, 0].add_patch(polygon1)
            ax[0, 0].imshow(reference_retina)
            max_val, min_val, mean_val, sd_val = self._get_masked_stats(
                reference_retina, boundary_polygon
            )
            ax[0, 0].set_title(
                f"original retina\nmax = {max_val:.2f}\nmin = {min_val:.2f}\nmean = {mean_val:.2f}\nsd = {sd_val:.2f}"
            )
            ax[0, 0].set_xlabel("")
            ax[0, 0].set_ylabel("")
            ax[0, 0].set_xticks([])
            ax[0, 0].set_yticks([])

            if stage == "final":
                masked_data = self._get_masked_data(reference_retina, boundary_polygon)
                ax[1, 0].hist(masked_data.flatten(), bins=100)
                ax[1, 0].set_title("Histogram of original retina")
                # Colorbar
                divider = make_axes_locatable(ax[0, 0])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(ax[0, 0].imshow(reference_retina), cax=cax)

            # Set new data and redraw
            ax[0, 1].clear()
            ax[0, 1].add_patch(polygon2)
            ax[0, 1].imshow(center_mask)
            max_val, min_val, mean_val, sd_val = self._get_masked_stats(
                center_mask, boundary_polygon
            )
            ax[0, 1].set_title(
                f"center mask iteration {iteration}\nmax = {max_val:.2f}\nmin = {min_val:.2f}\nmean = {mean_val:.2f}\nsd = {sd_val:.2f}"
            )
            ax[0, 1].set_xlabel("")
            ax[0, 1].set_ylabel("")
            ax[0, 1].set_xticks([])
            ax[0, 1].set_yticks([])
            if stage == "final":
                masked_data = self._get_masked_data(center_mask, boundary_polygon)
                ax[1, 1].hist(masked_data.flatten(), bins=100)
                ax[1, 1].set_title("Histogram of center mask")
                # Colorbar
                divider = make_axes_locatable(ax[0, 1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(ax[0, 1].imshow(center_mask), cax=cax)

            ax[0, 2].clear()
            ax[0, 2].add_patch(polygon3)
            ax[0, 2].imshow(new_retina)
            max_val, min_val, mean_val, sd_val = self._get_masked_stats(
                new_retina, boundary_polygon
            )
            ax3_title = f"new retina iteration {iteration}\nmax = {max_val:.2f}\nmin = {min_val:.2f}\nmean = {mean_val:.2f}\nsd = {sd_val:.2f}"
            ax[0, 2].set_xlabel("")
            ax[0, 2].set_ylabel("")
            ax[0, 2].set_xticks([])
            ax[0, 2].set_yticks([])
            ax[0, 2].set_title(ax3_title)
            if stage == "final":
                masked_data = self._get_masked_data(new_retina, boundary_polygon)
                ax[1, 2].hist(masked_data.flatten(), bins=100)
                ax[1, 2].set_title("Histogram of new retina")
                # Colorbar
                divider = make_axes_locatable(ax[0, 2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(ax[0, 2].imshow(new_retina), cax=cax)

            if "additional_points" in fig_args:
                points = fig_args["additional_points"]
                unit_idx = fig_args["unit_idx"]
                ax[0, 2].plot(points[0], points[1], ".r")
                ax3_title = (
                    ax3_title + f"\nPoint is the center of mass for unit {unit_idx}"
                )
                ax[0, 2].set_title(ax3_title)

            # Redraw and pause
            fig.canvas.draw()
            plt.pause(0.001)

            if stage == "final":
                if savefigname:
                    self._figsave(figurename=savefigname)

    def show_cones_linked_to_gc(self, gc_list=None, n_samples=None, savefigname=None):
        """
        Visualize a ganglion cell and its connected cones.
        """

        gc_df = self.data_io.get_data(self.context.retina_parameters["mosaic_file"])
        gc_npz = self.data_io.get_data(
            self.context.retina_parameters["spatial_rfs_file"]
        )

        x_mm, y_mm = self.pol2cart(
            gc_df[["pos_ecc_mm"]].values, gc_df[["pos_polar_deg"]].values
        )
        gc_pos_mm = np.column_stack((x_mm, y_mm))
        X_grid_cen_mm = gc_npz["X_grid_cen_mm"]
        Y_grid_cen_mm = gc_npz["Y_grid_cen_mm"]
        gc_img_mask = gc_npz["gc_img_mask"]

        ret_npz = self.data_io.get_data(self.context.retina_parameters["ret_file"])
        weights = ret_npz["cones_to_gcs_weights"]
        cone_positions = ret_npz["cone_optimized_pos_mm"]

        if isinstance(gc_list, list):
            pass  # gc_list supercedes n_samples
        elif isinstance(n_samples, int):
            gc_list = np.random.choice(range(len(gc_df)), n_samples, replace=False)
        else:
            raise ValueError("Either gc_list or n_samples must be provided.")

        fig, ax = plt.subplots(1, len(gc_list), figsize=(12, 10))
        if len(gc_list) == 1:
            ax = [ax]

        for idx, this_sample in enumerate(gc_list):
            gc_position = gc_pos_mm[this_sample, :]
            if self.context.retina_parameters["dog_model_type"] == "circular":
                DoG_patch = Circle(
                    xy=gc_position,
                    radius=gc_df.loc[this_sample, "rad_c_mm"],
                    edgecolor="g",
                    facecolor="none",
                )
            elif self.context.retina_parameters["dog_model_type"] in [
                "ellipse_independent",
                "ellipse_fixed",
            ]:

                # Create ellipse patch for visualizing the RF
                DoG_patch = Ellipse(
                    xy=gc_position,
                    width=2 * gc_df.loc[this_sample, "semi_xc_mm"],
                    height=2 * gc_df.loc[this_sample, "semi_yc_mm"],
                    angle=gc_df.loc[this_sample, "orient_cen_rad"] * 180 / np.pi,
                    edgecolor="g",
                    facecolor="none",
                )
            ax[idx].scatter(*gc_position, color="red", label="ganglion cell")

            # Plot each cone with alpha based on connection probability
            connection_probs = weights[:, this_sample]
            n_connected = np.sum(connection_probs > 0)

            # Normalize the connection probabilities to the range [0, 1]
            norm = mcolors.Normalize(vmin=0, vmax=1)
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "white_blue", [(0, "white"), (1, "blue")]
            )

            # Map the probabilities to colors
            colors = cmap(norm(connection_probs))

            # Plot the cones with colors representing the connection probabilities
            ax[idx].scatter(
                cone_positions[:, 0], cone_positions[:, 1], c=colors, alpha=0.6
            )

            mask = gc_img_mask[this_sample, ...]
            x_mm = X_grid_cen_mm[this_sample, ...] * mask
            y_mm = Y_grid_cen_mm[this_sample, ...] * mask
            x_mm = x_mm[x_mm != 0]
            y_mm = y_mm[y_mm != 0]

            ax[idx].plot(x_mm, y_mm, ".g", label="RF center pixel midpoints")
            ax[idx].add_patch(DoG_patch)
            ax[idx].set_xlabel("X Position (mm)")
            ax[idx].set_ylabel("Y Position (mm)")
            ax[idx].set_title(
                f"ganglion cell {this_sample} and Connected {n_connected} cones"
            )
            ax[idx].legend()
            ax[idx].set_aspect("equal", adjustable="box")

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_bipolars_linked_to_gc(
        self, gc_list=None, n_samples=None, savefigname=None
    ):
        """
        Visualize a ganglion cell and its connected bipolars.
        """

        gc_df = self.data_io.get_data(self.context.retina_parameters["mosaic_file"])
        gc_npz = self.data_io.get_data(
            self.context.retina_parameters["spatial_rfs_file"]
        )

        x_mm, y_mm = self.pol2cart(
            gc_df[["pos_ecc_mm"]].values, gc_df[["pos_polar_deg"]].values
        )
        gc_pos_mm = np.column_stack((x_mm, y_mm))
        X_grid_cen_mm = gc_npz["X_grid_cen_mm"]
        Y_grid_cen_mm = gc_npz["Y_grid_cen_mm"]
        gc_img_mask = gc_npz["gc_img_mask"]

        ret_npz = self.data_io.get_data(self.context.retina_parameters["ret_file"])

        weights = ret_npz["bipolar_to_gcs_cen_weights"]
        bipolar_positions = ret_npz["bipolar_optimized_pos_mm"]

        if isinstance(gc_list, list):
            pass  # gc_list supercedes n_samples
        elif isinstance(n_samples, int):
            gc_list = np.random.choice(range(len(gc_df)), n_samples, replace=False)
        else:
            raise ValueError("Either gc_list or n_samples must be provided.")

        fig, ax = plt.subplots(1, len(gc_list), figsize=(12, 10))
        if len(gc_list) == 1:
            ax = [ax]

        for idx, this_sample in enumerate(gc_list):
            gc_position = gc_pos_mm[this_sample, :]
            if self.context.retina_parameters["dog_model_type"] == "circular":
                DoG_patch = Circle(
                    xy=gc_position,
                    radius=gc_df.loc[this_sample, "rad_c_mm"],
                    edgecolor="g",
                    facecolor="none",
                )
            elif self.context.retina_parameters["dog_model_type"] in [
                "ellipse_independent",
                "ellipse_fixed",
            ]:
                # Create ellipse patch for visualizing the RF
                DoG_patch = Ellipse(
                    xy=gc_position,
                    width=2 * gc_df.loc[this_sample, "semi_xc_mm"],
                    height=2 * gc_df.loc[this_sample, "semi_yc_mm"],
                    angle=gc_df.loc[this_sample, "orient_cen_rad"] * 180 / np.pi,
                    edgecolor="g",
                    facecolor="none",
                )
            ax[idx].scatter(*gc_position, color="red", label="ganglion cell")

            # Plot each bipolar with alpha based on connection normalized probability
            connection_probs = weights[:, this_sample] / weights[:, this_sample].max()
            n_connected = np.sum(connection_probs > 0)

            for bipolar_pos, prob in zip(bipolar_positions, connection_probs):
                ax[idx].scatter(*bipolar_pos, alpha=prob, color="blue")

            mask = gc_img_mask[this_sample, ...]
            x_mm = X_grid_cen_mm[this_sample, ...] * mask
            y_mm = Y_grid_cen_mm[this_sample, ...] * mask
            x_mm = x_mm[x_mm != 0]
            y_mm = y_mm[y_mm != 0]

            ax[idx].plot(x_mm, y_mm, ".g", label="RF center pixel midpoints")

            # Add DoG_patch to the plot
            ax[idx].add_patch(DoG_patch)

            ax[idx].set_xlabel("X Position (mm)")
            ax[idx].set_ylabel("Y Position (mm)")
            ax[idx].set_title(
                f"Ganglion cell {this_sample} and connected {n_connected} bipolars, relative conn prob"
            )
            ax[idx].legend()
            # Set equal aspect ratio
            ax[idx].set_aspect("equal", adjustable="box")

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_cones_linked_to_bipolars(
        self, bipo_list=None, n_samples=None, savefigname=None
    ):
        """
        Visualize a ganglion cell and its connected bipolars.

        Parameters
        ----------
        bipo_list : list, optional
            List of bipolar indices to show. This supercedes n_samples.
        n_samples : int, optional
            Number of bipolar cells to show.
        savefigname : str, optional
            If provided, the figure will be saved with this filename.
        """

        ret_npz = self.data_io.get_data(self.context.retina_parameters["ret_file"])

        cones_to_bipolars_center_weights = ret_npz["cones_to_bipolars_center_weights"]
        cones_to_bipolars_surround_weights = ret_npz[
            "cones_to_bipolars_surround_weights"
        ]
        weights = cones_to_bipolars_center_weights - cones_to_bipolars_surround_weights
        cone_positions = ret_npz["cone_optimized_pos_mm"]
        bipolar_positions = ret_npz["bipolar_optimized_pos_mm"]

        if isinstance(bipo_list, list):
            pass  # bipo_list supercedes n_samples
        elif isinstance(n_samples, int):
            bipo_list = np.random.choice(
                range(bipolar_positions.shape[0]), n_samples, replace=False
            )
        else:
            raise ValueError("Either bipo_list or n_samples must be provided.")

        fig, ax = plt.subplots(1, len(bipo_list), figsize=(12, 10))
        cmap = cm.coolwarm
        divnorm = colors.TwoSlopeNorm(vmin=weights.min(), vcenter=0, vmax=weights.max())
        if len(bipo_list) == 1:
            ax = [ax]

        for idx, this_sample in enumerate(bipo_list):
            this_bipo_pos = bipolar_positions[this_sample, :]

            ax[idx].scatter(*this_bipo_pos, color="black", label="bipolar cell")

            # Plot each bipolar with alpha based on connection probability
            connection_probs = weights[:, this_sample]
            n_connected = np.sum(connection_probs > 0)

            # Scatter plot
            sc = ax[idx].scatter(
                cone_positions[:, 0],  # x positions
                cone_positions[:, 1],  # y positions
                c=connection_probs,  # Color values
                cmap=cmap,  # Colormap
                norm=divnorm,  # Normalization centered at 0
            )

            ax[idx].set_xlabel("X Position (mm)")
            ax[idx].set_ylabel("Y Position (mm)")
            ax[idx].set_title(f"Bipolar {this_sample} sums {n_connected} cones")
            ax[idx].legend()
            # Set equal aspect ratio
            ax[idx].set_aspect("equal", adjustable="box")

        # Mark weights.min() as the tick to min colorbar value
        cbar = fig.colorbar(sc, ax=ax, extend="both")
        cbar.set_ticks([weights.min(), 0, weights.max()])
        cbar.set_ticklabels([f"{weights.min():.2f}", "0", f"{weights.max():.2f}"])

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_connection_histograms(self, weight_cutoff=0.001, savefigname=None):
        """
        Display histograms and heatmaps of retinal connection weights.

        Parameters
        ----------
        weight_cutoff : float, optional
            Weights below this value will be considered zero.
        savefigname : str, optional
            If provided, the figure will be saved with this filename.
        """
        ret_data = self.data_io.get_data(self.context.retina_parameters["ret_file"])

        weights = {
            "cones_to_bipolars_center": ret_data["cones_to_bipolars_center_weights"],
            "cones_to_bipolars_surround": ret_data[
                "cones_to_bipolars_surround_weights"
            ],
            "bipolar_to_gcs": ret_data["bipolar_to_gcs_cen_weights"],
            "cones_to_gcs": ret_data["cones_to_gcs_weights"],
        }

        flat_weights = {key: w.flatten() for key, w in weights.items()}
        non_zero_idxs = {
            key: np.where(w > weight_cutoff)[0] for key, w in flat_weights.items()
        }
        zero_counts = {
            key: len(w) - len(non_zero_idxs[key]) for key, w in flat_weights.items()
        }
        zero_annots = {
            key: f"Weights < {weight_cutoff} ({zero_counts[key]/len(flat_weights[key])*100:.1f}%) not shown"
            for key in flat_weights
        }

        fig, ax = plt.subplots(2, 4, figsize=(20, 10))

        for i, (key, w) in enumerate(flat_weights.items()):
            ax[0, i].hist(w[non_zero_idxs[key]], bins=100)
            ax[0, i].set_title(key.replace("_", " "))
            ax[0, i].text(
                0.1, 0.9, zero_annots[key], transform=ax[0, i].transAxes, fontsize=10
            )

        for i, (key, w) in enumerate(weights.items()):
            ax[1, i].imshow(w, aspect="auto", interpolation="none")

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_fan_in_out_distributions(self, weight_cutoff=0.0, savefigname=None):
        """
        Display fan-in and fan-out distributions for retinal connection weights.

        Parameters
        ----------
        weight_cutoff : float, optional
            Weights below this value will be considered zero.
        savefigname : str, optional
            If provided, the figure will be saved with this filename.
        """
        ret_data = self.data_io.get_data(self.context.retina_parameters["ret_file"])

        weights = {
            "cones_to_bipolars_center": ret_data["cones_to_bipolars_center_weights"],
            "cones_to_bipolars_surround": ret_data[
                "cones_to_bipolars_surround_weights"
            ],
            "bipolar_to_gcs": ret_data["bipolar_to_gcs_cen_weights"],
            "cones_to_gcs": ret_data["cones_to_gcs_weights"],
        }

        fan_in = {key: np.sum(w > weight_cutoff, axis=0) for key, w in weights.items()}
        fan_out = {key: np.sum(w > weight_cutoff, axis=1) for key, w in weights.items()}

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))

        for i, (key, w) in enumerate(weights.items()):
            # Fan-In histograms and violin plots
            sns.histplot(fan_in[key], bins=30, kde=False, ax=axes[i, 0])
            axes[i, 0].set_title(f"{key.replace('_', ' ')} Fan-In Histogram")
            mean_fan_in = np.mean(fan_in[key])
            range_fan_in = (np.min(fan_in[key]), np.max(fan_in[key]))
            axes[i, 0].annotate(
                f"Mean: {mean_fan_in:.2f}\nRange: {range_fan_in[0]} - {range_fan_in[1]}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                fontsize=10,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
            )
            sns.violinplot(y=fan_in[key], ax=axes[i, 1])
            axes[i, 1].set_title(f"{key.replace('_', ' ')} Fan-In Violin Plot")

            # Fan-Out histograms and violin plots
            sns.histplot(fan_out[key], bins=30, kde=False, ax=axes[i, 2])
            axes[i, 2].set_title(f"{key.replace('_', ' ')} Fan-Out Histogram")
            mean_fan_out = np.mean(fan_out[key])
            range_fan_out = (np.min(fan_out[key]), np.max(fan_out[key]))
            axes[i, 2].annotate(
                f"Mean: {mean_fan_out:.2f}\nRange: {range_fan_out[0]} - {range_fan_out[1]}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                fontsize=10,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
            )
            sns.violinplot(y=fan_out[key], ax=axes[i, 3])
            axes[i, 3].set_title(f"{key.replace('_', ' ')} Fan-Out Violin Plot")

        plt.tight_layout()

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_DoG_img_grid(self, gc_list=None, n_samples=None, savefigname=None):
        """
        Visualize a ganglion cell image, DoG fit and center grid points.
        """

        gc_df = self.data_io.get_data(self.context.retina_parameters["mosaic_file"])
        gc_npz = self.data_io.get_data(
            self.context.retina_parameters["spatial_rfs_file"]
        )

        x_mm, y_mm = self.pol2cart(
            gc_df[["pos_ecc_mm"]].values, gc_df[["pos_polar_deg"]].values
        )
        gc_pos_mm = np.column_stack((x_mm, y_mm))
        X_grid_cen_mm = gc_npz["X_grid_cen_mm"]
        Y_grid_cen_mm = gc_npz["Y_grid_cen_mm"]
        gc_img_mask = gc_npz["gc_img_mask"]
        gc_img = gc_npz["gc_img"]

        gc_df = self.data_io.get_data(self.context.retina_parameters["mosaic_file"])
        half_pix_mm = (gc_npz["um_per_pix"] / 1000) / 2

        if isinstance(gc_list, list):
            pass  # gc_list supercedes n_samples
        elif isinstance(n_samples, int):
            gc_list = np.random.choice(range(len(gc_df)), n_samples, replace=False)
        else:
            raise ValueError("Either gc_list or n_samples must be provided.")

        fig, ax = plt.subplots(1, len(gc_list), figsize=(12, 10))
        if len(gc_list) == 1:
            ax = [ax]

        for idx, this_sample in enumerate(gc_list):
            # Plot each rf image

            # extent (left, right, bottom, top)
            left = X_grid_cen_mm[this_sample, ...].min() - half_pix_mm
            right = X_grid_cen_mm[this_sample, ...].max() + half_pix_mm
            bottom = Y_grid_cen_mm[this_sample, ...].min() - half_pix_mm
            top = Y_grid_cen_mm[this_sample, ...].max() + half_pix_mm
            extent = (left, right, bottom, top)
            ax[idx].imshow(gc_img[this_sample, ...], extent=extent)

            # Center grid points
            mask = gc_img_mask[this_sample, ...]
            x_mm = X_grid_cen_mm[this_sample, ...] * mask
            y_mm = Y_grid_cen_mm[this_sample, ...] * mask
            x_mm = x_mm[x_mm != 0]
            y_mm = y_mm[y_mm != 0]
            ax[idx].plot(x_mm, y_mm, ".g", label="center mask")

            # Create circle/ellipse patch for visualizing the RF
            gc_position = gc_pos_mm[this_sample, :]
            if self.context.retina_parameters["dog_model_type"] == "circular":
                DoG_patch = Circle(
                    xy=gc_position,
                    radius=gc_df.loc[this_sample, "rad_c_mm"],
                    edgecolor="g",
                    facecolor="none",
                )
            elif self.context.retina_parameters["dog_model_type"] in [
                "ellipse_independent",
                "ellipse_fixed",
            ]:
                DoG_patch = Ellipse(
                    xy=gc_position,
                    width=2 * gc_df.loc[this_sample, "semi_xc_mm"],
                    height=2 * gc_df.loc[this_sample, "semi_yc_mm"],
                    angle=gc_df.loc[this_sample, "orient_cen_rad"] * 180 / np.pi,
                    edgecolor="g",
                    facecolor="none",
                )

            # Center point
            ax[idx].scatter(*gc_position, color="red", label="DoG midpoint")

            # Add DoG_patch to the plot
            ax[idx].add_patch(DoG_patch)

            ax[idx].set_xlabel("X Position (mm)")
            ax[idx].set_ylabel("Y Position (mm)")
            ax[idx].set_title(f"ganglion cell {this_sample}")
            ax[idx].legend()
            # Set equal aspect ratio
            ax[idx].set_aspect("equal", adjustable="box")

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_cell_density_vs_ecc(
        self,
        unit_type="gc",
        show_control_data=False,
        savefigname=None,
    ):
        cell_density_dict = self.project_data.construct_retina[f"{unit_type}_n_vs_ecc"]
        cell_eccentricity = cell_density_dict["cell_eccentricity"]
        cell_density = cell_density_dict["cell_density"]
        this_function = cell_density_dict["function"]
        fit_parameters = cell_density_dict["fit_parameters"]

        ecc_limits_deg = self.context.retina_parameters["ecc_limits_deg"]
        deg_per_mm = self.context.retina_parameters["deg_per_mm"]
        ecc_lim_mm = np.asarray(ecc_limits_deg) / deg_per_mm
        mean_ecc = np.mean(ecc_lim_mm)

        # Get density at mean eccentricity
        mean_density = this_function(mean_ecc, *fit_parameters)

        fig, ax = plt.subplots()
        ax.plot(cell_eccentricity, cell_density, "o", label="data")
        ax.plot(
            cell_eccentricity,
            this_function(cell_eccentricity, *fit_parameters),
            label="fit",
        )
        ax.set_xlabel("Eccentricity (mm)")
        ax.set_ylabel(f"{unit_type} density")
        title_text = f"{unit_type} density vs eccentricity"
        title_text += f": {mean_density:.0f}/mm^2 at {mean_ecc:.2f} mm"
        # Make new line with average cone surface are
        mean_surface_area = 1 / mean_density
        title_text += f"\nAverage unit surface area: {mean_surface_area:.2e} mm^2"
        ax.set_title(title_text)
        ax.legend()

        # Format fit_parameters to two decimal places in exponential notation
        formatted_fit_parameters = ", ".join(
            f"{param:.2e}\n" for param in fit_parameters
        )

        # Write onto figure axes the function name, equation, and the formatted fit parameters
        ax.text(
            0.60,
            0.30,
            f"{this_function.__name__}\nfit parameters: {formatted_fit_parameters}",
            transform=ax.transAxes,
        )

        if show_control_data is True:
            control_density_dict = self.project_data.construct_retina[
                f"{unit_type}_control_n_vs_ecc"
            ]
            ax.plot(
                control_density_dict["cell_eccentricity"],
                control_density_dict["cell_density"],
                "o",
                label="control data",
            )
            ax.legend()

        # set x and y axes logarithmic
        ax.set_xscale("log")
        ax.set_yscale("log")

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_bipolar_nonlinearity(self, savefigname=None):

        ret_file_npz = self.data_io.get_data(self.context.retina_parameters["ret_file"])
        popt = ret_file_npz["bipolar_nonlinearity_parameters"]
        bipolar_g_sur_scaled = ret_file_npz["bipolar_g_sur_scaled"]
        bipolar_RI_values = ret_file_npz["bipolar_RI_values"]
        bipolar_nonlinearity_fit = ret_file_npz["bipolar_nonlinearity_fit"]

        x = np.linspace(-1, 1, 100)
        y = bipolar_nonlinearity_fit

        plt.plot(bipolar_g_sur_scaled, bipolar_RI_values, "o")
        plt.plot(bipolar_g_sur_scaled, y)
        # Annotate as text the 3 second order polynomial parameters
        plt.text(
            0.05, 0.95, f"Parameters: {popt[0]:.2f}x^2 + {popt[1]:.2f}x + {popt[2]:.2f}"
        )

        if savefigname:
            self._figsave(figurename=savefigname)

    # SimulateRetina visualization
    def show_stimulus_with_gcs(
        self,
        example_gc=5,
        frame_number=0,
        ax=None,
        show_rf_id=False,
        savefigname=None,
    ):
        """
        Plots the 1SD ellipses of the RGC mosaic. This method is a SimulateRetina call.

        Parameters
        ----------
        retina : object
            The retina object that contains all the relevant information about the stimulus video and ganglion cells.
        frame_number : int, optional
            The index of the frame from the stimulus video to be displayed. Default is 0.
        ax : matplotlib.axes.Axes, optional
            The axes object to draw the plot on. If None, the current axes is used. Default is None.
        example_gc : int, optional
            The index of the ganglion cell to be highlighted. Default is 5.
        show_rf_id : bool, optional
            If True, the index of each ganglion cell will be printed at the center of its ellipse. Default is False..
        """
        stim_to_show = self.project_data.simulate_retina["stim_to_show"]

        stimulus_video = stim_to_show["stimulus_video"]
        df_stimpix = stim_to_show["df_stimpix"]
        stimulus_height_pix = stim_to_show["stimulus_height_pix"]
        pix_per_deg = stim_to_show["pix_per_deg"]
        deg_per_mm = stim_to_show["deg_per_mm"]
        retina_center = stim_to_show["retina_center"]

        dog_model_type = self.context.retina_parameters["dog_model_type"]
        fig = plt.figure()
        ax = ax or plt.gca()
        ax.imshow(stimulus_video.frames[frame_number, :, :])
        ax = plt.gca()

        gc_rot_deg = df_stimpix["orient_cen_rad"] * (-1) * 180 / np.pi

        for index, gc in df_stimpix.iterrows():
            if index == example_gc:
                facecolor = "yellow"
            else:
                facecolor = "None"

            match dog_model_type:
                case "ellipse_independent" | "ellipse_fixed":
                    circ = Ellipse(
                        (gc.q_pix, gc.r_pix),
                        width=2 * gc.semi_xc,
                        height=2 * gc.semi_yc,
                        angle=gc_rot_deg[index],  # Rotation in degrees anti-clockwise.
                        edgecolor="blue",
                        facecolor=facecolor,
                    )
                case "circular":
                    circ = Ellipse(
                        (gc.q_pix, gc.r_pix),
                        width=2 * gc.rad_c,
                        height=2 * gc.rad_c,
                        angle=gc_rot_deg[index],  # Rotation in degrees anti-clockwise.
                        edgecolor="blue",
                        facecolor=facecolor,
                    )

            ax.add_patch(circ)

            # If show_rf_id is True, annotate each ellipse with the index
            if show_rf_id:
                ax.annotate(
                    str(index),
                    (gc.q_pix, gc.r_pix),
                    color="black",
                    weight="bold",
                    fontsize=8,
                    ha="center",
                    va="center",
                )

        locs, labels = plt.yticks()

        locs = locs[locs < stimulus_height_pix]
        locs = locs[locs > 0]

        left_y_labels = locs.astype(int)
        plt.yticks(ticks=locs)
        ax.set_ylabel("pix")

        xlocs = locs - np.mean(locs)
        down_x_labels = np.round(xlocs / pix_per_deg, decimals=2) + np.real(
            retina_center
        )
        plt.xticks(ticks=locs, labels=down_x_labels)
        ax.set_xlabel("deg")

        ax2 = ax.twinx()
        ax2.tick_params(axis="y")
        right_y_labels = np.round((locs / pix_per_deg) / deg_per_mm, decimals=2)
        plt.yticks(ticks=locs, labels=right_y_labels)
        ax2.set_ylabel("mm")

        fig.tight_layout()

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_single_gc_view(
        self, unit_index, frame_number=0, ax=None, savefigname=None
    ):
        """
        Overlays the receptive field center of the specified retinal ganglion cell (RGC) on top of
        a given stimulus frame which is cropped around the RGC.

        Parameters
        ----------
        unit_index : int
            Index of the RGC for which the view is to be shown.
        frame_number : int, optional
            Frame number of the stimulus to display. Default is 0.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If not provided, uses the current axis.
        """

        stim_to_show = self.project_data.simulate_retina["stim_to_show"]
        stimulus_video = stim_to_show["stimulus_video"]
        df_stimpix = stim_to_show["df_stimpix"]
        qmin_all, qmax_all, rmin_all, rmax_all = stim_to_show["qr_min_max"]
        qmin = qmin_all[unit_index]
        qmax = qmax_all[unit_index]
        rmin = rmin_all[unit_index]
        rmax = rmax_all[unit_index]

        if ax is None:
            fig, ax = plt.subplots()

        gc = df_stimpix.iloc[unit_index]

        # Show stimulus frame cropped to RGC surroundings & overlay 1SD center RF on top of that
        ax.imshow(
            stimulus_video.frames[frame_number, :, :],
            cmap=self.cmap_stim,
            vmin=0,
            vmax=255,
        )
        ax.set_xlim([qmin, qmax])
        ax.set_ylim([rmax, rmin])

        # When in pixel coordinates, positive value in Ellipse angle is clockwise. Thus minus here.
        # Note that Ellipse angle is in degrees.
        # Width and height in Ellipse are diameters, thus x2.
        circ = Ellipse(
            (gc.q_pix, gc.r_pix),
            width=2 * gc.semi_xc,
            height=2 * gc.semi_yc,
            angle=gc.orient_cen_rad * (-1),  # Rotation in degrees anti-clockwise.
            edgecolor="white",
            facecolor="yellow",
        )
        ax.add_patch(circ)
        plt.xticks([])
        plt.yticks([])

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_temporal_kernel_frequency_response(
        self, unit_index=0, ax=None, savefigname=None
    ):
        """
        Plot the frequency response of the temporal kernel for a specified or all retinal ganglion cells (RGCs).

        Parameters
        ----------
        unit_index : int, optional
            Index of the RGC for which the frequency response is to be shown.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If not provided, uses the current axis.
        """

        assert (
            self.context.retina_parameters["temporal_model_type"] == "fixed"
        ), "No fixed temporal filter for dynamic temporal model, aborting..."

        spat_temp_filter_to_show = self.project_data.simulate_retina[
            "spat_temp_filter_to_show"
        ]
        temporal_filters = spat_temp_filter_to_show["temporal_filters"]
        data_filter_duration = spat_temp_filter_to_show["data_filter_duration"]

        tf = temporal_filters[unit_index, :]

        if ax is None:
            fig, ax = plt.subplots()

        ft_tf = np.fft.fft(tf)
        timestep = data_filter_duration / len(tf) / 1000  # in seconds
        freqs = np.fft.fftfreq(tf.size, d=timestep)
        ampl_s = np.abs(ft_tf)

        ax.set_xscale("log")
        ax.set_xlim([0.1, 100])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain")
        ax.plot(freqs, ampl_s, ".")

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def plot_midpoint_contrast(self, unit_index=0, ax=None, savefigname=None):
        """
        Plot the contrast at the midpoint pixel of the stimulus cropped to a specified RGC's surroundings.

        Parameters
        ----------
        unit_index : int, optional
            Index of the RGC for which to plot the contrast. Default is 0.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If not provided, uses the current axis.

        """
        stim_to_show = self.project_data.simulate_retina["stim_to_show"]
        spatial_filter_sidelen = stim_to_show["spatial_filter_sidelen"]
        stimulus_video = stim_to_show["stimulus_video"]
        stimulus_cropped_all = stim_to_show["stimulus_cropped"]
        stimulus_cropped = stimulus_cropped_all[unit_index]

        midpoint_ix = (spatial_filter_sidelen - 1) // 2
        signal = stimulus_cropped[midpoint_ix, midpoint_ix, :]

        video_dt = (1 / stimulus_video.fps) * b2u.second
        tvec = np.arange(0, len(signal)) * video_dt

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(tvec, signal)
        ax.set_ylim([-1, 1])

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def plot_local_rms_contrast(self, unit_index=0, ax=None, savefigname=None):
        """
        Plot the local RMS contrast for the stimulus cropped to a specified RGC's surroundings.

        Parameters
        ----------
        unit_index : int, optional
            Index of the RGC for which to plot the local RMS contrast. Default is 0.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If not provided, uses the current axis.
        """
        stim_to_show = self.project_data.simulate_retina["stim_to_show"]
        stimulus_cropped_all = stim_to_show["stimulus_cropped"]
        stimulus_cropped = stimulus_cropped_all[unit_index]
        stimulus_video = stim_to_show["stimulus_video"]
        spatial_filter_sidelen = stim_to_show["spatial_filter_sidelen"]
        # Invert from Weber contrast
        stimulus_cropped = 127.5 * (stimulus_cropped + 1.0)

        n_frames = stimulus_video.video_n_frames
        sidelen = spatial_filter_sidelen
        signal = np.zeros(n_frames)

        for t in range(n_frames):
            frame_mean = np.mean(stimulus_cropped[:, :, t])
            squared_sum = np.sum((stimulus_cropped[:, :, t] - frame_mean) ** 2)
            signal[t] = np.sqrt(1 / (frame_mean**2 * sidelen**2) * squared_sum)

        video_dt = (1 / stimulus_video.fps) * b2u.second
        tvec = np.arange(0, len(signal)) * video_dt

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(tvec, signal)
        ax.set_ylim([0, 1])

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def plot_local_michelson_contrast(self, unit_index=0, ax=None, savefigname=None):
        """
        Plot the local Michelson contrast for the stimulus cropped to a specified RGC's surroundings.

        Parameters
        ----------
        unit_index : int, optional
            Index of the RGC for which to plot the local Michelson contrast. Default is 0.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If not provided, uses the current axis.
        """
        stim_to_show = self.project_data.simulate_retina["stim_to_show"]
        stimulus_cropped_all = stim_to_show["stimulus_cropped"]
        stimulus_cropped = stimulus_cropped_all[unit_index]
        stimulus_video = stim_to_show["stimulus_video"]

        # Invert from Weber contrast
        stimulus_cropped = 127.5 * (stimulus_cropped + 1.0)

        n_frames = stimulus_video.video_n_frames
        signal = np.zeros(n_frames)

        # unsigned int will overflow when frame_max + frame_min = 256
        stimulus_cropped = stimulus_cropped.astype(np.uint16)
        for t in range(n_frames):
            frame_min = np.min(stimulus_cropped[:, :, t])
            frame_max = np.max(stimulus_cropped[:, :, t])
            signal[t] = (frame_max - frame_min) / (frame_max + frame_min)

        video_dt = (1 / stimulus_video.fps) * b2u.second
        tvec = np.arange(0, len(signal)) * video_dt

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(tvec, signal)
        ax.set_ylim([0, 1])

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_gain_calibration(
        self, threshold, folder_pattern, signal_gain=1.0, savefigname=None
    ):
        """
        Show signal gain calibration plot based on the provided threshold and folder pattern. Calls
        `ana.get_gain_calibration_df` to retrieve the data and then plots the gain calibration.

        Parameters
        ----------
        threshold : float
            The threshold value for gain calibration.
        folder_pattern : str
            The folder pattern to search for gain calibration data.
        signal_gain : float, optional
            The gain applied to the signal. Default is 1.0.
        savefigname : str, optional
            The name of the file where the figure will be saved. If None, the figure is not saved.
        """
        df, peak_column, most_frequent_peak_idx = self.ana.get_gain_calibration_df(
            threshold, folder_pattern, signal_gain=signal_gain
        )
        df_at_peak = df.loc[:, [peak_column, "gain"]]

        # Raise warning if threshold is not in the range of peak values
        if (
            threshold < df_at_peak[peak_column].min()
            or threshold > df_at_peak[peak_column].max()
        ):
            print(
                f"Warning: Threshold {threshold} is out of range of peak values "
                f"(range: {df_at_peak[peak_column].min():.2f} - {df_at_peak[peak_column].max():.2f}),"
                " extrapolating gain..."
            )

        # Assuming df_at_peak and peak_column are already defined
        peak_fr_raw = df_at_peak[peak_column].values
        gain_values_raw = df_at_peak["gain"].values

        # Create a linear interpolation function with extrapolation
        slope, intercept, r, p, se = stats.linregress(gain_values_raw, peak_fr_raw)

        # Calculate residuals
        residuals = peak_fr_raw - (slope * gain_values_raw + intercept)

        # Calculate the modified Z-scores
        median_residuals = np.median(residuals)
        median_absolute_deviation = np.median(np.abs(residuals - median_residuals))
        modified_z_scores = (
            0.6745 * (residuals - median_residuals) / median_absolute_deviation
        )

        # Define a threshold for the modified Z-scores
        outlier_threshold = (
            3.29  # Corresponds to a p-value of 0.001 in a two-tailed test
        )

        # Identify outliers
        mask = np.abs(modified_z_scores) <= outlier_threshold
        gain_values = gain_values_raw[mask]
        peak_fr = peak_fr_raw[mask]

        # Recalculate slope and intercept after dropping outliers
        slope, intercept, r, p, se = stats.linregress(gain_values, peak_fr)

        # Calculate gain at threshold
        gain_at_threshold = (threshold - intercept) / slope

        # Ensure the gain column is numeric
        df["gain"] = pd.to_numeric(df["gain"])
        # breakpoint()
        # Melt the DataFrame to long format for seaborn
        df_melted = df.melt(
            id_vars=["gain"],
            value_vars=[f"tf{i}" for i in range(16)],
            var_name="tf",
            value_name="value",
        )

        # Create a plot
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        sns.lineplot(
            data=df_melted, x="tf", y="value", hue="gain", ax=ax[0], legend="full"
        )

        # Customize the plot
        ax[0].set_title("Temporal Frequency vs Response")
        ax[0].set_xlabel("Temporal Frequency")
        ax[0].set_ylabel("Response")

        # Draw a dashed line at peak column
        ax[0].axvline(
            x=most_frequent_peak_idx, color="r", linestyle="--", label="Peak response"
        )
        # Plot the regression line, show datapoints, and threshold line
        sns.scatterplot(
            x=gain_values,
            y=peak_fr,
            ax=ax[1],
            label="Data points",
            color="blue",
        )
        # Add regression line using the slope and intercept
        x_fit = np.linspace(gain_values.min(), gain_values.max(), 100)
        y_fit = slope * x_fit + intercept
        ax[1].plot(x_fit, y_fit, color="orange", label="Regression line")
        # Plot the outlier points not included into the regression
        outlier_mask = ~mask
        sns.scatterplot(
            x=gain_values_raw[outlier_mask],
            y=peak_fr_raw[outlier_mask],
            ax=ax[1],
            label="Outliers",
            color="gray",
        )

        # PLot vertical line at gain at threshold
        ax[1].axvline(
            x=gain_at_threshold,
            color="g",
            linestyle="--",
            label=f"Gain at threshold {gain_at_threshold:.2f}",
        )

        ax[1].set_xlabel("Gain")
        ax[1].set_ylabel("Peak Response")
        ax[1].set_title("Gain vs Peak Response")
        ax[1].axhline(
            y=threshold, color="r", linestyle="--", label=f"Threshold {threshold:.2f}"
        )
        ax[1].set_title("Gain vs Peak Response with Threshold")
        ax[1].legend()

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_spikes_from_gz_file(
        self, filename: str, savefigname: Optional[str] = None
    ) -> None:
        """
        Visualize ganglion cell (gc) responses loaded from file.

        Parameters
        ----------
        filename : str
            The path to the file containing the gc responses.
        savefigname : str, optional
            The name of the file where the figure will be saved. If None, the figure is not saved.
        """
        # breakpoint()
        data_dict = self.data_io.get_data(filename)
        spiketrains = data_dict["spikes_0"]
        n_units = data_dict["n_units"]
        dt = data_dict["dt"]
        spike_idx = spiketrains[0]
        spike_times = spiketrains[1] / b2u.second

        # Make an eventplot of the spikes
        for_eventplot = []
        for this_idx in range(n_units):
            unit_events = spike_times[spike_idx == this_idx]
            for_eventplot.append(unit_events)

        # Infer duration from last spike time
        duration = np.round(np.max(spike_times)) * b2u.second

        # Create subplots
        fig, ax = plt.subplots(2, 1, sharex=True)

        # Event plot on first subplot
        ax[0].eventplot(for_eventplot)
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("unit #")

        # # Average firing rate on second subplot
        bin_width = 10 * b2u.ms

        num_bins = int(np.ceil(duration / bin_width))
        bin_edges = np.linspace(0, duration / b2u.second, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist, _ = np.histogram(spike_times, bins=bin_edges)

        avg_fr = hist / (n_units * (bin_width / b2u.second))

        ax[1].plot(bin_centers, avg_fr, label="Measured")
        ax[1].set_ylabel("Mean firing rate (Hz)")
        ax[1].set_xlabel("Time (s)")
        ax[1].legend()

        # Set Path(filename).parent.name as figure title
        fig.suptitle(Path(filename).parent.name)

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_all_gc_responses(self, sweep_idx=0, savefigname=None):
        """
        Visualize ganglion cell (gc) responses based on the data in the SimulateRetina object.

        Parameters
        ----------
        savefigname : str, optional
            The name of the file where the figure will be saved. If None, the figure is not saved.

        Attributes Accessed
        --------------------
        project_data.simulate_retina : dict
            Dictionary attached to ProjectData class instance containing the gc responses
            and other information to show.
        """
        gc_responses_to_show = self.project_data.simulate_retina["gc_responses_to_show"]
        n_sweeps = gc_responses_to_show["n_sweeps"]
        assert (
            sweep_idx < n_sweeps
        ), f"show_sweep {sweep_idx} is larger than n_sweeps {n_sweeps}."

        n_units = gc_responses_to_show["n_units"]
        spiketrains = gc_responses_to_show["all_spiketrains"][sweep_idx]
        duration = gc_responses_to_show["duration"]
        # requested firing_rates before spike generation.
        firing_rates = gc_responses_to_show["firing_rates"][..., sweep_idx]
        video_dt = gc_responses_to_show["video_dt"]
        # tvec_new = gc_responses_to_show["tvec_new"]

        cone_responses_to_show = self.project_data.simulate_retina[
            "cone_responses_to_show"
        ]
        if "cone_signal" in cone_responses_to_show.keys():
            cone_signal = cone_responses_to_show["cone_signal"]
        photodiode_to_show = self.project_data.simulate_retina["photodiode_to_show"]
        photodiode_response = photodiode_to_show["photodiode_response"]

        # Prepare data for manual visualization
        for_eventplot = spiketrains.copy()  # list of different leght arrays

        for_histogram = np.concatenate(spiketrains)
        firing_rate_mean = np.nanmean(firing_rates, axis=0)
        sample_name = "unit #"

        # Create subplots
        fig, ax = plt.subplots(3, 1, sharex=True)

        # Event plot on first subplot
        ax[0].eventplot(for_eventplot)
        ax[0].set_xlim([0, duration / b2u.second])
        ax[0].set_ylabel(sample_name)

        # Generator potential and average firing rate on second subplot
        tvec = np.arange(0, firing_rates.shape[-1], 1) * video_dt
        ax[1].plot(tvec, firing_rate_mean, label="Generator")
        ax[1].set_xlim([0, duration / b2u.second])

        # Given bin_width in ms, convert it to the correct unit
        bin_width = 10 * b2u.ms

        # Find the nearest integer number of simulation_dt units for hist_dt
        simulation_dt = self.context.run_parameters["simulation_dt"] * b2u.second
        hist_dt = np.round(bin_width / simulation_dt) * simulation_dt

        # Update bin_edges based on the new hist_dt
        num_bins = int(np.ceil(duration / hist_dt))
        bin_edges = np.linspace(0, duration / b2u.second, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist, _ = np.histogram(for_histogram, bins=bin_edges)

        # Update average firing rate calculation based on the new hist_dt
        avg_fr = hist / (n_units * (hist_dt / b2u.second))

        ax[1].plot(bin_centers, avg_fr, label="Measured")
        ax[1].set_ylabel("Firing rate (Hz)")
        ax[1].set_xlabel("Time (s)")
        ax[1].legend()

        # Plot cone signal axis ticks to left and photoreceptor to right
        if "cone_signal" in cone_responses_to_show.keys():
            cone_signal_mean = cone_signal.mean(axis=0)
            ax[2].plot(tvec, cone_signal_mean, label="cone_signal")
        ax[2].set_xlim([0, duration / b2u.second])
        ax[2].set_ylabel("Cone signal")
        ax[2].set_xlabel("Time (s)")
        ax[2].legend()
        ax2 = ax[2].twinx()
        ax2.plot(tvec, photodiode_response, label="Photodiode", color="r")
        ax2.set_ylabel("cd/m2")
        ax2.legend()

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_all_generator_potentials(self, savefigname=None):
        """ """
        gc_responses_to_show = self.project_data.simulate_retina["gc_responses_to_show"]
        n_sweeps = gc_responses_to_show["n_sweeps"]
        n_units = gc_responses_to_show["n_units"]

        duration = gc_responses_to_show["duration"]
        # requested firing_rates before spike generation.
        generator_potentials = gc_responses_to_show["generator_potentials"]
        video_dt = gc_responses_to_show["video_dt"]
        tvec_new = gc_responses_to_show["tvec_new"]

        cone_responses_to_show = self.project_data.simulate_retina[
            "cone_responses_to_show"
        ]
        if "cone_signal" in cone_responses_to_show.keys():
            cone_signal = cone_responses_to_show["cone_signal"]
        photodiode_to_show = self.project_data.simulate_retina["photodiode_to_show"]
        photodiode_response = photodiode_to_show["photodiode_response"]

        # Prepare data for visualization
        generator_potentials_mean = np.nanmean(generator_potentials, axis=0)

        # Create subplots
        fig, ax = plt.subplots(3, 1, sharex=True)
        tvec = np.arange(0, generator_potentials.shape[-1], 1) * video_dt
        ax[0].plot(tvec, generator_potentials.T)
        ax[0].set_xlim([0, duration / b2u.second])
        ax[0].set_ylabel("Generator potentials")
        ax[0].set_xlabel("Time (s)")

        # Generator potential and average firing rate on second subplot
        ax[1].plot(tvec, generator_potentials_mean)
        ax[1].set_xlim([0, duration / b2u.second])
        ax[1].set_ylabel("Generator mean (Hz)")
        ax[1].set_xlabel("Time (s)")
        ax[1].legend()

        # Plot cone signal axis ticks to left and photoreceptor to right
        if "cone_signal" in cone_responses_to_show.keys():
            cone_signal_mean = cone_signal.mean(axis=0)
            ax[2].plot(tvec, cone_signal_mean, label="cone_signal")
        ax[2].set_xlim([0, duration / b2u.second])
        ax[2].set_ylabel("Cone signal")
        ax[2].set_xlabel("Time (s)")
        ax[2].legend()
        ax2 = ax[2].twinx()
        ax2.plot(tvec, photodiode_response, label="Photodiode", color="r")
        ax2.set_ylabel("cd/m2")
        ax2.legend()

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_generator_potential_histogram(self, savefigname=None):
        """
        Display a histogram of generator potentials and related plots.

        Parameters
        ----------
        savefigname : str, optional
            Filename to save the figure. If None, the figure is not saved.

        Notes
        -----
        This method creates a figure with four subplots:
        1. Histogram of generator potentials after the baseline start time.
        2. Plot of the mean generator potential over time.
        3. Plot of the mean requested firing rate over time (before spike generation)
        4. Plot of the photodiode response over time.

        The figure is titled with the temporal model being used, and the
        histogram subplot includes a vertical dashed line to indicate the
        median generator potential value. If NaNs are present in the
        generator potentials, their count and percentage are annotated on the
        histogram.
        """

        gc_responses_to_show = self.project_data.simulate_retina["gc_responses_to_show"]
        generator_potentials = gc_responses_to_show["generator_potentials"]
        video_dt = gc_responses_to_show["video_dt"]
        firing_rates = gc_responses_to_show["firing_rates"]
        visual_stimulus_parameters = self.context.visual_stimulus_parameters
        fps = visual_stimulus_parameters["fps"]
        baseline_start_seconds = visual_stimulus_parameters["baseline_start_seconds"]

        # Prepare data for manual visualization
        generator_potential_mean = np.nanmean(generator_potentials, axis=0)
        firing_rate_mean = np.nanmean(firing_rates, axis=0)
        baseline_start_tp = int(baseline_start_seconds * fps)

        photodiode_to_show = self.project_data.simulate_retina["photodiode_to_show"]
        photodiode_response = photodiode_to_show["photodiode_response"]

        temporal_model_type = self.context.retina_parameters["temporal_model_type"]

        # Create subplots
        fig, ax = plt.subplots(4, 1, sharex=False, figsize=(12, 8))

        # Generator potential and average firing rate on second subplot
        ax[0].hist(
            generator_potentials[:, baseline_start_tp:].flatten(),
            bins=100,
        )
        ax[0].set_ylabel("Count", fontsize=14)
        # Mark the median value with vertical dashed line
        ax[0].axvline(
            np.nanmedian(generator_potentials[:, baseline_start_tp:]),
            color="r",
            linestyle="--",
        )
        n_nans = np.isnan(generator_potentials[:, baseline_start_tp:]).sum()
        if n_nans > 0:
            ax[0].annotate(
                f"{n_nans} nans ({100 * n_nans / generator_potentials.size:.2f}%)",
                xy=(0.8, 0.8),
                xycoords="axes fraction",
                fontsize=12,
                ha="center",
            )

        tvec = np.arange(0, generator_potentials.shape[-1], 1) * video_dt
        ax[1].plot(
            tvec[baseline_start_tp:],
            generator_potential_mean[baseline_start_tp:],
        )
        ax[1].set_ylabel("Generator mean", fontsize=14)

        ax[2].plot(
            tvec[baseline_start_tp:],
            firing_rate_mean[baseline_start_tp:],
        )
        ax[2].set_ylabel("Firing rate", fontsize=14)

        ax[3].plot(
            tvec[baseline_start_tp:],
            photodiode_response[baseline_start_tp:],
        )
        ax[3].set_ylabel("Photodiode", fontsize=14)
        ax[3].set_xlabel("Time (s)", fontsize=14)

        # Set suptitle to temporal_model_type
        fig.suptitle(
            f"Generator potentials for {temporal_model_type} temporal model",
            fontsize=16,
        )

        # Adjust tick parameters for all subplots
        for axis in ax:
            axis.tick_params(axis="both", which="major", labelsize=12)

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_cone_responses(self, time_range=None, savefigname=None):
        # Load data
        simulate_retina = self.project_data.simulate_retina
        gc_responses = simulate_retina["gc_responses_to_show"]
        photodiode = simulate_retina["photodiode_to_show"]
        cone_responses = simulate_retina["cone_responses_to_show"]

        if "cone_signal" not in cone_responses.keys():
            raise ValueError(
                "Cone responses not available. Requires simulation w/ subunit temporal_model. Aborting..."
            )
        visual_stimulus_parameters = self.context.visual_stimulus_parameters

        # Prepare data
        duration = gc_responses["duration"]
        photodiode_response = photodiode["photodiode_response"]
        cone_signal_u = cone_responses["cone_signal_u"]  # u for unit, pA or mV
        unit = cone_responses["unit"]
        tvec_mean = np.linspace(0, duration / b2u.second, len(photodiode_response))
        fps = visual_stimulus_parameters["fps"]
        baseline_start_seconds = visual_stimulus_parameters["baseline_start_seconds"]
        baseline_start_tp = int(baseline_start_seconds * fps)

        # Calculate baselines and adjust signals
        bl_photodiode = np.mean(photodiode_response[:baseline_start_tp])
        bl_cone = np.mean(cone_signal_u[:, :10])
        adjusted_photodiode = photodiode_response - bl_photodiode
        mean_cone_signal = cone_signal_u.mean(axis=0)
        adjusted_cone_signal = mean_cone_signal - bl_cone
        adjusted_cone_signal = adjusted_cone_signal / abs(adjusted_cone_signal).max()
        adjusted_photodiode = adjusted_photodiode / abs(adjusted_photodiode).max()
        r_argmax = np.abs(adjusted_cone_signal).argmax()
        # get value at argmax
        cone_signal_peak = mean_cone_signal[r_argmax]
        delta_val = cone_signal_peak - bl_cone

        # Plotting
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(tvec_mean, photodiode_response, label="Photodiode Response")
        ax[1].plot(tvec_mean, adjusted_cone_signal)
        ax[1].plot(tvec_mean, adjusted_photodiode, linestyle="--")

        # Set labels and legends
        ax[0].set_ylabel("Luminance (cd/m2)")
        ax[1].grid(True)

        # Plot unadjusted cone responses to ax[2]. Cone responses are averaged across all cones.
        ax[2].plot(tvec_mean, cone_signal_u.mean(axis=0), label="Cone Signal")
        ax[2].set_ylabel(f"Cone Response ({unit})")
        ax[2].set_xlabel("Time (s)")

        # Add vertical line to r_argmax position for plots ax[1] and ax[2]
        ax[1].axvline(tvec_mean[r_argmax], color="r", linestyle="--")
        ax[2].axvline(tvec_mean[r_argmax], color="r", linestyle="--")

        # Annotate the delta value
        match unit:
            case "mV":
                anno_text = r"$\Delta Vm$: {:.2f} mV".format(delta_val)
            case "pA":
                anno_text = r"$\Delta I$: {:.2f} pA".format(delta_val)
        ax[2].annotate(
            anno_text,
            xy=(tvec_mean[r_argmax], cone_signal_peak),
            xytext=(tvec_mean[r_argmax] - 0.05, cone_signal_peak),
            ha="right",  # Align text to the right
            arrowprops=dict(facecolor="black", arrowstyle="->"),
        )

        # Handling optional save and time range
        if time_range:
            ax[0].set_xlim(time_range)
            ax[1].set_xlim(time_range)
        if savefigname:
            plt.savefig(savefigname)

    def show_gc_noise(self, savefigname=None):
        """
        Display the noise in ganglion cell (gc) responses.

        Parameters
        ----------
        savefigname : str, optional
            The name of the file where the figure will be saved. If None, the figure is not saved.

        Attributes Accessed
        --------------------
        project_data.simulate_retina : dict
            Dictionary attached to ProjectData class instance containing the gc responses
            and other information to analyze.
        """
        gc_responses_to_show = self.project_data.simulate_retina["gc_responses_to_show"]
        gc_synaptic_noise = gc_responses_to_show["gc_synaptic_noise"]
        n_sweeps = gc_responses_to_show["n_sweeps"]
        n_gcs = gc_responses_to_show["n_units"]

        cov_matrix = np.zeros((n_gcs, n_gcs, n_sweeps))
        for trial in range(n_sweeps):
            cov_matrix[..., trial] = np.cov(gc_synaptic_noise[..., trial])

        cov_matrix_mean = np.mean(cov_matrix, axis=2)
        # breakpoint()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(gc_synaptic_noise.flatten(), 100)
        ax[1].imshow(cov_matrix_mean, cmap="viridis")

        if savefigname is not None:
            plt.savefig(savefigname)

    def show_all_gc_histogram(
        self, start_time=None, end_time=None, sweep_idx=0, savefigname=None
    ):
        """
        Display histograms for the mean and SD of spike rates of ganglion cell (gc) responses.

        Parameters
        ----------
        start_time : float, optional
            The start time in seconds for limiting the analysis period. Defaults to None, indicating the start of the dataset.
        end_time : float, optional
            The end time in seconds for limiting the analysis period. Defaults to None, indicating the end of the dataset.
        savefigname : str, optional
            The name of the file where the figure will be saved. If None, the figure is not saved.

        Attributes Accessed
        --------------------
        project_data.simulate_retina : dict
            Dictionary attached to ProjectData class instance containing the gc responses
            and other information to analyze.
        """
        gc_responses = self.project_data.simulate_retina["gc_responses_to_show"]
        spiketrains = gc_responses["all_spiketrains"][sweep_idx]

        # Optional time limiting
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = gc_responses["duration"]

        # Adjusting spiketrains according to the specified time period
        adjusted_spiketrains = [
            spiketrain[(spiketrain >= start_time) & (spiketrain <= end_time)]
            for spiketrain in spiketrains
        ]

        # Calculating mean spike rates
        spike_rates = [
            len(spiketrain) / (end_time - start_time)
            for spiketrain in adjusted_spiketrains
        ]

        # Change to arrays

        mean_spike_rates = np.array(spike_rates)
        # sd_spike_rates = np.std(spike_rates)

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.hist(
            mean_spike_rates,
            bins="auto",
            color="skyblue",
            alpha=0.7,
            label="Mean Spike Rates",
        )

        ax.set_title("Mean Spike Rates")
        ax.set_xlabel("Spike Rate (Hz)")
        ax.set_ylabel("Frequency")

        # for a in ax:
        ax.legend()

        plt.tight_layout()

        if savefigname is not None:
            plt.savefig(savefigname)

    def show_spatiotemporal_filter(self, unit_index=0, savefigname=None):
        """
        Display the spatiotemporal filter for a given unit in the retina.

        This method retrieves the specified unit's spatial and temporal filters
        from the 'simulate_retina' attribute of the 'project_data' object.

        Parameters
        ----------
        unit_index : int, optional
            Index of the unit for which the spatiotemporal filter is to be shown.
        savefigname : str or None, optional
            If a string is provided, the figure will be saved with this filename.
        """

        assert (
            self.context.retina_parameters["temporal_model_type"] == "fixed"
        ), "No fixed temporal filter for dynamic temporal model, aborting..."

        spat_temp_filter_to_show = self.project_data.simulate_retina[
            "spat_temp_filter_to_show"
        ]
        spatial_filters = spat_temp_filter_to_show["spatial_filters"]
        temporal_filters = spat_temp_filter_to_show["temporal_filters"]
        gc_type = spat_temp_filter_to_show["gc_type"]
        response_type = spat_temp_filter_to_show["response_type"]
        temporal_filter_len = spat_temp_filter_to_show["temporal_filter_len"]
        spatial_filter_sidelen = spat_temp_filter_to_show["spatial_filter_sidelen"]

        temporal_filter = temporal_filters[unit_index, :]
        spatial_filter = spatial_filters[unit_index, :]
        spatial_filter = spatial_filter.reshape(
            (spatial_filter_sidelen, spatial_filter_sidelen)
        )

        vmax = np.max(np.abs(spatial_filter))
        vmin = -vmax

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        plt.suptitle(gc_type + " " + response_type + " / unit ix " + str(unit_index))
        plt.subplot(121)
        im = ax[0].imshow(
            spatial_filter, cmap=self.cmap_spatial_filter, vmin=vmin, vmax=vmax
        )
        ax[0].grid(True)
        plt.colorbar(im, ax=ax[0])

        plt.subplot(122)
        if self.context.retina_parameters["temporal_model_type"] == "dynamic":
            # Print text to middle of ax[1]: "No fixed temporal filter for dynamic temporal model"
            ax[1].text(
                0.5,
                0.5,
                "No fixed temporal filter for dynamic temporal model",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
            )

        else:
            ax[1].plot(range(temporal_filter_len), np.flip(temporal_filter))

        plt.tight_layout()

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_spatiotemporal_filter_summary(self, savefigname=None):
        """
        Display the spatiotemporal filter for all units in the current simulation.


        Parameters
        ----------
        savefigname : str or None, optional
            If a string is provided, the figure will be saved with this filename.
        """

        # assert (
        #     self.context.retina_parameters["temporal_model_type"] == "fixed"
        # ), "No fixed temporal filter for dynamic temporal model, aborting..."

        spat_temp_filter_to_show = self.project_data.simulate_retina[
            "spat_temp_filter_to_show"
        ]
        spatial_filters = spat_temp_filter_to_show["spatial_filters"]
        temporal_filters = spat_temp_filter_to_show["temporal_filters"]
        gc_type = spat_temp_filter_to_show["gc_type"]
        response_type = spat_temp_filter_to_show["response_type"]
        temporal_filter_len = spat_temp_filter_to_show["temporal_filter_len"]
        spatial_filter_sidelen = spat_temp_filter_to_show["spatial_filter_sidelen"]

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        plt.subplot(121)
        im = ax[0].plot(spatial_filters.sum(axis=1), ".")
        ax[0].grid(True)
        ax[0].set_title("Spatial filter sums")
        ax[0].set_xlabel("Unit #")

        plt.subplot(122)
        ax[1].plot(temporal_filters.sum(axis=1), ".")
        ax[1].grid(True)
        ax[1].set_title("Temporal filter sums")
        ax[1].set_xlabel("Unit #")

        plt.tight_layout()

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_impulse_response(self, savefigname=None):
        viz_dict = self.project_data.simulate_retina["impulse_to_show"]

        tvec = viz_dict["tvec"]  # in seconds
        svec = viz_dict["svec"]

        contrasts = viz_dict["contrasts"]  # contrasts_for_impulse
        impulse_responses = viz_dict["impulse_responses"]  # yvecs
        idx_start_delay = viz_dict["idx_start_delay"]  # in milliseconds

        start_delay = tvec[idx_start_delay] * 1000  # convert to milliseconds
        tvec = tvec * 1000  # convert to milliseconds
        tvec = tvec - start_delay  # shift to start at 0

        unit_index = viz_dict["Unit idx"]
        ylims = np.array([np.min(impulse_responses), np.max(impulse_responses)])

        plt.figure()

        for u_idx, this_unit in enumerate(unit_index):
            for c_idx, this_contrast in enumerate(contrasts):
                if len(contrasts) > 1:
                    label = f"Unit {this_unit}, contrast {this_contrast}"
                else:
                    label = f"Unit {this_unit}"
                plt.plot(
                    tvec[:-1],
                    impulse_responses[u_idx, :-1, c_idx],
                    label=label,
                )

        # Set vertical dashed line at max (svec) time point, i.e. at the impulse time
        plt.axvline(x=tvec[np.argmax(np.abs(svec))], color="k", linestyle="--")
        plt.legend()
        plt.ylim(ylims[0] * 1.1, ylims[1] * 1.1)

        gc_type = viz_dict["gc_type"]
        response_type = viz_dict["response_type"]
        temporal_model_type = viz_dict["temporal_model_type"]

        plt.title(
            f"{gc_type} {response_type} ({temporal_model_type} model) impulse response(s)"
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Normalized response")
        # Put grid on
        plt.grid(True)

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_unity(self, savefigname=None):
        """
        Display the uniformity regions of a working retina and optionally save the figure.

        This method visualizes the total region covered by Delaunay triangulation mask,
        the unity region where the sum of unit regions and the Delaunay mask equals 1,
        and the sum of unit center regions.

        Parameters
        ----------
        savefigname : str, optional
            The filename for saving the figure. If None, the figure is not saved.
        """

        uniformify_data = self.project_data.simulate_retina["uniformify_data"]
        uniformify_index = uniformify_data["uniformify_index"]
        total_region = uniformify_data["total_region"]
        unity_region = uniformify_data["unity_region"]
        unit_region = uniformify_data["unit_region"]
        mask_threshold = uniformify_data["mask_threshold"]

        fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        ax[0].set_title("Total region (Delaunay_mask)")
        plt.colorbar(ax[0].imshow(total_region, cmap="gray"), ax=ax[0])
        plt.colorbar(ax[1].imshow(unity_region, cmap="gray"), ax=ax[1])
        ax[1].set_title(
            f"Unity region, where:\nSum of unit regions * Delaunay mask == 1\nth = {mask_threshold:.2f}\nuniformify_index = {uniformify_index:.2f}"
        )
        ax[2].set_title("Sum of unit regions")
        plt.colorbar(ax[2].imshow(unit_region, cmap="gray"), ax=ax[2])

        plt.tight_layout()

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    # Cone filtering (natural images and videos) and cone noise visualization
    def show_cone_filter_response(self, image, image_after_optics, cone_response):
        """
        PreGCProcessing call.
        """
        fig, ax = plt.subplots(nrows=2, ncols=3)
        axs = ax.ravel()
        axs[0].hist(image.flatten(), 20)
        axs[1].hist(image_after_optics.flatten(), 20)
        axs[2].hist(cone_response.flatten(), 20)

        axs[3].imshow(image)
        axs[3].set_title("Original image")
        axs[4].imshow(image_after_optics)
        axs[4].set_title("Image after optics")
        axs[5].imshow(cone_response)
        axs[5].set_title("Nonlinear cone response")

    def plot_analog_stimulus(self, analog_input):
        data = analog_input.Input

        plt.figure()
        plt.plot(data.T)

    def show_retina_img(self, savefigname=None):
        """
        Show the image of whole retina with all the receptive fields summed up.
        """

        gen_ret = self.project_data.construct_retina["ret"]

        img_ret = gen_ret["img_ret"]
        img_ret_masked = gen_ret["img_ret_masked"]
        img_ret_adjusted = gen_ret["img_ret_adjusted"]

        plt.figure()
        plt.subplot(221)
        plt.imshow(img_ret, cmap="gray")
        plt.colorbar()
        plt.title("Original coverage")
        plt.subplot(222)

        cmap = plt.cm.get_cmap("viridis")
        custom_cmap = mcolors.ListedColormap(
            cmap(np.linspace(0, 1, int(np.max(img_ret_masked)) + 1))
        )
        plt.imshow(img_ret_masked, cmap=custom_cmap)
        plt.colorbar()
        plt.title("Summed masks")

        plt.subplot(223)
        plt.imshow(img_ret_adjusted, cmap="gray")
        plt.colorbar()
        plt.title("Adjusted coverage")

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_rf_imgs(self, n_samples=10, savefigname=None):
        """
        Show the individual RFs of the VAE retina

        gc_vae_img: (n_units, n_pix, n_pix)
        gc_vae_img_mask: (n_units, n_pix, n_pix)
        gc_vae_img_final: (n_units, n_pix, n_pix)
        """

        gen_rfs = self.project_data.construct_retina["gen_rfs"]

        gc_vae_img = gen_rfs["gc_vae_img"]
        gc_vae_img_mask = gen_rfs["gc_vae_img_mask"]
        gc_vae_img_final = gen_rfs["gc_vae_img_final"]

        fig, axs = plt.subplots(3, n_samples, figsize=(n_samples, 3))
        samples = np.random.choice(gc_vae_img.shape[0], n_samples, replace=False)
        for i, sample in enumerate(samples):
            axs[0, i].imshow(gc_vae_img[sample], cmap="gray")
            axs[0, i].axis("off")
            axs[0, i].set_title("unit " + str(sample))

            axs[1, i].imshow(gc_vae_img_mask[sample], cmap="gray")
            axs[1, i].axis("off")

            axs[2, i].imshow(gc_vae_img_final[sample], cmap="gray")
            axs[2, i].axis("off")

        # On the left side of the first axis of each row, set text labels.
        axs[0, 0].set_ylabel("RF")
        axs[0, 0].axis("on")
        axs[1, 0].set_ylabel("Mask")
        axs[1, 0].axis("on")
        axs[2, 0].set_ylabel("Adjusted RF")
        axs[2, 0].axis("on")

        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])

        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])

        axs[2, 0].set_xticks([])
        axs[2, 0].set_yticks([])

        # # Adjust the layout so labels are visible
        # fig.subplots_adjust(left=0.15)
        if savefigname:
            self._figsave(figurename=savefigname)

    def show_rf_violinplot(self):
        """
        Show the individual RFs of the VAE retina

        gc_vae_img: (n_units, n_pix, n_pix)
        gc_vae_img_final: (n_units, n_pix, n_pix)
        """

        gen_rfs = self.project_data.construct_retina["gen_rfs"]

        gc_vae_img = gen_rfs["gc_vae_img"]
        gc_vae_img_final = gen_rfs["gc_vae_img_final"]

        fig, axs = plt.subplots(
            2, 1, figsize=(10, 10)
        )  # I assume you want a bigger figure size.

        # reshape and transpose arrays so that we have one row per unit
        df_rf = pd.DataFrame(gc_vae_img.reshape(gc_vae_img.shape[0], -1).T)
        df_pruned = pd.DataFrame(
            gc_vae_img_final.reshape(gc_vae_img_final.shape[0], -1).T
        )

        # Show seaborn boxplot with RF values, one box for each unit
        sns.violinplot(data=df_rf, ax=axs[0])
        axs[0].set_title("RF values")
        # Put grid on
        axs[0].grid(True)
        # ...and RF_adjusted values
        sns.violinplot(data=df_pruned, ax=axs[1])
        axs[1].set_title("RF adjusted values")
        axs[1].grid(True)

    # Experiment visualization
    def _string_on_plot(
        self, ax, variable_name=None, variable_value=None, variable_unit=None, idx=0
    ):
        plot_str = f"{variable_name} = {variable_value:6.2f} {variable_unit}"
        ax.text(
            0.05,
            0.95 - idx * 0.2,
            plot_str,
            fontsize=8,
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="Square,pad=0.2", fc="white", ec="white", lw=1),
        )

    def _get_exp_variables(self, experiment_df):
        """Get experimental variables from the experiment dataframe."""
        multiple_values_idx = experiment_df.nunique() > 1
        columns = experiment_df.loc[:, multiple_values_idx].columns.tolist()
        exp_variables = [
            column for column in columns if column != "stimulus_video_name"
        ]
        return exp_variables

    def fr_response(
        self, filename, exp_variables, xlog=False, ylog=False, savefigname=None
    ):
        """
        Plot the mean firing rate response curve.
        """

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)
        assert (
            len(exp_variables) == 1
        ), "Only one variable can be plotted at a time, aborting..."

        experiment_df = pd.read_csv(data_folder / filename, index_col=0)
        data_df = pd.read_csv(
            data_folder / f"{cond_names_string}_population_means.csv", index_col=0
        )
        data_df_units = pd.read_csv(
            data_folder / f"{cond_names_string}_unit_means.csv", index_col=0
        )

        response_levels_s = experiment_df.loc[:, "contrast"]
        mean_response_levels_s = data_df.mean()
        response_levels_s = pd.to_numeric(response_levels_s)
        response_levels_s = response_levels_s.round(decimals=2)

        response_function_df = pd.DataFrame(
            {cond_names_string: response_levels_s, "response": mean_response_levels_s}
        )

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        sns.lineplot(
            data=response_function_df,
            x=cond_names_string,
            y="response",
            marker="o",
            color="black",
            ax=ax[0],
        )

        # Title
        ax[0].set_title(f"{cond_names_string} response function (population mean)")

        if xlog:
            ax[0].set_xscale("log")
        if ylog:
            ax[0].set_yscale("log")

        sns.boxplot(data=data_df_units, color="white", linewidth=2, whis=100, ax=ax[1])
        sns.swarmplot(data=data_df_units, color="black", size=3, ax=ax[1])

        # Title
        ax[1].set_title(f"{cond_names_string} response function (individual units)")

        if savefigname:
            self._figsave(figurename=savefigname)

    def F1F2_popul_response(
        self,
        filename,
        exp_variables,
        xlog=False,
        ylog=False,
        savefigname=None,
    ):
        """
        Plot F1 and  F2 frequency response curves (central tendency)
        for all conditions.Population response, i.e. mean across units.

        Parameters
        ----------
        filename : str
            The name of the file containing the data to be plotted.
        exp_variables : list of str
            List of experimental variable names to be used for fetching the data and plotting.
        xlog : bool
            If True, the x-axis is shown in logarithmic scale.
        ylog : bool
            If True, the y-axis is shown in logarithmic scale.
        savefigname : str or None
            If not empty, the figure is saved to this filename.
        """

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)
        n_variables = len(exp_variables)

        if n_variables > 2:
            raise ValueError("Can only plot up to 2 variables at a time, aborting...")

        experiment_df = pd.read_csv(data_folder / filename, index_col=0)
        F_popul_df = pd.read_csv(
            data_folder / f"{cond_names_string}_F1F2_population_means.csv", index_col=0
        )

        F_popul_long_df = pd.melt(
            F_popul_df,
            id_vars=["sweep", "F_peak"],
            value_vars=F_popul_df.columns[:-2],
            var_name=f"{cond_names_string}_names",
            value_name="amplitudes",
        )

        # Make new columns with conditions' levels
        for cond in exp_variables:
            levels_s = experiment_df.loc[:, cond]
            levels_s = pd.to_numeric(levels_s)
            levels_s = levels_s.round(decimals=2)
            F_popul_long_df[cond] = F_popul_long_df[f"{cond_names_string}_names"].map(
                levels_s
            )
        if n_variables == 2:
            F_popul_long_df = F_popul_long_df[F_popul_long_df["F_peak"] == "F1"]
            title_prefix = "Population mean F1 for "
        else:
            title_prefix = "Population mean F1 and F2 for "

        fig, ax = plt.subplots(1, n_variables, figsize=(8, 4))

        if n_variables == 1:
            sns.lineplot(
                data=F_popul_long_df,
                x=exp_variables[0],
                y="amplitudes",
                hue="F_peak",
                palette="tab10",
                errorbar=None,
                ax=ax,
            )
            ax.set_title(title_prefix + exp_variables[0])
            if xlog:
                ax.set_xscale("log")
            if ylog:
                ax.set_yscale("log")

        else:  # 2 variables
            for i, cond in enumerate(exp_variables):
                F_popul_long_df = F_popul_long_df[F_popul_long_df["F_peak"] == "F1"]
                secondary_condition = set(exp_variables) - {cond}
                sns.lineplot(
                    data=F_popul_long_df,
                    x=cond,
                    y="amplitudes",
                    hue=secondary_condition.pop(),
                    palette="tab10",
                    errorbar=None,
                    ax=ax[i],
                )
                ax[i].set_title(title_prefix + cond)
                if xlog:
                    ax[i].set_xscale("log")
                if ylog:
                    ax[i].set_yscale("log")

        if savefigname:
            self._figsave(figurename=savefigname)

    def F1F2_unit_response(
        self,
        filename,
        exp_variables,
        xlog=False,
        ylog=False,
        savefigname=None,
    ):
        """
        Plot F1 and  F2 frequency response curves (central tendency and a confidence interval)
        for all conditions. Unit response, i.e. mean across trials. If hue is not "F_peak",
        only F1 response is shown.

        Parameters
        ----------
        filename : str
            The name of the file containing the data to be plotted.
        exp_variables : list of str
            List of experimental variable names to be used for fetching the data and plotting.
        xlog : bool
            If True, the x-axis is shown in logarithmic scale.
        ylog : bool
            If True, the y-axis is shown in logarithmic scale.
        savefigname : str or None
            If not empty, the figure is saved to this filename."""

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)
        n_variables = len(exp_variables)

        if n_variables > 2:
            raise ValueError("Can only plot up to 2 variables at a time, aborting...")

        experiment_df = pd.read_csv(data_folder / filename, index_col=0)

        F_unit_ampl_df = pd.read_csv(
            data_folder / f"{cond_names_string}_F1F2_unit_ampl_means.csv", index_col=0
        )
        F_unit_long_df = pd.melt(
            F_unit_ampl_df,
            id_vars=["unit", "F_peak"],
            value_vars=F_unit_ampl_df.columns[:-2],
            var_name=f"{cond_names_string}_names",
            value_name="amplitudes",
        )

        # Make new columns with conditions' levels
        for cond in exp_variables:
            levels_s = experiment_df.loc[:, cond]
            levels_s = pd.to_numeric(levels_s)
            levels_s = levels_s.round(decimals=2)
            F_unit_long_df[cond] = F_unit_long_df[f"{cond_names_string}_names"].map(
                levels_s
            )

        if n_variables == 2:
            F_unit_long_df = F_unit_long_df[F_unit_long_df["F_peak"] == "F1"]
            title_prefix = "Unit mean F1 for "
        else:
            title_prefix = "Unit mean F1 and F2 for "

        fig, ax = plt.subplots(1, n_variables, figsize=(8, 4))

        if n_variables == 1:
            sns.lineplot(
                data=F_unit_long_df,
                x=exp_variables[0],
                y="amplitudes",
                hue="F_peak",
                palette="tab10",
                ax=ax,
            )
            ax.set_title(title_prefix + exp_variables[0])
            if xlog:
                ax.set_xscale("log")
            if ylog:
                ax.set_yscale("log")

        else:
            for i, cond in enumerate(exp_variables):
                F_unit_long_df = F_unit_long_df[F_unit_long_df["F_peak"] == "F1"]
                secondary_condition = set(exp_variables) - {cond}
                sns.lineplot(
                    data=F_unit_long_df,
                    x=cond,
                    y="amplitudes",
                    hue=secondary_condition.pop(),
                    palette="tab10",
                    ax=ax[i],
                )

                # Title
                ax[i].set_title(title_prefix + cond)
                if xlog:
                    ax[i].set_xscale("log")
                if ylog:
                    ax[i].set_yscale("log")

        if savefigname:
            self._figsave(figurename=savefigname)

    def ptp_response(
        self, filename, exp_variables, x_of_interest=None, savefigname=None
    ):
        """
        Plot the peak-to-peak firing rate magnitudes across conditions.

        Parameters
        ----------
        filename : str
            The name of the file containing the data to be plotted.
        exp_variables : list of str
            List of experimental variable names to be used for fetching the data and plotting.
        x_of_interest : list of str or None
            If provided, the data will be filtered to include only these variables.
            If None, all data will be included.
        savefigname : str or None
            If not empty, the figure is saved to this filename.
        """

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)
        assert (
            len(exp_variables) == 1
        ), "Only one variable can be plotted at a time, aborting..."

        experiment_df = pd.read_csv(data_folder / filename, index_col=0)
        data_df = pd.read_csv(
            data_folder / f"{cond_names_string}_PTP_population_means.csv", index_col=0
        )

        if x_of_interest is None:
            data_df_selected = data_df
        else:
            data_df_selected = data_df.loc[:, x_of_interest]

        # Turn series into array of values
        x_values_df = experiment_df.loc[:, exp_variables]
        x_values_series = x_values_df.iloc[0, :]
        x_values_series = pd.to_numeric(x_values_series)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        plt.subplot(121)
        plt.plot(
            x_values_series.values,
            data_df.mean().values,
            color="black",
        )

        sns.boxplot(
            data=data_df_selected,
            color="black",
            ax=ax[1],
        )

        # Title
        ax[0].set_title(f"{cond_names_string} ptp (population mean)")
        ax[1].set_title(f"{cond_names_string} ptp at two peaks and a through")

        if savefigname:
            self._figsave(figurename=savefigname)

    def spike_raster_response(self, filename, sweeps_to_show=[0], savefigname=None):
        """
        Show spikes from a results file.

        Parameters
        ----------
        filename : str
            The name of the file containing the data to be plotted.
        savefigname : str or None
            If not empty, the figure is saved to this filename.
        """

        experiment_df = self.data_io.get_data(filename=filename)
        cond_names = experiment_df.index.values
        exp_variables = self._get_exp_variables(experiment_df)

        if experiment_df["n_sweeps"].max() < max(sweeps_to_show) + 1:
            raise ValueError(
                f"Requested sweep indices {sweeps_to_show} exceed the maximum number of sweeps in the data: {experiment_df['n_sweeps'].max()}"
            )

        for sweep_idx in sweeps_to_show:
            # Visualize
            fig, ax = plt.subplots(len(experiment_df.index), 1, figsize=(8, 4))
            # make sure ax is subscriptable
            ax = np.array(ax, ndmin=1)

            gc_type = self.context.retina_parameters["gc_type"]
            response_type = self.context.retina_parameters["response_type"]
            # Loop conditions
            for idx, cond_name in enumerate(cond_names):
                gz_filename = f"Response_{gc_type}_{response_type}_{cond_name}.gz"

                data_dict = self.data_io.get_data(gz_filename)

                cond_s = experiment_df.loc[cond_name, :]
                duration_seconds = pd.to_numeric(cond_s.loc["duration_seconds"])
                baseline_start_seconds = pd.to_numeric(
                    cond_s.loc["baseline_start_seconds"]
                )
                baseline_end_seconds = pd.to_numeric(cond_s.loc["baseline_end_seconds"])
                duration_tot = (
                    duration_seconds + baseline_start_seconds + baseline_end_seconds
                )
                t_min = -baseline_start_seconds
                t_max = duration_seconds + baseline_end_seconds
                units, times = self.ana._get_spikes_by_interval(
                    data_dict, sweep_idx, 0, duration_tot
                )
                # Shift times by baseline start
                times = (times / b2u.second) - baseline_start_seconds
                ax[idx].plot(
                    times,
                    units,
                    ".",
                )
                # Set x min and max to t min and max
                ax[idx].set_xlim([t_min, t_max])
                this_value = experiment_df.loc[cond_name, exp_variables]
                index_list = this_value.index.to_list()
                value_list = this_value.to_list()
                title_substring = ", ".join(
                    [f"{index_list[i]} {value_list[i]}" for i in range(len(index_list))]
                )
                title_string = f"{this_value.name}: {title_substring}"

                ax[idx].set_title(
                    title_string,
                    fontsize=10,
                    loc="left",
                )

                mean_fr = self.ana._analyze_meanfr(
                    data_dict, sweep_idx, 0, duration_tot
                )
                sd_fr = self.ana._analyze_sd_fr(data_dict, sweep_idx, 0, duration_tot)

                self._string_on_plot(
                    ax[idx],
                    variable_name="Mean FR",
                    variable_value=mean_fr,
                    variable_unit="Hz",
                    idx=0,
                )
                self._string_on_plot(
                    ax[idx],
                    variable_name="SD FR",
                    variable_value=sd_fr,
                    variable_unit="Hz",
                    idx=1,
                )

            if savefigname:
                savefigname = (
                    Path(savefigname).stem
                    + f"_sweep{sweep_idx}"
                    + Path(savefigname).suffix
                )
                self._figsave(figurename=savefigname)

    def show_relative_gain(self, filename, exp_variables, savefigname=None):
        """
        Display the relative gain of the cone, bipolar, and ganglion cell responses.

        The cone and bipolar responses are available with the subunit temporal model.

        Parameters
        ----------
        exp_variables : list of str
            List of experimental variable names to be used for fetching the data and plotting.
        savefigname : str, optional
            Filename to save the figure. If None, the figure will not be saved.
        """
        cond_names_string = "_".join(exp_variables)
        experiment_df = self.data_io.get_data(filename=filename)
        cond_names = experiment_df.index.values
        gc_type = self.context.retina_parameters["gc_type"]
        response_type = self.context.retina_parameters["response_type"]
        data_folder = self.context.output_folder

        pattern = f"exp_results_{gc_type}_{response_type}_{cond_names_string}_*.csv"
        data_fullpath = self.data_io.most_recent_pattern(data_folder, pattern)
        df = self.data_io.get_data(data_fullpath)
        available_data = df.columns.to_list()[3:]
        n_data = len(available_data)

        fig, ax = plt.subplots(n_data, 1, figsize=(8, 4))

        # Determine the unique values for the hue variable
        hue_values = df[cond_names_string + "_R"].unique()

        if "True" in experiment_df.loc[:, "logarithmic"][0]:
            # Use a logarithmic colormap
            norm = colors.LogNorm(vmin=hue_values.min(), vmax=hue_values.max())
        else:
            norm = colors.Normalize(vmin=hue_values.min(), vmax=hue_values.max())
        cmap = plt.get_cmap("viridis")

        for i, data_name in enumerate(available_data):
            sns.lineplot(
                data=df,
                x="time",
                y=data_name,
                hue=cond_names_string + "_R",
                ax=ax[i],
                palette=sns.color_palette(cmap(norm(hue_values))),
            )

            # Format legend to show integers only
            handles, labels = ax[i].get_legend_handles_labels()
            int_labels = [str(int(float(label))) for label in labels]
            ax[i].legend(handles, int_labels, title=cond_names_string + "_R")

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_unit_correlation(
        self, filename, exp_variables, time_window=None, savefigname=None
    ):
        """ """

        def _exp_func(x, a, b, c):
            return a * np.exp(-b * x) + c

        cond_names_string = "_".join(exp_variables)
        experiment_df = self.data_io.get_data(filename=filename)
        data_folder = self.context.output_folder

        # Load results
        filename_in = f"{cond_names_string}_correlation.npz"
        npy_save_path = data_folder / filename_in
        npz_file = np.load(npy_save_path, allow_pickle=True)
        ccf_mtx_mean = npz_file["ccf_mtx_mean"]
        ccf_mtx_SEM = npz_file["ccf_mtx_SEM"]
        lags = npz_file["lags"]

        if time_window is not None:
            idx_start = np.argmin(np.abs(lags - time_window[0]))
            idx_end = np.argmin(np.abs(lags - time_window[1]))
            lags = lags[idx_start:idx_end]
            ccf_mtx_mean = ccf_mtx_mean[:, :, idx_start:idx_end]
            ccf_mtx_SEM = ccf_mtx_SEM[:, :, idx_start:idx_end]

        unit_vec = npz_file["unit_vec"]

        filename_in = f"{cond_names_string}_correlation_neighbors.csv"
        neighbor_unique_df = pd.read_csv(data_folder / filename_in)
        n_corr = len(neighbor_unique_df)

        filename_in = f"{cond_names_string}_correlation_distances.csv"
        dist_df = pd.read_csv(data_folder / filename_in)

        # Visualize
        fig, ax = plt.subplots(n_corr, 2, figsize=(8, 4))
        ax = np.array(ax, ndmin=1)  # make sure ax is subscriptable

        # Loop conditions
        for idx, row in neighbor_unique_df.iterrows():
            yx_idx_s = row["yx_idx"]
            yx_str = yx_idx_s.strip("[]").split()
            yx_list = [int(num) for num in yx_str]
            yx_name = row["yx_name"]

            ax[idx, 0].plot(
                lags,
                ccf_mtx_mean[yx_list[0], yx_list[1], :],
                color="black",
            )
            ax[idx, 0].fill_between(
                lags,
                ccf_mtx_mean[yx_list[0], yx_list[1], :]
                - ccf_mtx_SEM[yx_list[0], yx_list[1], :],
                ccf_mtx_mean[yx_list[0], yx_list[1], :]
                + ccf_mtx_SEM[yx_list[0], yx_list[1], :],
                color="black",
                alpha=0.2,
            )
            ax[idx, 0].set_title(
                f"{yx_name}",
                fontsize=10,
            )

        ax[0, 1].plot(dist_df["distance_mm"], dist_df["ccoef"], ".", color="black")
        # Make exponential regression to the data
        x = dist_df["distance_mm"].values
        y = dist_df["ccoef"].values
        popt, pcov = opt.curve_fit(_exp_func, x, y, p0=(1, 1e-6, 1))
        y_fit = _exp_func(x, *popt)
        ax[0, 1].plot(x, y_fit, "--", color="black")
        ax[0, 1].set_title(
            f"Correlation vs distance\ny = {popt[0]:.2f} exp(-{popt[1]:.2f}x) + {popt[2]:.2f}",
            fontsize=10,
        )
        # set the rest of the axes off
        for i in range(1, n_corr):
            ax[i, 1].axis("off")

        if savefigname:
            self._figsave(figurename=savefigname)

    def tf_vs_fr_cg(
        self,
        filename,
        exp_variables,
        n_contrasts=None,
        xlog=False,
        ylog=False,
        savefigname=None,
    ):
        """
        Plot F1 frequency response curves for 2D frequency-contrast experiment.
        Unit response, i.e. mean across trials.
        Subplot 1: temporal frequency vs firing rate at n_contrasts
        Subplot 2: temporal frequency vs contrast gain (cg) at n_contrasts. Contrast gain is defined as the F1 response divided by contrast.

        Parameters
        ----------
        filename : str
            Name of the file containing the experiment metadata.
        exp_variables : list of str
            List of experiment variables to be plotted.
        n_contrasts : int
            Number of contrasts to be plotted. If None, all contrasts are plotted.
        xlog : bool
            If True, x-axis is logarithmic.
        ylog : bool
            If True, y-axis is logarithmic.
        """

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)

        # Experiment metadata
        experiment_df = pd.read_csv(data_folder / filename, index_col=0)

        # Results
        F_unit_ampl_df = pd.read_csv(
            data_folder / f"{cond_names_string}_F1F2_unit_ampl_means.csv", index_col=0
        )

        F_unit_long_df = pd.melt(
            F_unit_ampl_df,
            id_vars=["unit", "F_peak"],
            value_vars=F_unit_ampl_df.columns[:-2],
            var_name=f"{cond_names_string}_names",
            value_name="amplitudes",
        )

        # Make new columns with conditions' levels
        for cond in exp_variables:
            levels_s = experiment_df.loc[:, cond]
            levels_s = pd.to_numeric(levels_s)
            levels_s = levels_s.round(decimals=2)
            F_unit_long_df[cond] = F_unit_long_df[f"{cond_names_string}_names"].map(
                levels_s
            )

        # Make new columns cg and phase.
        F_unit_long_df["cg"] = F_unit_long_df["amplitudes"] / F_unit_long_df["contrast"]

        F_unit_long_df = F_unit_long_df[F_unit_long_df["F_peak"] == "F1"].reset_index(
            drop=True
        )

        # Select only the desired number of contrasts at about even intervals, including the lowest and the highest contrast
        if n_contrasts:
            contrasts = F_unit_long_df["contrast"].unique()
            contrasts.sort()
            contrasts = contrasts[:: int(len(contrasts) / n_contrasts)]
            contrasts = np.append(contrasts, F_unit_long_df["contrast"].max())
            F_unit_long_df = F_unit_long_df.loc[
                F_unit_long_df["contrast"].isin(contrasts)
            ]

        fig, ax = plt.subplots(2, 1, figsize=(8, 12))

        # Make the three subplots using seaborn lineplot
        sns.lineplot(
            data=F_unit_long_df,
            x="temporal_frequency",
            y="amplitudes",
            hue="contrast",
            palette="tab10",
            ax=ax[0],
        )
        ax[0].set_title("Firing rate vs temporal frequency")
        if xlog:
            ax[0].set_xscale("log")
        if ylog:
            ax[0].set_yscale("log")

        sns.lineplot(
            data=F_unit_long_df,
            x="temporal_frequency",
            y="cg",
            hue="contrast",
            palette="tab10",
            ax=ax[1],
        )
        ax[1].set_title("Contrast gain vs temporal frequency")
        if xlog:
            ax[1].set_xscale("log")
        if ylog:
            ax[1].set_yscale("log")

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_response_vs_background_experiment(self, unit="cd/m2", savefigname=None):
        """
        Plots the unit responses as a function of flash intensity for different background light levels.
        If a savefigname is provided, the figure is saved.

        Parameters:
        savefigname (str, optional): Name of the figure to save. Defaults to None.
        """

        gc_type = self.context.retina_parameters["gc_type"]
        response_type = self.context.retina_parameters["response_type"]

        # Load results
        filename = f"exp_results_{gc_type}_{response_type}_response_vs_background.csv"
        df = self.data_io.get_data(filename=filename)

        match unit:
            case "R*":
                bg_str = "background_R"
                flash_str = "flash_R"
            case "cd/m2":
                bg_str = "background"
                flash_str = "flash"

        # from df.columns, select column names which do not include substrings "flash" or "background"
        data_columns = df.loc[:, ~df.columns.str.contains("flash|background")].columns
        df_melted = pd.melt(
            df,
            id_vars=[bg_str, flash_str],
            value_vars=data_columns,
        )

        g = sns.FacetGrid(
            df_melted,
            row="variable",
            hue=bg_str,
            margin_titles=True,
            sharex=True,
            sharey=False,
            height=3,
            aspect=1.5,
        )

        g.map(sns.lineplot, flash_str, "value")
        [i[0].set_xscale("log") for i in g.axes]

        handles, labels = g.axes[0, 0].get_legend_handles_labels()
        plt.legend(handles, labels, loc="upper left")

        plt.xlabel(f"Log Flash Intensity ({unit})")
        plt.ylabel("Response")
        plt.suptitle("Onset-response vs background illumination")

        if savefigname:
            self._figsave(figurename=savefigname)

    # Validation viz
    def _build_param_plot(self, coll_ana_df_in, param_plot_dict, to_spa_dict_in):
        """
        Prepare for parametric plotting of multiple conditions.

        Parameters
        ----------
        coll_ana_df_in : pandas.DataFrame
            Mapping from to_spa_dict in conf file to dataframes which include parameter and analysis details.
        param_plot_dict : dict
            Dictionary guiding the parametric plot. See :func:`show_catplot` for details.
        to_spa_dict_in : dict
            Dictionary containing the startpoints, parameters and analyzes which are active in conf file.

        Returns
        -------
        data_list : list
            A nested list of data for each combination of outer and inner conditions.
        data_name_list : list
            A nested list of names for each combination of outer and inner conditions.
        outer_name_list : list
            A list of names for the outer conditions.
        """

        to_spa_dict = to_spa_dict_in
        coll_ana_df = coll_ana_df_in

        [title] = to_spa_dict[param_plot_dict["title"]]
        outer_list = to_spa_dict[param_plot_dict["outer"]]
        inner_list = to_spa_dict[param_plot_dict["inner"]]

        # if paths to data provided, take inner names from distinct list
        if param_plot_dict["inner_paths"] is True:
            inner_list = param_plot_dict["inner_path_names"]

        mid_idx = list(param_plot_dict.values()).index("startpoints")
        par_idx = list(param_plot_dict.values()).index("parameters")
        ana_idx = list(param_plot_dict.values()).index("analyzes")

        key_list = list(param_plot_dict.keys())

        # Create dict whose key is folder hierarchy and value is plot hierarchy
        hdict = {
            "mid": key_list[mid_idx],
            "par": key_list[par_idx],
            "ana": key_list[ana_idx],
        }

        data_list = []  # nested list, N items = N outer x N inner
        data_name_list = []  # nested list, N items = N outer x N inner
        outer_name_list = []  # list , N items = N outer

        for outer in outer_list:
            inner_data_list = []  # list , N items = N inner
            inner_name_list = []  # list , N items = N inner

            for in_idx, inner in enumerate(inner_list):
                # Nutcracker. eval to "outer", "inner" and "title"
                mid = eval(f"{hdict['mid']}")
                par = eval(f"{hdict['par']}")
                this_folder = f"{mid}_{par}"
                this_ana = eval(f"{hdict['ana']}")

                this_ana_col = coll_ana_df.loc[this_ana]["csv_col_name"]
                if param_plot_dict["compiled_results"] is True:
                    this_folder = f"{this_folder}_compiled_results"
                    this_ana_col = f"{this_ana_col}_mean"

                if param_plot_dict["inner_paths"] is True:
                    csv_path_tuple = param_plot_dict["paths"][in_idx]
                    csv_path = reduce(
                        lambda acc, y: Path(acc).joinpath(y), csv_path_tuple
                    )
                else:
                    csv_path = None

                # get data
                (
                    data0_df,
                    data_df_compiled,
                    independent_var_col_list,
                    dependent_var_col_list,
                    timestamp,
                ) = self.data_io.get_csv_as_df(
                    folder_name=this_folder, csv_path=csv_path, include_only=None
                )

                df = data_df_compiled[this_ana_col]
                inner_data_list.append(df)
                inner_name_list.append(inner)

            data_list.append(inner_data_list)
            data_name_list.append(inner_name_list)
            outer_name_list.append(outer)

        return (
            data_list,
            data_name_list,
            outer_name_list,
        )

    def validate_gc_rf_size(self, savefigname=None):
        gen_rfs = self.project_data.construct_retina["gen_rfs"]

        if self.context.retina_parameters["spatial_model_type"] == "VAE":
            gen_rfs = gen_rfs
            gc_vae_img = gen_rfs["gc_vae_img"]

            new_um_per_pix = self.construct_retina.updated_vae_um_per_pix

            # Get ellipse DOG and VAE DOG values
            gc_df = self.construct_retina.gc_df_original
            gc_vae_df = self.construct_retina.gc_vae_df

            fit = self.construct_retina.Fit(
                self.context.dog_metadata_parameters,
                self.construct_retina.gc_type,
                self.construct_retina.response_type,
                spatial_data=gc_vae_img,
                fit_type="concentric_rings",
                new_um_per_pix=new_um_per_pix,
            )

            all_data_fits_df = fit.all_data_fits_df
            gen_spat_filt = fit.gen_spat_filt
            good_idx_rings = fit.good_idx_rings
        else:
            raise ValueError(
                "Only VAE spatial_model_type is supported for validate_gc_rf_size, it shows DOG values, too."
            )

        # cr for concentric rings, i.e. the symmetric DoG model
        # Scales pix to mm for semi_xc i.e. central radius for cd fits
        gc_vae_cr_df = self.construct_retina._update_vae_gc_df(
            all_data_fits_df, new_um_per_pix
        )

        # Center radius and eccentricity for cr
        deg_per_mm = self.construct_retina.deg_per_mm
        cen_mm_cr = gc_vae_cr_df["semi_xc"].values
        cen_deg_cr = cen_mm_cr * deg_per_mm
        cen_min_arc_cr = cen_deg_cr * 60
        ecc_mm_cr = self.construct_retina.gc_vae_df["pos_ecc_mm"].values
        ecc_deg_cr = ecc_mm_cr * deg_per_mm

        # Center radius and eccentricity for ellipse fit
        cen_mm_fit = np.sqrt(gc_df["semi_xc"].values * gc_df["semi_yc"].values)
        cen_deg_fit = cen_mm_fit * deg_per_mm
        cen_min_arc_fit = cen_deg_fit * 60
        ecc_mm_fit = gc_df["pos_ecc_mm"].values
        ecc_deg_fit = ecc_mm_fit * deg_per_mm

        # Center radius and eccentricity for vae ellipse fit
        cen_mm_vae = np.sqrt(gc_vae_df["semi_xc"].values * gc_vae_df["semi_yc"].values)
        cen_deg_vae = cen_mm_vae * deg_per_mm
        cen_min_arc_vae = cen_deg_vae * 60
        ecc_mm_vae = gc_vae_df["pos_ecc_mm"].values
        ecc_deg_vae = ecc_mm_vae * deg_per_mm

        # Read in corresponding data from literature
        spatial_DoG_path = self.context.literature_data_files["spatial_DoG_path"]
        spatial_DoG_data = self.data_io.get_data(spatial_DoG_path)

        lit_ecc_deg = spatial_DoG_data["Xdata"]  # ecc (deg)
        lit_cen_min_arc = spatial_DoG_data["Ydata"]  # rf center radius (min of arc)

        # Plot the results
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plt.plot(ecc_deg_fit, cen_min_arc_fit, "o", label="Ellipse fit")
        plt.plot(ecc_deg_vae, cen_min_arc_vae, "o", label="VAE ellipse fit")
        plt.plot(ecc_deg_cr, cen_min_arc_cr, "o", label="VAE concentric rings fit")
        plt.plot(lit_ecc_deg, lit_cen_min_arc, "o", label="Schottdorf_2021_JPhysiol")
        plt.xlabel("Eccentricity (deg)")
        plt.ylabel("Center radius (min of arc)")
        plt.legend()
        plt.title(
            f"GC dendritic diameter vs eccentricity, {self.construct_retina.gc_type} type"
        )

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_catplot(self, param_plot_dict):
        """
        Visualization of parameter values in different categories. Data is collected in _build_param_plot, and all plotting is here.

        Definitions for parametric plotting of multiple conditions/categories.

        First, define what data is going to be visualized in to_spa_dict.
        Second, define how it is visualized in param_plot_dict.

        Limitations:
            You cannot have analyzes as title AND inner_sub = True.
            For violinplot and inner_sub = True, N bin edges MUST be two (split view)

        outer : panel (distinct subplots) # analyzes, startpoints, parameters, controls
        inner : inside one axis (subplot) # startpoints, parameters, controls

        The dictionary xy_plot_dict contains:

        title : str
            Title-level of plot, e.g. "parameters". Multiple allowed => each in separate figure
        outer : str
            Panel-level of plot, e.g. "analyzes". Multiple allowed => plt subplot panels
        inner : str
            Inside one axis (subplot) level of plot, e.g. "startpoints". Multiple allowed => direct comparison
        inner_sub : bool
            Further subdivision by value, such as mean firing rate
        inner_sub_ana : str
            Name of ana. This MUST be included into to_spa_dict "analyzes". E.g. "Excitatory Firing Rate"
        bin_edges : list of lists
            Binning of data. E.g. [[0.001, 150], [150, 300]]
        plot_type : str
            Parametric plot type. Allowed types include "box", "violin", "strip", "swarm", "boxen", "point" and "bar".
        compiled_results : bool
            Data at compiled_results folder, mean over iterations
        sharey : bool
            Share y-axis between subplots
        inner_paths : bool
            Provide comparison from arbitrary paths, e.g. controls
        paths : list of tuples
            Provide list of tuples of full path parts to data folder.
            E.g. [(root_path, 'Single_narrow_iterations_control', 'Bacon_gL_compiled_results'),]
            The number of paths MUST be the same as the number of corresponding inner variables.
        """

        coll_ana_df = copy.deepcopy(self.coll_spa_dict["coll_ana_df"])
        to_spa_dict = copy.deepcopy(self.context.to_spa_dict)

        titles = to_spa_dict[param_plot_dict["title"]]

        if param_plot_dict["save_description"] is True:
            describe_df_list = []
            describe_df_columns_list = []
            describe_folder_full = Path.joinpath(self.context.path, "Descriptions")
            describe_folder_full.mkdir(parents=True, exist_ok=True)

        # If param_plot_dict["inner_paths"] is True, replace titles with and [""] .
        if param_plot_dict["inner_paths"] is True:
            titles = [""]

        # Recursive call for multiple titles => multiple figures
        for this_title in titles:
            this_title_list = [this_title]
            to_spa_dict[param_plot_dict["title"]] = this_title_list

            (
                data_list,
                data_name_list,
                data_sub_list,
                outer_name_list,
                sub_col_name,
            ) = self._build_param_plot(coll_ana_df, param_plot_dict, to_spa_dict)

            sharey = param_plot_dict["sharey"]
            palette = param_plot_dict["palette"]

            if param_plot_dict["display_optimal_values"] is True:
                optimal_value_foldername = param_plot_dict["optimal_value_foldername"]
                optimal_description_name = param_plot_dict["optimal_description_name"]

                # read optimal values to dataframe from path/optimal_values/optimal_unfit_description.csv
                optimal_df = pd.read_csv(
                    Path.joinpath(
                        self.context.path,
                        optimal_value_foldername,
                        optimal_description_name,
                    )
                )
                # set the first column as index
                optimal_df.set_index(optimal_df.columns[0], inplace=True)

            fig, [axs] = plt.subplots(1, len(data_list), sharey=sharey, squeeze=False)

            if (
                "divide_by_frequency" in param_plot_dict
                and param_plot_dict["divide_by_frequency"] is True
            ):
                # Divide by frequency

                frequency_names_list = [
                    "Excitatory Firing Rate",
                    "Inhibitory Firing Rate",
                ]

                # Get the index of the frequency names
                out_fr_idx = np.array([], dtype=int)
                for out_idx, this_name in enumerate(outer_name_list):
                    if this_name in frequency_names_list:
                        out_fr_idx = np.append(out_fr_idx, out_idx)

                # Sum the two frequency values, separately for each inner
                frequencies = np.zeros([len(data_list[0][0]), len(data_list[0])])
                # Make a numpy array of zeros, whose shape is length
                for this_idx in out_fr_idx:
                    this_fr_list = data_list[this_idx]
                    for fr_idx, this_fr in enumerate(this_fr_list):
                        frequencies[:, fr_idx] += this_fr.values

                # Drop the out_fr_idx from the data_list
                data_list = [
                    l for idx, l in enumerate(data_list) if idx not in out_fr_idx
                ]
                outer_name_list = [
                    l for idx, l in enumerate(outer_name_list) if idx not in out_fr_idx
                ]

                # Divide the values by the frequencies
                for out_idx, this_data_list in enumerate(data_list):
                    for in_idx, this_data in enumerate(this_data_list):
                        new_values = this_data.values / frequencies[:, in_idx]
                        # Assign this_data.values back to the data_list
                        data_list[out_idx][in_idx] = pd.Series(
                            new_values, name=this_data.name
                        )

            for out_idx, inner_data_list in enumerate(data_list):
                outer_name = outer_name_list[out_idx]
                inner_df_coll = pd.DataFrame()
                sub_df_coll = pd.DataFrame()
                for in_idx, inner_series in enumerate(inner_data_list):
                    inner_df_coll[data_name_list[out_idx][in_idx]] = inner_series
                    if param_plot_dict["inner_sub"] is True:
                        sub_df_coll[data_name_list[out_idx][in_idx]] = data_sub_list[
                            out_idx
                        ][in_idx]

                self.data_is_valid(inner_df_coll.values, accept_empty=False)

                # For backwards compatibility in FCN22 project 221209 SV
                if outer_name == "Coherence":
                    if inner_df_coll.max().max() > 1:
                        inner_df_coll = inner_df_coll / 34
                if param_plot_dict["save_description"] is True:
                    describe_df_list.append(inner_df_coll)  # for saving
                    describe_df_columns_list.append(f"{outer_name}")

                # We use axes level plots instead of catplot which is figure level plot.
                # This way we can control the plotting order and additional arguments
                if param_plot_dict["inner_sub"] is False:
                    # wide df--each column plotted
                    boxprops = dict(
                        linestyle="-", linewidth=1, edgecolor="black", facecolor=".7"
                    )

                    if param_plot_dict["plot_type"] == "box":
                        g1 = sns.boxplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            boxprops=boxprops,
                            whis=[0, 100],
                            showfliers=False,
                            showbox=True,
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "violin":
                        g1 = sns.violinplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "strip":
                        g1 = sns.stripplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "swarm":
                        g1 = sns.swarmplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "boxen":
                        g1 = sns.boxenplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "point":
                        g1 = sns.pointplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "bar":
                        g1 = sns.barplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )

                elif param_plot_dict["inner_sub"] is True:
                    inner_df_id_vars = pd.DataFrame().reindex_like(inner_df_coll)
                    # Make a “long-form” DataFrame
                    for this_bin_idx, this_bin_limits in enumerate(
                        param_plot_dict["bin_edges"]
                    ):
                        # Apply bin edges to sub data
                        inner_df_id_vars_idx = sub_df_coll.apply(
                            lambda x: (x > this_bin_limits[0])
                            & (x < this_bin_limits[1]),
                            raw=True,
                        )
                        inner_df_id_vars[inner_df_id_vars_idx] = this_bin_idx

                    inner_df_id_values_vars = pd.concat(
                        [
                            inner_df_coll.stack(dropna=False),
                            inner_df_id_vars.stack(dropna=False),
                        ],
                        axis=1,
                    )

                    inner_df_id_values_vars = inner_df_id_values_vars.reset_index()
                    inner_df_id_values_vars.drop(columns="level_0", inplace=True)
                    inner_df_id_values_vars.columns = [
                        "Title",
                        outer_name,
                        sub_col_name,
                    ]

                    bin_legends = [
                        f"{m}-{n}" for [m, n] in param_plot_dict["bin_edges"]
                    ]
                    inner_df_id_values_vars[sub_col_name].replace(
                        to_replace=[*range(0, len(param_plot_dict["bin_edges"]))],
                        value=bin_legends,
                        inplace=True,
                    )

                    if param_plot_dict["plot_type"] == "box":
                        g1 = sns.boxplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            whis=[0, 100],
                            showfliers=False,
                            showbox=True,
                        )
                    elif param_plot_dict["plot_type"] == "violin":
                        g1 = sns.violinplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            split=True,
                        )
                    elif param_plot_dict["plot_type"] == "strip":
                        g1 = sns.stripplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )
                    elif param_plot_dict["plot_type"] == "swarm":
                        g1 = sns.swarmplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )
                    elif param_plot_dict["plot_type"] == "boxen":
                        g1 = sns.boxenplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )
                    elif param_plot_dict["plot_type"] == "point":
                        g1 = sns.pointplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )
                    elif param_plot_dict["plot_type"] == "bar":
                        g1 = sns.barplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )

                g1.set(xlabel=None, ylabel=None)
                fig.suptitle(this_title, fontsize=16)

                labels = data_name_list[out_idx]
                axs[out_idx].set_xticklabels(labels, rotation=60)

                if param_plot_dict["display_optimal_values"] is True:
                    # Get column name from coll_ana_df
                    col_name = coll_ana_df.loc[outer_name, "csv_col_name"]
                    matching_column = [
                        c
                        for c in optimal_df.columns
                        if c.startswith(col_name) and c.endswith("_mean")
                    ]
                    if len(matching_column) > 0:
                        min_value = optimal_df.loc["min", matching_column[0]]
                        max_value = optimal_df.loc["max", matching_column[0]]
                        # draw a horizontal dashed line to axs[out_idx] at y=min_value and y=max_value
                        axs[out_idx].axhline(y=min_value, color="black", linestyle="--")
                        axs[out_idx].axhline(y=max_value, color="black", linestyle="--")

                # To get min max etc if necessary
                # print(inner_df_coll.describe())

                # If statistics is tested, set statistics value and name to each axs subplot
                if param_plot_dict["inner_stat_test"] is True:
                    """
                    Apply the statistical test to inner_df_coll
                    If len(inner_data_list) == 2, apply Wilcoxon signed-rank test.
                    Else if len(inner_data_list) > 2, apply Friedman test.
                    Set stat_name to the test name.
                    """
                    if len(inner_data_list) == 2:
                        stat_test_name = "Wilcoxon signed-rank test"
                        statistics, stat_p_value = self.ana.stat_tests.wilcoxon_test(
                            inner_df_coll.values[:, 0], inner_df_coll.values[:, 1]
                        )
                    elif len(inner_data_list) > 2:
                        stat_test_name = "Friedman test"
                        statistics, stat_p_value = self.ana.stat_tests.friedman_test(
                            inner_df_coll.values
                        )
                    else:
                        raise ValueError(
                            "len(inner_data_list) must be 2 or more for stat_test, aborting..."
                        )

                    # Find the column with largest median value, excluding nans
                    median_list = []
                    for this_idx, this_column in enumerate(inner_df_coll.columns):
                        median_list.append(
                            np.nanmedian(inner_df_coll.values[:, this_idx])
                        )
                    max_median_idx = np.argmax(median_list)

                    # If p-value is less than 0.05, append
                    # the str(max_median_idx) to stat_p_value
                    if stat_p_value < 0.05:
                        stat_corrected_p_value_str = f"{stat_p_value:.3f} (max median at {data_name_list[out_idx][max_median_idx]})"
                    else:
                        stat_corrected_p_value_str = f"{stat_p_value:.3f}"

                    axs[out_idx].set_title(
                        f"{outer_name}\n{stat_test_name} =\n{stat_corrected_p_value_str}\n{statistics:.1f}\nN = {inner_df_coll.shape[0]}"
                    )
                else:
                    axs[out_idx].set_title(outer_name)

            if param_plot_dict["save_description"] is True:
                describe_df_columns_list = [
                    c.replace(" ", "_") for c in describe_df_columns_list
                ]
                describe_df_all = pd.DataFrame()
                for this_idx, this_column in enumerate(describe_df_columns_list):
                    # Append the describe_df_all data describe_df_list[this_idx]
                    this_describe_df = describe_df_list[this_idx]
                    # Prepend the column names with this_column
                    this_describe_df.columns = [
                        this_column + "_" + c for c in this_describe_df.columns
                    ]

                    describe_df_all = pd.concat(
                        [describe_df_all, this_describe_df], axis=1
                    )

                filename_full = Path.joinpath(
                    describe_folder_full,
                    param_plot_dict["save_name"] + "_" + this_title + ".csv",
                )

                # Save the describe_df_all dataframe .to_csv(filename_full, index=False)
                describe_df_all_df = describe_df_all.describe()
                describe_df_all_df.insert(
                    0, "description", describe_df_all.describe().index
                )
                describe_df_all_df.to_csv(filename_full, index=False)
                describe_df_list = []
                describe_df_columns_list = []

            if self.save_figure_with_arrayidentifier is not None:
                id = "box"

                self._figsave(
                    figurename=f"{self.save_figure_with_arrayidentifier}_{id}_{this_title}",
                    myformat="svg",
                    subfolderpath=self.save_figure_to_folder,
                )
                self._figsave(
                    figurename=f"{self.save_figure_with_arrayidentifier}_{id}_{this_title}",
                    myformat="png",
                    subfolderpath=self.save_figure_to_folder,
                )


class VizResponse:
    """
    Show spiking rate dynamically together with the stimulus.
    """

    def __init__(self, context, data_io, project_data, VisualSignal):
        self.context = context
        self.data_io = data_io
        self.project_data = project_data

        self.VisualSignal = VisualSignal  # Attach class but do not init yet.

    def _get_convolved_spike_matrix(
        self, spike_idx, spike_times, n_units, n_tp, video_dt, bin_edges, window_length
    ):
        spike_mtx = np.zeros((n_units, n_tp), dtype=float)

        # Make one-dimensional envelope for smoothing
        window_length_tp = np.ceil(window_length / video_dt)
        window_length_tp = int(window_length_tp)
        # Create an exponential decay envelope
        time_points = np.linspace(0, window_length_tp - 1, window_length_tp)
        exponential_window = np.exp(-time_points / (window_length_tp / 5))

        # Normalize the window so that it sums to 1
        exponential_window /= np.sum(exponential_window)

        for this_unit in range(n_units):
            # Convolve each unit spike train with the Gaussian window
            this_unit_mask = spike_idx == this_unit
            this_unit_spikes = spike_times[this_unit_mask]
            this_unit_histogram, _ = np.histogram(
                this_unit_spikes,
                bins=bin_edges,
            )
            spike_mtx[this_unit, :] = np.convolve(
                this_unit_histogram, exponential_window, mode="same"
            )
        spike_mtx /= video_dt  # convert to Hz

        return spike_mtx

    def client(
        self,
        video_file_name: str,
        response_file_name: str,
        window_length: float,
        rate_scale: float = 10,
    ):
        """
        Show the response of the retina to a video stimulus.

        Parameters
        ----------
        video_file_name : str
            The name of the video file.
        response_file_name : str
            The name of the response file.
        window_length : float
            The length of the window for the response visualization in seconds.
        """

        stimulus_video = self.data_io.load_stimulus_from_videofile(video_file_name)

        self.vs = self.VisualSignal(
            self.context.visual_stimulus_parameters,
            self.context.retina_parameters["retina_center"],
            self.data_io.load_stimulus_from_videofile,
            self.context.run_parameters["simulation_dt"],
            self.context.retina_parameters["deg_per_mm"],
            self.context.retina_parameters["optical_aberration"],
            self.context.visual_stimulus_parameters["pix_per_deg"],
            stimulus_video=stimulus_video,
        )
        stimulus_video = self.vs.load_stimulus_from_videofile(video_file_name)

        # Extract retina center and unit positions
        retina_center = self.context.retina_parameters["retina_center"]
        retina_center_deg = (retina_center.real, retina_center.imag)

        # normalize frames to [0, 1] for visualization
        frames = stimulus_video.frames / 256

        fps = stimulus_video.fps
        baseline_len_tp = stimulus_video.baseline_len_tp
        video_height_deg = stimulus_video.video_height_deg
        video_width_deg = stimulus_video.video_width_deg
        stim_len_tp = stimulus_video.n_stim_tp

        # Set stimulus onset at 0 seconds
        n_tp = frames.shape[0]
        video_dt = 1.0 / fps  # seconds
        tp_in_seconds = np.arange(0, n_tp * video_dt, video_dt)
        bin_edges_in_seconds = np.append(tp_in_seconds, tp_in_seconds[-1] + video_dt)
        baseline_in_seconds = video_dt * baseline_len_tp
        frame_seconds = tp_in_seconds - baseline_in_seconds
        stim_start_seconds = frame_seconds[baseline_len_tp]
        stim_stop_seconds = frame_seconds[stim_len_tp + baseline_len_tp - 1]

        # Get spike data
        response_dict = self.data_io.get_data(filename=response_file_name)
        spike_idx = response_dict["spikes_0"][0]
        spike_times = response_dict["spikes_0"][1] / b2u.second
        n_units = response_dict["n_units"]

        # Convolve the spike matrix with a exponential decay window
        spike_mtx = self._get_convolved_spike_matrix(
            spike_idx,
            spike_times,
            n_units,
            n_tp,
            video_dt,
            bin_edges_in_seconds,
            window_length,
        )
        unit_positions = response_dict["z_coord"]
        unit_x_deg = np.real(unit_positions)
        unit_y_deg = np.imag(unit_positions)

        plt.ion()

        fig, ax = plt.subplots(
            2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [5, 1]}
        )

        # Make colorbar
        cmap = plt.get_cmap("hot")
        norm = plt.Normalize(vmin=0, vmax=rate_scale)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Spiking Rate (Hz)", rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=10)

        # Scale the spike matrix: rate_scale => maximum value in the colorbar
        spike_mtx_norm = norm(spike_mtx)
        spike_mtx_norm_mean = spike_mtx_norm.mean(axis=0)
        # Calculate window width in terms of frame indices
        window_width_frames = int(window_length / video_dt)

        # Create a slider for frame index
        axcolor = "lightgoldenrodyellow"
        ax_frame = plt.axes([0.2, 0.01, 0.6, 0.03], facecolor=axcolor)
        slider_frame = widgets.Slider(
            ax_frame, "Frame", 0, n_tp - window_width_frames, valinit=0, valstep=1
        )

        def update(val):
            frame_idx = int(slider_frame.val)
            frame = frames[frame_idx]
            ax[0].clear()
            ax[1].clear()

            # Show the stimulus frame
            ax[0].imshow(
                frame,
                extent=[
                    -video_width_deg / 2 + retina_center_deg[0],
                    video_width_deg / 2 + retina_center_deg[0],
                    -video_height_deg / 2 + retina_center_deg[1],
                    video_height_deg / 2 + retina_center_deg[1],
                ],
                aspect="equal",
                vmin=0,
                vmax=1,
            )

            # Plot the spikes as a scatter plot
            ax[0].scatter(
                unit_x_deg,
                unit_y_deg,
                c=spike_mtx_norm[:, frame_idx],
                cmap=cmap,
                norm=norm,
                s=20,
                edgecolor="none",
                alpha=0.7,
            )

            ax[0].set_xlim(
                -video_width_deg / 2 + retina_center_deg[0],
                video_width_deg / 2 + retina_center_deg[0],
            )
            ax[0].set_ylim(
                -video_height_deg / 2 + retina_center_deg[1],
                video_height_deg / 2 + retina_center_deg[1],
            )
            ax[0].set_xlabel("X (deg)")
            ax[0].set_ylabel("Y (deg)")

            ax[0].set_title(f"Frame {frame_seconds[frame_idx]:.2f} s")

            ax[1].plot(frame_seconds, spike_mtx_norm_mean, color="black", linewidth=0.2)

            ax[1].vlines(
                x=np.array([stim_start_seconds, stim_stop_seconds]),
                ymin=0,
                ymax=rate_scale,
                color="red",
                linewidth=1,
            )

            # Show sliding window as a transparent rectangle
            window_rect = patches.Rectangle(
                (frame_seconds[frame_idx] - window_length, 0),
                window_length,
                rate_scale,
                linewidth=1,
                edgecolor="black",
                facecolor="gray",
                alpha=0.3,
            )
            ax[1].add_patch(window_rect)

            ax[1].set_xlim(
                frame_seconds[0],
                frame_seconds[-1],
            )
            ax[1].set_ylim(0, rate_scale)
            ax[1].set_xlabel("Time (s)")

            fig.canvas.draw_idle()

        slider_frame.on_changed(update)

        # Initial plot
        update(0)

        plt.show(block=True)
