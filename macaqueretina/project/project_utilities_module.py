# Built-in
import copy
import inspect
import pdb
import sys
import time
from argparse import ArgumentError
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import pandas as pd


class ProjectUtilitiesMixin:
    """
    Utilities for ProjectManager class. This class is not instantiated. It serves as a container for project independent helper functions.
    """

    def pp_df_full(self, df):
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.max_colwidth",
            None,
        ):
            print(df)

    def get_xy_from_npz(self, npz_data):
        """
        Return sorted and squeezed data from an npz data file.
        """
        data_set_x = np.squeeze(npz_data["Xdata"])
        data_set_y = np.squeeze(npz_data["Ydata"])

        data_set_x_index = np.argsort(data_set_x)
        x_data = data_set_x[data_set_x_index]
        y_data = data_set_y[data_set_x_index]

        return x_data, y_data

    def countlines(self, startpath, lines=0, header=True, begin_start=None):
        """
        Counts lines in folder .py files.
        From https://stackoverflow.com/questions/38543709/count-lines-of-code-in-directory-using-python

        Usage:
        lines = PM.countlines(Path("macaqueretina"))
        """

        startpath = Path(startpath).resolve()

        if header:
            print("{:>10} |{:>10} | {:<20}".format("ADDED", "TOTAL", "FILE"))
            print("{:->11}|{:->11}|{:->20}".format("", "", ""))

        for thing in Path.iterdir(startpath):
            thing = Path.joinpath(startpath, thing)
            if thing.is_file():
                if str(thing).endswith(".py"):
                    with open(thing, "r") as f:
                        newlines = f.readlines()
                        newlines = len(newlines)
                        lines += newlines

                        if begin_start is not None:
                            reldir_of_thing = "." + str(thing).replace(
                                str(begin_start), ""
                            )
                        else:
                            reldir_of_thing = "." + str(thing).replace(
                                str(startpath), ""
                            )

                        print(
                            "{:>10} |{:>10} | {:<20}".format(
                                newlines, lines, reldir_of_thing
                            )
                        )

        for thing in Path.iterdir(startpath):
            thing = Path.joinpath(startpath, thing)
            if Path.is_dir(thing):
                lines = self.countlines(
                    thing, lines, header=False, begin_start=startpath
                )

        return lines


class DataSampler:
    """
    Class for digitizing data points from published images.

    Attributes
    ----------
    filename : str
        Path to the image file.
    min_X : float
        Minimum x-axis value for calibration.
    max_X : float
        Maximum x-axis value for calibration.
    min_Y : float
        Minimum y-axis value for calibration.
    max_Y : float
        Maximum y-axis value for calibration.
    logX : bool
        If True, indicates x-axis is logarithmic.
    logY : bool
        If True, indicates y-axis is logarithmic.
    calibration_points : list
        List of calibration points selected from the image.
    data_points : list
        List of data points selected from the image.

    Methods
    -------
    quality_control()
        Displays the original image with calibration and data points.
    collect_and_save_points()
        Interactively collect calibration and data points from the image.
    """

    def __init__(self, filename, min_X, max_X, min_Y, max_Y, logX=False, logY=False):
        self.filename = filename
        self.min_X = min_X
        self.max_X = max_X
        self.min_Y = min_Y
        self.max_Y = max_Y
        self.logX = logX
        self.logY = logY

        self.calibration_points = []
        self.data_points = []

    def _add_calibration_point(self, point):
        """Adds a point to the list of calibration points."""
        self.calibration_points.append(point)

    def _add_data_point(self, point):
        """Adds a point to the list of data points."""
        Xdata, Ydata = self._to_data_units(point[0], point[1])
        self.data_points.append((Xdata, Ydata))

    def _validate_calibration(self):
        """Validates the calibration point input."""
        assert (
            len(self.calibration_points) == 3
        ), "Three calibration points are required."
        x_min, y_min = self.calibration_points[0]
        _, y_max = self.calibration_points[1]
        x_max, _ = self.calibration_points[2]
        assert (
            x_min < x_max and y_min > y_max
        ), "The calibration set is not valid, 1. origo, 2. y max, 3. x max"

    def _to_data_units(self, x, y):
        """Converts image units (pixels) to data units."""
        pix_x_min, pix_y_min = self.calibration_points[0]
        _, pix_y_max = self.calibration_points[1]
        pix_x_max, _ = self.calibration_points[2]
        x_range_pix = pix_x_max - pix_x_min
        y_range_pix = self.calibration_points[0][1] - pix_y_max  # inverted scale

        x_rightwards = x - pix_x_min
        # From bottom upwards, y min is the largest pixel value
        y_upright = pix_y_min - y
        if self.logX:
            x_scaled_log = (x_rightwards / x_range_pix) * (
                np.log10(self.max_X) - np.log10(self.min_X)
            ) + np.log10(self.min_X)
            x_scaled = np.power(10, x_scaled_log)
        else:
            x_scaled = (
                x_rightwards / x_range_pix * (self.max_X - self.min_X) + self.min_X
            )
        if self.logY:
            y_scaled_log = (y_upright / y_range_pix) * (
                np.log10(self.max_Y) - np.log10(self.min_Y)
            ) + np.log10(self.min_Y)

            y_scaled = np.power(10, y_scaled_log)
        else:
            y_scaled = (y_upright / y_range_pix) * (
                self.max_Y - self.min_Y
            ) + self.min_Y

        return x_scaled, y_scaled

    def _to_image_units(self, x_data, y_data):
        """Converts data units back to image units (pixels)."""
        pix_x_min, pix_y_min = self.calibration_points[0]
        _, pix_y_max = self.calibration_points[1]
        pix_x_max, _ = self.calibration_points[2]
        x_range_pix = pix_x_max - pix_x_min
        y_range_pix = pix_y_min - pix_y_max  # inverted scale
        if self.logX:
            x_data_log = np.log10(x_data)
            x_rightwards_log = x_data_log - np.log10(self.min_X)
            x = (
                x_rightwards_log / (np.log10(self.max_X) - np.log10(self.min_X))
            ) * x_range_pix + pix_x_min
        else:
            x = (x_data - self.min_X) / (
                self.max_X - self.min_X
            ) * x_range_pix + pix_x_min
        if self.logY:
            y_data_log = np.log10(y_data)
            y_upwards_log = y_data_log - np.log10(self.min_Y)
            y_upwards_pix = (
                y_upwards_log / (np.log10(self.max_Y) - np.log10(self.min_Y))
            ) * y_range_pix
            y = pix_y_min - y_upwards_pix
        else:
            y_offset = y_data - self.min_Y
            y_upright = y_offset / (self.max_Y - self.min_Y) * y_range_pix
            y = pix_y_min - y_upright

        return x, y

    def _save_data(self):
        """Saves the digitized data to a file."""
        Xdata, Ydata = zip(*[(x, y) for x, y in self.data_points])
        calib_x, calib_y = zip(*[(x, y) for x, y in self.calibration_points])
        nameout = self.filename.stem + "_c"
        filename_full = self.filename.parent / nameout
        np.savez(
            filename_full, Xdata=Xdata, Ydata=Ydata, calib_x=calib_x, calib_y=calib_y
        )
        print(f"Saved data into {filename_full}.npz")

    def _load_data(self):
        """Restores data from file."""
        namein = self.filename.stem + "_c.npz"
        data_filename_full = self.filename.parent / namein
        data = np.load(data_filename_full)
        self.data_points = [(x, y) for x, y in zip(*[data["Xdata"], data["Ydata"]])]
        self.calibration_points = [
            (x, y) for x, y in zip(*[data["calib_x"], data["calib_y"]])
        ]

    def quality_control(self):
        """Displays the original image with calibration and data points."""
        imagedata = plt.imread(self.filename)

        self._load_data()

        # Convert data points to image units for plotting
        data_x, data_y = zip(*[self._to_image_units(x, y) for x, y in self.data_points])
        calib_x = [pt[0] for pt in self.calibration_points]
        calib_y = [pt[1] for pt in self.calibration_points]

        # Print data to screen in beautiful format
        print("Calibration points (pixel coordinates):")
        for i, (x, y) in enumerate(self.calibration_points):
            print(f"Point {i}: x={x:.2e}, y={y:.2e}")
        print("Data points (data coordinates):")
        for i, (x, y) in enumerate(self.data_points):
            print(f"Point {i}: x={x:.2e}, y={y:.2e}")

        fig, ax = plt.subplots()
        ax.imshow(imagedata, cmap="gray")
        ax.scatter(calib_x, calib_y, color="red", s=50, label="Calibration Points")
        ax.scatter(data_x, data_y, color="blue", s=30, label="Data Points")
        ax.legend()

    def collect_and_save_points(self):
        """Interactively collect calibration and data points from the image."""
        imagedata = plt.imread(self.filename)
        fig, ax = plt.subplots()
        ax.imshow(imagedata, cmap="gray")

        # Set the cursor to a crosshair
        cursor = widgets.Cursor(ax, useblit=True, color="red", linewidth=0.5)

        print("Calibrate 1. origo, 2. y max, 3. x max")
        calib_points = plt.ginput(3, timeout=0)
        for point in calib_points:
            self._add_calibration_point(point)

        self._validate_calibration()

        print(
            "And now the data points: left click to add, right click to remove, middle to stop."
        )
        data_points = plt.ginput(-1, timeout=0)
        for point in data_points:
            self._add_data_point(point)
        plt.close(fig)

        self._save_data()

    def get_data_arrays(self):
        """Loads the data for plotting."""
        self._load_data()
        return (
            np.array([point[0].item() for point in self.data_points]),
            np.array([point[1].item() for point in self.data_points]),
        )


class PrintableMixin:
    """
    Mixin class to add pretty-printing capabilities to classes.
    """

    def __str__(self):
        class_info = f"Instance of {self.__class__.__name__}, ID: {id(self)}\n"

        # Getting class, module, and line number information
        class_name = self.__class__.__name__
        module_name = inspect.getmodule(self).__name__
        line_number = inspect.getsourcelines(self.__class__)[1]
        class_info += f"\nClass name: {class_name}\nCreated at: {module_name} line {line_number}\n"

        # Getting signature
        signature = inspect.signature(self.__init__)
        class_info += f"\nSignature:\n{signature}\n"

        # Preparing attributes and methods for pretty printing
        attributes = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]
        methods = [
            method
            for method in dir(self)
            if callable(getattr(self, method)) and not method.startswith("__")
        ]

        # Determining the max length for pretty alignment
        max_attr_name_len = max(len(attr) for attr in attributes) if attributes else 0
        max_attr_type_len = (
            max(len(str(type(getattr(self, attr)))) for attr in attributes)
            if attributes
            else 0
        )
        if max_attr_type_len > 32:
            max_attr_type_len = 32
        max_attr_val_len = 30

        # Compiling attributes information
        attributes_info = "\nAttributes:\n"
        for attr in attributes:
            attr_instance = getattr(self, attr)
            module_name = attr_instance.__class__.__module__
            class_name = attr_instance.__class__.__name__
            full_type_name = (
                f"{module_name}.{class_name}"
                if module_name not in ("__builtin__", "builtins")
                else class_name
            )

            mem_size = ""
            match full_type_name:
                case "pandas.core.frame.DataFrame":
                    mem_size = f"{attr_instance.values.nbytes / 1e6:.2f} MB"
                case "numpy.ndarray":
                    mem_size = f"{attr_instance.nbytes / 1e6:.2f} MB"

            match full_type_name:
                case "pandas.core.frame.DataFrame" | "numpy.ndarray":
                    attr_value = f"shape: {attr_instance.shape}"
                case "dict":
                    attr_value = f"n keys: {len(attr_instance.keys())}"
                case "list" | "tuple":
                    attr_value = len(attr_instance)
                case "str" | "int":
                    attr_value = attr_instance
                case "float":
                    attr_value = f"{attr_instance:.2f}"
                case "brian2.units.fundamentalunits.Quantity":
                    attr_value = f"shape: {attr_instance.shape}, unit: {attr_instance.get_best_unit()}"
                case "data_io.data_io_module.DummyVideoClass":
                    attr_value = ""
                case _:
                    attr_value = attr_instance

            attr_value = str(attr_value)

            attributes_info += f"{attr:<{max_attr_name_len}}\t{full_type_name:<{max_attr_type_len}}\t{attr_value:<{max_attr_val_len}}\t{mem_size}\n"

        # Compiling methods information
        methods_info = "\nMethods:\n" + ",\n".join(methods)

        return class_info + attributes_info + methods_info
