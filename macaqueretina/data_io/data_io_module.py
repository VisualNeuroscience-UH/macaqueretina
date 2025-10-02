# Built-in
import logging
import os
import pickle
import types
import zlib

# from functools import lru_cache
from pathlib import Path

# Third-party
import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as scprs
import yaml
from brian2.input.timedarray import TimedArray

# Local
from macaqueretina.context.context_module import Context


class DataIO:
    def __init__(self, context) -> None:
        self.context = context

        # Attach other methods/packages
        self.savemat = sio.savemat
        self.csr_matrix = scprs.csr_matrix

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        if isinstance(value, Context):
            self._context = value
        else:
            raise AttributeError(
                "Trying to set improper context. Context must be a context object."
            )

    def _write_to_file(self, save_path, data):
        def _remove_timed_arrays(obj):
            # Timed arrays of brian contain some expressions which cannot be saved by pickle,
            if not isinstance(obj, dict):
                return obj
            keys = list(obj.keys())
            for key in keys:
                if isinstance(obj[key], TimedArray):
                    del obj[key]
                elif isinstance(obj[key], dict):
                    obj[key] = _remove_timed_arrays(obj[key])
            return obj

        save_path = Path(save_path)
        data = _remove_timed_arrays(data)
        with open(save_path.as_posix(), "wb") as fb:
            fb.write(zlib.compress(pickle.dumps(data, pickle.HIGHEST_PROTOCOL), 9))

    def _check_candidate_file(self, path, filename):
        candidate_data_fullpath_filename = Path.joinpath(path, filename)
        if candidate_data_fullpath_filename.is_file():
            data_fullpath_filename = candidate_data_fullpath_filename
            return data_fullpath_filename
        else:
            return None

    def listdir_loop(self, path, substring=None, exclude_substring=None):
        """
        Find files and folders in path with key substring substring and exclusion substring exclude_substring
        """
        files = []
        for f in Path.iterdir(path):
            if substring is not None and exclude_substring is not None:
                if (
                    substring.lower() in f.as_posix().lower()
                    and exclude_substring.lower() not in f.as_posix().lower()
                ):
                    files.append(f)
            elif substring is not None and exclude_substring is None:
                if substring.lower() in f.as_posix().lower():
                    files.append(f)
            elif substring is None and exclude_substring is not None:
                if exclude_substring.lower() not in f.as_posix().lower():
                    files.append(f)
            else:
                files.append(f)

        paths = [Path.joinpath(path, basename) for basename in files]

        return paths

    def most_recent(self, path, substring=None, exclude_substring=None):
        paths = self.listdir_loop(path, substring, exclude_substring)

        if not paths:
            return None
        else:
            data_fullpath_filename = max(paths, key=os.path.getmtime)
            return data_fullpath_filename

    def most_recent_pattern(self, path, pattern):
        files = list(Path(path).glob(pattern))

        if not files:
            raise FileNotFoundError(f"\nNo {pattern} found in {path}")
        else:
            # Take most recent matching file
            return max(files, key=os.path.getmtime)

    def all_patterns(self, path, pattern):
        """
        Find all files in path with key pattern.
        """
        files = list(Path(path).glob(pattern))

        if not files:
            raise FileNotFoundError(f"\nNo {pattern} found in {path}")
        else:
            return files

    def parse_path(self, filename, substring=None, exclude_substring=None):
        """
        This function returns full path to either given filename or to most recently
        updated file of given substring (a.k.a. containing key substring in filename).
        Note that the substring can be timestamp in the filename.
        """
        data_fullpath_filename = None
        path = self.context.path
        experiment = self.context.experiment
        input_folder = self.context.input_folder
        output_folder = self.context.output_folder
        stimulus_folder = self.context.stimulus_folder

        if output_folder is not None:
            # Check if output folder is absolute path
            if Path(output_folder).is_absolute():
                output_path = output_folder
            else:
                output_path = Path.joinpath(path, output_folder)
        else:
            # Set current path if run separately from project
            output_path = Path.joinpath(path, "./")
        if input_folder is not None:
            # Check if input folder is absolute path
            if Path(input_folder).is_absolute():
                input_path = input_folder
            else:
                input_path = Path.joinpath(path, input_folder)
        else:
            input_path = Path.joinpath(path, "./")
        if stimulus_folder is not None:
            # Check if stimulus folder is absolute path
            if Path(stimulus_folder).is_absolute():
                stimulus_path = stimulus_folder
            else:
                stimulus_path = Path.joinpath(path, Path(experiment), stimulus_folder)

        # Check first for direct load in current directory. E.g. for direct ipython testing
        # if isinstance(filename, str) and len(filename > 0):
        if len(str(filename)) > 0:
            data_fullpath_filename = self._check_candidate_file(Path("./"), filename)

            # Next check direct load in output path, stimulus_path, input path and project path in this order
            # Keep output path as first, because if you rebuild the retina in the same run, it will be at the output path.
            if not data_fullpath_filename:
                data_fullpath_filename = self._check_candidate_file(
                    output_path, filename
                )
            if not data_fullpath_filename:
                data_fullpath_filename = self._check_candidate_file(
                    stimulus_path, filename
                )
            if not data_fullpath_filename:
                data_fullpath_filename = self._check_candidate_file(
                    input_path, filename
                )
            if not data_fullpath_filename:
                data_fullpath_filename = self._check_candidate_file(path, filename)

        # Parse substring next in project/input and project paths
        elif substring is not None or exclude_substring is not None:
            # Parse output folder for given substring
            data_fullpath_filename = self.most_recent(
                output_path, substring=substring, exclude_substring=exclude_substring
            )
            if not data_fullpath_filename:
                # Check for substring next in stimulus folder
                data_fullpath_filename = self.most_recent(
                    stimulus_path,
                    substring=substring,
                    exclude_substring=exclude_substring,
                )
            if not data_fullpath_filename:
                # Check for substring first in input folder
                data_fullpath_filename = self.most_recent(
                    input_path, substring=substring, exclude_substring=exclude_substring
                )
            if not data_fullpath_filename:
                # Check for substring next in project folder
                data_fullpath_filename = self.most_recent(
                    path, substring=substring, exclude_substring=exclude_substring
                )

        return data_fullpath_filename

    # @lru_cache(maxsize=128)
    def get_data(
        self,
        filename=None,
        substring=None,
        exclude_substring=None,
        return_filename=False,
        full_path=None,
    ):
        """
        Open requested file and get data.
        :param filename: str, filename
        :param substring: str, keyword in filename
        :param exclude_substring: str, exclusion keyword, exclude_substring despite substring keyword in filename
        """

        if full_path is None:
            if substring is not None:
                substring = substring.lower()
            # Explore which is the most recent file in path of substring and add full path to filename
            data_fullpath_filename = self.parse_path(
                filename, substring=substring, exclude_substring=exclude_substring
            )
        else:
            if isinstance(full_path, str):
                full_path = Path(full_path)
            if not full_path.is_file():
                data_fullpath_filename = None

            data_fullpath_filename = full_path

        if data_fullpath_filename is None:
            raise FileNotFoundError(f"I could not find file {filename}, aborting...")

        # Open file by extension type
        filename_extension = data_fullpath_filename.suffix

        if filename_extension in [".gz", ".pkl"]:
            try:
                fi = open(data_fullpath_filename, "rb")
                data_pickle = zlib.decompress(fi.read())
                data = pickle.loads(data_pickle)
            except:
                with open(data_fullpath_filename, "rb") as data_pickle:
                    data = pickle.load(data_pickle)
        elif ".mat" in filename_extension:
            data = {}
            sio.loadmat(data_fullpath_filename, data)
        elif "csv" in filename_extension:
            data = pd.read_csv(data_fullpath_filename)
            if "Unnamed: 0" in data.columns:
                # data = data.drop(["Unnamed: 0"], axis=1)
                data.set_index("Unnamed: 0", inplace=True)
                data.index.name = None
        elif filename_extension in [".jpg", ".png"]:
            image = cv2.imread(
                str(data_fullpath_filename), 0
            )  # The 0-flag calls for grayscale. Comes in as uint8 type

            # Normalize image intensity to 0-1, if RGB value
            if np.ptp(image) > 1:
                data = np.float32(image / 256)
            else:
                data = np.float32(image)  # 16 bit to save space and memory
        elif filename_extension in [".avi", ".mp4"]:
            video_data = cv2.VideoCapture(str(data_fullpath_filename))
            data = self._video_capture2numpy_array(video_data)
        elif filename_extension in [".npy"]:
            data = np.load(data_fullpath_filename)
        elif filename_extension in [".npz"]:
            data = np.load(data_fullpath_filename, allow_pickle=True)
        elif filename_extension in [".h5", ".hdf5"]:
            data = self.load_dict_from_hdf5(data_fullpath_filename)
        elif filename_extension in [".yaml"]:
            data = self.load_dict_from_yaml(data_fullpath_filename)
        else:
            raise TypeError("U r trying to input unknown filetype, aborting...")

        print(f"Loaded file {data_fullpath_filename}")
        # Check for existing loggers (python builtin, called from other modules, such as the run_script.py)
        if logging.getLogger().hasHandlers():
            logging.info(f"Loaded file {data_fullpath_filename}")

        if return_filename is True:
            return data, data_fullpath_filename
        else:
            return data

    def _video_capture2numpy_array(self, video_data):
        """
        Convert video data to numpy array
        :param video_data: cv2.VideoCapture object
        :return: numpy array
        """

        frames = []

        # Read frames in a loop
        while True:
            ret, frame = video_data.read()
            if not ret:
                break
            frames.append(frame)

        # Release the VideoCapture object
        video_data.release()
        # Convert list of frames to a numpy array
        data = np.array(frames)

        return data

    def save_dict_to_hdf5(self, filename, dic):
        """
        Save a dictionary to an hdf5 file.

        Parameters
        ----------
        filename : str
            The filename for the hdf5 file to be saved.
        dic : dict
            The dictionary to be saved.

        Notes
        -----
        The function opens an hdf5 file and calls the helper method
        _recursively_save_dict_contents_to_group to store the dictionary contents
        into the hdf5 file.
        """
        with h5py.File(filename, "w") as h5file:
            # self._recursively_save_dict_contents_to_group(h5file, "/", dic)
            self._recursively_save_dict_contents_to_group(h5file, "/", dic)

    def save_dict_to_yaml(self, filename, dic, overwrite=True):
        """
        Save a dictionary to a yaml file.

        Parameters
        ----------
        filename : str
            The filename for the yaml file to be saved.
        dic : dict
            The dictionary to be saved.
        """
        if filename.exists() and not overwrite:
            return
        with open(filename, "w") as file:
            yaml.dump(dic, file, default_flow_style=False)

    def load_dict_from_yaml(self, filename):
        """
        Load a dictionary from a yaml file.

        Parameters
        ----------
        filename : str
            The filename of the yaml file to be loaded.

        Returns
        -------
        dict
            The dictionary loaded from the yaml file.
        """
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
        return data

    def _recursively_save_dict_contents_to_group(
        self, h5file, path, dic, compression="gzip", compression_opts=9
    ):
        """
        Recursively save dictionary contents to a group in an hdf5 file.

        Parameters
        ----------
        h5file : File
            The hdf5 File object to which the data needs to be written.
        path : str
            The path in the hdf5 file where the data needs to be saved.
        dic : dict
            The dictionary whose contents need to be saved.
        compression : str, optional
            The compression strategy to be used while saving data. Default is 'gzip'.
        compression_opts : int, optional
            Specifies a compression preset if gzip is used. Default is 9.

        Raises
        ------
        ValueError
            If an unsupported datatype is provided in the dictionary, a ValueError is raised.

        Notes
        -----
        This function saves all non-None and non-dictionary items in the input dictionary
        into an hdf5 group. The function calls itself recursively for dictionary items
        in the input dictionary. For numerical data (types np.uint64, np.float64, int, float,
        tuple, np.ndarray), gzip compression is used by default.
        """

        for key, item in dic.items():
            if isinstance(item, dict):
                self._recursively_save_dict_contents_to_group(
                    h5file, path + key + "/", item, compression, compression_opts
                )
            elif item is not None:  # If item is None, we skip it
                if path + key in h5file:  # If dataset already exists, delete it
                    del h5file[path + key]
                # Use create_dataset for all non-dictionary items
                if isinstance(item, (np.uint64, np.float64, int, float)):
                    # For numpy types and int and float, we need to wrap them in a numpy array
                    h5file.create_dataset(
                        path + key,
                        data=np.array([item]),
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                elif isinstance(item, str):
                    # For string type, we create a special dtype=h5py.string_dtype() dataset
                    str_type = h5py.string_dtype(encoding="utf-8")
                    h5file.create_dataset(
                        path + key, data=np.array(item, dtype=str_type)
                    )
                elif isinstance(item, tuple):
                    # For tuple, we convert it to list first
                    h5file.create_dataset(
                        path + key,
                        data=np.array(list(item)),
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                elif isinstance(item, np.ndarray):
                    h5file.create_dataset(
                        path + key,
                        data=item,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                elif isinstance(item, type):
                    continue
                elif isinstance(item, types.MethodType):
                    continue
                else:
                    raise ValueError("Cannot save %s type" % type(item))

    def load_dict_from_hdf5(self, filename):
        """
        Load a dictionary from hdf5 file.

        Parameters
        ----------
        filename: hdf5 file name

        Returns
        -------
        dict
            The dictionary loaded from the hdf5 file.
        """
        with h5py.File(filename, "r") as h5file:
            return self._recursively_load_dict_contents_from_group(h5file, "/")

    def _recursively_load_dict_contents_from_group(self, h5file, path):
        data_dict = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                # Convert single item numpy arrays to their corresponding scalars
                val = item[()]
                if isinstance(val, bytes):  # If it's bytes, decode it
                    val = val.decode()
                if isinstance(val, np.ndarray):
                    # If it's a string, decode it
                    if val.dtype == np.dtype("O"):
                        val = val[0].decode()
                    # If it's a size-1 array, convert to python scalar
                    elif val.shape == (1,):
                        val = val[0]
                data_dict[key] = val
            elif isinstance(item, h5py._hl.group.Group):
                data_dict[key] = self._recursively_load_dict_contents_from_group(
                    h5file, path + key + "/"
                )
        return data_dict

    def save_array_to_hdf5(self, filename, array):
        """
        Save a numpy array to hdf5 file.

        Parameters
        ----------
        filename : str
            The name of the hdf5 file to be saved.
        array : np.ndarray
            The numpy array to be saved.
        """
        assert isinstance(array, np.ndarray), f"Cannot save {type(array)} type"
        with h5py.File(filename, "w") as hdf5_file_handle:
            # highest compression as default
            dset = hdf5_file_handle.create_dataset(
                "array", data=array, compression="gzip", compression_opts=6
            )

    def load_array_from_hdf5(self, filename):
        """
        Load a numpy array from hdf5 file.
        :param filename: hdf5 file name
        """

        with h5py.File(filename, "r") as hdf5_file_handle:
            array = hdf5_file_handle["array"][...]
        return array

    def _write_frames_to_mp4_videofile(self, pl_fullpath_filename, stimulus):
        """Write frames to videofile"""

        # Convert frames to uint8
        frames = stimulus.frames.copy()

        if frames.dtype != np.uint8:
            # Scale the frames to 8-bit depth
            frames = (frames / frames.max()) * 255

            frames = np.around(frames).astype(np.uint8)

            print("The mp4 videofile for viewing is saved as uint8 type.")

        # Init openCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*stimulus.options["codec"])
        fullpath_filename = str(pl_fullpath_filename)
        print(f"Saving video to {fullpath_filename}")
        video = cv2.VideoWriter(
            fullpath_filename,
            fourcc,
            float(stimulus.options["fps"]),
            (stimulus.options["image_width"], stimulus.options["image_height"]),
            isColor=False,
        )  # path, codec, fps, size. Note, the isColor the flag is currently supported on Windows only

        # Write frames to videofile frame-by-frame
        for index in np.arange(frames.shape[0]):
            video.write(frames[index, :, :])

        video.release()

    def _check_output_folder(self):
        """
        Create output directory if it does not exist.
        Return full path to output directory.
        """
        output_path = Path.joinpath(self.context.path, self.context.output_folder)
        if not Path(output_path).exists():
            Path(output_path).mkdir(parents=True)

        return output_path

    def _check_stimulus_folder(self):
        """
        Create output directory if it does not exist.
        Return full path to output directory.
        """
        stimulus_path = Path.joinpath(self.context.path, self.context.stimulus_folder)
        if not Path(stimulus_path).exists():
            Path(stimulus_path).mkdir(parents=True)

        return stimulus_path

    def _get_filename_stem_and_suffix(self, filename):
        """
        Get filename stem and suffix.
        """
        filename_stem = Path(filename).stem
        filename_suffix = Path(filename).suffix
        return filename_stem, filename_suffix

    def get_video_full_name(self, filename):
        """
        Add full path to video name.

        Then check if video name has correct extension.
        If not, add correct extension.

        :param filename: video name
        :return: full path to video name
        """

        parent_path = self._check_stimulus_folder()

        if not isinstance(filename, Path):
            filename = Path(filename)

        filename_stem, filename_suffix = self._get_filename_stem_and_suffix(filename)

        if filename_suffix in [".mp4", ".avi"]:
            fullpath_filename = Path.joinpath(
                parent_path, filename_stem + filename_suffix
            )
        elif filename_suffix == "":
            fullpath_filename = Path.joinpath(parent_path, filename_stem + ".mp4")
        else:
            fullpath_filename = Path.joinpath(parent_path, filename_stem + ".mp4")

        return fullpath_filename

    def save_stimulus_to_videofile(self, filename, stimulus):
        """
        Save stimulus to a videofile and an hdf5 file.

        This function saves two different files:
        1. A video file in mp4 or comparable video format for viewing.
        2. A hdf5 file for reloading.

        Parameters
        ----------
        filename : str
            The name of the video file to be saved.
        stimulus : object
            The stimulus object to be saved.
        """

        fullpath_filename = self.get_video_full_name(filename)

        if fullpath_filename.is_file():
            return  # If file already exists, do not overwrite

        # To display to user
        self._write_frames_to_mp4_videofile(fullpath_filename, stimulus)

        # save all stimulus object attributes in the same format
        stimulus_dict = {
            key: value
            for key, value in stimulus.__dict__.items()
            if key not in ["_context", "_data_io", "_cones"]
        }

        full_path_out = f"{fullpath_filename}.hdf5"
        self.save_dict_to_hdf5(full_path_out, stimulus_dict)

    def load_stimulus_from_videofile(self, filename):
        """
        Load stimulus data from an hdf5 video file.

        This method retrieves the full path of the specified video file and
        loads its contents into a dictionary. A dummy VideoBaseClass object is
        then created to represent the stimulus, with its attributes populated
        from the dictionary. If any of the `options` attribute values of the
        stimulus are numpy arrays, they are converted to tuples.

        Parameters
        ----------
        filename : str
            Name of the video file from which to load the stimulus.

        Returns
        -------
        stimulus : DummyVideoClass instance
            An instance of the dummy VideoBaseClass that represents the loaded
            stimulus. Its attributes are populated from the hdf5 file contents.

        Notes
        -----
        - If logging handlers are set, an informational log is produced indicating
        the file that has been loaded.
        - The function assumes that the file extension is `.hdf5` and appends it to the filename.
        """

        fullpath_filename = self.get_video_full_name(filename)

        # load video from hdf5 file
        full_path_in = f"{fullpath_filename}.hdf5"
        try:
            data_dict = self.load_dict_from_hdf5(full_path_in)
        except Exception as e:
            raise FileNotFoundError(f"Stimulus not available at {full_path_in}")

        # Create a dummy VideoBaseCLass object to create a stimulus object
        class DummyVideoClass:
            def __init__(self, data_dict):
                for key, value in data_dict.items():
                    setattr(self, key, value)
                # self.options = options

        stimulus = DummyVideoClass(data_dict)

        # Iterate stimulus.options dict as key, value pairs. If value is numpy array, convert it to tuple
        for key, value in stimulus.options.items():
            if isinstance(value, np.ndarray):
                stimulus.options[key] = tuple(value)

        print(f"Loaded file {full_path_in}")
        # Check for existing loggers (python builtin, called from other modules, such as the run_script.py)
        if logging.getLogger().hasHandlers():
            logging.info(f"Loaded file {full_path_in}")

        return stimulus

    def save_cone_response_to_hdf5(self, filename, cone_response):
        """
        Save cone response to hdf5 file.

        Parameters
        ----------
        filename : str
            The name of the hdf5 file to be saved.
        cone_response : np.ndarray
            The cone response data to be saved.
        """
        parent_path = self._check_output_folder()

        filename_stem, filename_suffix = self._get_filename_stem_and_suffix(filename)

        filename_extension = "_cone_response"
        stem_extension = filename_stem + filename_extension

        fullpath_filename = Path.joinpath(parent_path, stem_extension + ".hdf5")
        self.save_array_to_hdf5(fullpath_filename, cone_response)

    def load_cone_response_from_hdf5(self, filename):
        """
        Load cone response from hdf5 file.
        :param filename: hdf5 file name
        :return: cone response
        """

        parent_path = self.context.output_folder

        filename_stem, filename_suffix = self._get_filename_stem_and_suffix(filename)
        if filename_suffix in ["hdf5"]:
            fullpath_filename = Path.joinpath(
                parent_path, filename_stem + filename_suffix
            )
        else:
            fullpath_filename = Path.joinpath(parent_path, filename_stem + ".hdf5")

        cone_response = self.load_array_from_hdf5(fullpath_filename)

        return cone_response

    def save_analog_stimulus(
        self,
        filename_out=None,
        Input=None,
        z_coord=None,
        w_coord=None,
        frameduration=None,
    ):
        assert all(
            [
                filename_out is not None,
                Input is not None,
                z_coord is not None,
                w_coord is not None,
                frameduration is not None,
            ]
        ), "Some input missing from save_analog_stimulus, aborting..."

        total_duration = Input.shape[1] * frameduration / 1000
        # mat['stimulus'].shape should be (Nunits, Ntimepoints)
        mat_out_dict = {
            "z_coord": z_coord,
            "w_coord": w_coord,
            "stimulus": Input,
            "frameduration": frameduration,
            "stimulus_duration_in_seconds": total_duration,
        }

        filename_out_full = self.context.output_folder.joinpath(filename_out)

        sio.savemat(filename_out_full, mat_out_dict)
        print(f"Duration of stimulus is {total_duration} seconds")

    def save_np_dict_to_npz(
        self, data_dict, output_path, filename_stem="", overwrite=True
    ):
        """
        Save a dictionary of numerical values to an NPZ file.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing numerical data to be saved.
            Each key-value pair corresponds to a variable name and its associated numpy array.

        output_path : str or Path
            The path to the output folder where the NPZ file will be saved.

        filename_stem : str, optional
            The stem of the filename for the saved file. Default is "data_values".
        """

        if isinstance(output_path, str):
            output_path = Path(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        if filename_stem.endswith(".npz"):
            filename = filename_stem
        else:
            filename = filename_stem + ".npz"

        # Construct the filename for saving
        npz_filename = output_path / filename

        if npz_filename.exists() and not overwrite:
            return

        # Save all items in the dictionary as separate arrays in an NPZ file
        # np.savez(npz_filename, **data_dict)
        np.savez_compressed(npz_filename, allow_pickle=True, **data_dict)

    def _save_spikes_for_cxsystem(
        self,
        spikearrays,
        n_units,
        w_coord,
        z_coord,
        filename=None,
        analog_signal=None,
        dt=None,
    ):

        print(" -  Saving spikes, rgc coordinates and analog signal (if not None)...")

        data_to_save = {}
        for ii in range(len(spikearrays)):
            data_to_save["spikes_" + str(ii)] = []
            data_to_save["spikes_" + str(ii)].append(spikearrays[ii][0])
            data_to_save["spikes_" + str(ii)].append(spikearrays[ii][1])
        data_to_save["w_coord"] = w_coord
        data_to_save["z_coord"] = z_coord

        data_to_save["n_units"] = n_units

        if analog_signal is not None:
            data_to_save["analog_signal"] = analog_signal

        if dt is not None:
            data_to_save["dt"] = dt

        if filename is None:
            save_path = self.context.output_folder.joinpath("most_recent_spikes")
        else:
            save_path = self.context.output_folder.joinpath(filename)

        filename_full = save_path.with_suffix(".gz")

        self._write_to_file(filename_full, data_to_save)

    def _save_spikes_csv(self, simulated_spiketrains, n_cells, filename=None):
        """
        Saves spikes as a csv file with rows of the form unit_index and spike_time.
        This file can be used in ViSimpl:
        visimpl.AppImage -csv parasol_structure.csv parasol_spikes.csv

        Parameters
        ----------
        filename: str, optional
            Name of the file to save the spikes to. If None, the filename will be
            generated automatically.
        """
        if len(simulated_spiketrains) == 0:
            print("There are no simulated spiketrains to save")
            return

        spikes_df = pd.DataFrame(columns=["unit_index", "spike_time"])
        for unit_index in range(n_cells):
            spiketrain = simulated_spiketrains[unit_index]
            index_array = unit_index * np.ones(len(spiketrain))
            temp_df = pd.DataFrame(
                np.column_stack((index_array, spiketrain)),
                columns=["unit_index", "spike_time"],
            )
            spikes_df = pd.concat([spikes_df, temp_df], axis=0)

        spikes_df["unit_index"] = spikes_df["unit_index"].astype(int)
        spikes_df = spikes_df.sort_values(by="spike_time")

        if filename is None:
            save_path = self.context.output_folder.joinpath("most_recent_spikes")
        else:
            save_path = self.context.output_folder.joinpath(filename)
        filename_full = save_path.with_suffix(".csv")

        spikes_df.to_csv(filename_full, index=False, header=False)

    def _save_structure_csv(self, rgc_coords, filename=None):
        """
        Saves x,y coordinates of model units to a csv file (for use in ViSimpl).

        Parameters
        ----------
        filename: str, optional
            Name of the file to save the structure to. If None, the filename will be
            generated automatically.
        """
        if filename is None:
            save_path = self.context.output_folder.joinpath("most_recent_structure")
        else:
            save_path = self.context.output_folder.joinpath(
                str(filename) + "_structure"
            )
        filename_full = save_path.with_suffix(".csv")

        rgc_coords["z_deg"] = 0.0

        rgc_coords.to_csv(filename_full, header=False, index=False)

    def _get_w_z_coords(self, gcs):
        """
        Create w_coord, z_coord for cortical and visual coordinates, respectively

        Parameters
        ----------
        None

        Returns
        -------
        w_coord : np.ndarray
            Cortical coordinates
        z_coord : np.ndarray
            Visual coordinates
        """

        # Create w_coord, z_coord for cortical and visual coordinates, respectively
        z_coord = gcs.df["x_deg"].values + 1j * gcs.df["y_deg"].values

        a = gcs.visual2cortical_params["a"]
        k = gcs.visual2cortical_params["k"]
        w_coord = k * np.log(z_coord + a)

        return w_coord, z_coord

    def _save_additional_variables(self, vs, filename, variables, overwrite=True):
        """
        Save additional variables to a file.

        Parameters
        ----------
        vs : object
            The object containing the variables to be saved.
        filename : str
            The name of the file to save the variables to.
        """
        data_dict = {}
        filename_stem = filename
        for variable in variables:
            if hasattr(vs, variable):
                data = getattr(vs, variable)
                data_dict[variable] = data

        self.save_np_dict_to_npz(
            data_dict,
            self.context.output_folder,
            filename_stem=filename_stem,
            overwrite=overwrite,
        )

    def save_retina_output(self, vs, gcs, filename, save_variables=None):

        vs.w_coord, vs.z_coord = self._get_w_z_coords(gcs)

        for this_variable in save_variables:
            match this_variable:
                case "spikes":
                    self._save_spikes_for_cxsystem(
                        vs.spikearrays,
                        vs.n_units,
                        vs.w_coord,
                        vs.z_coord,
                        filename=filename,
                        dt=vs.simulation_dt,
                    )

                case "cone_noise":
                    cone_noise_hash = self.context.retina_parameters["cone_noise_hash"]
                    self._save_additional_variables(
                        vs,
                        f"cone_noise_{cone_noise_hash}",
                        ["cone_noise", "cone_noise_u"],
                        overwrite=False,
                    )

                case "gc_synaptic_noise":
                    cone_noise_hash = self.context.retina_parameters["cone_noise_hash"]
                    gc_type = self.context.retina_parameters["gc_type"]
                    response_type = self.context.retina_parameters["response_type"]

                    filename_gc_noise = (
                        f"{gc_type}_{response_type}_noise_{cone_noise_hash}"
                    )

                    self._save_additional_variables(
                        vs,
                        filename_gc_noise,
                        ["gc_synaptic_noise"],
                        overwrite=False,
                    )

                case "cone_bipo_gen_fir":
                    self._save_additional_variables(
                        vs,
                        filename,
                        [
                            "cone_signal",
                            "bipolar_signal",
                            "generator_potentials",
                            "firing_rates",
                        ],
                    )
