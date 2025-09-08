# Third-party
import numpy as np

# Local
from macaqueretina.analysis.analysis_module import Analysis
from macaqueretina.context.context_module import Context
from macaqueretina.data_io.data_io_module import DataIO
from macaqueretina.project.project_utilities_module import ProjectUtilities
from macaqueretina.retina.construct_retina_module import ConstructRetina
from macaqueretina.retina.fit_module import Fit
from macaqueretina.retina.retina_math_module import RetinaMath
from macaqueretina.retina.simulate_retina_module import (
    ReceptiveFieldsBase,
    SimulateRetina,
    VisualSignal,
)
from macaqueretina.retina.vae_module import RetinaVAE
from macaqueretina.stimuli.experiment_module import Experiment
from macaqueretina.stimuli.visual_stimulus_module import AnalogInput, VisualStimulus
from macaqueretina.viz.viz_module import Viz, VizResponse

"""
Module on retina management

We use dependency injection to make the code more modular and easier to test.
It means that during construction here at the manager level, we can inject
an object instance to constructor of a "client", which becomes an attribute
of the instance.

Simo Vanni 2022
"""


class ProjectData:
    """
    This is a container for project piping data for internal use, such as visualizations.
    """

    def __init__(self) -> None:
        self.construct_retina = {}
        self.simulate_retina = {}
        self.fit = {}


class ProjectManager(ProjectUtilities):
    def __init__(self, config):
        """
        Main project manager.
        In init we construct other classes and inject necessary dependencies.
        This class is allowed to house project-dependent data and methods.
        """

        context = Context(config.as_dict())

        self.context = context

        data_io = DataIO(context)
        self.data_io = data_io

        project_data = ProjectData()

        retina_math = RetinaMath()

        ana = Analysis(
            # Interfaces
            context,
            data_io,
            # Methods which are needed also elsewhere
            pol2cart=retina_math.pol2cart,
            get_photoisomerizations_from_luminance=retina_math.get_photoisomerizations_from_luminance,
        )

        self.ana = ana

        viz = Viz(
            # Interfaces
            context,
            data_io,
            project_data,
            ana,
            # Methods which are needed also elsewhere
            round_to_n_significant=self.round_to_n_significant,
            DoG2D_fixed_surround=retina_math.DoG2D_fixed_surround,
            DoG2D_independent_surround=retina_math.DoG2D_independent_surround,
            DoG2D_circular=retina_math.DoG2D_circular,
            pol2cart=retina_math.pol2cart,
            sector2area_mm2=retina_math.sector2area_mm2,
            interpolate_data=retina_math.interpolate_data,
            lorenzian_function=retina_math.lorenzian_function,
        )

        self.viz = viz

        self.viz_spikes_with_stimulus = VizResponse(
            context,
            data_io,
            project_data,
            VisualSignal,
        )

        fit = Fit(project_data, context.dog_metadata_parameters)

        retina_vae = RetinaVAE(context)

        self.construct_retina = ConstructRetina(
            context,
            data_io,
            viz,
            fit,
            retina_vae,
            retina_math,
            project_data,
            self.get_xy_from_npz,
        )
        self.viz.construct_retina = self.construct_retina

        stimulate = VisualStimulus(context, data_io, self.get_xy_from_npz)
        self.stimulate = stimulate

        simulate_retina = SimulateRetina(
            context, data_io, project_data, retina_math, context.device, stimulate
        )
        self.simulate_retina = simulate_retina

        experiment = Experiment(
            context, data_io, stimulate, simulate_retina)
        self.experiment = experiment

        analog_input = AnalogInput(
            context,
            data_io,
            viz,
            ReceptiveFields=ReceptiveFieldsBase,
            pol2cart_df=self.simulate_retina.pol2cart_df,
            get_w_z_coords=self.simulate_retina.get_w_z_coords,
        )

        self.analog_input = analog_input

        # Set numpy random seed
        np.random.seed(self.context.numpy_seed)

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

    @property
    def data_io(self):
        return self._data_io

    @data_io.setter
    def data_io(self, value):
        if isinstance(value, DataIO):
            self._data_io = value
        else:
            raise AttributeError(
                "Trying to set improper data_io. Data_io must be a DataIO object."
            )

    @property
    def construct_retina(self):
        return self._construct_retina

    @construct_retina.setter
    def construct_retina(self, value):
        if isinstance(value, ConstructRetina):
            self._construct_retina = value
        else:
            raise AttributeError(
                "Trying to set improper construct_retina. construct_retina must be a ConstructRetina instance."
            )

    @property
    def simulate_retina(self):
        return self._working_retina

    @simulate_retina.setter
    def simulate_retina(self, value):
        if isinstance(value, SimulateRetina):
            self._working_retina = value
        else:
            raise AttributeError(
                "Trying to set improper simulate_retina. simulate_retina must be a SimulateRetina instance."
            )

    @property
    def stimulate(self):
        return self._stimulate

    @stimulate.setter
    def stimulate(self, value):
        if isinstance(value, VisualStimulus):
            self._stimulate = value
        else:
            raise AttributeError(
                "Trying to set improper stimulate. stimulate must be a VisualStimulus instance."
            )

    @property
    def analog_input(self):
        return self._analog_input

    @analog_input.setter
    def analog_input(self, value):
        if isinstance(value, AnalogInput):
            self._analog_input = value
        else:
            raise AttributeError(
                "Trying to set improper analog_input. analog_input must be a AnalogInput instance."
            )

