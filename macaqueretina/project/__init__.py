# Built-in
from typing import TYPE_CHECKING

# Local
from macaqueretina.project.project_manager_module import create_data_sampler_instance
from macaqueretina.project.project_utilities_module import ProjectUtilitiesMixin

if TYPE_CHECKING:
    from macaqueretina.project.project_utilities_module import DataSampler

_cached_project_utlities_mixin_instance = None


class _ProjectUtilitiesMixinWrapper:
    """Wrapper class that routes attribute access to the ProjectUtilitiesMixin instance."""

    def __getattr__(self, name):
        global _cached_project_utlities_mixin_instance

        if _cached_project_utlities_mixin_instance is None:
            _cached_project_utlities_mixin_instance = ProjectUtilitiesMixin()

        return getattr(_cached_project_utlities_mixin_instance, name)

    def __dir__(self):
        global _cached_project_utlities_mixin_instance

        if _cached_project_utlities_mixin_instance is None:
            _cached_project_utlities_mixin_instance = ProjectUtilitiesMixin()

        return dir(_cached_project_utlities_mixin_instance)


project_utilities = _ProjectUtilitiesMixinWrapper()

_cached_data_sampler_instance = None


class _DataSamplerWrapper:
    """Wrapper (and factory) class that routes attribute access to the DataSampler instance."""

    def __call__(
        self,
        filename: str,
        min_X: float,
        max_X: float,
        min_Y: float,
        max_Y: float,
        logX: bool = False,
        logY: bool = False,
    ) -> DataSampler:
        self.filename = filename
        self.min_X = min_X
        self.max_X = max_X
        self.min_Y = min_Y
        self.max_Y = max_Y
        self.logX = logX
        self.logY = logY

        global _cached_data_sampler_instance
        if _cached_data_sampler_instance is None:
            _cached_data_sampler_instance = create_data_sampler_instance(
                self.filename,
                self.min_X,
                self.max_X,
                self.min_Y,
                self.max_Y,
                self.logX,
                self.logY,
            )
        return _cached_data_sampler_instance

    def __getattr__(self, name):
        global _cached_data_sampler_instance
        if _cached_data_sampler_instance is None:
            _cached_data_sampler_instance = create_data_sampler_instance(
                self.filename,
                self.min_X,
                self.max_X,
                self.min_Y,
                self.max_Y,
                self.logX,
                self.logY,
            )
        return getattr(_cached_data_sampler_instance, name)

    def __dir__(self):
        global _cached_data_sampler_instance
        if _cached_data_sampler_instance is None:
            _cached_data_sampler_instance = create_data_sampler_instance(
                self.filename,
                self.min_X,
                self.max_X,
                self.min_Y,
                self.max_Y,
                self.logX,
                self.logY,
            )
        return dir(_cached_data_sampler_instance)


data_sampler = _DataSamplerWrapper()
