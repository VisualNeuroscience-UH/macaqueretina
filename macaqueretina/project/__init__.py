from macaqueretina.project.project_utilities_module import ProjectUtilitiesMixin

_cached_project_utilities_mixin = None


def countlines(startpath, lines=0, header=True, begin_start=None):
    """Wrapper for ProjectUtilitiesMixin.countilines()."""
    global _cached_project_utilities_mixin

    if _cached_project_utilities_mixin is None:
        _cached_project_utilities_mixin = ProjectUtilitiesMixin()
    return _cached_project_utilities_mixin.countlines(
        startpath, lines, header, begin_start
    )
