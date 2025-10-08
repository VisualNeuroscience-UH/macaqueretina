class Context:
    """
    Set all keyword arguments to ProjectManager constructor into context.
    Variable names with either 'file' or 'folder' in the name become pathlib.Path object in construction (__init__)
    All variables become available in [obj].context.[variable_name], e.g. self.context.path for experiment path.
    """

    def __init__(self, all_properties) -> None:
        pass

    