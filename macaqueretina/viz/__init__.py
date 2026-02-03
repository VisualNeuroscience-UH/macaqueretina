from macaqueretina.project.project_manager_module import create_viz


# TODO: module-level cache of the Viz instance, compare the config hash, re-instance only when hash is different


def __getattr__(name):
    from macaqueretina import config

    inst = create_viz(config)
    return getattr(inst, name)


def __dir__():
    from macaqueretina import config

    return dir(create_viz(config))
