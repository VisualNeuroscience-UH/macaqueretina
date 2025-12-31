"""
This module runs the core_parameters.yaml defined 'run' pipeline when called directly as python macaqueretina.
If environment has YAML_TMPDIR defined (as in HPC run), selected to level keys are explored for futher replacements
before running the pipeline.
"""

# Built-in
import os
import sys
import time
from pathlib import Path

# Third party
import yaml


def update_yaml_with_env_vars(yaml_path, top_level_key):
    """
    Environment variables whose name match the keys under the given
    top level key are updated to the env variable value.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if top_level_key not in data:
        print(f"Error: '{top_level_key}' not found in YAML file.")
        sys.exit(1)

    updated_keys = 0
    for key, _ in data[top_level_key].items():
        env_var = key.upper()
        if env_var in os.environ:
            data[top_level_key][key] = os.environ[env_var]
            updated_keys += 1

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(
        f"Updated '{yaml_path}' with environment {updated_keys} variables for top-level key '{top_level_key}'."
    )


def copy_and_update_yaml(yaml_tmpdir):
    """
    Copies all files from parameters folder to temporary parameters folder.
    Updates parameters which should be updated via env vars
    """
    main_path = Path(__file__).resolve()
    git_repo_root_path = main_path.parent.parent
    parameters_folder: Path = git_repo_root_path.joinpath("macaqueretina/parameters/")
    all_files = list(parameters_folder.glob("*.*"))

    yaml_tmpdir.mkdir(parents=True, exist_ok=True)
    for src in all_files:
        dst = yaml_tmpdir / src.name

        dst.write_bytes(src.read_bytes())  # Copy file content
        print(f"Copied {src.name} to {dst}")

    # Works when the filename and top-level key have the same name.
    top_level_keys = ["retina_parameters", "visual_stimulus_parameters"]
    for top_level_key in top_level_keys:
        param_file = f"{top_level_key}.yaml"
        yaml_path = yaml_tmpdir / param_file
        update_yaml_with_env_vars(yaml_path, top_level_key)


def main():
    start_time = time.time()

    if os.environ.get("YAML_TMPDIR"):
        yaml_tmpdir = Path(os.environ.get("YAML_TMPDIR"))
        copy_and_update_yaml(yaml_tmpdir)

    from project.project_manager_module import (
        ProjectManager,
        run_core_parameter_pipeline,
    )

    PM = ProjectManager(yaml_path=yaml_tmpdir)

    run_core_parameter_pipeline(PM)

    end_time = time.time()
    print(
        "Total time taken: ",
        time.strftime(
            "%H hours %M minutes %S seconds", time.gmtime(end_time - start_time)
        ),
    )


if __name__ == "__main__":
    main()
