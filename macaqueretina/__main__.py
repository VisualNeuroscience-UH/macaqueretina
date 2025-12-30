" This module runs the core_parameters.yaml defined 'run' pipeline when called directly as python macaqueretina. " ""

import time

# Local
from project.project_manager_module import ProjectManager, run_core_parameter_pipeline


def main():
    start_time = time.time()

    PM = ProjectManager()

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
