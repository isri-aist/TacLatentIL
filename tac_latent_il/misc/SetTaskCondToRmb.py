import argparse
import re
from pathlib import Path

from robo_manip_baselines.common import RmbData, find_rmb_files


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.hdf5 or *.rmb) or directory containing them",
    )
    parser.add_argument(
        "--task_cond",
        type=int,
        required=True,
        help="integer task condition to set (stored as int)",
    )
    parser.add_argument(
        "--filename_regex",
        type=str,
        default=".*",
        help="regular expression to filter target rmb filenames",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing value if it exists",
    )

    return parser.parse_args()


class SetTaskCond:
    def __init__(self, path, task_cond, filename_regex=".*", overwrite=False):
        self.path = path
        self.task_cond_new = int(task_cond)
        self.filename_regex = re.compile(filename_regex)
        self.overwrite = overwrite

    def run(self):
        rmb_path_list = find_rmb_files(self.path)

        for rmb_path in rmb_path_list:
            rmb_path = rmb_path
            fname = Path(rmb_path).name
            if not self.filename_regex.search(fname):
                continue

            print(f"[{self.__class__.__name__}] Open {rmb_path}")
            with RmbData(rmb_path, mode="r+") as rmb_data:
                task_cond_orig = rmb_data.attrs.get("task_cond", None)

                if task_cond_orig is not None and not self.overwrite:
                    raise ValueError(
                        f"[{self.__class__.__name__}] task_cond already exists: "
                        f"{task_cond_orig} (use --overwrite to replace)"
                    )

                print(
                    f"Set task_cond from '{task_cond_orig}' to '{self.task_cond_new}'"
                )
                # Explicitly store as int
                rmb_data.attrs["task_cond"] = int(self.task_cond_new)


if __name__ == "__main__":
    set_task = SetTaskCond(**vars(parse_argument()))
    set_task.run()
