"""Output path helpers for XYZ and batch runs."""

import os
from datetime import datetime

DEFAULT_OUTPUT_BASE = "outputs"

MODE_LABELS = {
    "pair": "pair",
    "batch": "batch",
    "align": "align",
    "align-batch": "align",
}


def data_folder_name(path: str) -> str:
    """Return the leaf folder name for a file path or directory path."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        name = os.path.basename(path.rstrip(os.sep))
    else:
        name = os.path.basename(os.path.dirname(path))
    return name or "data"


def make_run_output_dir(
    data_path: str,
    mode: str,
    base_output: str = DEFAULT_OUTPUT_BASE,
) -> str:
    """
    Create a timestamped run folder: {base}/{data_folder}_{mode}_{MM-DD-YYYY-HH-MM}.

    Parameters
    ----------
    data_path : str
        Input file or folder used to infer the data folder name.
    mode : str
        One of pair, batch, align (align-batch is normalized to align).
    base_output : str
        Root output directory (default: outputs/).
    """
    mode_key = mode.replace("_", "-")
    mode_label = MODE_LABELS.get(mode_key, mode_key)
    folder_name = data_folder_name(data_path)
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M")
    run_name = f"{folder_name}_{mode_label}_{timestamp}"
    run_path = os.path.join(base_output, run_name)
    os.makedirs(run_path, exist_ok=True)
    return run_path


def resolve_run_output(
    mode: str,
    *,
    output: str | None = None,
    target: str | None = None,
    query: str | None = None,
    folder: str | None = None,
    base_output: str = DEFAULT_OUTPUT_BASE,
) -> str:
    """Return explicit output path, or create a default timestamped run folder."""
    if output is not None:
        os.makedirs(output, exist_ok=True)
        return output

    if mode in ("pair", "align"):
        if not query:
            raise ValueError("query path is required to resolve the output directory")
        data_path = query
    elif mode in ("batch", "align-batch"):
        if not folder:
            raise ValueError("folder path is required to resolve the output directory")
        data_path = folder
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return make_run_output_dir(data_path, mode, base_output=base_output)


def gen_output_folder(files_list, base_output=DEFAULT_OUTPUT_BASE):
    """Deprecated: use make_run_output_dir instead."""
    if not files_list:
        return make_run_output_dir(".", "batch", base_output=base_output)
    return make_run_output_dir(files_list[0], "batch", base_output=base_output)


def gen_out_path(path, comment="", output_directory=DEFAULT_OUTPUT_BASE):
    os.makedirs(output_directory, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(output_directory, f"{base_name}_{comment}.xyz")


def gen_out_path_legacy(path, comment="", output_directory=DEFAULT_OUTPUT_BASE):
    """Legacy path format used by ring renumbering."""
    os.makedirs(output_directory, exist_ok=True)
    base_name = os.path.basename(path)
    if len(base_name) > 14:
        base_name = base_name[10:-4]
    else:
        base_name = os.path.splitext(base_name)[0]
    return os.path.join(output_directory, f"{base_name}_{comment}.xyz")
