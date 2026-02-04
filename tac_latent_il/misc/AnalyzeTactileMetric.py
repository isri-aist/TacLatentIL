import argparse
import os
import pickle
from pathlib import Path

import yaml


def find_runs(root: Path):
    runs = []
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        files = set(filenames)

        has_tb = any(fn.startswith("events.out.tfevents.") for fn in files)
        has_meta = "model_meta_info.pkl" in files
        has_yaml = "tactile_metric.yaml" in files

        if has_tb and has_meta and has_yaml:
            runs.append(p)
    return runs


def load_min_loss_infonce(yaml_path: Path):
    with open(yaml_path, "r") as f:
        rows = yaml.safe_load(f) or []

    best_loss = None
    best_epoch = None
    for r in rows:
        if isinstance(r, dict) and "loss_infonce" in r:
            try:
                loss = float(r["loss_infonce"])
            except Exception:
                continue
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_epoch = r.get("epoch", None)

    return best_loss, best_epoch


def load_model_meta(meta_path: Path):
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def get_policy_name(meta: dict):
    policy_name = meta.get("policy", {}).get("name", "UnknownPolicy")
    policy_name = policy_name.removeprefix("TactileMetric")

    # Sparsh encoder type (if exists)
    tactile_kwargs = meta.get("policy", {}).get("args", {}).get("tactile_kwargs", {})
    encoder_type = tactile_kwargs.get("encoder_type", None)

    if encoder_type:
        return f"{policy_name}(encoder_type={encoder_type})"
    return policy_name


def get_temperature(meta: dict):
    return meta.get("policy", {}).get("temperature", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str, help="root directory to search")
    parser.add_argument(
        "--print-dirname",
        action="store_true",
        help="also print leaf directory name",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    runs = find_runs(root)

    rows = []
    for run_dir in runs:
        meta_path = run_dir / "model_meta_info.pkl"
        yaml_path = run_dir / "tactile_metric.yaml"

        try:
            meta = load_model_meta(meta_path)
        except Exception:
            meta = {}

        min_loss, min_epoch = load_min_loss_infonce(yaml_path)
        rows.append(
            {
                "policy": get_policy_name(meta),
                "loss": min_loss,
                "epoch": min_epoch,
                "temperature": get_temperature(meta),
                "dirname": run_dir.name,
            }
        )

    # sort by InfoNCE
    rows = sorted(
        rows,
        key=lambda r: float("inf") if r["loss"] is None else r["loss"],
    )

    # column widths
    policy_width = max(len("policy"), *(len(r["policy"]) for r in rows))
    loss_header = "min_loss_infonce"
    epoch_header = "min_epoch"
    temp_header = "temperature"
    loss_width = max(len(loss_header), 10)  # enough for "12345.67"
    epoch_width = max(len(epoch_header), 6)
    temp_width = max(len(temp_header), 11)  # enough for typical floats/None
    dir_header = "dir_name"
    dir_width = (
        max(len(dir_header), *(len(r["dirname"]) for r in rows))
        if args.print_dirname
        else 0
    )

    # header
    header_parts = [
        "policy".ljust(policy_width),
        loss_header.rjust(loss_width),
        epoch_header.rjust(epoch_width),
        temp_header.rjust(temp_width),
    ]
    if args.print_dirname:
        header_parts.append(dir_header.ljust(dir_width))
    print("  ".join(header_parts))

    # rows
    for r in rows:
        loss_str = f"{r['loss']:.2f}" if r["loss"] is not None else "None"
        epoch_str = str(r["epoch"]) if r["epoch"] is not None else "None"
        temp_str = str(r["temperature"]) if r["temperature"] is not None else "None"
        parts = [
            r["policy"].ljust(policy_width),
            loss_str.rjust(loss_width),
            epoch_str.rjust(epoch_width),
            temp_str.rjust(temp_width),
        ]
        if args.print_dirname:
            parts.append(r["dirname"].ljust(dir_width))
        print("  ".join(parts))


if __name__ == "__main__":
    main()
