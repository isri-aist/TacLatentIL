import argparse
import os
import pickle
from collections import defaultdict
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
        return f"{policy_name}({encoder_type})"
    return policy_name


def get_temperature(meta: dict):
    return meta.get("policy", {}).get("temperature", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str, help="root directory to search")
    parser.add_argument(
        "--print-dirname",
        action="store_true",
        help="also print leaf directory name (only in raw mode)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="print each run without aggregation",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output yaml path (aggregated). if omitted, no yaml is saved",
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
                "n": 1,
            }
        )

    # drop None losses for aggregation/sorting stability
    rows_valid = [r for r in rows if r["loss"] is not None]

    if args.raw:
        # sort by InfoNCE (smaller is better)
        rows_valid = sorted(rows_valid, key=lambda r: r["loss"])

        policy_width = max(len("policy"), *(len(r["policy"]) for r in rows_valid))
        loss_header = "min_loss_infonce"
        epoch_header = "min_epoch"
        temp_header = "temperature"
        n_header = "n"
        loss_width = max(len(loss_header), 10)
        epoch_width = max(len(epoch_header), 6)
        temp_width = max(len(temp_header), 11)
        n_width = max(len(n_header), 3)
        dir_header = "dir_name"
        dir_width = (
            max(len(dir_header), *(len(r["dirname"]) for r in rows_valid))
            if args.print_dirname
            else 0
        )

        header_parts = [
            "policy".ljust(policy_width),
            loss_header.rjust(loss_width),
            epoch_header.rjust(epoch_width),
            temp_header.rjust(temp_width),
            n_header.rjust(n_width),
        ]
        if args.print_dirname:
            header_parts.append(dir_header.ljust(dir_width))
        print("  ".join(header_parts))

        for r in rows_valid:
            loss_str = f"{r['loss']:.2f}"
            epoch_str = str(r["epoch"]) if r["epoch"] is not None else "None"
            temp_str = str(r["temperature"]) if r["temperature"] is not None else "None"
            parts = [
                r["policy"].ljust(policy_width),
                loss_str.rjust(loss_width),
                epoch_str.rjust(epoch_width),
                temp_str.rjust(temp_width),
                str(r["n"]).rjust(n_width),
            ]
            if args.print_dirname:
                parts.append(r["dirname"].ljust(dir_width))
            print("  ".join(parts))
        return

    # ---- aggregated mode (policy-wise mean/var) ----
    groups = defaultdict(list)
    for r in rows_valid:
        # group by policy (and temperature if you want strict separation)
        groups[(r["policy"], r["temperature"])].append(r["loss"])

    agg_rows = []
    for (policy, temp), losses in groups.items():
        n = len(losses)
        mean = sum(losses) / n
        var = sum((x - mean) ** 2 for x in losses) / n  # population variance
        agg_rows.append(
            {
                "policy": policy,
                "mean": mean,
                "var": var,
                "temperature": temp,
                "n": n,
            }
        )

    # sort by mean (smaller is better)
    agg_rows = sorted(agg_rows, key=lambda r: r["mean"])

    policy_width = max(len("policy"), *(len(r["policy"]) for r in agg_rows))
    mean_header = "mean_loss"
    var_header = "var_loss"
    temp_header = "temperature"
    n_header = "n"
    mean_width = max(len(mean_header), 10)
    var_width = max(len(var_header), 10)
    temp_width = max(len(temp_header), 11)
    n_width = max(len(n_header), 3)

    header_parts = [
        "policy".ljust(policy_width),
        mean_header.rjust(mean_width),
        var_header.rjust(var_width),
        temp_header.rjust(temp_width),
        n_header.rjust(n_width),
    ]
    print("  ".join(header_parts))

    for r in agg_rows:
        temp_str = str(r["temperature"]) if r["temperature"] is not None else "None"
        parts = [
            r["policy"].ljust(policy_width),
            f"{r['mean']:.2f}".rjust(mean_width),
            f"{r['var']:.4f}".rjust(var_width),
            temp_str.rjust(temp_width),
            str(r["n"]).rjust(n_width),
        ]
        print("  ".join(parts))

    # Save yaml
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        with open(out_path, "w") as f:
            yaml.safe_dump(
                [
                    {
                        "policy": r["policy"],
                        "mean_loss_infonce": float(r["mean"]),
                        "var_loss_infonce": float(r["var"]),
                        "temperature": r["temperature"],
                        "n": r["n"],
                    }
                    for r in agg_rows
                ],
                f,
                sort_keys=False,
            )

        print(f"Save {out_path}")

    # Plot
    import matplotlib.pyplot as plt
    import numpy as np

    policies = [r["policy"] for r in agg_rows]
    means = np.array([r["mean"] for r in agg_rows])
    stds = np.sqrt(np.array([r["var"] for r in agg_rows]))

    x = np.arange(len(policies))

    plt.figure(figsize=(0.6 * len(policies) + 2, 4))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, policies, rotation=45, ha="right")
    plt.ylabel("InfoNCE loss (mean ± std)")
    plt.title(f"{root.name}")
    plt.tight_layout()

    if args.out:
        png_path = out_path.with_suffix(".png")
        plt.savefig(png_path, dpi=150)
        print(f"Save {png_path}")

    plt.show()


if __name__ == "__main__":
    main()
