import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def find_policy_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def find_yaml_files(policy_dir: Path) -> List[Path]:
    return sorted(policy_dir.rglob("*.yaml")) + sorted(policy_dir.rglob("*.yml"))


def load_yaml_dict(path: Path) -> Dict[str, float]:
    with open(path, "r") as f:
        obj = yaml.safe_load(f)

    if obj is None:
        return {}

    if not isinstance(obj, dict):
        raise ValueError(f"YAML must be a dict (key: value) : {path}")

    out: Dict[str, float] = {}
    for k, v in obj.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def get_policy_name_from_dirname(dirname: str) -> str:
    name = dirname

    # drop common prefix
    if name.startswith("DiffusionPolicy"):
        name = name.removeprefix("DiffusionPolicy")

    # Sparsh encoder type (DiffusionPolicySparsh_dino, etc.)
    if name.startswith("Sparsh_"):
        enc = name.removeprefix("Sparsh_")
        if enc:
            return f"Sparsh({enc})"
        return "Sparsh"

    return name


def aggregate_policy_metrics(
    yaml_dicts: List[Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    if not yaml_dicts:
        return {}, {}, 0

    ref_keys = set(yaml_dicts[0].keys())
    for i, d in enumerate(yaml_dicts[1:], start=1):
        keys = set(d.keys())
        if keys != ref_keys:
            missing = sorted(ref_keys - keys)
            extra = sorted(keys - ref_keys)
            raise ValueError(
                "YAML keys mismatch while aggregating.\n"
                f"  ref_keys({len(ref_keys)}): {sorted(ref_keys)}\n"
                f"  idx={i} keys({len(keys)}): {sorted(keys)}\n"
                f"  missing_from_idx: {missing}\n"
                f"  extra_in_idx: {extra}"
            )

    values: Dict[str, List[float]] = defaultdict(list)
    for d in yaml_dicts:
        for k, v in d.items():
            values[k].append(v)

    mean_by_key: Dict[str, float] = {}
    var_by_key: Dict[str, float] = {}
    n = len(yaml_dicts)

    for k, xs in values.items():
        if len(xs) == 0:
            continue
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        mean_by_key[k] = mean
        var_by_key[k] = var

    return mean_by_key, var_by_key, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str, help="root directory of eval results")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output yaml path (aggregated). if omitted, no yaml is saved",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    policy_dirs = find_policy_dirs(root)

    rows = []
    all_metric_keys = set()

    for pdir in policy_dirs:
        yaml_paths = find_yaml_files(pdir)
        yaml_dicts = []
        for yp in yaml_paths:
            try:
                yaml_dicts.append(load_yaml_dict(yp))
            except Exception:
                continue

        if len(yaml_dicts) == 0:
            continue

        mean_by_key, var_by_key, n = aggregate_policy_metrics(yaml_dicts)
        all_metric_keys.update(mean_by_key.keys())
        all_metric_keys.update(var_by_key.keys())

        rows.append(
            {
                "policy": get_policy_name_from_dirname(pdir.name),
                "mean": mean_by_key,
                "var": var_by_key,
                "n": n,
                "dirname": pdir.name,
            }
        )

    # stable sort by policy name
    rows = sorted(rows, key=lambda r: r["policy"])

    # Decide column order: metric keys sorted
    metric_keys = sorted(all_metric_keys)

    # ---- print (table) ----
    policy_width = max(len("policy"), *(len(r["policy"]) for r in rows))
    n_header = "n"
    n_width = max(len(n_header), 3)

    # headers like mean_act_err / var_act_err ...
    mean_headers = [f"mean_{k}" for k in metric_keys]
    var_headers = [f"var_{k}" for k in metric_keys]

    # widths
    col_widths = {}
    col_widths["policy"] = policy_width
    for h in mean_headers + var_headers:
        col_widths[h] = max(len(h), 10)
    col_widths[n_header] = n_width

    # print header
    parts = []
    parts.append("policy".ljust(col_widths["policy"]))
    for h in mean_headers:
        parts.append(h.rjust(col_widths[h]))
    for h in var_headers:
        parts.append(h.rjust(col_widths[h]))
    parts.append(n_header.rjust(col_widths[n_header]))
    print("  ".join(parts))

    # print rows
    for r in rows:
        parts = []
        parts.append(r["policy"].ljust(col_widths["policy"]))

        for k, h in zip(metric_keys, mean_headers):
            v = r["mean"].get(k, None)
            s = f"{v:.6g}" if v is not None else "None"
            parts.append(s.rjust(col_widths[h]))

        for k, h in zip(metric_keys, var_headers):
            v = r["var"].get(k, None)
            s = f"{v:.6g}" if v is not None else "None"
            parts.append(s.rjust(col_widths[h]))

        parts.append(str(r["n"]).rjust(col_widths[n_header]))
        print("  ".join(parts))

    # ---- save yaml ----
    if args.out:
        out_path = Path(args.out).expanduser().resolve()

        out_rows = []
        for r in rows:
            o = {"policy": r["policy"]}
            for k in metric_keys:
                if k in r["mean"]:
                    o[f"mean_{k}"] = float(r["mean"][k])
                if k in r["var"]:
                    o[f"var_{k}"] = float(r["var"][k])
            o["n"] = int(r["n"])
            out_rows.append(o)

        with open(out_path, "w") as f:
            yaml.safe_dump(out_rows, f, sort_keys=False)

        print(f"Save {out_path}")


if __name__ == "__main__":
    main()
