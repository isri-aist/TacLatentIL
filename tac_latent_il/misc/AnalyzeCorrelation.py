import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats


def _load_yaml_list(path: Path):
    with open(path, "r") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return []
    if not isinstance(obj, list):
        raise ValueError(f"YAML must be a list of dicts: {path}")
    return obj


def _index_by_policy(rows, required_keys):
    out = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        if "policy" not in r:
            continue
        for k in required_keys:
            if k not in r:
                raise ValueError(
                    f"Missing key '{k}' in row for policy={r.get('policy')}"
                )
        out[str(r["policy"])] = r
    return out


def _corrs(x, y):
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    kendall_tau, kendall_p = stats.kendalltau(x, y, variant="b")
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "kendall_tau_b": float(kendall_tau),
        "kendall_p": float(kendall_p),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--task_name", type=str, default=None)

    ap.add_argument("--eval_yaml", required=False)
    ap.add_argument("--success_yaml", required=False)

    ap.add_argument("--title", type=str, default="Eval Metric vs Success")

    args = ap.parse_args()

    if (args.in_dir is None) ^ (args.task_name is None):
        raise ValueError("--in_dir and --task_name must be specified together.")
    elif args.in_dir is not None and args.out_dir is None:
        raise ValueError("--out_dir must be specified when using --in_dir/--task_name.")

    if args.task_name is not None:
        in_dir = Path(args.in_dir).expanduser().resolve()
        out_dir = Path(args.out_dir).expanduser().resolve() / args.task_name
        eval_path = in_dir / f"Eval-{args.task_name}.yaml"
        success_path = in_dir / f"Success-{args.task_name}.yaml"
        args.title = args.task_name
    else:
        if args.eval_yaml is None or args.success_yaml is None:
            raise ValueError(
                "Either specify --in_dir/--out_dir/--task_name, "
                "or specify --eval_yaml and --success_yaml."
            )
        eval_path = Path(args.eval_yaml).expanduser().resolve()
        success_path = Path(args.success_yaml).expanduser().resolve()
        out_dir = Path(args.out_dir or "corr_out").expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load ----
    eval_rows = _load_yaml_list(eval_path)
    success_rows = _load_yaml_list(success_path)

    success_by_policy = _index_by_policy(
        success_rows,
        required_keys=["mean_success_rate", "var_success_rate", "n"],
    )

    # eval metric keys (mean_xxx)
    metric_keys = sorted(
        k.removeprefix("mean_") for k in eval_rows[0].keys() if k.startswith("mean_")
    )

    # ---- per-metric analysis ----
    for key in metric_keys:
        mean_key = f"mean_{key}"
        var_key = f"var_{key}"

        eval_by_policy = _index_by_policy(
            eval_rows,
            required_keys=[mean_key, var_key, "n"],
        )

        policies = sorted(set(eval_by_policy.keys()) & set(success_by_policy.keys()))
        if len(policies) < 3:
            print(f"[SKIP] {key}: not enough overlapping policies")
            continue

        x = np.array([eval_by_policy[p][mean_key] for p in policies])
        y = np.array([success_by_policy[p]["mean_success_rate"] for p in policies])

        xerr = np.sqrt(np.array([eval_by_policy[p][var_key] for p in policies]))
        yerr = np.sqrt(
            np.array([success_by_policy[p]["var_success_rate"] for p in policies])
        )

        corrs = _corrs(x, y)

        # ---- print ----
        print(f"\n=== Metric: {key} ===")
        for p in policies:
            print(
                f"- {p}: "
                f"{mean_key}={eval_by_policy[p][mean_key]:.4f}±{math.sqrt(eval_by_policy[p][var_key]):.4f}, "
                f"success={success_by_policy[p]['mean_success_rate']:.3f}"
            )

        print(f"  Pearson r = {corrs['pearson_r']:.4f} (p={corrs['pearson_p']:.3g})")
        print(
            f"  Spearman rho = {corrs['spearman_rho']:.4f} (p={corrs['spearman_p']:.3g})"
        )
        print(
            f"  Kendall tau-b = {corrs['kendall_tau_b']:.4f} (p={corrs['kendall_p']:.3g})"
        )

        # ---- plot ----
        plt.figure(figsize=(6.0, 4.5))
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=3)

        for i, p in enumerate(policies):
            plt.annotate(
                p, (x[i], y[i]), textcoords="offset points", xytext=(5, 4), fontsize=9
            )

        plt.xlabel(key)
        plt.ylabel("rollout success rate")
        plt.title(args.title)

        txt = (
            f"Pearson r={corrs['pearson_r']:.3f}\n"
            f"Spearman ρ={corrs['spearman_rho']:.3f}\n"
            f"Kendall τ-b={corrs['kendall_tau_b']:.3f}"
        )
        plt.gca().text(
            0.02, 0.02, txt, transform=plt.gca().transAxes, fontsize=9, va="bottom"
        )

        plt.ylim(0.0, 1.0)
        plt.tight_layout()

        out_png = out_dir / f"Corr-{key}.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
