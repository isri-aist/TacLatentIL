import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    from scipy import stats
except Exception:
    stats = None


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
    if stats is None:
        raise RuntimeError("scipy is required (scipy.stats). Please install scipy.")
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


def _bootstrap_ci(
    x,
    y,
    n_boot=5000,
    seed=0,
):
    """
    Bootstrap over policies (resample indices with replacement).
    Returns percentile CIs for (pearson, spearman, kendall).
    """
    if stats is None:
        raise RuntimeError("scipy is required (scipy.stats). Please install scipy.")

    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 policies for bootstrap correlation CI.")

    pearsons = []
    spearmans = []
    kendalls = []

    idxs = np.arange(n)
    for _ in range(n_boot):
        bs = rng.choice(idxs, size=n, replace=True)
        xb = x[bs]
        yb = y[bs]

        # correlations can be nan if constant; skip those samples
        pr = stats.pearsonr(xb, yb)[0]
        sr = stats.spearmanr(xb, yb)[0]
        kr = stats.kendalltau(xb, yb, variant="b")[0]

        if not (np.isfinite(pr) and np.isfinite(sr) and np.isfinite(kr)):
            continue

        pearsons.append(pr)
        spearmans.append(sr)
        kendalls.append(kr)

    def ci(a):
        a = np.asarray(a, dtype=np.float64)
        if a.size < 100:
            # too many degenerate samples
            return {
                "mean": float(np.nan),
                "lo": float(np.nan),
                "hi": float(np.nan),
                "n_eff": int(a.size),
            }
        lo, hi = np.percentile(a, [2.5, 97.5])
        return {
            "mean": float(a.mean()),
            "lo": float(lo),
            "hi": float(hi),
            "n_eff": int(a.size),
        }

    return {
        "pearson_r": ci(pearsons),
        "spearman_rho": ci(spearmans),
        "kendall_tau_b": ci(kendalls),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--task_name", type=str, default=None)

    ap.add_argument(
        "--metric_yaml",
        required=False,
        help="tactile_metric_aggregated.yaml from AnalyzeTactileMetric.py",
    )
    ap.add_argument(
        "--success_yaml",
        required=False,
        help="aggregated success yaml from AnalyzeRolloutSuccess.py",
    )

    ap.add_argument("--bootstrap", type=int, default=5000, help="num bootstrap samples")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    ap.add_argument(
        "--title", type=str, default="Metric vs Rollout Success", help="plot title"
    )
    args = ap.parse_args()

    if (args.in_dir is None) ^ (args.task_name is None):
        raise ValueError("--in_dir and --task_name must be specified together.")
    elif args.in_dir is not None and args.out_dir is None:
        raise ValueError("--out_dir must be specified when using --in_dir/--task_name.")

    if args.task_name is not None:
        in_dir = Path(args.in_dir).expanduser().resolve()
        out_dir = Path(args.out_dir).expanduser().resolve() / args.task_name
        metric_path = in_dir / f"Metric-{args.task_name}.yaml"
        success_path = in_dir / f"Success-{args.task_name}.yaml"
        out_yaml = out_dir / f"Corr-{args.task_name}.yaml"
        out_png = out_dir / f"Corr-{args.task_name}.png"
        args.title = args.task_name
    else:
        if args.metric_yaml is None or args.success_yaml is None:
            raise ValueError(
                "Either specify --in_dir/--out_dir/--task_name, or specify --metric_yaml and --success_yaml."
            )
        metric_path = Path(args.metric_yaml).expanduser().resolve()
        success_path = Path(args.success_yaml).expanduser().resolve()
        out_dir = Path(args.out_dir or "corr_out").expanduser().resolve()
        out_yaml = out_dir / "Corr.yaml"
        out_png = out_dir / "Cor.png"

    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = _load_yaml_list(metric_path)
    success_rows = _load_yaml_list(success_path)

    metric_by_policy = _index_by_policy(
        metric_rows, required_keys=["mean_loss_infonce", "var_loss_infonce", "n"]
    )
    success_by_policy = _index_by_policy(
        success_rows, required_keys=["mean_success_rate", "var_success_rate", "n"]
    )

    # join
    policies = sorted(set(metric_by_policy.keys()) & set(success_by_policy.keys()))
    if len(policies) == 0:
        raise ValueError(
            "No overlapping policies between metric_yaml and success_yaml (policy field mismatch)."
        )

    # arrays
    x = np.array(
        [float(metric_by_policy[p]["mean_loss_infonce"]) for p in policies],
        dtype=np.float64,
    )
    y = np.array(
        [float(success_by_policy[p]["mean_success_rate"]) for p in policies],
        dtype=np.float64,
    )
    xerr = np.sqrt(
        np.array(
            [float(metric_by_policy[p]["var_loss_infonce"]) for p in policies],
            dtype=np.float64,
        )
    )
    yerr = np.sqrt(
        np.array(
            [float(success_by_policy[p]["var_success_rate"]) for p in policies],
            dtype=np.float64,
        )
    )

    # correlations
    corrs = _corrs(x, y)
    cis = _bootstrap_ci(x, y, n_boot=args.bootstrap, seed=args.seed)

    # print summary
    print(f"Policies joined: {len(policies)}")
    for p in policies:
        print(
            f"- {p}: "
            f"metric={metric_by_policy[p]['mean_loss_infonce']:.4f}±{math.sqrt(float(metric_by_policy[p]['var_loss_infonce'])):.4f} "
            f"(n={metric_by_policy[p]['n']}), "
            f"success={success_by_policy[p]['mean_success_rate']:.4f}±{math.sqrt(float(success_by_policy[p]['var_success_rate'])):.4f} "
            f"(n={success_by_policy[p]['n']})"
        )

    print("\nCorrelations:")
    print(f"  Spearman rho = {corrs['spearman_rho']:.4f} (p={corrs['spearman_p']:.3g})")
    print(
        f"  Kendall tau-b = {corrs['kendall_tau_b']:.4f} (p={corrs['kendall_p']:.3g})"
    )
    print(f"  Pearson r = {corrs['pearson_r']:.4f} (p={corrs['pearson_p']:.3g})")

    print("\nBootstrap 95% CI (percentile):")
    print(
        f"  Spearman rho: mean={cis['spearman_rho']['mean']:.4f}, "
        f"[{cis['spearman_rho']['lo']:.4f}, {cis['spearman_rho']['hi']:.4f}] "
        f"(n_eff={cis['spearman_rho']['n_eff']})"
    )
    print(
        f"  Kendall tau-b: mean={cis['kendall_tau_b']['mean']:.4f}, "
        f"[{cis['kendall_tau_b']['lo']:.4f}, {cis['kendall_tau_b']['hi']:.4f}] "
        f"(n_eff={cis['kendall_tau_b']['n_eff']})"
    )
    print(
        f"  Pearson r: mean={cis['pearson_r']['mean']:.4f}, "
        f"[{cis['pearson_r']['lo']:.4f}, {cis['pearson_r']['hi']:.4f}] "
        f"(n_eff={cis['pearson_r']['n_eff']})"
    )

    # save yaml
    with open(out_yaml, "w") as f:
        yaml.safe_dump(
            {
                "n_policies": int(len(policies)),
                "policies": policies,
                "corr": corrs,
                "bootstrap_ci_95": cis,
                "inputs": {
                    "metric_yaml": str(metric_path),
                    "success_yaml": str(success_path),
                },
            },
            f,
            sort_keys=False,
        )
    print(f"\nSaved: {out_yaml}")

    # plot scatter with error bars
    plt.figure(figsize=(6.0, 4.5))
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=3)

    # annotate
    for i, p in enumerate(policies):
        plt.annotate(
            p, (x[i], y[i]), textcoords="offset points", xytext=(5, 4), fontsize=9
        )

    plt.xlabel("surrogate metric (mean InfoNCE loss; lower is better)")
    plt.ylabel("rollout success rate (mean)")
    plt.title(args.title)

    # put correlations in the plot
    txt = (
        f"Spearman ρ={corrs['spearman_rho']:.3f} "
        f"[{cis['spearman_rho']['lo']:.3f},{cis['spearman_rho']['hi']:.3f}]\n"
        f"Kendall τ-b={corrs['kendall_tau_b']:.3f} "
        f"[{cis['kendall_tau_b']['lo']:.3f},{cis['kendall_tau_b']['hi']:.3f}]\n"
        f"Pearson r={corrs['pearson_r']:.3f} "
        f"[{cis['pearson_r']['lo']:.3f},{cis['pearson_r']['hi']:.3f}]"
    )
    plt.gca().text(
        0.02, 0.02, txt, transform=plt.gca().transAxes, fontsize=9, va="bottom"
    )

    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")

    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")

    plt.show()


if __name__ == "__main__":
    main()
