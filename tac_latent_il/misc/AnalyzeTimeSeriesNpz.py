import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def _effective_rank(X: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    total = float(np.sum(s))
    if total <= 0.0:
        return 0.0
    p = np.maximum(s / total, 1e-12)
    ent = -float(np.sum(p * np.log(p)))
    return float(np.exp(ent))


def _compute_lat_windowed(z: np.ndarray, W: int):
    """
    z: [T, Dz]
    return:
      lat_std_w: [T]
      lat_rank_w: [T]
    """
    T = int(z.shape[0])
    lat_std_w = np.zeros((T,), dtype=np.float32)
    lat_rank_w = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        t0, t1 = max(0, t - W), min(T, t + W + 1)
        zw = z[t0:t1]
        lat_std_w[t] = float(np.mean(np.std(zw, axis=0)))
        lat_rank_w[t] = float(_effective_rank(zw))
    return lat_std_w, lat_rank_w


def _compute_series_per_episode(
    z: np.ndarray,  # [T, Dz]
    action_gt: np.ndarray,  # [T, Ta, Da]
    action_pred: np.ndarray,  # [T, Ns, Ta, Da]
    grad_feat: np.ndarray,  # [T, Ns, Dz]
    sigma: np.ndarray,  # [Dz] (global per-policy)
    W: int,
):
    # action
    err = action_pred - action_gt[:, None, :, :]
    rmse = np.sqrt(np.mean(err**2, axis=(2, 3)))  # [T, Ns]
    action_err = rmse.mean(axis=1).astype(np.float32)  # [T]

    action_var = action_pred.var(axis=1).mean(axis=(1, 2)).astype(np.float32)  # [T]

    # grad sensitivity
    grad = np.linalg.norm(grad_feat, axis=-1)  # [T, Ns]
    grad_raw = grad.mean(axis=1).astype(np.float32)  # [T]

    grad_znorm = np.linalg.norm(grad_feat * sigma[None, None, :], axis=-1)  # [T, Ns]
    grad_norm = grad_znorm.mean(axis=1).astype(np.float32)  # [T]

    # latent dynamics
    dz_raw = np.linalg.norm(z[1:] - z[:-1], axis=1)
    dz_raw = np.concatenate([[0.0], dz_raw]).astype(np.float32)

    dz_norm = np.linalg.norm((z[1:] - z[:-1]) / sigma[None, :], axis=1)
    dz_norm = np.concatenate([[0.0], dz_norm]).astype(np.float32)

    # windowed
    lat_std_w, lat_rank_w = _compute_lat_windowed(z, W)

    return {
        "action_err": action_err,
        "action_var": action_var,
        "grad_raw": grad_raw,
        "grad_norm": grad_norm,
        "lat_std_w": lat_std_w,
        "lat_rank_w": lat_rank_w,
        "dz_raw": dz_raw,
        "dz_norm": dz_norm,
    }


def _collect_npz_paths_under_dir(p: Path):
    paths = sorted(p.rglob("*.npz"))
    if not paths:
        raise ValueError(f"No .npz found under directory: {p}")
    return paths


def _bin_aggregate(x_list, t_list, bins: int):
    bucket = [[] for _ in range(bins)]
    for x, tt in zip(x_list, t_list):
        b = np.clip((tt * bins).astype(int), 0, bins - 1)
        for i in range(len(tt)):
            bucket[b[i]].append(float(x[i]))

    out = {
        k: np.full((bins,), np.nan, dtype=np.float32) for k in ["median", "q25", "q75"]
    }
    n = np.zeros((bins,), dtype=np.int64)

    for i in range(bins):
        if not bucket[i]:
            continue
        arr = np.array(bucket[i], dtype=np.float32)
        out["median"][i] = float(np.median(arr))
        out["q25"][i] = float(np.quantile(arr, 0.25))
        out["q75"][i] = float(np.quantile(arr, 0.75))
        n[i] = int(arr.size)

    out["n"] = n
    return out


def _bin_mean(x_list, t_list, bins: int) -> np.ndarray:
    bucket = [[] for _ in range(bins)]
    for x, tt in zip(x_list, t_list):
        b = np.clip((tt * bins).astype(int), 0, bins - 1)
        for i in range(len(tt)):
            bucket[b[i]].append(float(x[i]))
    mean = np.full((bins,), np.nan, dtype=np.float32)
    for i in range(bins):
        if bucket[i]:
            mean[i] = float(np.mean(np.asarray(bucket[i], dtype=np.float32)))
    return mean


def _scalar_summary(x_list):
    allv = np.concatenate([np.asarray(x, dtype=np.float32) for x in x_list], axis=0)
    return {
        "mean": float(np.mean(allv)),
        "q90": float(np.quantile(allv, 0.90)),
        "max": float(np.max(allv)),
    }


def _plot_band(agg, title, out_png):
    xs = np.linspace(0.0, 1.0, len(agg["median"]), endpoint=False)
    plt.figure()
    plt.plot(xs, agg["median"])
    plt.fill_between(xs, agg["q25"], agg["q75"], alpha=0.2)
    plt.title(title)
    plt.xlabel("progress")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_taskcond_overlaid(taskcond_to_agg, title, out_png):
    plt.figure()
    xs = None
    for tc, agg in sorted(taskcond_to_agg.items()):
        if xs is None:
            xs = np.linspace(0.0, 1.0, len(agg["median"]), endpoint=False)
        plt.plot(xs, agg["median"], label=f"task_cond={tc}")
    plt.title(title)
    plt.xlabel("progress")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _discover_policies(root: Path):
    """
    Policy dirs are the top-level subdirectories under root.
    If none exist but root contains npz, treat root itself as a single policy.
    """
    if not root.exists():
        raise ValueError(f"Invalid input path: {root}")

    subdirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    policies = []
    for d in subdirs:
        if any(d.rglob("*.npz")):
            policies.append(d)

    if policies:
        return policies

    raise ValueError(f"No policy directories (with npz) found under: {root}")


def _load_policy_npzs(policy_dir: Path):
    npz_paths = _collect_npz_paths_under_dir(policy_dir)

    episode_id_list = []
    task_cond_list = []
    t_norm_list = []
    z_list = []
    action_gt_list = []
    action_pred_list = []
    grad_feat_list = []

    for p in npz_paths:
        npz = np.load(p, allow_pickle=True)
        episode_id_list.extend(list(npz["episode_id"]))
        task_cond_list.extend(list(npz["task_cond"]))
        t_norm_list.extend(list(npz["t_norm"]))
        z_list.extend(list(npz["z"]))
        action_gt_list.extend(list(npz["action_gt"]))
        action_pred_list.extend(list(npz["action_pred"]))
        grad_feat_list.extend(list(npz["grad_feat"]))

    episode_id = np.asarray(episode_id_list)
    task_cond = np.asarray(task_cond_list, dtype=int)

    # per-policy sigma (for z-normalized grad and dz_norm)
    z_all = np.concatenate([np.asarray(z, dtype=np.float32) for z in z_list], axis=0)
    sigma = np.maximum(z_all.std(axis=0).astype(np.float32), 1e-8)

    return {
        "npz_paths": npz_paths,
        "episode_id": episode_id,
        "task_cond": task_cond,
        "t_norm_list": t_norm_list,
        "z_list": z_list,
        "action_gt_list": action_gt_list,
        "action_pred_list": action_pred_list,
        "grad_feat_list": grad_feat_list,
        "sigma": sigma,
    }


def _analyze_one_policy(
    policy_name: str, policy_dir: Path, out_dir: Path, bins: int, window: int
):
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_policy_npzs(policy_dir)
    npz_paths = data["npz_paths"]
    episode_id = data["episode_id"]
    task_cond = data["task_cond"]
    t_norm_list = data["t_norm_list"]
    z_list = data["z_list"]
    action_gt_list = data["action_gt_list"]
    action_pred_list = data["action_pred_list"]
    grad_feat_list = data["grad_feat_list"]
    sigma = data["sigma"]

    series_by_ep = []
    for z, ag, apred, gf in zip(
        z_list, action_gt_list, action_pred_list, grad_feat_list
    ):
        series_by_ep.append(
            _compute_series_per_episode(z, ag, apred, gf, sigma, window)
        )

    metrics = [
        "action_err",
        "action_var",
        "grad_raw",
        "grad_norm",
        "lat_std_w",
        "lat_rank_w",
        "dz_raw",
        "dz_norm",
    ]

    out_yaml = {
        "policy": policy_name,
        "policy_dir": str(policy_dir),
        "num_npz": int(len(npz_paths)),
        "bins": int(bins),
        "window": int(window),
        "num_episodes": int(len(episode_id)),
        "metrics": {},
        "by_task_cond": {},
    }

    # overall
    for m in metrics:
        xs = [s[m] for s in series_by_ep]
        agg = _bin_aggregate(xs, t_norm_list, bins)
        _plot_band(agg, m, out_dir / f"{m}.png")
        out_yaml["metrics"][m] = {
            "scalar": _scalar_summary(xs),
            "bin_n_min": int(np.nanmin(agg["n"])) if np.any(agg["n"] > 0) else 0,
        }

    # by task_cond (overlay)
    uniq_tc = sorted(set([int(x) for x in task_cond]))
    for tc in uniq_tc:
        out_yaml["by_task_cond"][int(tc)] = {"num_episodes": 0, "metrics": {}}
    for tc in uniq_tc:
        idx = [i for i in range(len(task_cond)) if int(task_cond[i]) == tc]
        out_yaml["by_task_cond"][int(tc)]["num_episodes"] = int(len(idx))

    for m in metrics:
        tc_to_agg = {}
        for tc in uniq_tc:
            idx = [i for i in range(len(task_cond)) if int(task_cond[i]) == tc]
            if len(idx) == 0:
                continue
            t_tc = [t_norm_list[i] for i in idx]
            x_tc = [series_by_ep[i][m] for i in idx]
            agg_tc = _bin_aggregate(x_tc, t_tc, bins)
            tc_to_agg[int(tc)] = agg_tc
            out_yaml["by_task_cond"][int(tc)]["metrics"][m] = {
                "scalar": _scalar_summary(x_tc),
                "bin_n_min": int(np.nanmin(agg_tc["n"]))
                if np.any(agg_tc["n"] > 0)
                else 0,
            }

        _plot_taskcond_overlaid(
            tc_to_agg,
            title=f"{m} (by task_cond)",
            out_png=out_dir / f"{m}_by_task_cond.png",
        )

    with open(out_dir / "summary.yaml", "w") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False)

    return {
        "policy": policy_name,
        "t_norm_list": t_norm_list,
        "series_by_ep": series_by_ep,
        "metrics": metrics,
        "out_yaml": out_yaml,
    }


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "in_dir", type=str, help="root directory whose top-level subdirs are policies"
    )
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--task_name", type=str, default=None)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--window", type=int, default=4)
    args = ap.parse_args()

    in_dir = Path(os.path.expanduser(args.in_dir))
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_dirs = _discover_policies(in_dir)

    # name resolution: if policy_dir is root itself, use root name
    policies = []
    for d in policy_dirs:
        if d.resolve() == in_dir.resolve():
            policies.append(d.name)
        else:
            policies.append(d.name)

    print(f"[AnalyzeTimeSeriesNpz] in_dir={in_dir}")
    print(f"[AnalyzeTimeSeriesNpz] policies={policies}")

    # 1) per-policy evaluation (same spirit as AnalyzeTimeSeriesNpz)
    per_policy = {}
    for pol, pol_dir in zip(policies, policy_dirs):
        pol_out = out_dir / pol
        print(f"[AnalyzeTimeSeriesNpz] Analyze policy={pol} dir={pol_dir}")
        per_policy[pol] = _analyze_one_policy(
            pol, pol_dir, pol_out, args.bins, args.window
        )

    # 2) cross-policy comparison (mean line only, same spirit as Compare script)
    compare_dir = out_dir / "_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    # metrics list (assume consistent across policies)
    metrics = per_policy[policies[0]]["metrics"]
    xs = np.linspace(0.0, 1.0, args.bins, endpoint=False)

    compare_yaml = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "task_name": args.task_name,
        "bins": int(args.bins),
        "window": int(args.window),
        "policies": list(policies),
        "metrics": {},
    }

    for m in metrics:
        plt.figure()
        compare_yaml["metrics"][m] = {}
        for pol in policies:
            series_by_ep = per_policy[pol]["series_by_ep"]
            t_norm_list = per_policy[pol]["t_norm_list"]
            x_list = [s[m] for s in series_by_ep]
            mean_line = _bin_mean(x_list, t_norm_list, args.bins)
            plt.plot(xs, mean_line, label=pol)

            compare_yaml["metrics"][m][pol] = {
                "scalar": _scalar_summary(x_list),
            }

        title = f"{m}"
        if args.task_name is not None:
            title = f"{args.task_name}: {m}"
        plt.title(title)
        plt.xlabel("progress")
        plt.ylabel("mean")
        plt.legend()
        plt.tight_layout()
        plt.savefig(compare_dir / f"{m}_policies_mean.png", dpi=200)
        plt.close()

    with open(compare_dir / "summary_compare.yaml", "w") as f:
        yaml.safe_dump(compare_yaml, f, sort_keys=False)

    print(f"[AnalyzeTimeSeriesNpz] Save {out_dir}")


if __name__ == "__main__":
    main()
