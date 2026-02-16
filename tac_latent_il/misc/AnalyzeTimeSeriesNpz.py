import argparse
import os

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


def _window_indices(T: int, t: int, W: int):
    return max(0, t - W), min(T, t + W + 1)


def _compute_windowed_mean(series: np.ndarray, W: int) -> np.ndarray:
    """
    series: [T]
    return: [T], out[t] = mean(series[t-W : t+W])
    """
    T = int(series.shape[0])
    out = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        t0, t1 = _window_indices(T, t, W)
        out[t] = float(np.mean(series[t0:t1]))
    return out


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
        t0, t1 = _window_indices(T, t, W)
        zw = z[t0:t1]
        lat_std_w[t] = float(np.mean(np.std(zw, axis=0)))
        lat_rank_w[t] = float(_effective_rank(zw))
    return lat_std_w, lat_rank_w


def _compute_series_per_episode(
    z: np.ndarray,  # [T, Dz]
    action_gt: np.ndarray,  # [T, Ta, Da]
    action_pred: np.ndarray,  # [T, Ns, Ta, Da]
    grad_feat: np.ndarray,  # [T, Ns, Dz]
    sigma: np.ndarray,  # [Dz] (global)
    W: int,
):
    T, Dz = z.shape
    Ns = int(action_pred.shape[1])
    _ = Ns  # silence lint if needed

    # action error (RMSE), averaged over Ns samples
    err = action_pred - action_gt[:, None, :, :]
    rmse = np.sqrt(np.mean(err**2, axis=(2, 3)))  # [T, Ns]
    action_err = rmse.mean(axis=1).astype(np.float32)  # [T]

    # action variance across Ns samples
    action_var = action_pred.var(axis=1).mean(axis=(1, 2)).astype(np.float32)  # [T]

    # grad sensitivity across Ns samples
    grad = np.linalg.norm(grad_feat, axis=-1)  # [T, Ns]
    grad_mean = grad.mean(axis=1).astype(np.float32)  # [T]
    grad_var = grad.var(axis=1).astype(np.float32)  # [T]

    # grad sensitivity with z-normalization (sigma weighting)
    grad_znorm = np.linalg.norm(grad_feat * sigma[None, None, :], axis=-1)  # [T, Ns]
    grad_znorm_mean = grad_znorm.mean(axis=1).astype(np.float32)  # [T]
    grad_znorm_var = grad_znorm.var(axis=1).astype(np.float32)  # [T]

    # latent dynamics
    dz = np.linalg.norm(z[1:] - z[:-1], axis=1)
    dz = np.concatenate([[0.0], dz]).astype(np.float32)

    # windowed
    lat_std_w, lat_rank_w = _compute_lat_windowed(z, W)

    dz_w = _compute_windowed_mean(dz, W)
    action_err_w = _compute_windowed_mean(action_err, W)
    action_var_w = _compute_windowed_mean(action_var, W)
    grad_mean_w = _compute_windowed_mean(grad_mean, W)
    grad_var_w = _compute_windowed_mean(grad_var, W)
    grad_znorm_mean_w = _compute_windowed_mean(grad_znorm_mean, W)
    grad_znorm_var_w = _compute_windowed_mean(grad_znorm_var, W)

    return {
        "action_err": action_err,
        "action_err_w": action_err_w,
        "action_var": action_var,
        "action_var_w": action_var_w,
        "grad_mean": grad_mean,
        "grad_mean_w": grad_mean_w,
        "grad_var": grad_var,
        "grad_var_w": grad_var_w,
        "grad_znorm_mean": grad_znorm_mean,
        "grad_znorm_mean_w": grad_znorm_mean_w,
        "grad_znorm_var": grad_znorm_var,
        "grad_znorm_var_w": grad_znorm_var_w,
        "lat_std_w": lat_std_w,
        "lat_rank_w": lat_rank_w,
        "dz": dz,
        "dz_w": dz_w,
    }


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


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("in_npz", type=str)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--window", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    npz = np.load(args.in_npz, allow_pickle=True)
    episode_id = np.asarray(npz["episode_id"])
    task_cond = np.asarray(npz["task_cond"], dtype=int)

    t_norm_list = list(npz["t_norm"])
    z_list = list(npz["z"])
    action_gt_list = list(npz["action_gt"])
    action_pred_list = list(npz["action_pred"])
    grad_feat_list = list(npz["grad_feat"])

    # global sigma for z-normalized grad
    z_all = np.concatenate(
        [np.asarray(z, dtype=np.float32) for z in z_list], axis=0
    )  # [T_all, Dz]
    sigma = z_all.std(axis=0).astype(np.float32)
    sigma = np.maximum(sigma, 1e-8)

    series_by_ep = []
    for z, ag, apred, gf in zip(
        z_list, action_gt_list, action_pred_list, grad_feat_list
    ):
        series_by_ep.append(
            _compute_series_per_episode(z, ag, apred, gf, sigma, args.window)
        )

    metrics = [
        "action_err",
        "action_err_w",
        "action_var",
        "action_var_w",
        "grad_mean",
        "grad_mean_w",
        "grad_var",
        "grad_var_w",
        "grad_znorm_mean",
        "grad_znorm_mean_w",
        "grad_znorm_var",
        "grad_znorm_var_w",
        "lat_std_w",
        "lat_rank_w",
        "dz",
        "dz_w",
    ]

    out_yaml = {
        "in_npz": args.in_npz,
        "bins": int(args.bins),
        "window": int(args.window),
        "num_episodes": int(len(episode_id)),
        "metrics": {},
        "by_task_cond": {},
    }

    # overall
    for m in metrics:
        xs = [s[m] for s in series_by_ep]
        agg = _bin_aggregate(xs, t_norm_list, args.bins)
        _plot_band(agg, m, os.path.join(args.out_dir, f"{m}.png"))
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
            agg_tc = _bin_aggregate(x_tc, t_tc, args.bins)
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
            out_png=os.path.join(args.out_dir, f"{m}_by_task_cond.png"),
        )

    out_path = os.path.join(args.out_dir, "summary.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False)

    print(f"[AnalyzeTimeSeriesNpz] Save {args.out_dir}")


if __name__ == "__main__":
    main()
