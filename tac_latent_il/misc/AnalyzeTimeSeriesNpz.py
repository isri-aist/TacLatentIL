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
    t0 = max(0, t - W)
    t1 = min(T, t + W + 1)
    return t0, t1


def _compute_series_per_episode(
    z: np.ndarray,  # [T, D]
    grad_feat: np.ndarray,  # [T, D]
    sigma: np.ndarray,  # [D] (global)
    W: int,
):
    Tz, D = z.shape

    grad_sens = np.linalg.norm(grad_feat, axis=-1).astype(np.float32)  # [T]

    grad_znorm_feat = grad_feat * sigma[None, :]
    grad_sens_znorm = np.linalg.norm(grad_znorm_feat, axis=-1).astype(np.float32)

    dz = np.linalg.norm(z[1:] - z[:-1], axis=1).astype(np.float32)
    dz = np.concatenate([np.array([0.0], dtype=np.float32), dz], axis=0)

    lat_std_w = np.zeros((Tz,), dtype=np.float32)
    lat_rank_w = np.zeros((Tz,), dtype=np.float32)
    grad_w = np.zeros((Tz,), dtype=np.float32)
    grad_znorm_w = np.zeros((Tz,), dtype=np.float32)
    dz_w = np.zeros((Tz,), dtype=np.float32)

    for t in range(Tz):
        t0, t1 = _window_indices(Tz, t, W)
        zw = z[t0:t1]
        lat_std_w[t] = float(np.mean(np.std(zw, axis=0)))
        lat_rank_w[t] = float(_effective_rank(zw))
        grad_w[t] = float(np.mean(grad_sens[t0:t1]))
        grad_znorm_w[t] = float(np.mean(grad_sens_znorm[t0:t1]))
        dz_w[t] = float(np.mean(dz[t0:t1]))

    return {
        "grad_sens": grad_sens,
        "grad_sens_w": grad_w,
        "grad_sens_znorm": grad_sens_znorm,
        "grad_sens_znorm_w": grad_znorm_w,
        "lat_std_w": lat_std_w,
        "lat_rank_w": lat_rank_w,
        "dz_norm": dz,
        "dz_norm_w": dz_w,
    }


def _bin_aggregate(x_list, t_list, bins: int):
    bucket = [[] for _ in range(bins)]
    for x, tt in zip(x_list, t_list):
        b = np.clip((tt * bins).astype(int), 0, bins - 1)
        for i in range(len(tt)):
            bucket[b[i]].append(float(x[i]))

    med = np.full((bins,), np.nan, dtype=np.float32)
    q25 = np.full((bins,), np.nan, dtype=np.float32)
    q75 = np.full((bins,), np.nan, dtype=np.float32)
    n = np.zeros((bins,), dtype=np.int64)

    for i in range(bins):
        if len(bucket[i]) == 0:
            continue
        arr = np.array(bucket[i], dtype=np.float32)
        med[i] = float(np.median(arr))
        q25[i] = float(np.quantile(arr, 0.25))
        q75[i] = float(np.quantile(arr, 0.75))
        n[i] = int(arr.size)

    return {"median": med, "q25": q25, "q75": q75, "n": n}


def _scalar_summary(x_list):
    allv = np.concatenate([np.asarray(x, dtype=np.float32) for x in x_list], axis=0)
    if allv.size == 0:
        return {}
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
    xs = None
    plt.figure()
    for tc, agg in sorted(taskcond_to_agg.items(), key=lambda kv: kv[0]):
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
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    npz = np.load(args.in_npz, allow_pickle=True)
    episode_id = np.asarray(npz["episode_id"])
    task_cond = np.asarray(npz["task_cond"], dtype=object)
    t_norm_list = list(npz["t_norm"])
    z_list = list(npz["z"])
    grad_feat_list = list(npz["grad_feat"])

    z_all = np.concatenate(z_list, axis=0)  # [T_all, D]
    sigma = z_all.std(axis=0)
    sigma = np.maximum(sigma, 1e-8)

    series_by_ep = []
    for z, gfeat in zip(z_list, grad_feat_list):
        series_by_ep.append(_compute_series_per_episode(z, gfeat, sigma, args.window))

    metrics = [
        "grad_sens",
        "grad_sens_w",
        "grad_sens_znorm",
        "grad_sens_znorm_w",
        "lat_std_w",
        "lat_rank_w",
        "dz_norm",
        "dz_norm_w",
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
        x_list = [s[m] for s in series_by_ep]
        agg = _bin_aggregate(x_list, t_norm_list, args.bins)
        _plot_band(agg, title=m, out_png=os.path.join(args.out_dir, f"{m}.png"))
        out_yaml["metrics"][m] = {
            "scalar": _scalar_summary(x_list),
            "bin_n_min": int(np.nanmin(agg["n"])) if np.any(agg["n"] > 0) else 0,
        }

    # by task_cond
    uniq_tc = sorted(set([int(x) for x in task_cond if x is not None]))
    for tc in uniq_tc:
        idx = [i for i in range(len(task_cond)) if int(task_cond[i]) == tc]
        if len(idx) == 0:
            continue
        out_yaml["by_task_cond"][int(tc)] = {
            "num_episodes": int(len(idx)),
            "metrics": {},
        }

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
