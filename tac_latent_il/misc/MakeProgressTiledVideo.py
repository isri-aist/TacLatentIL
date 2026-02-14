import argparse
from pathlib import Path

import cv2
import numpy as np
from robo_manip_baselines.common import find_rmb_files
from tqdm import tqdm


def _list_rmb_dirs(dataset_dir: Path):
    return [Path(p) for p in find_rmb_files(str(dataset_dir))]


def _open_video(rmb_dir: Path, camera_name: str):
    mp4 = rmb_dir / f"{camera_name}_rgb_image.rmb.mp4"
    if not mp4.exists():
        raise FileNotFoundError(mp4)
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {mp4}")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, length, fps


def _make_grid(imgs, grid_w):
    H, W, C = imgs[0].shape
    N = len(imgs)
    grid_h = int(np.ceil(N / grid_w))
    canvas = np.zeros((grid_h * H, grid_w * W, C), dtype=np.uint8)
    for i, img in enumerate(imgs):
        r = i // grid_w
        c = i % grid_w
        canvas[r * H : (r + 1) * H, c * W : (c + 1) * W] = img
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--camera_name", type=str, default="front")
    ap.add_argument("--grid_w", type=int, default=6)
    ap.add_argument("--out_fps", type=float, default=10.0)
    ap.add_argument("--progress_bar_h", type=int, default=40)
    ap.add_argument("--max_frames", type=int, default=200)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    rmb_dirs = _list_rmb_dirs(dataset_dir)
    assert len(rmb_dirs) > 0

    caps = []
    lengths = []
    for rmb in rmb_dirs:
        cap, L, fps = _open_video(rmb, args.camera_name)
        caps.append(cap)
        lengths.append(L)

    # use min length only to define output progress resolution
    F = min(min(lengths), args.max_frames)

    # read one frame to get size
    caps[0].set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame0 = caps[0].read()
    assert ok
    H, W, _ = frame0.shape

    grid_h = int(np.ceil(len(caps) / args.grid_w))
    out_H = grid_h * H + args.progress_bar_h
    out_W = args.grid_w * W

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, args.out_fps, (out_W, out_H))

    for f_out in tqdm(range(F), desc="Make tiled video"):
        progress = f_out / max(F - 1, 1)

        imgs = []
        for cap, L in zip(caps, lengths):
            t = int(round(progress * (L - 1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, img = cap.read()
            if not ok:
                img = np.zeros((H, W, 3), dtype=np.uint8)
            imgs.append(img)

        grid = _make_grid(imgs, args.grid_w)

        canvas = np.full((out_H, out_W, 3), 255, dtype=np.uint8)
        canvas[args.progress_bar_h :, :] = grid

        txt = f"progress = {progress:.2f}"
        cv2.putText(
            canvas,
            txt,
            (10, int(args.progress_bar_h * 0.8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        vw.write(canvas)

    vw.release()
    for cap in caps:
        cap.release()

    print(f"[MakeProgressTiledVideo] Saved {args.out}")


if __name__ == "__main__":
    main()
