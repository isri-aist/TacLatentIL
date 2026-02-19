#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/path/to/eval")

# task_name -> list of (start, end_inclusive, value)
TASK_RULES = {
    "DrawEqualLines": [
        (0, 7, 0),
        (8, 15, 1),
        (16, 23, 2),
        (24, 31, 3),
    ],
    "Pick2Spoon": [
        (0, 7, 0),
        (8, 15, 1),
        (16, 23, 2),
        (24, 31, 3),
    ],
    "Pick3ContainedHeavyBoxBasket": [
        (0, 9, 0),
        (10, 19, 1),
        (20, 29, 2),
    ],
    "PickPlaceBrownPaperBagOnFileTray": [
        (0, 14, 0),
        (15, 29, 1),
    ],
}


def build_task_cond(
    n: int, rules: list[tuple[int, int, int]], task_name: str
) -> np.ndarray:
    tc = np.full((n,), -1, dtype=np.int64)

    for s, e, v in rules:
        if s < 0 or e < 0 or s > e:
            raise ValueError(f"[{task_name}] invalid rule: {(s, e, v)}")
        if e >= n:
            raise ValueError(f"[{task_name}] rule end {e} exceeds length n={n}")
        tc[s : e + 1] = int(v)

    if np.any(tc < 0):
        idx = np.where(tc < 0)[0]
        if len(idx) > 20:
            raise ValueError(
                f"[{task_name}] uncovered indices: {idx[:20].tolist()} (and more)"
            )
        raise ValueError(f"[{task_name}] uncovered indices: {idx.tolist()}")

    return tc


def _infer_n(data: np.lib.npyio.NpzFile) -> int:
    keys = list(data.files)
    if "task_cond" in keys:
        return int(np.asarray(data["task_cond"]).shape[0])
    if "episode_id" in keys:
        return int(np.asarray(data["episode_id"]).shape[0])
    return int(np.asarray(data[keys[0]]).shape[0])


def rewrite_npz(path: Path, rules: list[tuple[int, int, int]], task_name: str) -> None:
    data = np.load(path, allow_pickle=True)
    keys = list(data.files)

    n = _infer_n(data)
    new_tc = build_task_cond(n, rules, task_name)

    # 既存 task_cond の長さチェック（存在する場合）
    if "task_cond" in keys:
        old_tc = np.asarray(data["task_cond"])
        if old_tc.shape != new_tc.shape:
            raise ValueError(
                f"{path}: task_cond shape mismatch old={old_tc.shape} new={new_tc.shape}"
            )

    out = {k: data[k] for k in keys}
    out["task_cond"] = new_tc

    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)

    # IMPORTANT: np.savez appends ".npz" if missing, so ensure suffix is ".npz"
    tmp = path.with_name(path.stem + ".tmp.npz")  # e.g., Eval1.tmp.npz
    np.savez(tmp, **out)
    tmp.replace(path)

    print(f"[OK] {path}  task_cond={tuple(new_tc.shape)}  backup={bak.name}")


def main():
    # ROOT/<task>/<policy>/Eval*.npz
    for task_name, rules in TASK_RULES.items():
        task_dir = ROOT / task_name
        if not task_dir.exists():
            print(f"[SKIP] missing task dir: {task_dir}")
            continue

        npz_list = sorted(task_dir.glob("*/Eval*.npz"))
        if not npz_list:
            print(f"[SKIP] no npz found under: {task_dir}")
            continue

        print(f"\n=== {task_name} ({len(npz_list)} files) ===")
        for p in npz_list:
            rewrite_npz(p, rules, task_name)


if __name__ == "__main__":
    main()
