import argparse
from collections import defaultdict
from pathlib import Path

import yaml


def policy_display_name(raw_policy_name: str, variant: str | None):
    if raw_policy_name == "DiffusionPolicy":
        return f"DP ({variant})"

    # DiffusionPolicyX -> X
    name = raw_policy_name.removeprefix("DiffusionPolicy")
    if name == "Sparsh" and variant is not None:
        return f"Sparsh({variant})"
    return name


def parse_success_rate(
    success_str: str, success_chars: set[str], partial_success_chars: set[str]
) -> float | None:
    """
    - Ignore spaces always.
    - Success if char in success_chars (counts as 1.0 point).
    - Partial success if char in partial_success_chars (counts as 0.5 point).
    - Failure if char is neither success nor partial (counts as 0.0 point, but included in denominator).
    - Return (points) / (num non-space chars). If denom == 0, return None.
    """
    points = 0.0
    denom = 0
    for ch in success_str:
        if ch == " ":
            continue
        denom += 1
        if ch in success_chars:
            points += 1.0
        elif ch in partial_success_chars:
            points += 0.5
        else:
            # failure: +0.0
            pass
    if denom == 0:
        return None
    return points / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="rollout success yaml")
    parser.add_argument(
        "--success_chars",
        type=str,
        default="1",
        help="characters treated as success (space is always ignored). default: '1'",
    )
    parser.add_argument(
        "--partial_success_chars",
        type=str,
        default="",
        help="characters treated as partial success (0.5 point; space is always ignored). default: ''",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output yaml path (aggregated). if omitted, no yaml is saved",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml_path).expanduser().resolve()
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    success_chars = set(args.success_chars)
    partial_success_chars = set(args.partial_success_chars)
    print("Args:")
    print(f"- success_chars: {success_chars}")
    print(f"- partial_success_chars: {partial_success_chars}\n")

    # Validation: no overlap between success and partial success
    overlap = success_chars & partial_success_chars
    if overlap:
        overlap_str = "".join(sorted(overlap))
        raise ValueError(
            f"success_chars and partial_success_chars must not overlap: {overlap_str!r}"
        )

    # Task keys (skip Meta)
    task_keys = [k for k in data.keys() if k != "Meta"]
    if not task_keys:
        raise ValueError("No task section found (expected keys other than 'Meta').")

    # groups[policy] -> list of per-run success rates
    groups = defaultdict(list)

    for task in task_keys:
        task_dict = data.get(task, {}) or {}
        for raw_policy, policy_dict in task_dict.items():
            if not isinstance(policy_dict, dict):
                continue

            if raw_policy == "DiffusionPolicy":
                # Bare DiffusionPolicy: camera / tactile
                for variant in ("camera", "tactile"):
                    if variant not in policy_dict or not isinstance(
                        policy_dict[variant], dict
                    ):
                        continue
                    disp = policy_display_name(raw_policy, variant)
                    success_list = policy_dict[variant].get("success", []) or []
                    for s in success_list:
                        r = parse_success_rate(
                            str(s), success_chars, partial_success_chars
                        )
                        if r is not None:
                            groups[disp].append(r)
            elif raw_policy.endswith("Sparsh"):
                # Sparsh: split by encoder variants
                for enc, enc_dict in policy_dict.items():
                    if not isinstance(enc_dict, dict):
                        continue
                    success_list = enc_dict.get("success", []) or []
                    disp = policy_display_name(raw_policy, enc)
                    for s in success_list:
                        r = parse_success_rate(
                            str(s), success_chars, partial_success_chars
                        )
                        if r is not None:
                            groups[disp].append(r)
            else:
                # Others: MUST have "default"
                if "default" not in policy_dict or not isinstance(
                    policy_dict["default"], dict
                ):
                    raise ValueError(
                        f"Non-sparsh policy '{raw_policy}' must have a dict entry 'default'."
                    )
                disp = policy_display_name(raw_policy, None)
                success_list = policy_dict["default"].get("success", []) or []
                for s in success_list:
                    r = parse_success_rate(str(s), success_chars, partial_success_chars)
                    if r is not None:
                        groups[disp].append(r)

    # Aggregate
    agg_rows = []
    for policy, rates in groups.items():
        n = len(rates)
        if n == 0:
            continue
        mean = sum(rates) / n
        var = sum((x - mean) ** 2 for x in rates) / n  # population variance
        agg_rows.append(
            {
                "policy": policy,
                "n": int(n),
                "mean_success_rate": float(mean),
                "var_success_rate": float(var),
            }
        )

    # Sort: higher success is better
    agg_rows = sorted(agg_rows, key=lambda r: r["mean_success_rate"], reverse=True)

    # Print table
    policy_width = (
        max(len("policy"), *(len(r["policy"]) for r in agg_rows))
        if agg_rows
        else len("policy")
    )
    mean_header = "mean_success"
    var_header = "var_success"
    n_header = "n"
    mean_width = max(len(mean_header), 10)
    var_width = max(len(var_header), 10)
    n_width = max(len(n_header), 3)

    header_parts = [
        "policy".ljust(policy_width),
        mean_header.rjust(mean_width),
        var_header.rjust(var_width),
        n_header.rjust(n_width),
    ]
    print("  ".join(header_parts))

    for r in agg_rows:
        parts = [
            r["policy"].ljust(policy_width),
            f"{r['mean_success_rate']:.2f}".rjust(mean_width),
            f"{r['var_success_rate']:.4f}".rjust(var_width),
            str(r["n"]).rjust(n_width),
        ]
        print("  ".join(parts))

    # Save yaml
    if args.out is not None:
        out_path = Path(args.out).expanduser().resolve()
        with open(out_path, "w") as f:
            yaml.safe_dump(agg_rows, f, sort_keys=False)
        print(f"Save {out_path}")

    # Plot
    import matplotlib.pyplot as plt
    import numpy as np

    policies = [r["policy"] for r in agg_rows]
    means = np.array([r["mean_success_rate"] for r in agg_rows], dtype=np.float64)
    stds = np.sqrt(
        np.array([r["var_success_rate"] for r in agg_rows], dtype=np.float64)
    )

    x = np.arange(len(policies))
    plt.figure(figsize=(0.6 * len(policies) + 2, 4))
    plt.bar(x, means, yerr=stds, capsize=4, color="#FF6A00")
    plt.xticks(x, policies, rotation=45, ha="right")
    plt.ylabel("Success rate (mean ± std)")
    plt.title(f"{task_keys[0]}")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    if args.out:
        png_path = out_path.with_suffix(".png")
        plt.savefig(png_path, dpi=150)
        print(f"Save {png_path}")

    plt.show()


if __name__ == "__main__":
    main()
