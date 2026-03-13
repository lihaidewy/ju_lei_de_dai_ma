import numpy as np


def init_stats():
    return {
        "sum_TP": 0,
        "sum_FP": 0,
        "sum_FN": 0,
        "all_center_err": [],
        "all_dx_err": [],
        "all_dy_err": [],
        "model_tot": {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0, 1, 2]},
    }


def update_stats(stats: dict, metrics: dict):
    stats["sum_TP"] += int(metrics["TP"])
    stats["sum_FP"] += int(metrics["FP"])
    stats["sum_FN"] += int(metrics["FN"])

    stats["all_center_err"] += list(metrics.get("center_errors", []))
    stats["all_dx_err"] += list(metrics.get("dx_errors", []))
    stats["all_dy_err"] += list(metrics.get("dy_errors", []))

    for mm in [0, 1, 2]:
        for k in ["TP", "FP", "FN"]:
            stats["model_tot"][mm][k] += int(metrics["model_counts"][mm][k])


def update_range_bias_stats(range_bias_stats: dict, range_bins: list, matches: list, gt_list: list):
    for mmatch in matches:
        gid = int(mmatch["gid"])
        dy = float(mmatch["dy"])

        gt_item = next((gg for gg in gt_list if int(gg["id"]) == gid), None)
        if gt_item is None:
            continue

        gy = float(gt_item["y"])
        for rb in range_bins:
            lo, hi = rb
            if lo <= gy < hi:
                range_bias_stats[rb].append(dy)
                break


def print_global_summary(frame_ids, stats, range_bins, range_bias_stats):
    print("\n" + "=" * 60)
    print("GLOBAL SUMMARY (over selected frames)")
    print("=" * 60)
    print(f"Frames evaluated: {len(frame_ids)}")
    print(f"Total: TP={stats['sum_TP']} FP={stats['sum_FP']} FN={stats['sum_FN']}")

    overall_P = stats["sum_TP"] / (stats["sum_TP"] + stats["sum_FP"]) if (stats["sum_TP"] + stats["sum_FP"]) else 1.0
    overall_R = stats["sum_TP"] / (stats["sum_TP"] + stats["sum_FN"]) if (stats["sum_TP"] + stats["sum_FN"]) else 1.0
    overall_F1 = (
        2 * overall_P * overall_R / (overall_P + overall_R)
        if (overall_P + overall_R)
        else 0.0
    )
    print(f"Overall: P={overall_P:.4f} R={overall_R:.4f} F1={overall_F1:.4f}")

    if len(stats["all_center_err"]) > 0:
        ce = np.asarray(stats["all_center_err"], dtype=float)
        dxe = np.asarray(stats["all_dx_err"], dtype=float)
        dye = np.asarray(stats["all_dy_err"], dtype=float)

        mean_ce = float(np.mean(ce))
        median_ce = float(np.median(ce))
        p90_ce = float(np.percentile(ce, 90))
        p95_ce = float(np.percentile(ce, 95))
        acc_0p3m = float(np.mean(ce <= 0.3))
        acc_0p5m = float(np.mean(ce <= 0.5))

        mean_dx = float(np.mean(dxe))
        mean_dy = float(np.mean(dye))
        median_dx = float(np.median(dxe))
        median_dy = float(np.median(dye))
        std_dx = float(np.std(dxe))
        std_dy = float(np.std(dye))

        print(
            f"Center error (TP only): "
            f"mean={mean_ce:.3f}  "
            f"median={median_ce:.3f}  "
            f"p90={p90_ce:.3f}  "
            f"p95={p95_ce:.3f}  "
            f"<=0.3m={acc_0p3m * 100:.2f}%  "
            f"<=0.5m={acc_0p5m * 100:.2f}%"
        )

        print(
            f"Center bias (GT - Pred, TP only): "
            f"mean_dx={mean_dx:.3f}  "
            f"mean_dy={mean_dy:.3f}  "
            f"median_dx={median_dx:.3f}  "
            f"median_dy={median_dy:.3f}  "
            f"std_dx={std_dx:.3f}  "
            f"std_dy={std_dy:.3f}"
        )
    else:
        print("Center error (TP only): n/a (no TP matches)")
        print("Center bias (GT - Pred, TP only): n/a")

    print("\nPer-model (GT-side TP/FN, cluster-side FP assigned by nearest GT within fp_assign_dist):")
    for mm in [0, 1, 2]:
        tp = stats["model_tot"][mm]["TP"]
        fp = stats["model_tot"][mm]["FP"]
        fn = stats["model_tot"][mm]["FN"]
        p = tp / (tp + fp) if (tp + fp) else 1.0
        r = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        print(f"  model={mm}: TP={tp} FP={fp} FN={fn} | P={p:.4f} R={r:.4f} F1={f1:.4f}")

    print("\nRange-wise Y bias (GT - Pred, TP only):")
    for rb in range_bins:
        vals = range_bias_stats[rb]
        lo, hi = rb
        hi_str = f"{hi:.0f}" if hi < 1e8 else "inf"

        if len(vals) == 0:
            print(f"  [{lo:.0f}, {hi_str}): n=0")
            continue

        arr = np.asarray(vals, dtype=float)
        print(
            f"  [{lo:.0f}, {hi_str}): "
            f"n={len(arr)}  "
            f"mean_dy={np.mean(arr):.3f}  "
            f"median_dy={np.median(arr):.3f}  "
            f"std_dy={np.std(arr):.3f}"
        )

    print("=" * 60 + "\n")
