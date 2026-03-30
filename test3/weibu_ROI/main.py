import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import Config
from data_pipeline import load_all_data

from analysis import build_valid_frames, run_analysis, print_summary
from visualization import launch_viewer


def main():

    params = {
        "START_FRAME": 100,
        "END_FRAME": 200,

        "ROI_OUTER": 1.0,
        "ROI_INNER": 1.0,

        "GT_DIM": {
            0: {"L": 5.06, "W": 2.22},
            1: {"L": 4.32, "W": 2.19},
            2: {"L": 3.55, "W": 2.58},
        },

        # True: 使用 Kalman
        # False: 不使用 Kalman，直接用 ROI 原始量测
        "USE_KALMAN": False,

        "KF_DT": 1.0,
        "KF_Q_POS": 0.01,
        "KF_Q_VEL": 0.01,
        "KF_R_POS": 1.50,
        "KF_INIT_POS_VAR": 4.0,
        "KF_INIT_VEL_VAR": 9.0,

        "MIN_ROI_POINTS": 1,
        "ENABLE_VIEWER": True,
        "SAVE_RESULT_CSV": False,
        "RESULT_CSV_PATH": "per_target_results.csv",
    }

    cfg = Config()
    radar_data, gt_df = load_all_data(cfg)
    print(gt_df.columns.tolist())

    frame_ids = build_valid_frames(
        radar_data,
        gt_df,
        params["START_FRAME"],
        params["END_FRAME"],
    )
    if not frame_ids:
        raise ValueError(
            f"在 {params['START_FRAME']} 到 {params['END_FRAME']} 帧之间没有找到公共帧。"
        )

    print("有效帧:")
    print(frame_ids)

    result_df, frame_cache = run_analysis(radar_data, gt_df, frame_ids, params)
    print_summary(result_df, params)

    if params.get("SAVE_RESULT_CSV", False):
        csv_path = params.get("RESULT_CSV_PATH", "per_target_results.csv")
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n逐目标结果已保存到: {csv_path}")

    if params["ENABLE_VIEWER"]:
        launch_viewer(frame_ids, frame_cache, params)


if __name__ == "__main__":
    main()
