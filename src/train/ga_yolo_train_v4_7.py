# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-05-26 (V4.7)
# Dependency: dataset_v3_1, yolov8n.pt
# Description: GA-YOLO V4.7 - CARRADA noisy stress dataset 재학습
# ================================================================================

from pathlib import Path

from ultralytics import YOLO


DATASET_YAML = "/home/kmin/RD_YOLO_GHOST/dataset_v3_1/dataset.yaml"
MODEL_NAME = "yolov8n.pt"
PROJECT_DIR = "/home/kmin/RD_YOLO_GHOST/runs/detect"
RUN_NAME = "ga_yolo_v4_7_noisy_stress"


# 학습 전에 YOLO가 읽을 dataset.yaml을 항상 같은 내용으로 맞춘다.
# 입력: yaml_path (dataset.yaml 저장 경로)
# 반환: 없음
def write_dataset_yaml(yaml_path):
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "path: /home/kmin/RD_YOLO_GHOST/dataset_v3_1\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 2\n"
        "names: ['real', 'ghost']\n"
    )
    yaml_path.write_text(text, encoding="utf-8")


# GPU가 없으면 학습이 중단되지 않도록 CPU로 자동 전환한다.
# 입력: 없음
# 반환: ultralytics train에 넘길 device 값
def select_device():
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


# Ultralytics 결과 객체와 results.csv에서 핵심 지표를 최대한 안전하게 읽는다.
# 입력: results (train 반환 객체)
# 반환: 요약 dict
def summarize_training(results):
    save_dir = Path(getattr(results, "save_dir", Path(PROJECT_DIR) / RUN_NAME))
    results_csv = save_dir / "results.csv"
    summary = {
        "best_map50": None,
        "best_map50_95": None,
        "final_epoch": None,
    }

    if not results_csv.exists():
        return summary

    rows = results_csv.read_text(encoding="utf-8").strip().splitlines()
    if len(rows) < 2:
        return summary

    headers = [header.strip() for header in rows[0].split(",")]
    data_rows = rows[1:]

    def find_column(candidates):
        for candidate in candidates:
            for idx, header in enumerate(headers):
                if candidate in header:
                    return idx
        return None

    epoch_idx = find_column(["epoch"])
    map50_idx = find_column(["metrics/mAP50(B)", "mAP50(B)"])
    map50_95_idx = find_column(["metrics/mAP50-95(B)", "mAP50-95(B)"])

    map50_values = []
    map50_95_values = []
    final_epoch = None
    for row in data_rows:
        values = [value.strip() for value in row.split(",")]
        if epoch_idx is not None and epoch_idx < len(values):
            final_epoch = int(float(values[epoch_idx]))
        if map50_idx is not None and map50_idx < len(values):
            map50_values.append(float(values[map50_idx]))
        if map50_95_idx is not None and map50_95_idx < len(values):
            map50_95_values.append(float(values[map50_95_idx]))

    summary["best_map50"] = max(map50_values) if map50_values else None
    summary["best_map50_95"] = max(map50_95_values) if map50_95_values else None
    summary["final_epoch"] = final_epoch
    return summary


# V4.7 noisy stress dataset 재학습을 실행한다.
# 입력: 없음
# 반환: 없음
def main():
    write_dataset_yaml(DATASET_YAML)
    device = select_device()

    print("=" * 80)
    print("GA-YOLO V4.7 noisy stress training")
    print("=" * 80)
    print(f"dataset.yaml: {DATASET_YAML}")
    print(f"device      : {device}")
    print(f"run name    : {RUN_NAME}")
    print("=" * 80)

    model = YOLO(MODEL_NAME)
    results = model.train(
        data=DATASET_YAML,
        epochs=200,
        imgsz=256,
        batch=16,
        device=device,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        patience=30,
        save=True,
        plots=True,
    )

    run_dir = Path(getattr(results, "save_dir", Path(PROJECT_DIR) / RUN_NAME))
    best_path = run_dir / "weights" / "best.pt"
    summary = summarize_training(results)

    print("=" * 80)
    print("V4.7 학습 결과 요약")
    print("=" * 80)
    print(f"best.pt 경로 : {best_path}")
    if summary["best_map50"] is None:
        print("best mAP50   : results.csv에서 확인 불가")
    else:
        print(f"best mAP50   : {summary['best_map50']:.6f}")
    if summary["best_map50_95"] is None:
        print("best mAP50-95: results.csv에서 확인 불가")
    else:
        print(f"best mAP50-95: {summary['best_map50_95']:.6f}")
    if summary["final_epoch"] is None:
        print("최종 epoch 수: results.csv에서 확인 불가")
    else:
        print(f"최종 epoch 수: {summary['final_epoch']}")
    print("V4.7 학습 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
