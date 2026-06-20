# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-05-26 (V4.8)
# Dependency: dataset_v3_2, yolov8n.pt
# Description: GA-YOLO V4.8 - negative scene 포함 데이터셋으로 FP 제어 재학습
#              V4.7 실패 원인(FP 폭증) 해결 목적
# ================================================================================

from pathlib import Path

from ultralytics import YOLO


DATASET_YAML = "/home/kmin/RD_YOLO_GHOST/dataset_v3_2/dataset.yaml"
MODEL_NAME = "yolov8n.pt"
PROJECT_DIR = "/home/kmin/RD_YOLO_GHOST/runs/detect"
RUN_NAME = "ga_yolo_v4_8_negative"

DATASET_ROOT = "/home/kmin/RD_YOLO_GHOST/dataset_v3_2"


# 학습 시작 전에 YOLO가 읽을 dataset.yaml을 명세와 같은 내용으로 고정한다.
# 입력: yaml_path (dataset.yaml 저장 경로/string)
# 반환: 없음
def write_dataset_yaml(yaml_path):
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "path: /home/kmin/RD_YOLO_GHOST/dataset_v3_2\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n"
        "nc: 2\n"
        "names: ['real', 'ghost']\n"
    )
    yaml_path.write_text(text, encoding="utf-8")


# 데이터셋이 비어 있으면 학습 결과가 무의미하므로 시작 전에 최소 구조를 확인한다.
# 입력: dataset_root (데이터셋 루트 경로/string)
# 반환: 없음
def validate_dataset_dirs(dataset_root):
    dataset_root = Path(dataset_root)
    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
        dataset_root / "labels" / "test",
    ]
    for required_dir in required_dirs:
        if not required_dir.is_dir():
            raise FileNotFoundError(f"필수 데이터셋 폴더가 없습니다: {required_dir}")


# CUDA가 없는 환경에서도 스크립트가 중단되지 않도록 CPU로 자동 전환한다.
# 입력: 없음
# 반환: ultralytics train에 전달할 device 값
def select_device():
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


# Ultralytics results.csv에서 핵심 지표를 읽어 실험 로그에 바로 남길 수 있게 한다.
# 입력: results (model.train 반환 객체)
# 반환: summary (best mAP와 최종 epoch dict)
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


# None 값도 콘솔에서 구분되게 출력해 results.csv 누락 여부를 숨기지 않는다.
# 입력: label (출력 이름/string), value (지표/float 또는 None)
# 반환: 없음
def print_metric(label, value):
    if value is None:
        print(f"{label}: results.csv에서 확인 불가")
    else:
        print(f"{label}: {value:.6f}")


# negative scene 포함 V3.2 데이터셋으로 GA-YOLO를 재학습한다.
# 입력: 없음
# 반환: 없음
def main():
    write_dataset_yaml(DATASET_YAML)
    validate_dataset_dirs(DATASET_ROOT)
    device = select_device()

    print("=" * 80)
    print("GA-YOLO V4.8 negative scene training")
    print("=" * 80)
    print(f"dataset.yaml: {DATASET_YAML}")
    print(f"model       : {MODEL_NAME}")
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
    print("V4.8 학습 결과 요약")
    print("=" * 80)
    print(f"best.pt 경로 : {best_path}")
    print_metric("best mAP50   ", summary["best_map50"])
    print_metric("best mAP50-95", summary["best_map50_95"])
    if summary["final_epoch"] is None:
        print("최종 epoch 수: results.csv에서 확인 불가")
    else:
        print(f"최종 epoch 수: {summary['final_epoch']}")
    print("V4.8 학습 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
