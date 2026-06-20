# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-05-21 (V4.1)
# Dependency: ultralytics, PIL, pandas, matplotlib, torch
# Description: GA-YOLO Ablation 실험 (viridis vs grayscale × YOLOv8n vs YOLOv8s)
# ================================================================================

import argparse
import json
import shutil
import signal
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO


STOP_REQUESTED = False

BASELINE_A0 = {
    "id": "A0",
    "name": "A0_v4_0_baseline",
    "mode": "viridis",
    "model_name": "YOLOv8n",
    "map50": 0.968,
    "map50_95": 0.775,
    "precision": 0.985,
    "recall": 0.962,
    "params_m": 3.0,
    "speed_ms": 3.4,
    "batch_used": 16,
    "note": "V4.0 baseline (aug default)",
}

EXPERIMENTS = [
    {
        "id": "A",
        "name": "A_vir_n",
        "mode": "viridis",
        "data": "dataset_v3_0b/data.yaml",
        "model": "yolov8n.pt",
        "model_name": "YOLOv8n",
        "note": "V4.1 retrain (aug off)",
    },
    {
        "id": "B",
        "name": "B_gray_n",
        "mode": "gray*",
        "data": "dataset_v4_1_gray/data.yaml",
        "model": "yolov8n.pt",
        "model_name": "YOLOv8n",
        "note": "V4.1 (aug off)",
    },
    {
        "id": "C",
        "name": "C_vir_s",
        "mode": "viridis",
        "data": "dataset_v3_0b/data.yaml",
        "model": "yolov8s.pt",
        "model_name": "YOLOv8s",
        "note": "V4.1 (aug off)",
    },
    {
        "id": "D",
        "name": "D_gray_s",
        "mode": "gray*",
        "data": "dataset_v4_1_gray/data.yaml",
        "model": "yolov8s.pt",
        "model_name": "YOLOv8s",
        "note": "V4.1 (aug off)",
    },
]


# Ctrl+C가 들어와도 이미 끝난 실험 결과를 보존하기 위해 전역 플래그만 바꾼다.
# 입력: signum(시그널 번호), frame(현재 프레임)
# 반환: 없음
def handle_interrupt(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[Interrupt] 현재 실험이 끝나는 지점에서 결과를 저장하고 종료합니다.")


# 사용자 경로의 ~ 표현을 WSL/리눅스 환경 기준 절대경로로 바꾼다.
# 입력: path_like(문자열 또는 Path)
# 반환: 확장된 Path
def expand_path(path_like):
    return Path(path_like).expanduser().resolve()


# viridis RGB PNG를 회색 표현으로 바꾸되 YOLO 입력 호환성을 위해 RGB 3채널을 유지한다.
# 입력: src_dir(dataset_v3_0b), dst_dir(dataset_v4_1_gray)
# 반환: 변환 소요 시간(초)
def convert_to_grayscale_dataset(src_dir, dst_dir):
    src_dir = expand_path(src_dir)
    dst_dir = expand_path(dst_dir)
    start = time.time()

    if not src_dir.exists():
        raise FileNotFoundError(f"원본 데이터셋이 없습니다: {src_dir}")

    if dst_dir.exists():
        if dst_dir.name != "dataset_v4_1_gray":
            raise ValueError(f"예상하지 않은 grayscale 출력 경로입니다: {dst_dir}")
        shutil.rmtree(dst_dir)

    for split in ("train", "val", "test"):
        src_image_dir = src_dir / "images" / split
        dst_image_dir = dst_dir / "images" / split
        dst_image_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(src_image_dir.glob("*.png"))
        if not image_paths:
            raise RuntimeError(f"변환할 이미지가 없습니다: {src_image_dir}")

        for image_path in image_paths:
            # 컬러맵 차이만 실험해야 하므로 크기와 좌표계를 바꾸지 않고 mode만 바꾼다.
            with Image.open(image_path) as img:
                gray_rgb = img.convert("L").convert("RGB")
                gray_rgb.save(dst_image_dir / image_path.name)

    labels_src = src_dir / "labels"
    labels_dst = dst_dir / "labels"
    if labels_src.exists():
        shutil.copytree(labels_src, labels_dst)
    else:
        raise FileNotFoundError(f"라벨 폴더가 없습니다: {labels_src}")

    meta_src = src_dir / "meta"
    if meta_src.exists():
        # 메타는 학습에는 필요 없지만 오류 분석에서 원본 scene 정보를 다시 보기 위해 보존한다.
        shutil.copytree(meta_src, dst_dir / "meta")

    data_yaml = (
        "# YOLOv8 dataset configuration\n"
        f"path: {dst_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "# Classes\n"
        "names:\n"
        "  0: real_target\n"
        "  1: ghost_target\n\n"
        "nc: 2\n"
    )
    (dst_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    return time.time() - start


# grayscale 변환이 시각적으로 정상인지 빠르게 확인할 수 있는 비교 이미지를 저장한다.
# 입력: src_dir, dst_dir, output_path, n_samples
# 반환: 없음
def save_grayscale_sample(src_dir, dst_dir, output_path, n_samples=3):
    src_dir = expand_path(src_dir)
    dst_dir = expand_path(dst_dir)
    output_path = expand_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    src_images = sorted((src_dir / "images" / "train").glob("*.png"))[:n_samples]
    if len(src_images) < n_samples:
        raise RuntimeError("grayscale sample을 만들 이미지 수가 부족합니다.")

    fig, axes = plt.subplots(n_samples, 2, figsize=(7, 3.2 * n_samples))
    if n_samples == 1:
        axes = np.asarray([axes])

    for row, src_image in enumerate(src_images):
        dst_image = dst_dir / "images" / "train" / src_image.name
        with Image.open(src_image) as src_img, Image.open(dst_image) as dst_img:
            axes[row, 0].imshow(src_img)
            axes[row, 1].imshow(dst_img)
        axes[row, 0].set_title(f"viridis: {src_image.name}")
        axes[row, 1].set_title("viridis-derived grayscale RGB")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# 이미지와 라벨 수가 맞는지 확인해 학습 중 조용히 깨지는 상황을 미리 막는다.
# 입력: dataset_dir
# 반환: split별 이미지 수 딕셔너리
def validate_dataset(dataset_dir):
    dataset_dir = expand_path(dataset_dir)
    expected_counts = {"train": 1600, "val": 200, "test": 200}
    counts = {}

    if not (dataset_dir / "data.yaml").exists():
        raise FileNotFoundError(f"data.yaml이 없습니다: {dataset_dir / 'data.yaml'}")

    for split, expected in expected_counts.items():
        image_paths = sorted((dataset_dir / "images" / split).glob("*.png"))
        label_paths = sorted((dataset_dir / "labels" / split).glob("*.txt"))
        counts[split] = len(image_paths)

        if len(image_paths) != expected:
            raise RuntimeError(f"{split} 이미지 수가 예상과 다릅니다: {len(image_paths)} != {expected}")
        if len(label_paths) != expected:
            raise RuntimeError(f"{split} 라벨 수가 예상과 다릅니다: {len(label_paths)} != {expected}")

        image_stems = {path.stem for path in image_paths}
        label_stems = {path.stem for path in label_paths}
        if image_stems != label_stems:
            missing_labels = sorted(image_stems - label_stems)[:5]
            missing_images = sorted(label_stems - image_stems)[:5]
            raise RuntimeError(
                f"{split} image/label stem 불일치: "
                f"missing_labels={missing_labels}, missing_images={missing_images}"
            )

        for image_path in image_paths:
            with Image.open(image_path) as img:
                if img.size != (256, 256):
                    raise RuntimeError(f"이미지 크기가 256x256이 아닙니다: {image_path}, size={img.size}")

    return counts


# Ultralytics 결과 객체의 필드명이 버전마다 조금 달라도 핵심 값을 안전하게 꺼낸다.
# 입력: metrics(ultralytics DetMetrics)
# 반환: 전체/클래스별 metric 딕셔너리
def extract_metrics(metrics):
    box = metrics.box

    overall = {
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
    }

    names = getattr(metrics, "names", {0: "real_target", 1: "ghost_target"})
    class_precision = np.asarray(getattr(box, "p", []), dtype=float)
    class_recall = np.asarray(getattr(box, "r", []), dtype=float)
    class_map50 = np.asarray(getattr(box, "ap50", []), dtype=float)
    class_map = np.asarray(getattr(box, "maps", []), dtype=float)

    class_metrics = {}
    for class_idx, class_name in names.items():
        idx = int(class_idx)
        class_metrics[str(class_name)] = {
            "precision": float(class_precision[idx]) if idx < len(class_precision) else None,
            "recall": float(class_recall[idx]) if idx < len(class_recall) else None,
            "map50": float(class_map50[idx]) if idx < len(class_map50) else None,
            "map50_95": float(class_map[idx]) if idx < len(class_map) else None,
        }

    speed = getattr(metrics, "speed", {}) or {}
    speed_ms = float(
        speed.get("preprocess", 0.0)
        + speed.get("inference", 0.0)
        + speed.get("postprocess", 0.0)
    )

    return overall, class_metrics, speed_ms


# 모델 객체에서 실제 파라미터 수를 읽어 표기값이 모델 head 변경에 흔들리지 않게 한다.
# 입력: model(YOLO)
# 반환: 백만 단위 파라미터 수
def count_params_m(model):
    try:
        params = sum(param.numel() for param in model.model.parameters())
    except AttributeError:
        return None
    return float(params / 1e6)


# 한 실험을 학습하고 best.pt 경로를 반환한다.
# 입력: exp_config, base_dir, batch, common_params
# 반환: 실험 결과 딕셔너리
def run_experiment(exp_config, base_dir, batch, common_params):
    base_dir = expand_path(base_dir)
    data_yaml = base_dir / exp_config["data"]
    start = time.time()

    print("=" * 80)
    print(f"[{exp_config['id']}] {exp_config['mode']} + {exp_config['model_name']} 학습 시작 (batch={batch})")
    print("=" * 80)

    model = YOLO(exp_config["model"])
    model.train(
        data=str(data_yaml),
        epochs=common_params["epochs"],
        imgsz=common_params["imgsz"],
        batch=batch,
        device=common_params["device"],
        workers=common_params["workers"],
        seed=common_params["seed"],
        deterministic=common_params["deterministic"],
        project=str(base_dir / "runs" / "detect"),
        name=f"ga_yolo_v4_1_{exp_config['name']}",
        exist_ok=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        translate=0.0,
        scale=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        erasing=0.0,
    )

    save_dir = Path(model.trainer.save_dir).resolve()
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best.pt가 생성되지 않았습니다: {best_path}")

    eval_metrics = evaluate_test(best_path, data_yaml, device=common_params["device"])
    elapsed = time.time() - start

    result = {
        "id": exp_config["id"],
        "name": exp_config["name"],
        "mode": exp_config["mode"],
        "model": exp_config["model"],
        "model_name": exp_config["model_name"],
        "data": str(data_yaml),
        "save_dir": str(save_dir),
        "best_path": str(best_path),
        "results_csv": str(save_dir / "results.csv"),
        "batch_used": int(batch),
        "train_time_sec": float(elapsed),
        "note": exp_config["note"],
        **eval_metrics,
    }
    return result


# OOM이 발생하면 더 작은 batch로 재시도해 긴 실험이 한 번에 멈추는 일을 줄인다.
# 입력: exp, base_dir, batch
# 반환: 성공한 실험 결과
def run_experiment_with_fallback(exp, base_dir, batch=16, common_params=None):
    if common_params is None:
        common_params = {}

    batch_candidates = []
    for candidate in (batch, 8, 4):
        if candidate not in batch_candidates:
            batch_candidates.append(candidate)

    for idx, current_batch in enumerate(batch_candidates):
        try:
            return run_experiment(exp, base_dir, batch=current_batch, common_params=common_params)
        except RuntimeError as error:
            error_text = str(error)
            if "CUDA out of memory" in error_text or "out of memory" in error_text.lower():
                next_batch = batch_candidates[idx + 1] if idx + 1 < len(batch_candidates) else "없음"
                print(f"⚠️  OOM: batch={current_batch} 실패, batch={next_batch}로 재시도")
                torch.cuda.empty_cache()
                continue
            raise

    raise RuntimeError("모든 batch 크기에서 OOM 발생")


# 학습 직후 상태 오염을 피하기 위해 best.pt를 새로 로드해서 test split만 평가한다.
# 입력: best_path, data_yaml, device
# 반환: test split 평가 결과 딕셔너리
def evaluate_test(best_path, data_yaml, device=0):
    eval_model = YOLO(str(best_path))
    metrics = eval_model.val(data=str(data_yaml), split="test", device=device, verbose=False, plots=False)
    overall, class_metrics, speed_ms = extract_metrics(metrics)
    params_m = count_params_m(eval_model)

    return {
        "map50": overall["map50"],
        "map50_95": overall["map50_95"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "class_metrics": class_metrics,
        "speed_ms": speed_ms,
        "params_m": params_m,
    }


# 완료된 실험 하나를 즉시 JSON에 반영해 중단 상황에서도 결과를 잃지 않게 한다.
# 입력: result, json_path
# 반환: 없음
def save_result_immediately(result, json_path):
    json_path = expand_path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"baseline": BASELINE_A0, "results": []}
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))

    results = [item for item in payload.get("results", []) if item.get("id") != result.get("id")]
    results.append(result)
    results.sort(key=lambda item: item["id"])
    payload["baseline"] = BASELINE_A0
    payload["results"] = results

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# 표에서 빈 실험도 명확하게 보이도록 숫자를 고정 폭 문자열로 바꾼다.
# 입력: value, digits
# 반환: 포맷 문자열
def fmt_float(value, digits=3):
    if value is None:
        return "?"
    return f"{float(value):.{digits}f}"


# 모델 크기를 논문 표에 쓰기 좋은 백만 단위 문자열로 바꾼다.
# 입력: value
# 반환: 포맷 문자열
def fmt_params(value):
    if value is None:
        return "?"
    return f"{float(value):.1f}M"


# 실험 결과를 한눈에 비교할 수 있는 표로 출력한다.
# 입력: results, baseline_a0
# 반환: 없음
def print_comparison_table(results, baseline_a0=BASELINE_A0):
    rows = [baseline_a0] + sorted(results, key=lambda item: item["id"])

    print("\n" + "=" * 105)
    print("🎯 GA-YOLO V4.1 Ablation Results (test split 기준)")
    print("=" * 105)
    print(
        "Exp | Mode      | Model    | mAP50    | mAP50-95 | P       | R       | "
        "Params  | Speed   | Batch | Note"
    )
    for item in rows:
        print(
            f"{item['id']:<3} | "
            f"{item['mode']:<9} | "
            f"{item['model_name']:<8} | "
            f"{fmt_float(item.get('map50')):<8} | "
            f"{fmt_float(item.get('map50_95')):<8} | "
            f"{fmt_float(item.get('precision')):<7} | "
            f"{fmt_float(item.get('recall')):<7} | "
            f"{fmt_params(item.get('params_m')):<7} | "
            f"{fmt_float(item.get('speed_ms'), 1)}ms  | "
            f"{str(item.get('batch_used', '?')):<5} | "
            f"{item.get('note', '')}"
        )
    print("=" * 105)
    print("* gray = viridis-derived grayscale (true dB grayscale은 V4.2에서 보완)")

    completed = sorted(results, key=lambda item: item["id"])
    if completed:
        best = max(completed, key=lambda item: item.get("map50_95", -1.0))
        fastest = min(completed, key=lambda item: item.get("speed_ms", float("inf")))
        lightweights = [item for item in completed if item["model_name"] == "YOLOv8n"]
        best_light = max(lightweights, key=lambda item: item.get("map50_95", -1.0)) if lightweights else None
        print(f"\n🏆 Best mAP50-95: Exp {best['id']} ({best['mode']}, {best['model_name']})")
        print(f"⚡ Best speed/accuracy: Exp {fastest['id']} ({fastest['mode']}, {fastest['model_name']})")
        if best_light:
            print(f"🪶 Best lightweight: Exp {best_light['id']} ({best_light['mode']}, {best_light['model_name']})")


# 클래스별 성능을 분리해 ghost 쪽 성능 저하가 있는지 바로 확인한다.
# 입력: results
# 반환: 없음
def print_class_breakdown(results):
    print("\n📊 클래스별 mAP50 (test split):")
    print("Exp | real_target | ghost_target")
    for item in sorted(results, key=lambda row: row["id"]):
        class_metrics = item.get("class_metrics", {})
        real_map = class_metrics.get("real_target", {}).get("map50")
        ghost_map = class_metrics.get("ghost_target", {}).get("map50")
        print(f"{item['id']:<3} | {fmt_float(real_map):>11} | {fmt_float(ghost_map):>12}")


# ablation의 핵심 질문 네 가지를 수치 차이로 요약한다.
# 입력: results
# 반환: 없음
def print_key_comparisons(results):
    by_id = {item["id"]: item for item in results}

    def delta(left_id, right_id):
        if left_id not in by_id or right_id not in by_id:
            return None
        return by_id[right_id]["map50_95"] - by_id[left_id]["map50_95"]

    comparisons = [
        ("viridis vs gray (n)", "A", "B"),
        ("viridis vs gray (s)", "C", "D"),
        ("n vs s (viridis)", "A", "C"),
        ("n vs s (gray)", "B", "D"),
    ]

    print("\n🔍 핵심 비교 (mAP50-95 기준):")
    for label, left_id, right_id in comparisons:
        value = delta(left_id, right_id)
        value_text = "?" if value is None else f"{value:+.3f}"
        print(f"  - {label:<22}: {left_id} vs {right_id} → Δ = {value_text}")


# results.csv의 epoch별 곡선을 겹쳐서 표현해 수렴 속도와 최종 성능을 함께 본다.
# 입력: results, baseline_a0, output_path
# 반환: 없음
def plot_ablation_curves(results, baseline_a0, output_path):
    output_path = expand_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("GA-YOLO V4.1 Ablation Comparison", fontsize=15)

    plot_specs = [
        (axes[0, 0], "val/box_loss", "Box Loss", None),
        (axes[0, 1], "val/cls_loss", "Cls Loss", None),
        (axes[1, 0], "metrics/mAP50(B)", "mAP50", baseline_a0["map50"]),
        (axes[1, 1], "metrics/mAP50-95(B)", "mAP50-95", baseline_a0["map50_95"]),
        (axes[2, 0], "metrics/precision(B)", "Precision", baseline_a0["precision"]),
        (axes[2, 1], "metrics/recall(B)", "Recall", baseline_a0["recall"]),
    ]

    for ax, column, title, baseline_value in plot_specs:
        for result in sorted(results, key=lambda item: item["id"]):
            csv_path = Path(result.get("results_csv", ""))
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            if "epoch" not in df.columns or column not in df.columns:
                continue
            df = df.drop_duplicates(subset="epoch", keep="last").sort_values("epoch")
            ax.plot(df["epoch"], df[column], label=f"{result['id']} {result['mode']} {result['model_name']}")

        if baseline_value is not None:
            ax.axhline(
                baseline_value,
                color="gray",
                linestyle="--",
                linewidth=1.2,
                label="A0 (V4.0 baseline, aug default)",
            )
        else:
            # A0의 loss curve는 남아 있지 않으므로 성능 baseline과 구분해 명시한다.
            ax.text(0.02, 0.92, "A0 loss curve unavailable", transform=ax.transAxes, fontsize=8, color="gray")

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# 초 단위 시간을 사람이 읽기 쉬운 문자열로 바꾼다.
# 입력: seconds
# 반환: 시간 문자열
def format_duration(seconds):
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remain = seconds % 60
    if hours:
        return f"{hours}시간 {minutes}분 {remain}초"
    if minutes:
        return f"{minutes}분 {remain}초"
    return f"{remain}초"


# 전체 실험 시간이 어디에 쓰였는지 출력해 다음 실험 비용을 예측한다.
# 입력: time_dict
# 반환: 없음
def print_time_summary(time_dict):
    print("\n" + "=" * 73)
    print("⏱️  V4.1 Ablation 소요 시간")
    print("=" * 73)
    for exp_id in ("A", "B", "C", "D"):
        if exp_id in time_dict.get("experiments", {}):
            label = time_dict["experiments"][exp_id]["label"]
            elapsed = time_dict["experiments"][exp_id]["elapsed"]
            print(f"  {exp_id} ({label}): {format_duration(elapsed)}")
    print("-" * 73)
    print(f"  📊 데이터셋 변환: {format_duration(time_dict.get('conversion_sec', 0.0))}")
    print(f"  🏃 학습 총 시간:  {format_duration(time_dict.get('training_sec', 0.0))}")
    print(f"  📈 평가 + 그래프: {format_duration(time_dict.get('postprocess_sec', 0.0))}")
    print("-" * 73)
    print(f"  ⏰ 전체 소요 시간: {format_duration(time_dict.get('total_sec', 0.0))}")
    print("=" * 73)


# 저장된 JSON이 있으면 중단 후 재실행할 때 이미 끝난 실험을 중복 실행하지 않도록 불러온다.
# 입력: json_path
# 반환: 완료 결과 리스트
def load_existing_results(json_path):
    json_path = expand_path(json_path)
    if not json_path.exists():
        return []
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return payload.get("results", [])


# CLI 인자를 해석해 실험 비용과 실행 경로를 명시적으로 통제한다.
# 입력: 없음
# 반환: argparse Namespace
def parse_args():
    parser = argparse.ArgumentParser(description="GA-YOLO V4.1 ablation runner")
    parser.add_argument("--base_dir", type=str, required=True, help="프로젝트 루트 경로")
    parser.add_argument("--batch", type=int, default=16, help="초기 batch 크기")
    parser.add_argument("--epochs", type=int, default=100, help="학습 epoch")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker 수")
    parser.add_argument("--seed", type=int, default=2026, help="재현성 확인용 seed")
    parser.add_argument("--device", type=str, default="0", help="Ultralytics device 값")
    return parser.parse_args()


def main():
    signal.signal(signal.SIGINT, handle_interrupt)
    args = parse_args()
    base_dir = expand_path(args.base_dir)
    src_dataset = base_dir / "dataset_v3_0b"
    gray_dataset = base_dir / "dataset_v4_1_gray"
    results_json = base_dir / "ablation_results.json"
    comparison_png = base_dir / "ablation_comparison.png"

    total_start = time.time()
    time_dict = {"experiments": {}}

    print(f"[V4.1] base_dir = {base_dir}")
    conversion_sec = convert_to_grayscale_dataset(src_dataset, gray_dataset)
    validate_dataset(src_dataset)
    validate_dataset(gray_dataset)
    save_grayscale_sample(src_dataset, gray_dataset, base_dir / "gray_conversion_sample.png", n_samples=3)
    time_dict["conversion_sec"] = conversion_sec

    common_params = {
        "epochs": int(args.epochs),
        "imgsz": 256,
        "device": args.device,
        "workers": int(args.workers),
        "seed": int(args.seed),
        "deterministic": True,
    }

    results = load_existing_results(results_json)
    completed_ids = {item["id"] for item in results}
    training_sec = sum(float(item.get("train_time_sec", 0.0)) for item in results)

    try:
        for exp in EXPERIMENTS:
            if STOP_REQUESTED:
                break
            if exp["id"] in completed_ids:
                print(f"[Skip] Exp {exp['id']}는 이미 완료되어 ablation_results.json에 있습니다.")
                continue

            exp_start = time.time()
            result = run_experiment_with_fallback(
                exp,
                base_dir,
                batch=int(args.batch),
                common_params=common_params,
            )
            save_result_immediately(result, results_json)
            results.append(result)
            completed_ids.add(exp["id"])

            elapsed = time.time() - exp_start
            training_sec += elapsed
            time_dict["experiments"][exp["id"]] = {
                "label": f"{exp['mode']} + {exp['model_name'].replace('YOLOv8', '')}",
                "elapsed": elapsed,
            }

            print_comparison_table(results, BASELINE_A0)
            print_class_breakdown(results)
            print_key_comparisons(results)

    except KeyboardInterrupt:
        print("\n[Interrupt] KeyboardInterrupt 감지. 완료된 실험까지만 저장합니다.")

    post_start = time.time()
    if results:
        print_comparison_table(results, BASELINE_A0)
        print_class_breakdown(results)
        print_key_comparisons(results)
        plot_ablation_curves(results, BASELINE_A0, comparison_png)
        print(f"\n[Saved] ablation comparison plot: {comparison_png}")
        print(f"[Saved] ablation results JSON: {results_json}")
    else:
        print("[Warning] 완료된 V4.1 실험 결과가 없습니다.")

    time_dict["training_sec"] = training_sec
    time_dict["postprocess_sec"] = time.time() - post_start
    time_dict["total_sec"] = time.time() - total_start
    print_time_summary(time_dict)


if __name__ == "__main__":
    main()
