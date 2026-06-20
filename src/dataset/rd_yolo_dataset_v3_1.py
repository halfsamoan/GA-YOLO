# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-05-26 (V3.1)
# Dependency: rd_yolo_sim_v2_2.py, carrada_noise_stats_v2_3.json
# Description: CARRADA 통계 기반 domain randomization 적용 혼합 데이터셋 생성
# ================================================================================

import csv
import json
import os
import random

import cv2
import numpy as np

from rd_yolo_dataset_v3_0a import (
    BW,
    N_chirps,
    N_samples,
    S,
    Tc,
    bbox_bin_height,
    bbox_bin_width,
    c,
    compute_ghost_targets,
    compute_rd_map,
    fc,
    fs,
    generate_beat_matrix,
    lam,
    range_bins,
    snap_to_peak,
    target_to_yolo_bbox,
)
from rd_yolo_dataset_v3_0b import (
    build_clutter_matrix,
    build_target_specs,
    check_overlap_snap,
    check_overlap_theory,
    yolo_bbox_is_valid,
)


NOISE_STATS_JSON = "/home/kmin/RD_YOLO_GHOST/carrada_noise_stats_v2_3.json"
OUTPUT_ROOT = "/home/kmin/RD_YOLO_GHOST/dataset_v3_1"
TOTAL_IMAGES = 2000
RANDOM_SEED = 42

IMAGE_SIZE = 256
MAX_SCENE_ATTEMPTS = 100
CARRADA_DYNAMIC_RANGE_DB = 60.15
TARGET_RANGE_MIN = 10.0
TARGET_RANGE_MAX = 110.0

LEVEL_COUNTS = {
    "clean": 600,
    "mild": 600,
    "medium": 500,
    "hard": 300,
}

SPLIT_COUNTS = {
    "clean": {"train": 480, "val": 60, "test": 60},
    "mild": {"train": 480, "val": 60, "test": 60},
    "medium": {"train": 400, "val": 50, "test": 50},
    "hard": {"train": 240, "val": 30, "test": 30},
}

RIDGE_SCALE = {
    "clean": 0.0,
    "mild": 0.3,
    "medium": 0.6,
    "hard": 1.0,
}

BLOB_COUNT_RANGE = {
    "clean": (0, 0),
    "mild": (1, 2),
    "medium": (2, 3),
    "hard": (3, 5),
}


# 입력 파일이 없으면 V3.1 노이즈 보정 기준이 사라지므로 초기에 중단한다.
# 입력: path (파일 경로)
# 반환: 없음
def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")


# 기존 결과와 섞이면 split/manifest가 오염되므로 비어 있지 않은 출력 폴더는 막는다.
# 입력: output_root (데이터셋 루트)
# 반환: 없음
def prepare_output_dirs(output_root):
    if os.path.isdir(output_root) and os.listdir(output_root):
        raise FileExistsError(
            f"출력 폴더가 이미 비어있지 않습니다: {output_root}\n"
            "기존 결과를 보존하려면 폴더명을 바꾸고, 재생성하려면 폴더를 직접 삭제하세요."
        )

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(output_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_root, "labels", split), exist_ok=True)
    os.makedirs(os.path.join(output_root, "samples"), exist_ok=True)


# CARRADA 분석 JSON을 읽어 noisy level별 dB domain 노이즈 파라미터로 사용한다.
# 입력: json_path (V2.3 통계 JSON 경로)
# 반환: 통계 dict
def load_noise_stats(json_path):
    require_file(json_path)
    with open(json_path, "r", encoding="utf-8") as file:
        stats = json.load(file)

    required_keys = ["recommended", "zero_doppler", "clutter_blob", "dynamic_range"]
    for key in required_keys:
        if key not in stats:
            raise KeyError(f"노이즈 통계 JSON에 필요한 키가 없습니다: {key}")
    return stats


# 고정 비율이 실제 파일 분할에도 유지되도록 noise level별 split 계획을 만든다.
# 입력: 없음
# 반환: split/noise_level 항목 리스트
def stratified_split():
    plan = []
    for noise_level in ("clean", "mild", "medium", "hard"):
        for split in ("train", "val", "test"):
            for _ in range(SPLIT_COUNTS[noise_level][split]):
                plan.append({"noise_level": noise_level, "split": split})

    if len(plan) != TOTAL_IMAGES:
        raise ValueError(f"split 계획 수가 TOTAL_IMAGES와 다릅니다: {len(plan)} != {TOTAL_IMAGES}")
    return plan


# V2.2 물리 모델을 유지하되 V3.1 범위 조건에 맞는 랜덤 scene을 만든다.
# 입력: scene_idx (0-based), master_rng (numpy RNG)
# 반환: scene_config, rng_dict
def sample_scene_config(scene_idx, master_rng):
    scene_seed = int(master_rng.integers(0, 2**31))
    scenario_rng = np.random.default_rng(scene_seed)

    n_real = int(scenario_rng.integers(1, 5))
    d_wall = float(scenario_rng.uniform(6.0, 10.0))
    real_targets = []

    for _ in range(n_real):
        real_targets.append(
            {
                "range": float(scenario_rng.uniform(TARGET_RANGE_MIN, TARGET_RANGE_MAX)),
                "velocity": float(scenario_rng.uniform(-15.0, 15.0)),
                "amp": 1.0,
                "x_pos": float(scenario_rng.uniform(0.3, 2.5)),
                "type": "real",
            }
        )

    clutter_config = {
        "type": "structured",
        "N_clutter": 30,
        "R_start": d_wall,
        "R_span": 60.0,
        "amp": 0.3,
        "amp_std": 0.05,
    }

    scene_config = {
        "scene_id": f"rd_{scene_idx + 1:06d}",
        "scene_idx": int(scene_idx),
        "scene_seed": int(scene_seed),
        "guardrail_present": True,
        "d_wall": d_wall,
        "wall_refl_coeff": 0.5,
        "real_targets": real_targets,
        "noise_power": 0.001,
        "clutter_config": clutter_config,
    }
    rng_dict = {
        "clutter_rng": np.random.default_rng(scene_seed + 1),
        "noise_rng": np.random.default_rng(scene_seed + 2),
        "carrada_noise_rng": np.random.default_rng(scene_seed + 3),
    }
    return scene_config, rng_dict


# CARRADA-like noise를 적용하기 전의 V2.2 기반 RD map과 GT 메타데이터를 만든다.
# 입력: scene_idx, master_rng, range_axis, velocity_axis
# 반환: scene_config, rng_dict, rd_image_db, targets_meta
def generate_rd_map(scene_idx, master_rng, range_axis, velocity_axis):
    for _ in range(MAX_SCENE_ATTEMPTS):
        scene_config, rng_dict = sample_scene_config(scene_idx, master_rng)
        ghosts_4b = compute_ghost_targets(
            scene_config["real_targets"],
            scene_config["d_wall"],
            scene_config["wall_refl_coeff"],
            "4bounce",
        )
        ghosts_3b = compute_ghost_targets(
            scene_config["real_targets"],
            scene_config["d_wall"],
            scene_config["wall_refl_coeff"],
            "3bounce",
        )
        target_specs = build_target_specs(scene_config["real_targets"], ghosts_4b, ghosts_3b)

        if check_overlap_theory(target_specs, range_axis, velocity_axis):
            continue

        targets = scene_config["real_targets"] + ghosts_4b + ghosts_3b
        beat_matrix = generate_beat_matrix(
            targets,
            c,
            fc,
            S,
            Tc,
            fs,
            N_chirps,
            N_samples,
            noise_power=scene_config["noise_power"],
            rng=rng_dict["noise_rng"],
        )
        clutter_matrix = build_clutter_matrix(scene_config["clutter_config"], rng_dict["clutter_rng"])
        total_matrix = beat_matrix + clutter_matrix

        _, rd_magnitude, rd_map_db, range_axis_local, velocity_axis_local, _ = compute_rd_map(
            total_matrix,
            c,
            S,
            fs,
            lam,
            Tc,
            N_chirps,
            N_samples,
        )

        targets_meta = []
        for target in target_specs:
            snap = snap_to_peak(
                float(target["range"]),
                float(target["velocity"]),
                rd_magnitude,
                range_axis_local,
                velocity_axis_local,
            )
            yolo_bbox = target_to_yolo_bbox(
                snap,
                N_chirps,
                range_bins,
                bbox_bin_width,
                bbox_bin_height,
            )
            if not yolo_bbox_is_valid(yolo_bbox):
                break

            target_meta = {
                "id": target["id"],
                "class": target["class"],
                "class_id": int(target["class_id"]),
                "theory_range": float(target["range"]),
                "theory_velocity": float(target["velocity"]),
                "bin_position": [int(snap["d_bin"]), int(snap["r_bin"])],
                "yolo_bbox": yolo_bbox,
                "family": int(target.get("family", -1)),
            }
            if target["class"] == "ghost":
                target_meta["bounce_type"] = target["bounce_type"]
                target_meta["parent"] = target["parent"]
            targets_meta.append(target_meta)
        else:
            if check_overlap_snap(targets_meta):
                continue
            scene_config["n_ghost"] = len(ghosts_4b) + len(ghosts_3b)
            # 이미지 좌표계는 y=range, x=Doppler이므로 dB map을 transpose한 뒤 노이즈를 넣는다.
            return scene_config, rng_dict, rd_map_db.T.copy(), targets_meta

    raise RuntimeError(f"scene_idx={scene_idx}: overlap 조건 때문에 scene 생성에 실패했습니다.")


# dB domain에서 CARRADA-like speckle/ridge/blob를 넣어 실제 RD domain gap을 흉내 낸다.
# 입력: rd_image_db, noise_level, stats, rng
# 반환: noisy_rd_image_db, noise_meta
def apply_noise(rd_image_db, noise_level, stats, rng):
    noisy = rd_image_db.astype(np.float64).copy()
    noise_meta = {
        "speckle_std": 0.0,
        "ridge_energy": 0.0,
        "blob_count": 0,
    }

    if noise_level == "clean":
        return noisy, noise_meta

    if noise_level not in ("mild", "medium", "hard"):
        raise ValueError(f"지원하지 않는 noise level입니다: {noise_level}")

    base_speckle_std = float(stats["recommended"][noise_level]["speckle_std"])
    speckle_std = base_speckle_std * float(rng.uniform(0.9, 1.1))
    noisy += rng.normal(0.0, speckle_std, size=noisy.shape)
    noise_meta["speckle_std"] = float(speckle_std)

    ridge_scale = RIDGE_SCALE[noise_level]
    ridge_energy = float(stats["zero_doppler"]["mean_energy"]) * ridge_scale
    ridge_half_width = int(max(1, 2 + rng.integers(-1, 2)))
    center_col = noisy.shape[1] // 2
    col_start = max(0, center_col - ridge_half_width)
    col_end = min(noisy.shape[1], center_col + ridge_half_width + 1)
    range_profile = rng.uniform(0.85, 1.15, size=(noisy.shape[0], 1))
    noisy[:, col_start:col_end] += ridge_energy * range_profile
    noise_meta["ridge_energy"] = float(ridge_energy)

    min_blobs, max_blobs = BLOB_COUNT_RANGE[noise_level]
    blob_count = int(rng.integers(min_blobs, max_blobs + 1))
    clutter_mean = float(stats["clutter_blob"]["mean"])
    clutter_std = float(stats["clutter_blob"]["std"])
    for _ in range(blob_count):
        blob_size = int(rng.integers(3, 8))
        row = int(rng.integers(0, noisy.shape[0]))
        col = int(rng.integers(0, noisy.shape[1]))
        energy = max(0.0, float(rng.normal(clutter_mean, clutter_std)))
        half = blob_size // 2
        r0 = max(0, row - half)
        r1 = min(noisy.shape[0], row + half + 1)
        c0 = max(0, col - half)
        c1 = min(noisy.shape[1], col + half + 1)
        blob = np.full((r1 - r0, c1 - c0), energy, dtype=np.float64)
        blob = cv2.GaussianBlur(blob, (0, 0), sigmaX=max(0.8, blob_size / 3.0))
        noisy[r0:r1, c0:c1] += blob
    noise_meta["blob_count"] = int(blob_count)

    return noisy, noise_meta


# dB map을 robust clipping 후 256x256 grayscale PNG로 저장할 uint8 이미지로 바꾼다.
# 입력: rd_image_db
# 반환: uint8 grayscale 이미지
def rd_db_to_uint8_image(rd_image_db):
    low = float(np.percentile(rd_image_db, 1.0))
    high = float(np.percentile(rd_image_db, 99.5))
    clipped = np.clip(rd_image_db, low, high)
    normalized = (clipped - low) / (high - low + 1e-12)
    image_128 = (normalized * 255.0).astype(np.uint8)
    return cv2.resize(image_128, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)


# YOLO 형식 라벨을 숫자 라인만 저장해 학습기가 바로 읽을 수 있게 한다.
# 입력: targets_meta, label_path
# 반환: 없음
def save_label(targets_meta, label_path):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    lines = []
    for target in targets_meta:
        cx, cy, w, h = target["yolo_bbox"]
        lines.append(f"{target['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(label_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


# GT bbox가 이미지와 같은 좌표계인지 눈으로 검증하기 위해 레벨별 샘플을 저장한다.
# 입력: image, targets_meta, output_path
# 반환: 없음
def visualize_sample(image, targets_meta, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for target in targets_meta:
        cx, cy, w, h = target["yolo_bbox"]
        x_center = cx * IMAGE_SIZE
        y_center = cy * IMAGE_SIZE
        box_w = w * IMAGE_SIZE
        box_h = h * IMAGE_SIZE
        x1 = int(round(x_center - box_w / 2.0))
        y1 = int(round(y_center - box_h / 2.0))
        x2 = int(round(x_center + box_w / 2.0))
        y2 = int(round(y_center + box_h / 2.0))
        color = (0, 0, 255) if target["class_id"] == 0 else (0, 165, 255)
        label = "real" if target["class_id"] == 0 else "ghost"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            overlay,
            label,
            (max(0, x1), max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(output_path, overlay)


# 이미지와 라벨을 저장하고 manifest에 들어갈 행을 만든다.
# 입력: scene_id, split, noise_level, image, targets_meta, noise_meta, dynamic_range
# 반환: manifest row
def save_scene(scene_id, split, noise_level, image, targets_meta, noise_meta, dynamic_range):
    image_rel = os.path.join("images", split, f"{scene_id}.png")
    label_rel = os.path.join("labels", split, f"{scene_id}.txt")
    image_path = os.path.join(OUTPUT_ROOT, image_rel)
    label_path = os.path.join(OUTPUT_ROOT, label_rel)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    cv2.imwrite(image_path, image)
    save_label(targets_meta, label_path)

    num_real = sum(1 for target in targets_meta if target["class_id"] == 0)
    num_ghost = sum(1 for target in targets_meta if target["class_id"] == 1)
    return {
        "image_id": scene_id,
        "split": split,
        "noise_level": noise_level,
        "num_real": num_real,
        "num_ghost": num_ghost,
        "speckle_std": noise_meta["speckle_std"],
        "ridge_energy": noise_meta["ridge_energy"],
        "blob_count": noise_meta["blob_count"],
        "dynamic_range": float(dynamic_range),
        "image": image_rel.replace("\\", "/"),
        "label": label_rel.replace("\\", "/"),
    }


# manifest.csv를 저장해 이후 clean/noisy ablation을 같은 split 기준으로 추적한다.
# 입력: manifest_rows
# 반환: 없음
def write_manifest(manifest_rows):
    output_path = os.path.join(OUTPUT_ROOT, "manifest.csv")
    columns = [
        "image_id",
        "split",
        "noise_level",
        "num_real",
        "num_ghost",
        "speckle_std",
        "ridge_energy",
        "blob_count",
        "dynamic_range",
        "image",
        "label",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({key: row.get(key, "") for key in columns})


# YOLOv8 학습이 바로 가능하도록 data.yaml을 함께 저장한다.
# 입력: 없음
# 반환: 없음
def write_data_yaml():
    text = (
        "# YOLOv8 dataset configuration\n"
        f"path: {OUTPUT_ROOT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: real_target\n"
        "  1: ghost_target\n\n"
        "nc: 2\n"
    )
    with open(os.path.join(OUTPUT_ROOT, "data.yaml"), "w", encoding="utf-8") as file:
        file.write(text)


# 생성 결과가 명세와 맞는지 콘솔에 요약해 바로 확인할 수 있게 한다.
# 입력: manifest_rows
# 반환: 없음
def print_summary(manifest_rows):
    level_counts = {level: 0 for level in LEVEL_COUNTS}
    split_counts = {"train": 0, "val": 0, "test": 0}
    dynamic_ranges = {level: [] for level in LEVEL_COUNTS}

    for row in manifest_rows:
        level_counts[row["noise_level"]] += 1
        split_counts[row["split"]] += 1
        dynamic_ranges[row["noise_level"]].append(float(row["dynamic_range"]))

    label_count = 0
    image_count = 0
    empty_label_count = 0
    for split in ("train", "val", "test"):
        image_dir = os.path.join(OUTPUT_ROOT, "images", split)
        label_dir = os.path.join(OUTPUT_ROOT, "labels", split)
        image_count += len([name for name in os.listdir(image_dir) if name.endswith(".png")])
        for name in os.listdir(label_dir):
            if name.endswith(".txt"):
                label_count += 1
                if os.path.getsize(os.path.join(label_dir, name)) == 0:
                    empty_label_count += 1

    print("=" * 80)
    print("V3.1 CARRADA-like domain randomization 데이터셋 생성 요약")
    print("=" * 80)
    print("[레벨별 생성 수]")
    for level in ("clean", "mild", "medium", "hard"):
        print(f"  {level:6s}: {level_counts[level]}")
    print("")
    print("[stratified split 결과]")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: {split_counts[split]}")
    print("")
    print(f"이미지 파일 수: {image_count}")
    print(f"라벨 파일 수  : {label_count}")
    print(f"라벨 파일 수 일치 여부: {image_count == label_count}")
    print("")
    print("[noise level별 dynamic range sanity check]")
    for level in ("clean", "mild", "medium", "hard"):
        arr = np.asarray(dynamic_ranges[level], dtype=np.float64)
        print(
            f"  {level:6s}: mean={float(np.mean(arr)):.3f} dB "
            f"(CARRADA 기준 {CARRADA_DYNAMIC_RANGE_DB:.2f} dB와 차이 {float(np.mean(arr) - CARRADA_DYNAMIC_RANGE_DB):+.3f} dB)"
        )
    print("")
    print(f"라벨 없는 이미지 수: {empty_label_count}")
    print("V3.1 데이터셋 생성 완료")
    print("=" * 80)


# 전체 데이터셋을 생성한다.
# 입력: 없음
# 반환: 없음
def main():
    if sum(LEVEL_COUNTS.values()) != TOTAL_IMAGES:
        raise ValueError("LEVEL_COUNTS 합계가 TOTAL_IMAGES와 다릅니다.")

    noise_stats = load_noise_stats(NOISE_STATS_JSON)
    prepare_output_dirs(OUTPUT_ROOT)

    split_plan = stratified_split()
    master_rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    range_axis = np.arange(N_samples // 2) * c * fs / (2.0 * S * N_samples)
    doppler_freq_axis = np.fft.fftshift(np.fft.fftfreq(N_chirps, d=Tc))
    velocity_axis = -doppler_freq_axis * lam / 2.0

    manifest_rows = []
    sample_saved = {level: False for level in LEVEL_COUNTS}

    for scene_idx, item in enumerate(split_plan):
        noise_level = item["noise_level"]
        split = item["split"]
        scene_id = f"rd_{scene_idx + 1:06d}"

        scene_config, rng_dict, rd_image_db, targets_meta = generate_rd_map(
            scene_idx,
            master_rng,
            range_axis,
            velocity_axis,
        )
        rd_noisy_db, noise_meta = apply_noise(
            rd_image_db,
            noise_level,
            noise_stats,
            rng_dict["carrada_noise_rng"],
        )
        dynamic_range = float(np.max(rd_noisy_db) - np.min(rd_noisy_db))
        image = rd_db_to_uint8_image(rd_noisy_db)
        row = save_scene(scene_id, split, noise_level, image, targets_meta, noise_meta, dynamic_range)
        row["num_real"] = len(scene_config["real_targets"])
        row["num_ghost"] = int(scene_config["n_ghost"])
        manifest_rows.append(row)

        if not sample_saved[noise_level]:
            sample_path = os.path.join(OUTPUT_ROOT, "samples", f"sample_{noise_level}.png")
            visualize_sample(image, targets_meta, sample_path)
            sample_saved[noise_level] = True

        if (scene_idx + 1) % 100 == 0:
            print(f"[Progress] {scene_idx + 1}/{TOTAL_IMAGES} 생성 완료")

    write_manifest(manifest_rows)
    write_data_yaml()
    print_summary(manifest_rows)


if __name__ == "__main__":
    main()
