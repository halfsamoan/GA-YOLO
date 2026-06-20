# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-05-26 (V3.2)
# Dependency: rd_yolo_sim_v2_2.py, carrada_noise_stats_v2_3.json
# Description: V3.2 - negative clutter scene 추가로 FP 제어 데이터셋 생성
#              V3.1 실패 원인(FP 폭증) 해결을 위한 설계
# ================================================================================

import csv
import json
import os
import random
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)

from rd_yolo_sim_v2_2 import compute_ghost_targets


NOISE_STATS_JSON = "/home/kmin/RD_YOLO_GHOST/carrada_noise_stats_v2_3.json"
OUTPUT_ROOT = "/home/kmin/RD_YOLO_GHOST/dataset_v3_2"
TOTAL_IMAGES = 2000
RANDOM_SEED = 42

c = 3e8
fc = 77e9
BW = 150e6
Tc = 50e-6
N_samples = 256
N_chirps = 128
S = BW / Tc
fs = N_samples / Tc
lam = c / fc

IMAGE_SIZE = 256
RANGE_BINS = N_samples
DOPPLER_BINS = N_chirps
BBOX_BIN_WIDTH = 5
BBOX_BIN_HEIGHT = 7
MAX_SCENE_ATTEMPTS = 100
CARRADA_DYNAMIC_RANGE_DB = 60.0

TARGET_RANGE_MIN = 10.0
TARGET_RANGE_MAX = 200.0
TARGET_VELOCITY_MIN = -15.0
TARGET_VELOCITY_MAX = 15.0

SCENE_TYPE_COUNTS = {
    "normal_target": 1400,
    "hard_target": 400,
    "negative": 200,
}

SPLIT_COUNTS = {
    "normal_target": {"train": 1120, "val": 140, "test": 140},
    "hard_target": {"train": 320, "val": 40, "test": 40},
    "negative": {"train": 160, "val": 20, "test": 20},
}

RIDGE_SCALE = {
    "clean": 0.0,
    "mild": 0.15,
    "medium": 0.30,
    "hard": 0.50,
}

BLOB_COUNT_RANGE = {
    "clean": (0, 0),
    "mild": (1, 2),
    "medium": (2, 3),
    "hard": (3, 5),
    "negative": (3, 6),
}

MANIFEST_COLUMNS = [
    "image_id",
    "split",
    "scene_type",
    "noise_level",
    "num_real",
    "num_ghost",
    "is_negative",
    "speckle_std",
    "ridge_energy",
    "blob_count",
]


# 필수 파일이 없으면 CARRADA 기반 noise 설정이 사라지므로 시작 단계에서 중단한다.
# 입력: path (파일 경로/string)
# 반환: 없음
def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"필수 파일을 찾을 수 없습니다: {path}")


# 기존 결과와 섞이면 negative 비율 검증이 깨지므로 비어 있는 출력 폴더만 허용한다.
# 입력: output_root (데이터셋 루트 경로/string)
# 반환: 없음
def prepare_output_dirs(output_root):
    if os.path.isdir(output_root) and os.listdir(output_root):
        raise FileExistsError(
            f"출력 폴더가 이미 비어 있지 않습니다: {output_root}\n"
            "기존 결과를 보존하려면 폴더명을 바꾸고, 재생성하려면 폴더를 직접 삭제하세요."
        )

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(output_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_root, "labels", split), exist_ok=True)
    os.makedirs(os.path.join(output_root, "samples"), exist_ok=True)


# V2.3에서 추출한 CARRADA 통계를 dB-domain noise 파라미터로 사용한다.
# 입력: json_path (통계 JSON 경로/string)
# 반환: stats (noise 통계/dict)
def load_noise_stats(json_path):
    require_file(json_path)
    with open(json_path, "r", encoding="utf-8") as file:
        stats = json.load(file)

    required_keys = ("recommended", "zero_doppler", "clutter_blob", "dynamic_range")
    for key in required_keys:
        if key not in stats:
            raise KeyError(f"noise 통계 JSON에 필수 key가 없습니다: {key}")
    for level in ("mild", "medium", "hard"):
        if level not in stats["recommended"]:
            raise KeyError(f"recommended 통계에 {level} 설정이 없습니다.")
    return stats


# scene 그룹별 80/10/10 비율을 고정해 FP 제어 샘플이 모든 split에 들어가게 한다.
# 입력: 없음
# 반환: split_plan (scene 생성 계획/list[dict])
def stratified_split():
    plan = []
    normal_noise_cycle = ["clean", "mild", "medium"]

    for split in ("train", "val", "test"):
        for idx in range(SPLIT_COUNTS["normal_target"][split]):
            plan.append(
                {
                    "scene_group": "normal_target",
                    "scene_type": "target",
                    "split": split,
                    "noise_level": normal_noise_cycle[idx % len(normal_noise_cycle)],
                    "is_negative": False,
                }
            )

        for _ in range(SPLIT_COUNTS["hard_target"][split]):
            plan.append(
                {
                    "scene_group": "hard_target",
                    "scene_type": "target",
                    "split": split,
                    "noise_level": "hard",
                    "is_negative": False,
                }
            )

        for idx in range(SPLIT_COUNTS["negative"][split]):
            plan.append(
                {
                    "scene_group": "negative",
                    "scene_type": "negative",
                    "split": split,
                    "noise_level": "medium" if idx % 2 == 0 else "hard",
                    "is_negative": True,
                }
            )

    if len(plan) != TOTAL_IMAGES:
        raise ValueError(f"split 계획 수가 TOTAL_IMAGES와 다릅니다: {len(plan)} != {TOTAL_IMAGES}")

    rng = random.Random(RANDOM_SEED)
    shuffled_plan = []
    for split in ("train", "val", "test"):
        split_items = [item for item in plan if item["split"] == split]
        rng.shuffle(split_items)
        shuffled_plan.extend(split_items)
    return shuffled_plan


# Doppler 부호 convention이 V1.1 이후의 접근 양수 convention과 맞는지 확인한다.
# 입력: 없음
# 반환: range_axis, velocity_axis (축 배열/np.ndarray)
def build_axes():
    range_axis = np.arange(RANGE_BINS) * c * fs / (2.0 * S * N_samples)
    doppler_freq_axis = np.fft.fftshift(np.fft.fftfreq(N_chirps, d=Tc))
    velocity_axis = -doppler_freq_axis * lam / 2.0
    return range_axis, velocity_axis


# V2.2와 같은 dechirped baseband 모델로 표적과 ghost의 beat matrix를 생성한다.
# 입력: targets_list (표적 목록/list), rng (난수 생성기), noise_power (복소 AWGN 전력/float)
# 반환: beat_matrix (N_chirps x N_samples 복소 배열)
def generate_beat_matrix(targets_list, rng, noise_power=0.001):
    t_fast = np.arange(N_samples) / fs
    beat_matrix = np.zeros((N_chirps, N_samples), dtype=np.complex128)

    if targets_list:
        reference_range = min(float(target["range"]) for target in targets_list)
        for target in targets_list:
            target_range = float(target["range"])
            target_velocity = float(target["velocity"])
            target_amp = float(target.get("amp", 1.0))

            for m in range(N_chirps):
                t_start_m = m * Tc
                range_start_m = target_range - target_velocity * t_start_m
                tau_start_m = 2.0 * range_start_m / c

                # chirp 내부 이동을 반영해야 fast-time peak가 고속 표적에서 물리적으로 어긋나지 않는다.
                range_m_t = target_range - target_velocity * (t_start_m + t_fast)
                tau_m_t = 2.0 * range_m_t / c
                f_beat_m_t = S * tau_m_t
                amp_eff_t = target_amp * (reference_range / np.maximum(range_m_t, 1e-6)) ** 2

                beat_matrix[m, :] += amp_eff_t * np.exp(
                    1j * 2.0 * np.pi * (f_beat_m_t * t_fast + fc * tau_start_m)
                )

    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal((N_chirps, N_samples))
        + 1j * rng.standard_normal((N_chirps, N_samples))
    )
    return beat_matrix + noise


# 가드레일성 정적 산란체를 넣어 V2.2의 ridge/clutter 배경을 유지한다.
# 입력: d_wall (가드레일 거리/m), rng (난수 생성기)
# 반환: clutter_matrix (N_chirps x N_samples 복소 배열)
def generate_guardrail_clutter(d_wall, rng):
    t_fast = np.arange(N_samples) / fs
    clutter_fast = np.zeros(N_samples, dtype=np.complex128)

    for k in range(30):
        range_k = d_wall + k * (70.0 / 30.0)
        amp_k = max(0.0, 0.3 + rng.standard_normal() * 0.05)
        tau_k = 2.0 * range_k / c
        f_beat_k = S * tau_k
        clutter_fast += amp_k * np.exp(1j * 2.0 * np.pi * (f_beat_k * t_fast + fc * tau_k))

    return np.tile(clutter_fast[np.newaxis, :], (N_chirps, 1))


# 200 m 표적까지 라벨링하기 위해 range FFT 전체 256 bin을 유지한다.
# 입력: beat_matrix (N_chirps x N_samples 복소 배열)
# 반환: rd_map, rd_magnitude, rd_map_db (Doppler x Range 배열)
def compute_rd_map_full_range(beat_matrix):
    range_window = np.hanning(N_samples)
    doppler_window = np.hanning(N_chirps)

    range_fft = np.fft.fft(beat_matrix * range_window[np.newaxis, :], axis=1)
    doppler_input = range_fft * doppler_window[:, np.newaxis]
    rd_map = np.fft.fftshift(np.fft.fft(doppler_input, axis=0), axes=0)

    rd_magnitude = np.abs(rd_map)
    rd_map_db = 20.0 * np.log10(rd_magnitude + 1e-12)
    return rd_map, rd_magnitude, rd_map_db


# 이론 위치 주변의 실제 peak로 라벨 중심을 보정한다.
# 입력: target (표적 dict), rd_magnitude, range_axis, velocity_axis
# 반환: snap (bin 위치 dict)
def snap_to_peak(target, rd_magnitude, range_axis, velocity_axis):
    r_bin_theory = int(np.argmin(np.abs(range_axis - float(target["range"]))))
    d_bin_theory = int(np.argmin(np.abs(velocity_axis - float(target["velocity"]))))

    r_lo = max(0, r_bin_theory - 2)
    r_hi = min(rd_magnitude.shape[1], r_bin_theory + 3)
    d_lo = max(0, d_bin_theory - 2)
    d_hi = min(rd_magnitude.shape[0], d_bin_theory + 3)

    local_window = rd_magnitude[d_lo:d_hi, r_lo:r_hi]
    local_peak = np.unravel_index(np.argmax(local_window), local_window.shape)
    return {
        "d_bin": int(d_lo + local_peak[0]),
        "r_bin": int(r_lo + local_peak[1]),
        "d_bin_theory": int(d_bin_theory),
        "r_bin_theory": int(r_bin_theory),
    }


# RD bin 좌표를 YOLO 정규화 bbox로 변환한다.
# 입력: snap (bin 위치 dict)
# 반환: yolo_bbox ([cx, cy, w, h]/list[float])
def target_to_yolo_bbox(snap):
    cx = (snap["d_bin"] + 0.5) / DOPPLER_BINS
    cy = (snap["r_bin"] + 0.5) / RANGE_BINS
    w = BBOX_BIN_WIDTH / DOPPLER_BINS
    h = BBOX_BIN_HEIGHT / RANGE_BINS
    return [float(cx), float(cy), float(w), float(h)]


# 서로 다른 target family가 너무 붙으면 YOLO 라벨 학습이 애매해지므로 해당 scene을 버린다.
# 입력: targets_meta (라벨 후보/list[dict]), min_sep (최소 bin 간격/int)
# 반환: 겹침 여부/bool
def has_bbox_overlap(targets_meta, min_sep=7):
    for i in range(len(targets_meta)):
        for j in range(i + 1, len(targets_meta)):
            if targets_meta[i]["family"] == targets_meta[j]["family"]:
                continue
            d_i, r_i = targets_meta[i]["bin_position"]
            d_j, r_j = targets_meta[j]["bin_position"]
            if abs(d_i - d_j) < min_sep and abs(r_i - r_j) < min_sep:
                return True
    return False


# V2.2 ghost 구조를 유지하는 target scene 설정을 샘플링한다.
# 입력: scene_idx (0-based/int), master_rng (난수 생성기)
# 반환: scene_config, rng_dict
def sample_target_scene_config(scene_idx, master_rng):
    scene_seed = int(master_rng.integers(0, 2**31))
    scenario_rng = np.random.default_rng(scene_seed)

    real_targets = []
    for _ in range(int(scenario_rng.integers(1, 5))):
        real_targets.append(
            {
                "range": float(scenario_rng.uniform(TARGET_RANGE_MIN, TARGET_RANGE_MAX)),
                "velocity": float(scenario_rng.uniform(TARGET_VELOCITY_MIN, TARGET_VELOCITY_MAX)),
                "amp": 1.0,
                "x_pos": float(scenario_rng.uniform(0.3, 2.5)),
                "type": "real",
            }
        )

    scene_config = {
        "scene_id": f"rd_{scene_idx + 1:06d}",
        "scene_idx": int(scene_idx),
        "scene_seed": int(scene_seed),
        "d_wall": float(scenario_rng.uniform(6.0, 10.0)),
        "wall_refl_coeff": 0.5,
        "real_targets": real_targets,
        "noise_power": 0.001,
    }
    rng_dict = {
        "beat_rng": np.random.default_rng(scene_seed + 1),
        "clutter_rng": np.random.default_rng(scene_seed + 2),
        "carrada_noise_rng": np.random.default_rng(scene_seed + 3),
    }
    return scene_config, rng_dict


# RD map 시뮬레이션과 real/ghost 라벨 메타데이터를 함께 생성한다.
# 입력: scene_idx, master_rng, range_axis, velocity_axis
# 반환: scene_config, rng_dict, rd_image_db, targets_meta
def generate_rd_map(scene_idx, master_rng, range_axis, velocity_axis):
    for _ in range(MAX_SCENE_ATTEMPTS):
        scene_config, rng_dict = sample_target_scene_config(scene_idx, master_rng)
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

        target_specs = []
        for idx, real_target in enumerate(scene_config["real_targets"]):
            target_specs.append({**real_target, "class_id": 0, "class": "real", "family": idx})
            target_specs.append({**ghosts_4b[idx], "class_id": 1, "class": "ghost", "family": idx})
            target_specs.append({**ghosts_3b[idx], "class_id": 1, "class": "ghost", "family": idx})

        if any(float(target["range"]) >= range_axis[-1] for target in target_specs):
            continue

        targets = scene_config["real_targets"] + ghosts_4b + ghosts_3b
        beat_matrix = generate_beat_matrix(
            targets,
            rng_dict["beat_rng"],
            noise_power=scene_config["noise_power"],
        )
        clutter_matrix = generate_guardrail_clutter(scene_config["d_wall"], rng_dict["clutter_rng"])
        _, rd_magnitude, rd_map_db = compute_rd_map_full_range(beat_matrix + clutter_matrix)

        targets_meta = []
        for target_idx, target in enumerate(target_specs):
            snap = snap_to_peak(target, rd_magnitude, range_axis, velocity_axis)
            yolo_bbox = target_to_yolo_bbox(snap)
            if not yolo_bbox_is_valid(yolo_bbox):
                break
            targets_meta.append(
                {
                    "id": f"T{target_idx}",
                    "class": target["class"],
                    "class_id": int(target["class_id"]),
                    "family": int(target["family"]),
                    "theory_range": float(target["range"]),
                    "theory_velocity": float(target["velocity"]),
                    "bin_position": [int(snap["d_bin"]), int(snap["r_bin"])],
                    "yolo_bbox": yolo_bbox,
                }
            )
        else:
            if has_bbox_overlap(targets_meta):
                continue
            scene_config["n_ghost"] = len(ghosts_4b) + len(ghosts_3b)
            return scene_config, rng_dict, rd_map_db.T.copy(), targets_meta

    raise RuntimeError(f"scene_idx={scene_idx}: target scene 생성에 실패했습니다.")


# target이 없는 배경을 만들어 밝은 clutter가 항상 target은 아니라는 학습 신호를 제공한다.
# 입력: scene_idx (0-based/int), master_rng (난수 생성기)
# 반환: scene_config, rng_dict, rd_image_db
def generate_negative_scene(scene_idx, master_rng):
    scene_seed = int(master_rng.integers(0, 2**31))
    scenario_rng = np.random.default_rng(scene_seed)
    beat_rng = np.random.default_rng(scene_seed + 1)
    clutter_rng = np.random.default_rng(scene_seed + 2)

    d_wall = float(scenario_rng.uniform(6.0, 10.0))
    clutter_matrix = generate_guardrail_clutter(d_wall, clutter_rng)
    background_noise = np.sqrt(0.001 / 2.0) * (
        beat_rng.standard_normal((N_chirps, N_samples))
        + 1j * beat_rng.standard_normal((N_chirps, N_samples))
    )
    _, _, rd_map_db = compute_rd_map_full_range(clutter_matrix + background_noise)

    scene_config = {
        "scene_id": f"rd_{scene_idx + 1:06d}",
        "scene_idx": int(scene_idx),
        "scene_seed": int(scene_seed),
        "d_wall": d_wall,
        "real_targets": [],
        "n_ghost": 0,
    }
    rng_dict = {
        "carrada_noise_rng": np.random.default_rng(scene_seed + 3),
    }
    return scene_config, rng_dict, rd_map_db.T.copy()


# noise 통계 JSON 스키마 차이를 흡수해 필요한 scalar 값을 안전하게 꺼낸다.
# 입력: stats (통계 dict), section/key/default
# 반환: value (float)
def get_stat_value(stats, section, key, default):
    value = stats.get(section, {}).get(key, default)
    if isinstance(value, dict):
        for candidate_key in ("mean", "value", "median"):
            if candidate_key in value:
                return float(value[candidate_key])
        return float(default)
    return float(value)


# dB domain에서 CARRADA-like speckle/ridge/blob을 추가한다.
# 입력: rd_image_db, noise_level, stats, rng, is_negative
# 반환: noisy_rd_image_db, noise_meta
def apply_noise(rd_image_db, noise_level, stats, rng, is_negative=False):
    noisy = rd_image_db.astype(np.float64).copy()
    noise_meta = {
        "speckle_std": 0.0,
        "ridge_energy": 0.0,
        "blob_count": 0,
    }

    if noise_level == "clean" and not is_negative:
        return noisy, noise_meta
    if noise_level not in ("clean", "mild", "medium", "hard"):
        raise ValueError(f"지원하지 않는 noise_level입니다: {noise_level}")

    level_for_stats = "medium" if noise_level == "clean" else noise_level
    base_speckle_std = get_stat_value(
        stats,
        "recommended",
        level_for_stats,
        2.0,
    )
    if isinstance(stats["recommended"].get(level_for_stats), dict):
        base_speckle_std = float(stats["recommended"][level_for_stats].get("speckle_std", base_speckle_std))

    speckle_std = base_speckle_std * float(rng.uniform(0.9, 1.1))
    noisy += rng.normal(0.0, speckle_std, size=noisy.shape)
    noise_meta["speckle_std"] = float(speckle_std)

    ridge_base = get_stat_value(stats, "zero_doppler", "mean_energy", 8.0)
    if is_negative:
        ridge_scale = float(rng.uniform(RIDGE_SCALE["medium"], RIDGE_SCALE["hard"]))
        ridge_half_width = 2
    else:
        ridge_scale = RIDGE_SCALE[level_for_stats]
        ridge_half_width = 2 if level_for_stats == "hard" else int(max(1, 1 + rng.integers(0, 2)))

    ridge_energy = ridge_base * ridge_scale
    center_col = noisy.shape[1] // 2
    col_start = max(0, center_col - ridge_half_width)
    col_end = min(noisy.shape[1], center_col + ridge_half_width + 1)
    range_profile = rng.uniform(0.85, 1.15, size=(noisy.shape[0], 1))
    noisy[:, col_start:col_end] += ridge_energy * range_profile
    noise_meta["ridge_energy"] = float(ridge_energy)

    if is_negative:
        min_blobs, max_blobs = BLOB_COUNT_RANGE["negative"]
    else:
        min_blobs, max_blobs = BLOB_COUNT_RANGE[level_for_stats]
    blob_count = int(rng.integers(min_blobs, max_blobs + 1))

    clutter_mean = get_stat_value(stats, "clutter_blob", "mean", 10.0)
    clutter_std = get_stat_value(stats, "clutter_blob", "std", 3.0)
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


# clipping 범위를 저장해 CARRADA 60 dB 기준과 비교 가능한 dynamic range를 계산한다.
# 입력: rd_image_db (dB RD map)
# 반환: low, high, dynamic_range
def compute_clip_bounds(rd_image_db):
    low = float(np.percentile(rd_image_db, 1.0))
    high = float(np.percentile(rd_image_db, 99.5))
    return low, high, float(high - low)


# dB map을 clipping 후 min-max normalize하여 256x256 grayscale PNG 배열로 바꾼다.
# 입력: rd_image_db (Range x Doppler dB map)
# 반환: image (256x256 uint8)
def rd_db_to_uint8_image(rd_image_db):
    low, high, _ = compute_clip_bounds(rd_image_db)
    clipped = np.clip(rd_image_db, low, high)
    normalized = (clipped - low) / (high - low + 1e-12)
    image = (normalized * 255.0).astype(np.uint8)
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)


# YOLO bbox가 학습기가 읽을 수 있는 정규화 범위인지 확인한다.
# 입력: yolo_bbox ([cx, cy, w, h])
# 반환: valid (bool)
def yolo_bbox_is_valid(yolo_bbox):
    cx, cy, w, h = yolo_bbox
    return 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0


# target scene은 YOLO 라벨을 쓰고 negative scene은 빈 txt를 반드시 만든다.
# 입력: targets_meta (라벨 메타/list), label_path (출력 경로/string)
# 반환: 없음
def save_label(targets_meta, label_path):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    if not targets_meta:
        open(label_path, "w", encoding="utf-8").close()
        return

    lines = []
    for target in targets_meta:
        cx, cy, w, h = target["yolo_bbox"]
        lines.append(f"{target['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(label_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


# sample 이미지에 GT bbox를 그려 라벨 좌표계가 이미지 좌표계와 맞는지 눈으로 확인한다.
# 입력: image (uint8 image), targets_meta (라벨 메타), output_path
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


# 이미지와 라벨을 저장하고 manifest에 필요한 최소 정보를 만든다.
# 입력: scene_id, split, scene_type, noise_level, image, targets_meta, noise_meta, is_negative
# 반환: row (manifest 행/dict)
def save_scene(scene_id, split, scene_type, noise_level, image, targets_meta, noise_meta, is_negative):
    image_path = os.path.join(OUTPUT_ROOT, "images", split, f"{scene_id}.png")
    label_path = os.path.join(OUTPUT_ROOT, "labels", split, f"{scene_id}.txt")
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    cv2.imwrite(image_path, image)
    save_label(targets_meta, label_path)

    return {
        "image_id": scene_id,
        "split": split,
        "scene_type": scene_type,
        "noise_level": noise_level,
        "num_real": int(sum(1 for target in targets_meta if target["class_id"] == 0)),
        "num_ghost": int(sum(1 for target in targets_meta if target["class_id"] == 1)),
        "is_negative": bool(is_negative),
        "speckle_std": float(noise_meta["speckle_std"]),
        "ridge_energy": float(noise_meta["ridge_energy"]),
        "blob_count": int(noise_meta["blob_count"]),
    }


# manifest를 고정 컬럼으로 저장해 후속 평가 스크립트의 파싱 흔들림을 막는다.
# 입력: manifest_rows (행 목록/list[dict])
# 반환: 없음
def write_manifest(manifest_rows):
    output_path = os.path.join(OUTPUT_ROOT, "manifest.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({key: row[key] for key in MANIFEST_COLUMNS})


# negative 라벨 파일이 모두 빈 파일인지 확인해 FP 제어 데이터의 조건을 보장한다.
# 입력: manifest_rows (행 목록/list[dict])
# 반환: status, empty_count
def validate_negative_labels(manifest_rows):
    negative_rows = [row for row in manifest_rows if row["is_negative"]]
    empty_count = 0
    for row in negative_rows:
        label_path = os.path.join(OUTPUT_ROOT, "labels", row["split"], f"{row['image_id']}.txt")
        if os.path.isfile(label_path) and os.path.getsize(label_path) == 0:
            empty_count += 1

    status = "PASS" if empty_count == len(negative_rows) == SCENE_TYPE_COUNTS["negative"] else "FAIL"
    return status, empty_count


# 생성 결과가 명세의 scene 수와 split 수를 만족하는지 콘솔에 요약한다.
# 입력: manifest_rows (행 목록), dynamic_ranges (scene type별 dB 범위)
# 반환: 없음
def print_summary(manifest_rows, dynamic_ranges):
    group_counts = {"normal_target": 0, "hard_target": 0, "negative": 0}
    split_counts = {"train": 0, "val": 0, "test": 0}

    for row in manifest_rows:
        if row["is_negative"]:
            group_counts["negative"] += 1
        elif row["noise_level"] == "hard":
            group_counts["hard_target"] += 1
        else:
            group_counts["normal_target"] += 1
        split_counts[row["split"]] += 1

    image_count = 0
    label_count = 0
    for split in ("train", "val", "test"):
        image_dir = os.path.join(OUTPUT_ROOT, "images", split)
        label_dir = os.path.join(OUTPUT_ROOT, "labels", split)
        image_count += len([name for name in os.listdir(image_dir) if name.endswith(".png")])
        label_count += len([name for name in os.listdir(label_dir) if name.endswith(".txt")])

    negative_status, empty_negative_count = validate_negative_labels(manifest_rows)

    print("=" * 80)
    print("V3.2 negative clutter dataset 생성 요약")
    print("=" * 80)
    print("[scene_type별 생성 수]")
    for group_name in ("normal_target", "hard_target", "negative"):
        print(f"  {group_name:13s}: {group_counts[group_name]}")

    print("")
    print("[stratified split 결과]")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: {split_counts[split]}")

    print("")
    print(f"이미지 파일 수: {image_count}")
    print(f"라벨 파일 수  : {label_count}")
    print(f"라벨 파일 수 이미지 파일 수 일치 여부: {image_count == label_count}")

    print("")
    print(f"negative 빈 라벨 수: {empty_negative_count}/{SCENE_TYPE_COUNTS['negative']}")
    print(f"negative label 검증 {negative_status}")

    print("")
    print("[scene_type별 dynamic range sanity check]")
    for scene_type in ("target", "negative"):
        values = np.asarray(dynamic_ranges[scene_type], dtype=np.float64)
        mean_dynamic_range = float(np.mean(values)) if values.size else 0.0
        print(
            f"  {scene_type:8s}: mean={mean_dynamic_range:.3f} dB "
            f"(CARRADA 기준 {CARRADA_DYNAMIC_RANGE_DB:.1f} dB 대비 {mean_dynamic_range - CARRADA_DYNAMIC_RANGE_DB:+.3f} dB)"
        )

    print("")
    print("V3.2 데이터셋 생성 완료")
    print("=" * 80)


# 전체 V3.2 데이터셋을 생성한다.
# 입력: 없음
# 반환: 없음
def main():
    if sum(SCENE_TYPE_COUNTS.values()) != TOTAL_IMAGES:
        raise ValueError("SCENE_TYPE_COUNTS 합계가 TOTAL_IMAGES와 다릅니다.")

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    noise_stats = load_noise_stats(NOISE_STATS_JSON)
    prepare_output_dirs(OUTPUT_ROOT)
    split_plan = stratified_split()
    range_axis, velocity_axis = build_axes()
    master_rng = np.random.default_rng(RANDOM_SEED)

    manifest_rows = []
    dynamic_ranges = {"target": [], "negative": []}
    sample_saved = {"normal": False, "hard": False, "negative": False}

    for scene_idx, plan_item in enumerate(split_plan):
        scene_id = f"rd_{scene_idx + 1:06d}"
        split = plan_item["split"]
        scene_type = plan_item["scene_type"]
        noise_level = plan_item["noise_level"]
        is_negative = bool(plan_item["is_negative"])

        if is_negative:
            _, rng_dict, rd_image_db = generate_negative_scene(scene_idx, master_rng)
            targets_meta = []
        else:
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
            is_negative=is_negative,
        )
        _, _, dynamic_range = compute_clip_bounds(rd_noisy_db)
        dynamic_ranges[scene_type].append(dynamic_range)

        image = rd_db_to_uint8_image(rd_noisy_db)
        row = save_scene(
            scene_id,
            split,
            scene_type,
            noise_level,
            image,
            targets_meta,
            noise_meta,
            is_negative,
        )
        manifest_rows.append(row)

        if scene_type == "target" and noise_level != "hard" and not sample_saved["normal"]:
            visualize_sample(image, targets_meta, os.path.join(OUTPUT_ROOT, "samples", "sample_normal.png"))
            sample_saved["normal"] = True
        elif scene_type == "target" and noise_level == "hard" and not sample_saved["hard"]:
            visualize_sample(image, targets_meta, os.path.join(OUTPUT_ROOT, "samples", "sample_hard.png"))
            sample_saved["hard"] = True
        elif scene_type == "negative" and not sample_saved["negative"]:
            visualize_sample(image, targets_meta, os.path.join(OUTPUT_ROOT, "samples", "sample_negative.png"))
            sample_saved["negative"] = True

        if (scene_idx + 1) % 100 == 0:
            print(f"[Progress] {scene_idx + 1}/{TOTAL_IMAGES} 생성 완료")

    write_manifest(manifest_rows)
    print_summary(manifest_rows, dynamic_ranges)


if __name__ == "__main__":
    main()
