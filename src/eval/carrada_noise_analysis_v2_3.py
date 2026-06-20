# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-05-26 (V2.3)
# Dependency: Carrada 폴더
# Description: CARRADA RD map에서 노이즈 통계 추출 → V3.1 시뮬레이터 보정용
# ================================================================================

import json
import os
import random

import numpy as np


CARRADA_ROOT = "/home/kmin/RD_YOLO_GHOST/carrada/Carrada"
OUTPUT_JSON = "/home/kmin/RD_YOLO_GHOST/carrada_noise_stats_v2_3.json"

RANDOM_SEED = 42
FRAMES_PER_SEQUENCE = 5
NOISE_PERCENTILE = 10.0
CLUTTER_PERCENTILE = 95.0


# 필수 폴더가 없으면 통계값 자체가 다른 데이터 기준이 되므로 초기에 중단한다.
# 입력: path (확인할 폴더 경로)
# 반환: 없음
def require_directory(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {path}")


# CARRADA 루트에서 range_doppler_numpy가 있는 시퀀스만 분석 대상으로 고른다.
# 입력: carrada_root (CARRADA 루트 경로)
# 반환: 유효 시퀀스명 리스트
def find_valid_sequences(carrada_root):
    require_directory(carrada_root)

    sequence_names = []
    for name in sorted(os.listdir(carrada_root)):
        sequence_dir = os.path.join(carrada_root, name)
        rd_dir = os.path.join(sequence_dir, "range_doppler_numpy")
        if os.path.isdir(sequence_dir) and os.path.isdir(rd_dir):
            npy_files = [filename for filename in os.listdir(rd_dir) if filename.endswith(".npy")]
            if npy_files:
                sequence_names.append(name)

    if not sequence_names:
        raise RuntimeError("range_doppler_numpy가 있는 CARRADA 시퀀스를 찾지 못했습니다.")

    return sequence_names


# 각 시퀀스에서 고정 seed로 프레임을 뽑아 V2.3 통계가 재현되도록 한다.
# 입력: sequence_dir (시퀀스 폴더 경로), rng (random.Random 객체)
# 반환: 선택된 .npy 파일 경로 리스트
def sample_rd_frames(sequence_dir, rng):
    rd_dir = os.path.join(sequence_dir, "range_doppler_numpy")
    require_directory(rd_dir)

    npy_files = sorted(filename for filename in os.listdir(rd_dir) if filename.endswith(".npy"))
    if len(npy_files) < FRAMES_PER_SEQUENCE:
        raise RuntimeError(f"샘플링할 RD map이 부족합니다: {rd_dir}, {len(npy_files)}개")

    selected_files = rng.sample(npy_files, FRAMES_PER_SEQUENCE)
    return [os.path.join(rd_dir, filename) for filename in selected_files]


# 하위 10% 픽셀을 신호가 약한 노이즈 floor 후보로 보고 평균/분산을 추출한다.
# 입력: rd_map (2D RD map 배열)
# 반환: 하위 10% 평균, 표준편차
def compute_noise_floor_stats(rd_map):
    threshold = np.percentile(rd_map, NOISE_PERCENTILE)
    noise_pixels = rd_map[rd_map <= threshold]
    return float(np.mean(noise_pixels)), float(np.std(noise_pixels))


# zero-Doppler ridge는 실제 RD에서 강한 clutter로 나타날 수 있어 따로 통계를 낸다.
# 입력: rd_map (2D RD map 배열)
# 반환: 중앙 Doppler bin의 평균, 최대값, 표준편차
def compute_zero_doppler_stats(rd_map):
    # CARRADA RD shape는 (range, doppler)이므로 Doppler=0 bin은 중앙 행이 아니라 중앙 열이다.
    zero_doppler_col = rd_map[:, rd_map.shape[1] // 2]
    return (
        float(np.mean(zero_doppler_col)),
        float(np.max(zero_doppler_col)),
        float(np.std(zero_doppler_col)),
    )


# 상위 5% 픽셀은 강한 clutter blob이나 ridge 영향을 근사하기 위해 사용한다.
# 입력: rd_map (2D RD map 배열)
# 반환: 상위 5% 픽셀 평균
def compute_clutter_blob_stat(rd_map):
    threshold = np.percentile(rd_map, CLUTTER_PERCENTILE)
    clutter_pixels = rd_map[rd_map >= threshold]
    return float(np.mean(clutter_pixels))


# 프레임 하나에서 V3.1 노이즈 모델에 필요한 통계량을 추출한다.
# 입력: rd_path (.npy RD map 경로)
# 반환: 통계 dict
def analyze_frame(rd_path):
    if not os.path.isfile(rd_path):
        raise FileNotFoundError(f"RD map 파일을 찾을 수 없습니다: {rd_path}")

    rd_map = np.load(rd_path)
    if rd_map.ndim != 2:
        raise ValueError(f"RD map은 2D 배열이어야 합니다: {rd_path}, shape={rd_map.shape}")

    noise_floor_mean, noise_floor_std = compute_noise_floor_stats(rd_map)
    zero_mean, zero_max, zero_std = compute_zero_doppler_stats(rd_map)
    clutter_mean = compute_clutter_blob_stat(rd_map)

    return {
        "noise_floor_mean": noise_floor_mean,
        "noise_floor_std": noise_floor_std,
        "zero_doppler_mean": zero_mean,
        "zero_doppler_max": zero_max,
        "zero_doppler_std": zero_std,
        "speckle_std": float(np.std(rd_map)),
        "clutter_blob_mean": clutter_mean,
        "dynamic_range": float(np.max(rd_map) - np.min(rd_map)),
    }


# 숫자 리스트를 평균/표준편차/min/max로 요약해 콘솔과 JSON에 같은 기준으로 남긴다.
# 입력: values (숫자 리스트)
# 반환: 통계 dict
def summarize_full(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# 숫자 리스트를 평균/표준편차로 요약해 권장 파라미터에 필요한 크기만 남긴다.
# 입력: values (숫자 리스트)
# 반환: 통계 dict
def summarize_mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


# CARRADA 통계를 V3.1 mild/medium/hard 노이즈 파라미터로 변환한다.
# 입력: noise_floor_summary (noise floor dict), speckle_summary (speckle dict)
# 반환: 권장 파라미터 dict
def build_recommended_params(noise_floor_summary, speckle_summary):
    noise_mean = noise_floor_summary["mean"]
    noise_std = noise_floor_summary["std"]
    speckle_mean = speckle_summary["mean"]
    speckle_std = speckle_summary["std"]

    return {
        "mild": {
            "noise_floor": float(noise_mean + 0.5 * noise_std),
            "speckle_std": float(speckle_mean + 0.5 * speckle_std),
        },
        "medium": {
            "noise_floor": float(noise_mean + 1.0 * noise_std),
            "speckle_std": float(speckle_mean + 1.0 * speckle_std),
        },
        "hard": {
            "noise_floor": float(noise_mean + 2.0 * noise_std),
            "speckle_std": float(speckle_mean + 2.0 * speckle_std),
        },
    }


# 사람이 바로 확인할 수 있도록 주요 통계와 V3.1 권장값을 콘솔에 출력한다.
# 입력: stats (저장할 전체 통계 dict), sequence_count (시퀀스 수), frame_count (프레임 수)
# 반환: 없음
def print_statistics(stats, sequence_count, frame_count):
    print("=" * 80)
    print("CARRADA RD map 노이즈 통계 분석 (V2.3)")
    print("=" * 80)
    print(f"분석 시퀀스 수: {sequence_count}")
    print(f"분석 프레임 수: {frame_count}")
    print("")

    print("[Noise floor: 하위 10% 픽셀 평균 기준]")
    print(f"  mean: {stats['noise_floor']['mean']:.6f}")
    print(f"  std : {stats['noise_floor']['std']:.6f}")
    print(f"  min : {stats['noise_floor']['min']:.6f}")
    print(f"  max : {stats['noise_floor']['max']:.6f}")
    print("")

    print("[Zero-Doppler 성분: 중앙 Doppler bin]")
    print(f"  mean energy: {stats['zero_doppler']['mean_energy']:.6f}")
    print(f"  mean max   : {stats['zero_doppler']['mean_max']:.6f}")
    print(f"  std        : {stats['zero_doppler']['std']:.6f}")
    print("")

    print("[Speckle: 전체 픽셀 표준편차]")
    print(f"  mean: {stats['speckle']['mean']:.6f}")
    print(f"  std : {stats['speckle']['std']:.6f}")
    print("")

    print("[Clutter blob: 상위 5% 픽셀 평균]")
    print(f"  mean: {stats['clutter_blob']['mean']:.6f}")
    print(f"  std : {stats['clutter_blob']['std']:.6f}")
    print("")

    print("[Dynamic range: max-min]")
    print(f"  mean: {stats['dynamic_range']['mean']:.6f}")
    print(f"  std : {stats['dynamic_range']['std']:.6f}")
    print("")

    print("V3.1 시뮬레이터 노이즈 파라미터 추천값")
    for level in ["mild", "medium", "hard"]:
        params = stats["recommended"][level]
        print(f"  {level:6s}: noise_floor={params['noise_floor']:.6f}, speckle_std={params['speckle_std']:.6f}")
    print("=" * 80)


# CARRADA 전체에서 샘플링한 RD map 통계를 JSON으로 저장한다.
# 입력: 없음
# 반환: 없음
def main():
    sequence_names = find_valid_sequences(CARRADA_ROOT)
    rng = random.Random(RANDOM_SEED)

    frame_stats = []
    sampled_frame_count = 0
    for sequence_name in sequence_names:
        sequence_dir = os.path.join(CARRADA_ROOT, sequence_name)
        rd_paths = sample_rd_frames(sequence_dir, rng)
        for rd_path in rd_paths:
            frame_stats.append(analyze_frame(rd_path))
            sampled_frame_count += 1

    noise_floor_summary = summarize_full([item["noise_floor_mean"] for item in frame_stats])
    zero_mean_energy = summarize_mean_std([item["zero_doppler_mean"] for item in frame_stats])
    zero_max = summarize_mean_std([item["zero_doppler_max"] for item in frame_stats])
    zero_std = summarize_mean_std([item["zero_doppler_std"] for item in frame_stats])
    speckle_summary = summarize_mean_std([item["speckle_std"] for item in frame_stats])
    clutter_summary = summarize_mean_std([item["clutter_blob_mean"] for item in frame_stats])
    dynamic_range_summary = summarize_mean_std([item["dynamic_range"] for item in frame_stats])

    stats = {
        "noise_floor": noise_floor_summary,
        "zero_doppler": {
            "mean_energy": zero_mean_energy["mean"],
            "mean_max": zero_max["mean"],
            "std": zero_std["mean"],
        },
        "speckle": speckle_summary,
        "clutter_blob": clutter_summary,
        "dynamic_range": dynamic_range_summary,
        "recommended": build_recommended_params(noise_floor_summary, speckle_summary),
    }

    output_dir = os.path.dirname(OUTPUT_JSON)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False)

    print_statistics(stats, len(sequence_names), sampled_frame_count)
    print(f"JSON 저장 완료: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
