# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-05-20 (bbox center +0.5 보정, class name 통일)
# Dependency: numpy, matplotlib, PIL, argparse, json, csv, signal
# Description: FMCW RD map YOLO 학습 데이터셋 자동 생성기 (2000장, dry-run 지원)
# ================================================================================

import argparse
import csv
import json
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

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
    generate_clutter_matrix,
    lam,
    range_bins,
    save_meta_json,
    save_rd_image,
    save_yolo_label,
    snap_to_peak,
    target_to_yolo_bbox,
)


STOP_REQUESTED = False


# Ctrl+C가 들어와도 지금까지 만든 manifest/stat을 저장할 수 있도록 플래그만 바꾼다.
# 입력: signum(시그널 번호), frame(현재 프레임)
# 반환: 없음
def handle_interrupt(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[Interrupt] 현재 scene 이후 manifest/stat을 저장하고 종료합니다.")


# V1.2부터 확정된 velocity axis convention을 시작 시점에 검증한다.
# 입력: 없음
# 반환: range_axis, velocity_axis
def validate_axes():
    range_axis = np.arange(N_samples // 2) * c * fs / (2.0 * S * N_samples)
    doppler_freq_axis = np.fft.fftshift(np.fft.fftfreq(N_chirps, d=Tc))
    velocity_axis = -doppler_freq_axis * lam / 2.0

    assert velocity_axis[0] > velocity_axis[-1], (
        "velocity_axis는 단조감소여야 함. "
        f"현재: [{velocity_axis[0]}, ..., {velocity_axis[-1]}]"
    )
    test_v_bin = int(np.argmin(np.abs(velocity_axis - 10.0)))
    assert velocity_axis[test_v_bin] > 0, (
        "v=+10 m/s에 가장 가까운 bin이 양수 velocity여야 함"
    )

    return range_axis, velocity_axis


# Negative scene용 약한 random clutter를 생성한다.
# 입력: N_clutter, range/velocity 범위, 진폭 통계, rng, radar parameter
# 반환: clutter_matrix(N_chirps x N_samples 복소수 배열)
def generate_random_clutter(
    N_clutter,
    R_min,
    R_max,
    V_min,
    V_max,
    amp,
    amp_std,
    rng,
    N_samples,
    N_chirps,
    c,
    fc,
    S,
    Tc,
    fs,
):
    # guardrail OFF 장면에도 약한 비구조 clutter를 넣어 "clutter=ghost" shortcut을 막는다.
    clutter_matrix = np.zeros((N_chirps, N_samples), dtype=np.complex128)
    t_fast = np.arange(N_samples) / fs

    for _ in range(N_clutter):
        range_k = float(rng.uniform(R_min, R_max))
        velocity_k = float(rng.uniform(V_min, V_max))
        amp_k = max(0.0, float(amp + rng.standard_normal() * amp_std))

        for m in range(N_chirps):
            range_m = range_k - velocity_k * (m * Tc)
            tau_m = 2.0 * range_m / c
            f_beat = S * tau_m
            phase = 2.0 * np.pi * (f_beat * t_fast + fc * tau_m)
            clutter_matrix[m, :] += amp_k * np.exp(1j * phase) / max(range_k**2, 1e-6)

    return clutter_matrix


# 하나의 랜덤 시나리오와 독립 RNG 묶음을 만든다.
# 입력: master_rng(마스터 RNG), scene_idx(0-based), force_options(dry-run 강제 옵션)
# 반환: scene_config(JSON 직렬화 가능), rng_dict(메모리 전용 RNG 객체)
def sample_random_scenario(master_rng, scene_idx, force_options=None):
    scene_seed = int(master_rng.integers(0, 2**31))
    scenario_rng = np.random.default_rng(scene_seed)
    clutter_rng = np.random.default_rng(scene_seed + 1)
    noise_rng = np.random.default_rng(scene_seed + 2)

    if force_options and "force_guardrail" in force_options:
        guardrail_present = bool(force_options["force_guardrail"])
    else:
        guardrail_present = bool(scenario_rng.random() < 0.7)

    if force_options and "force_n_real" in force_options:
        n_real = int(force_options["force_n_real"])
    else:
        if guardrail_present:
            n_real = int(scenario_rng.choice([1, 2, 3, 4], p=[0.30, 0.30, 0.25, 0.15]))
        else:
            n_real = int(scenario_rng.choice([1, 2, 3, 4], p=[0.10, 0.30, 0.35, 0.25]))

    d_wall = float(scenario_rng.uniform(6.0, 10.0)) if guardrail_present else None

    real_targets = []
    for _ in range(n_real):
        velocity = float(scenario_rng.uniform(-18.0, 18.0))
        while abs(velocity) < 1.0:
            velocity = float(scenario_rng.uniform(-18.0, 18.0))

        real_targets.append(
            {
                "range": float(scenario_rng.uniform(10.0, 70.0)),
                "velocity": velocity,
                "amp": 1.0,
                "x_pos": float(scenario_rng.uniform(0.3, 2.5)),
                "type": "real",
            }
        )

    if guardrail_present:
        clutter_config = {
            "type": "structured",
            "N_clutter": 30,
            "R_start": d_wall,
            "R_span": 60.0,
            "amp": 0.3,
            "amp_std": 0.05,
        }
    else:
        clutter_config = {
            "type": "random",
            "N_clutter": 10,
            "R_min": 5.0,
            "R_max": 70.0,
            "V_min": -5.0,
            "V_max": 5.0,
            "amp": 0.1,
            "amp_std": 0.02,
        }

    scene_config = {
        "scene_id": f"rd_{scene_idx + 1:06d}",
        "scene_idx": int(scene_idx),
        "scene_seed": scene_seed,
        "guardrail_present": guardrail_present,
        "d_wall": d_wall,
        "wall_refl_coeff": 0.5,
        "real_targets": real_targets,
        "noise_power": 0.001,
        "clutter_config": clutter_config,
        "force_options": force_options,
    }
    rng_dict = {
        "scenario_rng": scenario_rng,
        "clutter_rng": clutter_rng,
        "noise_rng": noise_rng,
    }

    return scene_config, rng_dict


# 이론 bin 기준으로 bbox가 겹칠 위험이 있는지 빠르게 검사한다.
# 입력: targets(이론 range/velocity 포함), range_axis, velocity_axis, min_separation
# 반환: True if overlap, False if 통과
def check_overlap_theory(targets, range_axis, velocity_axis, min_separation=7):
    bins = []
    for target in targets:
        r_bin = int(np.argmin(np.abs(range_axis - target["range"])))
        d_bin = int(np.argmin(np.abs(velocity_axis - target["velocity"])))
        bins.append((d_bin, r_bin, target.get("family", None)))

    return bins_have_overlap(bins, min_separation)


# snap-to-peak 이후 실제 라벨 bin 기준으로 bbox overlap을 검사한다.
# 입력: targets_with_snap(bin_position 포함), min_separation
# 반환: True if overlap, False if 통과
def check_overlap_snap(targets_with_snap, min_separation=7):
    bins = [
        (target["bin_position"][0], target["bin_position"][1], target.get("family", None))
        for target in targets_with_snap
    ]
    return bins_have_overlap(bins, min_separation)


# bin 리스트가 bbox 분리 기준을 만족하는지 검사한다.
# 입력: bins((d_bin, r_bin) 리스트), min_separation
# 반환: True if overlap
def bins_have_overlap(bins, min_separation=7):
    for i in range(len(bins)):
        for j in range(i + 1, len(bins)):
            # 같은 real target에서 파생된 real/3b/4b stack은 물리적으로 가까우므로 rejection하지 않는다.
            if bins[i][2] is not None and bins[i][2] == bins[j][2]:
                continue
            dr = abs(bins[i][1] - bins[j][1])
            dd = abs(bins[i][0] - bins[j][0])
            if dr < min_separation and dd < min_separation:
                return True
    return False


# scene_config에서 이론 검사용 target 리스트를 만든다.
# 입력: scene_config
# 반환: all_targets(real + optional ghosts)
def build_theory_targets(scene_config):
    real_targets = [
        {**target, "family": idx}
        for idx, target in enumerate(scene_config["real_targets"])
    ]
    if scene_config["guardrail_present"]:
        ghosts_4b = compute_ghost_targets(
            real_targets,
            scene_config["d_wall"],
            scene_config["wall_refl_coeff"],
            "4bounce",
        )
        ghosts_3b = compute_ghost_targets(
            real_targets,
            scene_config["d_wall"],
            scene_config["wall_refl_coeff"],
            "3bounce",
        )
        for ghost in ghosts_4b + ghosts_3b:
            ghost["family"] = int(ghost["parent"]) - 1
        return real_targets + ghosts_4b + ghosts_3b

    return real_targets


# 이론 overlap이 없는 시나리오가 나올 때까지 재샘플링한다.
# 입력: master_rng, scene_idx, axes, force_options, max/hard attempt
# 반환: scene_config, rng_dict, attempt_count, rejection_log
def attempt_scene(
    master_rng,
    scene_idx,
    range_axis,
    velocity_axis,
    force_options=None,
    max_attempts=10,
    hard_limit=100,
):
    rejection_log = []

    for attempt in range(hard_limit):
        scene_config, rng_dict = sample_random_scenario(master_rng, scene_idx, force_options)
        all_targets = build_theory_targets(scene_config)

        if not check_overlap_theory(all_targets, range_axis, velocity_axis):
            return scene_config, rng_dict, attempt, rejection_log

        rejection_log.append(
            {
                "scene_idx": int(scene_idx),
                "scene_id": scene_config["scene_id"],
                "attempt": int(attempt),
                "reason": "theory_overlap",
                "n_real": len(scene_config["real_targets"]),
                "guardrail": scene_config["guardrail_present"],
            }
        )

        if attempt >= max_attempts and force_options:
            return scene_config, rng_dict, attempt, rejection_log

    raise RuntimeError(
        f"scene_idx={scene_idx}: {hard_limit}회 시도 후에도 overlap 해결 실패. "
        f"force_options={force_options}, "
        f"마지막 n_real={len(scene_config['real_targets'])}, "
        f"guardrail={scene_config['guardrail_present']}"
    )


# 라벨 생성을 위한 target spec을 만든다.
# 입력: real_targets, ghosts_4b, ghosts_3b
# 반환: target_specs(라벨 순서가 고정된 리스트)
def build_target_specs(real_targets, ghosts_4b, ghosts_3b):
    target_specs = []
    for idx, real_target in enumerate(real_targets):
        target_specs.append(
            {
                **real_target,
                "id": f"T{idx}",
                "class": "real",
                "class_id": 0,
                "parent": None,
                "bounce_type": None,
                "family": idx,
            }
        )

        if idx < len(ghosts_4b):
            target_specs.append(
                {
                    **ghosts_4b[idx],
                    "id": f"G{idx}a",
                    "class": "ghost",
                    "class_id": 1,
                    "parent": f"T{idx}",
                    "bounce_type": "4bounce",
                    "family": idx,
                }
            )

        if idx < len(ghosts_3b):
            target_specs.append(
                {
                    **ghosts_3b[idx],
                    "id": f"G{idx}b",
                    "class": "ghost",
                    "class_id": 1,
                    "parent": f"T{idx}",
                    "bounce_type": "3bounce",
                    "family": idx,
                }
            )

    return target_specs


# V3.0b 단일 scene을 생성하고 파일로 저장한다.
# 입력: scene_config(JSON 가능), rng_dict(RNG 객체), output_dir, split, save_debug
# 반환: manifest_row, meta_dict 또는 None(2차 overlap skip)
def run_scene_v3b(scene_config, rng_dict, output_dir, split, save_debug=False):
    if scene_config["guardrail_present"]:
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
    else:
        ghosts_4b = []
        ghosts_3b = []

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

    _, rd_magnitude, rd_map_db, range_axis, velocity_axis, _ = compute_rd_map(
        total_matrix,
        c,
        S,
        fs,
        lam,
        Tc,
        N_chirps,
        N_samples,
    )

    target_specs = build_target_specs(scene_config["real_targets"], ghosts_4b, ghosts_3b)
    targets_meta = []
    snap_offsets = []

    for target in target_specs:
        snap = snap_to_peak(
            float(target["range"]),
            float(target["velocity"]),
            rd_magnitude,
            range_axis,
            velocity_axis,
        )
        yolo_bbox = target_to_yolo_bbox(snap, N_chirps, range_bins, bbox_bin_width, bbox_bin_height)
        if not yolo_bbox_is_valid(yolo_bbox):
            return None, None, "bbox_out_of_range"

        target_meta = {
            "id": target["id"],
            "class": target["class"],
            "class_id": int(target["class_id"]),
            "theory_range": float(target["range"]),
            "theory_velocity": float(target["velocity"]),
            "amp": float(target.get("amp", 1.0)),
            "x_pos": float(target.get("x_pos", 0.0)),
            "snap_offset_r": int(snap["offset_r"]),
            "snap_offset_d": int(snap["offset_d"]),
            "snap_status": snap["snap_status"],
            "bin_position": [int(snap["d_bin"]), int(snap["r_bin"])],
            "bin_theory": [int(snap["d_bin_theory"]), int(snap["r_bin_theory"])],
            "yolo_bbox": yolo_bbox,
            "family": int(target.get("family", -1)),
        }
        if target["class"] == "ghost":
            target_meta["bounce_type"] = target["bounce_type"]
            target_meta["parent"] = target["parent"]

        targets_meta.append(target_meta)
        snap_offsets.append(
            {
                "target": target["id"],
                "bin_offset_r": int(snap["offset_r"]),
                "bin_offset_d": int(snap["offset_d"]),
                "status": snap["snap_status"],
            }
        )

    if check_overlap_snap(targets_meta):
        return None, None, "snap_overlap"

    output_dir = Path(output_dir)
    scene_id = scene_config["scene_id"]
    image_rel = Path("images") / split / f"{scene_id}.png"
    label_rel = Path("labels") / split / f"{scene_id}.txt"
    meta_rel = Path("meta") / split / f"{scene_id}.json"
    debug_rel = Path("debug") / f"{scene_id}_bbox_overlay.png"

    save_rd_image(rd_map_db, output_dir / image_rel, image_size=256)
    save_yolo_label(targets_meta, output_dir / label_rel)

    offset_norms = [
        float(np.hypot(item["bin_offset_r"], item["bin_offset_d"]))
        for item in snap_offsets
    ]
    fallback_count = sum(1 for item in snap_offsets if item["status"] == "fallback_to_theory")

    meta_dict = {
        "scene_id": scene_id,
        "scene_idx": int(scene_config["scene_idx"]),
        "scene_seed": int(scene_config["scene_seed"]),
        "split": split,
        "image_path": str(image_rel).replace("\\", "/"),
        "label_path": str(label_rel).replace("\\", "/"),
        "image_size": [256, 256],
        "rd_map_shape": [N_chirps, range_bins],
        "radar_params": {
            "fc": fc,
            "BW": BW,
            "Tc": Tc,
            "N_samples": N_samples,
            "N_chirps": N_chirps,
        },
        "scenario": {
            "guardrail_present": scene_config["guardrail_present"],
            "d_wall": scene_config["d_wall"],
            "wall_refl_coeff": scene_config["wall_refl_coeff"],
            "noise_power": scene_config["noise_power"],
            "clutter_config": scene_config["clutter_config"],
            "force_options": scene_config.get("force_options"),
        },
        "targets": targets_meta,
        "snap_offsets": snap_offsets,
        "snap_summary": {
            "max_offset": float(max(offset_norms)) if offset_norms else 0.0,
            "mean_offset": float(np.mean(offset_norms)) if offset_norms else 0.0,
            "fallback_count": int(fallback_count),
        },
    }
    save_meta_json(meta_dict, output_dir / meta_rel)

    if save_debug:
        save_debug_overlay_v3b(rd_map_db, targets_meta, output_dir / debug_rel, range_axis, velocity_axis, scene_id)

    manifest_row = {
        "scene_id": scene_id,
        "split": split,
        "image": str(image_rel).replace("\\", "/"),
        "label": str(label_rel).replace("\\", "/"),
        "meta": str(meta_rel).replace("\\", "/"),
        "scenario_seed": int(scene_config["scene_seed"]),
        "clutter_seed": int(scene_config["scene_seed"] + 1),
        "noise_seed": int(scene_config["scene_seed"] + 2),
        "n_real": len(scene_config["real_targets"]),
        "n_ghost": len(ghosts_4b) + len(ghosts_3b),
        "guardrail": bool(scene_config["guardrail_present"]),
        "d_wall": "" if scene_config["d_wall"] is None else float(scene_config["d_wall"]),
        "attempt": int(scene_config.get("attempt", 0)),
        "meta_dict": meta_dict,
    }

    return manifest_row, meta_dict, None


# scene_config clutter type에 따라 clutter 행렬을 만든다.
# 입력: clutter_config, clutter_rng
# 반환: clutter_matrix
def build_clutter_matrix(clutter_config, clutter_rng):
    if clutter_config["type"] == "structured":
        return generate_clutter_matrix(
            clutter_config["N_clutter"],
            clutter_config["R_start"],
            clutter_config["R_span"],
            clutter_config["amp"],
            clutter_config["amp_std"],
            c,
            fc,
            S,
            fs,
            N_chirps,
            N_samples,
            rng=clutter_rng,
        )

    return generate_random_clutter(
        clutter_config["N_clutter"],
        clutter_config["R_min"],
        clutter_config["R_max"],
        clutter_config["V_min"],
        clutter_config["V_max"],
        clutter_config["amp"],
        clutter_config["amp_std"],
        clutter_rng,
        N_samples,
        N_chirps,
        c,
        fc,
        S,
        Tc,
        fs,
    )


# YOLO bbox 값이 학습기가 읽을 수 있는 정규화 범위인지 확인한다.
# 입력: yolo_bbox([cx, cy, w, h])
# 반환: True if valid
def yolo_bbox_is_valid(yolo_bbox):
    cx, cy, w, h = yolo_bbox
    return 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0


# V3.0b용 debug overlay를 저장한다.
# 입력: rd_map_db, targets_meta, output_path, axes, scene_id
# 반환: 없음
def save_debug_overlay_v3b(rd_map_db, targets_meta, output_path, range_axis, velocity_axis, scene_id):
    # dry-run에서 라벨 좌표계 버그를 눈으로 잡기 위해 scene_id별 overlay를 남긴다.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rd_image = rd_map_db.T
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    ax.imshow(rd_image, origin="upper", cmap="viridis", aspect="auto")

    for target in targets_meta:
        cx, cy, w, h = target["yolo_bbox"]
        x_center = cx * N_chirps
        y_center = cy * range_bins
        box_w = w * N_chirps
        box_h = h * range_bins
        x0 = x_center - box_w / 2.0
        y0 = y_center - box_h / 2.0

        color = "red" if target["class_id"] == 0 else "orange"
        ax.add_patch(
            patches.Rectangle(
                (x0, y0),
                box_w,
                box_h,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
            )
        )
        ax.scatter([target["bin_theory"][0]], [target["bin_theory"][1]], color="yellow", marker="x", s=35, linewidths=1.3)
        ax.scatter([target["bin_position"][0]], [target["bin_position"][1]], color="white", marker="+", s=45, linewidths=1.4)

        label = f"{target['id']} {target['class']} Δ({target['snap_offset_r']:+d},{target['snap_offset_d']:+d})"
        text = ax.text(x0 + box_w + 0.6, max(0.0, y0 + 1.2), label, color="white", fontsize=7, va="top")
        text.set_path_effects([path_effects.Stroke(linewidth=2.0, foreground="black"), path_effects.Normal()])

    handles = [
        patches.Patch(edgecolor="red", facecolor="none", label="red box = real"),
        patches.Patch(edgecolor="orange", facecolor="none", label="orange box = ghost"),
        plt.Line2D([0], [0], color="yellow", marker="x", linestyle="", label="yellow X = theory bin"),
        plt.Line2D([0], [0], color="white", marker="+", linestyle="", label="white + = snapped bin"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.75)
    ax.set_xlabel("Velocity bin (m/s)")
    ax.set_ylabel("Range bin (m)")
    x_ticks = np.linspace(0, N_chirps - 1, 5, dtype=int)
    y_ticks = np.linspace(0, range_bins - 1, 5, dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{idx}\n{velocity_axis[idx]:+.1f}" for idx in x_ticks], fontsize=7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{idx}\n{range_axis[idx]:.0f}" for idx in y_ticks], fontsize=7)
    ax.set_xlim(0, N_chirps - 1)
    ax.set_ylim(range_bins - 1, 0)
    ax.set_title(f"Scene {scene_id} - bbox vs peak verification", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


# 0-based scene_idx를 결정론적 split으로 배정한다.
# 입력: scene_idx, total_count
# 반환: split 문자열
def assign_split(scene_idx, total_count=2000):
    if scene_idx < int(total_count * 0.8):
        return "train"
    if scene_idx < int(total_count * 0.9):
        return "val"
    return "test"


# 출력 디렉토리 구조를 만든다.
# 입력: output_dir
# 반환: 없음
def create_output_dirs(output_dir):
    # YOLO 표준 디렉토리를 미리 만들면 중간 실패 후에도 경로 문제가 없다.
    output_dir = Path(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "meta" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "debug").mkdir(parents=True, exist_ok=True)


# output directory overwrite 정책을 적용한다.
# 입력: output_dir, overwrite
# 반환: 없음
def prepare_output_dir(output_dir, overwrite):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} 이미 존재합니다. --overwrite 옵션을 사용하세요.")
        shutil.rmtree(output_dir)
    create_output_dirs(output_dir)


# 시작 전 대략적인 디스크 여유 공간을 확인한다.
# 입력: output_dir, total_count
# 반환: 없음
def check_disk_space(output_dir, total_count):
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_dir.parent).free
    estimate_bytes = int(total_count * 120_000 + 80_000_000)
    if free_bytes < estimate_bytes:
        raise OSError(
            f"디스크 공간 부족 가능성: 필요 추정 {estimate_bytes / 1e6:.1f} MB, "
            f"가용 {free_bytes / 1e6:.1f} MB"
        )


# manifest CSV를 저장한다.
# 입력: manifest_rows
# 반환: 없음
def write_manifest_csv(manifest_rows, output_dir):
    output_path = Path(output_dir) / "manifest.csv"
    columns = [
        "scene_id",
        "split",
        "image",
        "label",
        "meta",
        "scenario_seed",
        "clutter_seed",
        "noise_seed",
        "n_real",
        "n_ghost",
        "guardrail",
        "d_wall",
        "attempt",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({key: row.get(key, "") for key in columns})


# rejected_scenarios.log를 저장한다.
# 입력: rejection_entries
# 반환: 없음
def write_rejection_log(rejection_entries, output_dir):
    output_path = Path(output_dir) / "rejected_scenarios.log"
    with output_path.open("w", encoding="utf-8") as f:
        for entry in rejection_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# dataset 통계를 계산한다.
# 입력: manifest_rows, master_seed, generation_time_sec, rejection_stats
# 반환: stats 딕셔너리
def compute_dataset_stats(manifest_rows, master_seed=2026, generation_time_sec=0.0, rejection_stats=None):
    split_counts = {"train": 0, "val": 0, "test": 0}
    class_counts = {
        "total": {"real": 0, "ghost": 0},
        "train": {"real": 0, "ghost": 0},
        "val": {"real": 0, "ghost": 0},
        "test": {"real": 0, "ghost": 0},
    }
    ghost_breakdown = {"3bounce": 0, "4bounce": 0}
    guardrail_distribution = {"on": 0, "off": 0}
    real_count_distribution = {"1": 0, "2": 0, "3": 0, "4": 0}
    ranges = []
    velocities = []
    snap_offsets = []
    fallback_count = 0

    for row in manifest_rows:
        split = row["split"]
        split_counts[split] += 1
        guardrail_distribution["on" if row["guardrail"] else "off"] += 1
        real_count_distribution[str(row["n_real"])] += 1

        meta = row["meta_dict"]
        for target in meta["targets"]:
            class_name = "real" if target["class_id"] == 0 else "ghost"
            class_counts["total"][class_name] += 1
            class_counts[split][class_name] += 1
            ranges.append(float(target["theory_range"]))
            velocities.append(float(target["theory_velocity"]))
            snap_offsets.append(float(np.hypot(target["snap_offset_r"], target["snap_offset_d"])))
            if target["snap_status"] == "fallback_to_theory":
                fallback_count += 1
            if class_name == "ghost":
                ghost_breakdown[target["bounce_type"]] += 1

    snap_total = max(1, len(snap_offsets))
    rejection_stats = rejection_stats or {}

    return {
        "total_scenes": len(manifest_rows),
        "split": split_counts,
        "class_counts": class_counts,
        "ghost_breakdown": ghost_breakdown,
        "guardrail_distribution": guardrail_distribution,
        "real_count_distribution": real_count_distribution,
        "range_distribution": describe_array(ranges),
        "velocity_distribution": describe_array(velocities),
        "snap_offset_stats": {
            "max": float(max(snap_offsets)) if snap_offsets else 0.0,
            "mean": float(np.mean(snap_offsets)) if snap_offsets else 0.0,
            "fallback_count": int(fallback_count),
            "fallback_rate": float(fallback_count / snap_total),
        },
        "rejection_stats": rejection_stats,
        "generation_time_sec": float(generation_time_sec),
        "master_seed": int(master_seed),
        "creation_date": datetime.now().date().isoformat(),
    }


# 숫자 리스트의 기본 분포를 계산한다.
# 입력: values
# 반환: mean/std/min/max 딕셔너리
def describe_array(values):
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# dataset_stats.json을 저장한다.
# 입력: stats, output_dir
# 반환: 없음
def write_dataset_stats(stats, output_dir):
    save_meta_json(stats, Path(output_dir) / "dataset_stats.json")


# YOLOv8 data.yaml을 저장한다.
# 입력: stats, output_dir
# 반환: 없음
def write_data_yaml(stats, output_dir):
    abs_path = str(Path(output_dir).resolve()).replace("\\", "/")
    text = (
        "# YOLOv8 dataset configuration\n"
        f"path: {abs_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "# Classes\n"
        "names:\n"
        "  0: real_target\n"
        "  1: ghost_target\n\n"
        "nc: 2\n"
    )
    Path(output_dir, "data.yaml").write_text(text, encoding="utf-8")


# 한 scene을 성공할 때까지 생성한다.
# 입력: master_rng, scene_idx, split, output_dir, axes, force_options, save_debug, trackers
# 반환: manifest_row 또는 None(STOP)
def generate_one_scene(
    master_rng,
    scene_idx,
    split,
    output_dir,
    range_axis,
    velocity_axis,
    force_options,
    save_debug,
    rejection_entries,
    tracker,
):
    for scene_try in range(100):
        scene_config, rng_dict, attempt, theory_rejections = attempt_scene(
            master_rng,
            scene_idx,
            range_axis,
            velocity_axis,
            force_options=force_options,
        )
        scene_config["attempt"] = int(attempt)
        tracker["total_attempts"] += attempt + 1
        tracker["theory_overlap_rejections"] += len(theory_rejections)
        rejection_entries.extend(theory_rejections)

        manifest_row, meta_dict, skip_reason = run_scene_v3b(
            scene_config,
            rng_dict,
            output_dir,
            split,
            save_debug=save_debug,
        )
        if manifest_row is not None:
            return manifest_row

        tracker["snap_overlap_skips"] += 1 if skip_reason == "snap_overlap" else 0
        rejection_entries.append(
            {
                "scene_idx": int(scene_idx),
                "scene_id": scene_config["scene_id"],
                "attempt": int(attempt),
                "reason": skip_reason,
                "n_real": len(scene_config["real_targets"]),
                "guardrail": scene_config["guardrail_present"],
            }
        )

    tracker["max_attempts_exceeded"] += 1
    raise RuntimeError(f"scene_idx={scene_idx}: snap/theory rejection으로 100회 생성 실패")


# dry-run 또는 full 데이터셋 생성을 실행한다.
# 입력: args(argparse 결과)
# 반환: stats 딕셔너리
def run_dataset_generation(args):
    validate_axes()
    total_count = 10 if args.dry_run else args.total
    output_dir = Path(args.output_dir)
    check_disk_space(output_dir, total_count)
    prepare_output_dir(output_dir, args.overwrite)

    master_rng = np.random.default_rng(args.seed)
    range_axis, velocity_axis = validate_axes()
    manifest_rows = []
    rejection_entries = []
    tracker = {
        "total_attempts": 0,
        "theory_overlap_rejections": 0,
        "snap_overlap_skips": 0,
        "max_attempts_exceeded": 0,
    }

    start_time = time.time()
    phase_logs = []

    try:
        if args.dry_run:
            phase_a_cases = [
                {"force_guardrail": True, "force_n_real": 1},
                {"force_guardrail": True, "force_n_real": 4},
                {"force_guardrail": False, "force_n_real": 1},
                {"force_guardrail": False, "force_n_real": 4},
                {"force_guardrail": True, "force_n_real": 2},
            ]
            all_force_options = phase_a_cases + [None] * 5
            for scene_idx, force_options in enumerate(all_force_options):
                split = assign_split(scene_idx, total_count)
                manifest_row = generate_one_scene(
                    master_rng,
                    scene_idx,
                    split,
                    output_dir,
                    range_axis,
                    velocity_axis,
                    force_options,
                    save_debug=scene_idx < 5,
                    rejection_entries=rejection_entries,
                    tracker=tracker,
                )
                manifest_rows.append(manifest_row)
                phase_logs.append((scene_idx, force_options, manifest_row))
        else:
            progress_marks = progress_indices(total_count)
            for scene_idx in range(total_count):
                if STOP_REQUESTED:
                    break
                split = assign_split(scene_idx, total_count)
                manifest_row = generate_one_scene(
                    master_rng,
                    scene_idx,
                    split,
                    output_dir,
                    range_axis,
                    velocity_axis,
                    force_options=None,
                    save_debug=False,
                    rejection_entries=rejection_entries,
                    tracker=tracker,
                )
                manifest_rows.append(manifest_row)

                if (scene_idx + 1) in progress_marks:
                    print_progress(scene_idx + 1, total_count, start_time, manifest_rows, tracker)
    except KeyboardInterrupt:
        print("\n[Interrupt] KeyboardInterrupt 감지. 현재까지 결과를 저장합니다.")

    elapsed = time.time() - start_time
    rejection_stats = build_rejection_stats(tracker)
    stats = compute_dataset_stats(manifest_rows, args.seed, elapsed, rejection_stats)
    write_manifest_csv(manifest_rows, output_dir)
    write_rejection_log(rejection_entries, output_dir)
    write_dataset_stats(stats, output_dir)
    write_data_yaml(stats, output_dir)

    if args.dry_run:
        print_dry_run_summary(stats, phase_logs, elapsed, output_dir)
    else:
        print_full_summary(stats, output_dir)

    return stats


# rejection tracker를 stats 형식으로 바꾼다.
# 입력: tracker
# 반환: rejection_stats
def build_rejection_stats(tracker):
    total_attempts = max(1, tracker["total_attempts"])
    return {
        "total_attempts": int(tracker["total_attempts"]),
        "theory_overlap_rejections": int(tracker["theory_overlap_rejections"]),
        "theory_rejection_rate": float(tracker["theory_overlap_rejections"] / total_attempts),
        "snap_overlap_skips": int(tracker["snap_overlap_skips"]),
        "max_attempts_exceeded": int(tracker["max_attempts_exceeded"]),
    }


# progress 출력할 scene index 집합을 만든다.
# 입력: total_count
# 반환: marks(set)
def progress_indices(total_count):
    marks = set()
    for p in range(10, 101, 10):
        marks.add(max(1, int(total_count * p / 100)))
    for idx in range(500, total_count + 1, 500):
        marks.add(idx)
    return marks


# full 생성 진행률을 출력한다.
# 입력: completed, total, start_time, manifest_rows, tracker
# 반환: 없음
def print_progress(completed, total, start_time, manifest_rows, tracker):
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else 0.0
    real_count = sum(row["n_real"] for row in manifest_rows)
    ghost_count = sum(row["n_ghost"] for row in manifest_rows)
    total_attempts = max(1, tracker["total_attempts"])
    theory_rate = tracker["theory_overlap_rejections"] / total_attempts
    print(
        f"[Progress] {completed}/{total} ({completed / total * 100:.1f}%) "
        f"- elapsed {elapsed:.1f}s, ETA {eta:.1f}s"
    )
    print(
        f"  Current: real={real_count}, ghost={ghost_count}, "
        f"theory_rej_rate={theory_rate * 100:.1f}%, "
        f"snap_skips={tracker['snap_overlap_skips']}"
    )


# dry-run 결과를 요구 형식에 맞게 출력한다.
# 입력: stats, phase_logs, elapsed, output_dir
# 반환: 없음
def print_dry_run_summary(stats, phase_logs, elapsed, output_dir):
    print("=" * 70)
    print("[V3.0b Dry-Run - 10장 검증]")
    print("=" * 70)
    print("  [Phase A: 강제 분배 5장]")
    for scene_idx, force_options, row in phase_logs[:5]:
        guardrail = "ON" if row["guardrail"] else "OFF"
        print(
            f"  {row['scene_id']}: guardrail={guardrail:<3}, "
            f"n_real={row['n_real']}, n_ghost={row['n_ghost']}, attempts={row['attempt']}"
        )

    print("\n  [Phase B: 자연 분포 5장]")
    for scene_idx, force_options, row in phase_logs[5:]:
        guardrail = "ON" if row["guardrail"] else "OFF"
        print(
            f"  {row['scene_id']}: guardrail={guardrail:<3}, "
            f"n_real={row['n_real']}, n_ghost={row['n_ghost']}, attempts={row['attempt']}"
        )

    real_total = stats["class_counts"]["total"]["real"]
    ghost_total = stats["class_counts"]["total"]["ghost"]
    ratio = ghost_total / max(1, real_total)
    rej = stats["rejection_stats"]
    print("\n  [통계 - Phase A+B]")
    print(f"  Total: {stats['total_scenes']}")
    print(f"  Class counts: real={real_total}, ghost={ghost_total}")
    print(f"  ratio (real:ghost): 1:{ratio:.2f}")
    print(
        f"  Guardrail: ON={stats['guardrail_distribution']['on']}, "
        f"OFF={stats['guardrail_distribution']['off']}"
    )
    print(f"  Real count distribution: {stats['real_count_distribution']}")
    print(
        f"  Snap max_offset: {stats['snap_offset_stats']['max']:.1f} bin, "
        f"mean: {stats['snap_offset_stats']['mean']:.2f} bin"
    )
    print(f"  fallback: {stats['snap_offset_stats']['fallback_count']}")
    print(
        f"  Theory overlap rejections: {rej['theory_overlap_rejections']} "
        f"(rate: {rej['theory_rejection_rate'] * 100:.1f}%)"
    )
    print(f"  Snap overlap skips: {rej['snap_overlap_skips']}")

    per_scene = elapsed / max(1, stats["total_scenes"])
    print("\n  [예상 풀 실행 (2000장)]")
    print(f"  예상 rejection rate: {rej['theory_rejection_rate'] * 100:.1f}%")
    print(f"  예상 소요 시간: {per_scene * 2000:.1f}초 (장당 {per_scene:.2f}초 기준)")
    print(f"\n  Debug overlay 5장: {Path(output_dir) / 'debug'}")
    print("  눈으로 확인 후 풀 실행:")
    print("    python rd_yolo_dataset_v3_0b.py --full")
    print("=" * 70)


# full 생성 결과를 요약 출력한다.
# 입력: stats, output_dir
# 반환: 없음
def print_full_summary(stats, output_dir):
    print("=" * 70)
    print("[V3.0b Full Dataset 생성 완료]")
    print("=" * 70)
    print(f"  Output: {Path(output_dir).resolve()}")
    print(f"  Total scenes: {stats['total_scenes']}")
    print(f"  Split: {stats['split']}")
    print(f"  Class counts: {stats['class_counts']['total']}")
    print(f"  Rejection stats: {stats['rejection_stats']}")
    print(f"  data.yaml: {Path(output_dir) / 'data.yaml'}")
    print("=" * 70)


# CLI 인자를 해석한다.
# 입력: 없음
# 반환: args
def parse_args():
    parser = argparse.ArgumentParser(description="FMCW RD YOLO dataset generator V3.0b")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="10장만 생성하고 debug overlay 저장")
    mode.add_argument("--full", action="store_true", help="전체 데이터셋 생성")
    parser.add_argument("--total", type=int, default=2000, help="full 생성 scene 수")
    parser.add_argument("--seed", type=int, default=2026, help="master seed")
    parser.add_argument("--output_dir", type=str, default="dataset_v3_0b", help="출력 디렉토리")
    parser.add_argument("--overwrite", action="store_true", help="기존 출력 디렉토리 삭제 후 재생성")
    return parser.parse_args()


def main():
    signal.signal(signal.SIGINT, handle_interrupt)
    args = parse_args()
    run_dataset_generation(args)


if __name__ == "__main__":
    main()
