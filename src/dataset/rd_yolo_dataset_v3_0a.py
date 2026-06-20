# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-05-20 (bbox center +0.5 보정, class name 통일)
# Dependency: numpy, matplotlib, scipy, PIL (or matplotlib only)
# Description: FMCW RD map → YOLO 학습 데이터셋 단일 시나리오 생성 및 라벨 검증
# ================================================================================

import json
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from rd_yolo_sim_v2_2 import compute_ghost_targets, compute_rd_map


# ================================================================================
# 1. FMCW 레이더 파라미터
# ================================================================================
c = 3e8
fc = 77e9
BW = 150e6
Tc = 50e-6
N_samples = 256
N_chirps = 128
S = BW / Tc
fs = N_samples / Tc
lam = c / fc

range_bins = N_samples // 2
bbox_bin_width = 5
bbox_bin_height = 5


# 표적 beat signal 행렬을 생성한다.
# 입력: targets_list(표적/ghost 딕셔너리 리스트), radar parameter, noise_power(잡음 전력), rng(default_rng)
# 반환: beat_matrix(N_chirps x N_samples 복소수 배열, 표적 신호 + AWGN)
def generate_beat_matrix(
    targets_list,
    c,
    fc,
    S,
    Tc,
    fs,
    N_chirps,
    N_samples,
    noise_power=0.001,
    rng=None,
):
    # V3부터는 scene별 seed를 분리해야 같은 시나리오에 다른 noise를 재현할 수 있다.
    if rng is None:
        rng = np.random.default_rng(42)

    t_fast = np.arange(N_samples) / fs
    beat_matrix = np.zeros((N_chirps, N_samples), dtype=np.complex128)
    reference_range = min(float(target["range"]) for target in targets_list)

    for target in targets_list:
        target_range = float(target["range"])
        target_velocity = float(target["velocity"])
        target_amp = float(target.get("amp", 1.0))

        for m in range(N_chirps):
            t_start_m = m * Tc

            # fc*tau 시작 위상은 slow-time 위상 연속성과 Doppler 부호를 유지하는 핵심 항이다.
            range_start_m = target_range - target_velocity * t_start_m
            tau_start_m = 2.0 * range_start_m / c

            # chirp 내부 이동까지 반영해야 fast-time peak가 실제 RD map 위치와 잘 맞는다.
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


# 가드레일 clutter 행렬을 생성한다.
# 입력: clutter 파라미터, radar parameter, rng(default_rng)
# 반환: clutter_matrix(N_chirps x N_samples 복소수 배열)
def generate_clutter_matrix(
    N_clutter,
    R_start,
    R_span,
    clutter_amp,
    clutter_amp_std,
    c,
    fc,
    S,
    fs,
    N_chirps,
    N_samples,
    rng=None,
):
    # clutter RNG를 noise와 분리해야 같은 scene에서 clutter만 고정하거나 noise만 바꾸는 실험이 가능하다.
    if rng is None:
        rng = np.random.default_rng(42)

    t_fast = np.arange(N_samples) / fs
    clutter_fast = np.zeros(N_samples, dtype=np.complex128)

    for k in range(N_clutter):
        range_k = R_start + k * (R_span / N_clutter)
        amp_k = max(0.0, clutter_amp + rng.standard_normal() * clutter_amp_std)
        tau_k = 2.0 * range_k / c
        f_beat_k = S * tau_k

        clutter_fast += amp_k * np.exp(
            1j * 2.0 * np.pi * (f_beat_k * t_fast + fc * tau_k)
        )

    return np.tile(clutter_fast[np.newaxis, :], (N_chirps, 1))


# 이론 bin 주변의 실제 RD peak로 라벨 중심을 보정한다.
# 입력: theory_range/velocity, rd_magnitude, range_axis, velocity_axis, search_half_width
# 반환: snapped bin, offset, snap 상태, 이론 bin 정보
def snap_to_peak(
    theory_range,
    theory_velocity,
    rd_magnitude,
    range_axis,
    velocity_axis,
    search_half_width=2,
):
    r_bin_theory = int(np.argmin(np.abs(range_axis - theory_range)))
    d_bin_theory = int(np.argmin(np.abs(velocity_axis - theory_velocity)))

    r_lo = max(0, r_bin_theory - search_half_width)
    r_hi = min(rd_magnitude.shape[1], r_bin_theory + search_half_width + 1)
    d_lo = max(0, d_bin_theory - search_half_width)
    d_hi = min(rd_magnitude.shape[0], d_bin_theory + search_half_width + 1)

    window = rd_magnitude[d_lo:d_hi, r_lo:r_hi]
    local_max = np.unravel_index(np.argmax(window), window.shape)

    r_bin_snapped = int(r_lo + local_max[1])
    d_bin_snapped = int(d_lo + local_max[0])
    offset_r = r_bin_snapped - r_bin_theory
    offset_d = d_bin_snapped - d_bin_theory

    theory_mag = rd_magnitude[d_bin_theory, r_bin_theory]
    snapped_mag = rd_magnitude[d_bin_snapped, r_bin_snapped]
    if snapped_mag < 0.5 * theory_mag:
        r_bin_snapped = r_bin_theory
        d_bin_snapped = d_bin_theory
        offset_r = 0
        offset_d = 0
        snap_status = "fallback_to_theory"
    else:
        snap_status = "snapped"

    return {
        "r_bin": r_bin_snapped,
        "d_bin": d_bin_snapped,
        "r_bin_theory": r_bin_theory,
        "d_bin_theory": d_bin_theory,
        "offset_r": int(offset_r),
        "offset_d": int(offset_d),
        "snap_status": snap_status,
    }


# target 정보를 YOLO bbox 좌표로 변환한다.
# 입력: target_info(스냅된 bin 포함), N_chirps, range_bins, bbox bin 크기
# 반환: yolo_bbox([cx, cy, w, h])
def target_to_yolo_bbox(
    target_info,
    N_chirps,
    range_bins,
    bbox_bin_width=5,
    bbox_bin_height=5,
):
    # 라벨과 overlay가 같은 중심을 쓰도록 snapped bin을 직접 정규화한다.
    cx = (target_info["d_bin"] + 0.5) / N_chirps
    cy = (target_info["r_bin"] + 0.5) / range_bins
    w = bbox_bin_width / N_chirps
    h = bbox_bin_height / range_bins

    return [
        float(np.clip(cx, 0.0, 1.0)),
        float(np.clip(cy, 0.0, 1.0)),
        float(w),
        float(h),
    ]


# RD map을 256x256 viridis PNG로 저장한다.
# 입력: rd_map_db(Doppler x Range), output_path, image_size
# 반환: 없음
def save_rd_image(rd_map_db, output_path, image_size=256):
    # nearest 확대는 FFT bin 경계를 보존해 YOLO 라벨과 이미지 peak가 어긋나지 않게 한다.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rd_image = rd_map_db.T
    scale = image_size // rd_image.shape[0]
    rd_up = np.repeat(np.repeat(rd_image, scale, axis=0), scale, axis=1)

    vmin = np.percentile(rd_up, 1.0)
    vmax = np.percentile(rd_up, 99.5)
    rd_norm = np.clip((rd_up - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    rgb = plt.get_cmap("viridis")(rd_norm)[..., :3]

    plt.imsave(output_path, rgb)


# YOLO 라벨 txt를 저장한다.
# 입력: targets_meta(라벨 메타 리스트), output_path
# 반환: 없음
def save_yolo_label(targets_meta, output_path):
    # YOLO 학습기는 주석을 읽지 않으므로 숫자 라인만 저장한다.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for target in targets_meta:
        cx, cy, w, h = target["yolo_bbox"]
        lines.append(f"{target['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# scene 메타 JSON을 저장한다.
# 입력: scene_info(메타 딕셔너리), output_path
# 반환: 없음
def save_meta_json(scene_info, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(scene_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# bbox 좌표가 peak를 감싸는지 확인하는 debug overlay를 저장한다.
# 입력: rd_map_db, targets_meta, output_path, range_axis, velocity_axis
# 반환: 없음
def save_debug_overlay(
    rd_map_db,
    targets_meta,
    output_path,
    range_axis,
    velocity_axis,
):
    # overlay는 YOLO txt와 같은 bbox 좌표를 그림으로 보여줘 라벨 버그를 빠르게 잡게 한다.
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
        rect = patches.Rectangle(
            (x0, y0),
            box_w,
            box_h,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.scatter(
            [target["bin_theory"][0]],
            [target["bin_theory"][1]],
            color="yellow",
            marker="x",
            s=35,
            linewidths=1.3,
        )
        ax.scatter(
            [target["bin_position"][0]],
            [target["bin_position"][1]],
            color="white",
            marker="+",
            s=45,
            linewidths=1.4,
        )

        label = (
            f"{target['id']} {target['class']} "
            f"Δ({target['snap_offset_r']:+d},{target['snap_offset_d']:+d})"
        )
        text = ax.text(
            x0 + box_w + 0.6,
            max(0.0, y0 + 1.2),
            label,
            color="white",
            fontsize=8,
            va="top",
        )
        text.set_path_effects(
            [path_effects.Stroke(linewidth=2.0, foreground="black"), path_effects.Normal()]
        )

    real_patch = patches.Patch(edgecolor="red", facecolor="none", label="red box = real")
    ghost_patch = patches.Patch(edgecolor="orange", facecolor="none", label="orange box = ghost")
    theory_marker = plt.Line2D([0], [0], color="yellow", marker="x", linestyle="", label="yellow X = theory bin")
    snapped_marker = plt.Line2D([0], [0], color="white", marker="+", linestyle="", label="white + = snapped bin")
    ax.legend(
        handles=[real_patch, ghost_patch, theory_marker, snapped_marker],
        loc="lower left",
        fontsize=7,
        framealpha=0.75,
    )

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
    ax.set_title("Scene rd_000001 - bbox vs peak verification", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


# 출력 디렉토리 구조를 만든다.
# 입력: output_dir(루트), split(train/val/test)
# 반환: paths(각 산출물 디렉토리)
def create_output_dirs(output_dir, split="train"):
    # V3.0b에서 split만 바꿔도 같은 구조를 재사용할 수 있게 미리 분리한다.
    output_dir = Path(output_dir)
    paths = {
        "image_dir": output_dir / "images" / split,
        "label_dir": output_dir / "labels" / split,
        "meta_dir": output_dir / "meta" / split,
        "debug_dir": output_dir / "debug",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


# 라벨 생성용 target 메타 리스트를 만든다.
# 입력: real_targets, ghosts_4b, ghosts_3b
# 반환: target_specs(T0/G0a/G0b 순서의 딕셔너리 리스트)
def build_target_specs(real_targets, ghosts_4b, ghosts_3b):
    # 라벨 순서를 고정해야 V3.0b 통계와 디버그가 항상 같은 의미를 갖는다.
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
            }
        )

        ghost_4b = ghosts_4b[idx]
        target_specs.append(
            {
                **ghost_4b,
                "id": f"G{idx}a",
                "class": "ghost",
                "class_id": 1,
                "parent": f"T{idx}",
                "bounce_type": "4bounce",
            }
        )

        ghost_3b = ghosts_3b[idx]
        target_specs.append(
            {
                **ghost_3b,
                "id": f"G{idx}b",
                "class": "ghost",
                "class_id": 1,
                "parent": f"T{idx}",
                "bounce_type": "3bounce",
            }
        )

    return target_specs


# 단일 시나리오를 실행해 이미지/라벨/meta/debug를 생성한다.
# 입력: scene_config(시나리오 딕셔너리), output_dir(루트), split(train/val/test)
# 반환: meta_dict(V3.0b manifest 생성용)
def run_scene(scene_config, output_dir, split="train"):
    scene_id = scene_config["scene_id"]
    scene_seed = int(scene_config["scene_seed"])
    d_wall = float(scene_config["d_wall"])
    wall_refl_coeff = float(scene_config["wall_refl_coeff"])
    real_targets = [
        {**target, "type": "real", "amp": float(target.get("amp", 1.0))}
        for target in scene_config["real_targets"]
    ]

    scenario_rng = np.random.default_rng(scene_seed)
    clutter_rng = np.random.default_rng(scene_seed + 1)
    noise_rng = np.random.default_rng(scene_seed + 2)
    _ = scenario_rng

    ghosts_4b = compute_ghost_targets(real_targets, d_wall, wall_refl_coeff, "4bounce")
    ghosts_3b = compute_ghost_targets(real_targets, d_wall, wall_refl_coeff, "3bounce")
    all_targets = real_targets + ghosts_4b + ghosts_3b

    beat_matrix = generate_beat_matrix(
        all_targets,
        c,
        fc,
        S,
        Tc,
        fs,
        N_chirps,
        N_samples,
        noise_power=scene_config["noise_power"],
        rng=noise_rng,
    )
    clutter_cfg = scene_config["clutter"]
    clutter_matrix = generate_clutter_matrix(
        clutter_cfg["N_clutter"],
        clutter_cfg["R_start"],
        clutter_cfg["R_span"],
        clutter_cfg["amp"],
        clutter_cfg["amp_std"],
        c,
        fc,
        S,
        fs,
        N_chirps,
        N_samples,
        rng=clutter_rng,
    )
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

    target_specs = build_target_specs(real_targets, ghosts_4b, ghosts_3b)
    targets_meta = []
    snap_offsets = []

    for target in target_specs:
        snap = snap_to_peak(
            float(target["range"]),
            float(target["velocity"]),
            rd_magnitude,
            range_axis,
            velocity_axis,
            search_half_width=2,
        )
        yolo_bbox = target_to_yolo_bbox(
            snap,
            N_chirps,
            range_bins,
            bbox_bin_width,
            bbox_bin_height,
        )

        target_meta = {
            "id": target["id"],
            "class": target["class"],
            "class_id": int(target["class_id"]),
            "theory_range": float(target["range"]),
            "theory_velocity": float(target["velocity"]),
            "amp": float(target.get("amp", 1.0)),
            "x_pos": float(target.get("x_pos", 0.0)),
            "snap_offset_r": snap["offset_r"],
            "snap_offset_d": snap["offset_d"],
            "snap_status": snap["snap_status"],
            "bin_position": [snap["d_bin"], snap["r_bin"]],
            "bin_theory": [snap["d_bin_theory"], snap["r_bin_theory"]],
            "yolo_bbox": yolo_bbox,
        }
        if target["class"] == "ghost":
            target_meta["bounce_type"] = target["bounce_type"]
            target_meta["parent"] = target["parent"]

        targets_meta.append(target_meta)
        snap_offsets.append(
            {
                "target": target["id"],
                "bin_offset_r": snap["offset_r"],
                "bin_offset_d": snap["offset_d"],
                "status": snap["snap_status"],
            }
        )

    paths = create_output_dirs(output_dir, split)
    image_rel = Path("images") / split / f"{scene_id}.png"
    label_rel = Path("labels") / split / f"{scene_id}.txt"
    meta_rel = Path("meta") / split / f"{scene_id}.json"
    debug_rel = Path("debug") / f"{scene_id}_bbox_overlay.png"

    save_rd_image(rd_map_db, Path(output_dir) / image_rel, image_size=256)
    save_yolo_label(targets_meta, Path(output_dir) / label_rel)
    save_debug_overlay(
        rd_map_db,
        targets_meta,
        Path(output_dir) / debug_rel,
        range_axis,
        velocity_axis,
    )

    offset_norms = [
        float(np.hypot(item["bin_offset_r"], item["bin_offset_d"]))
        for item in snap_offsets
    ]
    fallback_count = sum(1 for item in snap_offsets if item["status"] == "fallback_to_theory")

    meta_dict = {
        "scene_id": scene_id,
        "scene_seed": scene_seed,
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
            "d_wall": d_wall,
            "wall_refl_coeff": wall_refl_coeff,
            "noise_power": float(scene_config["noise_power"]),
        },
        "targets": targets_meta,
        "snap_offsets": snap_offsets,
        "snap_summary": {
            "max_offset": float(max(offset_norms)),
            "mean_offset": float(np.mean(offset_norms)),
            "fallback_count": int(fallback_count),
        },
    }
    save_meta_json(meta_dict, Path(output_dir) / meta_rel)

    return meta_dict


# 단일 scene 생성 결과를 콘솔에 요약한다.
# 입력: meta_dict(run_scene 반환값), output_dir
# 반환: 없음
def print_scene_summary(meta_dict, output_dir):
    output_dir = Path(output_dir)
    scene_id = meta_dict["scene_id"]
    scene_seed = meta_dict["scene_seed"]

    print("=" * 70)
    print("[V3.0a - 단일 시나리오 데이터셋 생성 검증]")
    print("=" * 70)
    print(f"  Scene ID: {scene_id}")
    print(f"  Seeds: scene={scene_seed}, clutter={scene_seed + 1}, noise={scene_seed + 2}")

    print("\n  [시나리오 구성]")
    real_targets = [target for target in meta_dict["targets"] if target["class_id"] == 0]
    ghost_targets = [target for target in meta_dict["targets"] if target["class_id"] == 1]
    print(f"  Real targets: {len(real_targets)}")
    for target in real_targets:
        print(
            f"    {target['id']}: range={target['theory_range']:.1f}m, "
            f"vel={target['theory_velocity']:+.1f}m/s"
        )

    print(f"  Ghost targets: {len(ghost_targets)}")
    for target in ghost_targets:
        bounce_label = "4b" if target["bounce_type"] == "4bounce" else "3b"
        print(
            f"    {target['id']} ({bounce_label}): range={target['theory_range']:.1f}m, "
            f"vel={target['theory_velocity']:+.1f}m/s"
        )

    print("\n  [Snap-to-Peak 결과]")
    for target in meta_dict["targets"]:
        print(
            f"  {target['id']}: "
            f"Δr={target['snap_offset_r']:+d}, "
            f"Δd={target['snap_offset_d']:+d}, "
            f"status={target['snap_status']}"
        )

    summary = meta_dict["snap_summary"]
    print(
        f"  Max offset: {summary['max_offset']:.1f} bin, "
        f"Mean offset: {summary['mean_offset']:.1f} bin, "
        f"Fallback: {summary['fallback_count']}"
    )

    image_path = output_dir / meta_dict["image_path"]
    label_path = output_dir / meta_dict["label_path"]
    meta_path = output_dir / "meta" / meta_dict["split"] / f"{scene_id}.json"
    debug_path = output_dir / "debug" / f"{scene_id}_bbox_overlay.png"
    print("\n  [출력 파일]")
    print(f"  Image: {image_path} (256×256)")
    print(f"  Label: {label_path} ({len(meta_dict['targets'])} objects)")
    print(f"  Meta : {meta_path}")
    print(f"  Debug: {debug_path} (512×512)")
    print("\n  V3.0a 검증 완료 - debug overlay 이미지를 눈으로 확인하세요.")
    print("=" * 70)


def main():
    scene_config = {
        "scene_id": "rd_000001",
        "scene_seed": 100001,
        "d_wall": 8.0,
        "wall_refl_coeff": 0.5,
        "real_targets": [
            {"range": 15.0, "velocity": 10.0, "amp": 1.0, "x_pos": 0.5},
            {"range": 50.0, "velocity": -5.0, "amp": 1.0, "x_pos": 0.5},
        ],
        "clutter": {
            "N_clutter": 30,
            "R_start": 8.0,
            "R_span": 60.0,
            "amp": 0.3,
            "amp_std": 0.05,
        },
        "noise_power": 0.001,
    }

    output_dir = "dataset_v3_0a"
    create_output_dirs(output_dir, split="train")
    meta = run_scene(scene_config, output_dir, split="train")
    print_scene_summary(meta, output_dir)


if __name__ == "__main__":
    main()
