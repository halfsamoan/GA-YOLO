# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-05-21 (V4.5)
# Dependency: train/val/test meta JSON, missed_ghost_analysis.csv(V4.4)
# Description: label bbox 크기 후보별 GT overlap을 분석해 적정 bbox 크기를 결정
# ================================================================================

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
META_BASE_CANDIDATES = [
    BASE_DIR / "dataset_v4_1_gray" / "meta",
    Path("/home/kmin/RD_YOLO_GHOST/dataset_v4_1_gray/meta"),
    BASE_DIR / "dataset_v3_0b" / "meta",
]
MISSED_GHOST_CSV = BASE_DIR / "family_meta_v4_4" / "missed_ghost_analysis.csv"
OUT_DIR = BASE_DIR / "bbox_sweep_v4_5"

SPLITS = ["train", "val", "test"]
CANDIDATE_SIZES = [5, 4, 3, 2, 1]
BIN_SIZE = 128
CONFLICT_IOU_THRESHOLD = 0.0
SEVERE_IOU_THRESHOLD = 0.5
DISTANCE_BUCKETS = ["[0-1)", "[1-2)", "[2-3)", "[3-4)", "[4-6)", "[6+)"]
REQUIRED_KEYS = {"id", "class_id", "bin_position", "family", "yolo_bbox"}
REQUIRED_GHOST_KEYS = {"parent", "bounce_type", "amp"}


@dataclass
class Target:
    split: str
    scene_id: str
    target_id: str
    class_id: int
    family: int
    bin_d: int
    bin_r: int
    bounce_type: str
    parent: str
    amp: float


@dataclass
class PairDistance:
    split: str
    scene_id: str
    target_a: str
    target_b: str
    distance_bins: float
    is_ghost_parent_pair: bool


@dataclass
class MissedGhostRef:
    scene_id: str
    target_id: str


@dataclass
class BoxSizeResult:
    box_size: int
    conflict_pair_count: int
    conflict_pair_ratio: float
    severe_conflict_pair_count: int
    severe_conflict_pair_ratio: float
    missed_ghost_resolved_count: int
    missed_ghost_resolved_ratio: float


# CLI에서 meta base 경로만 직접 지정할 수 있게 한다.
# 입력: 없음
# 반환: argparse Namespace
def parse_args():
    parser = argparse.ArgumentParser(description="GA-YOLO V4.5 bbox size sweep 진단")
    parser.add_argument("--meta_dir", default=None, help="train/val/test를 포함하는 meta base 경로")
    return parser.parse_args()


# split 하위 디렉터리를 가진 meta base 경로를 찾는다.
# 입력: cli_meta_dir (str 또는 None)
# 반환: meta base Path
def resolve_meta_base(cli_meta_dir):
    candidates = [Path(cli_meta_dir).expanduser().resolve()] if cli_meta_dir else META_BASE_CANDIDATES
    for candidate in candidates:
        if all((candidate / split).exists() for split in SPLITS):
            return candidate
    raise FileNotFoundError("train/val/test meta 디렉터리를 모두 가진 경로를 찾을 수 없습니다.")


# meta target이 V4.5 분석에 필요한 키를 갖는지 확인한다.
# 입력: target dict
# 반환: 없음
def validate_target_schema(target):
    missing = REQUIRED_KEYS - set(target)
    if int(target.get("class_id", -1)) == 1:
        missing.update(REQUIRED_GHOST_KEYS - set(target))
    if missing:
        raise ValueError(f"V4.5 분석에 필요한 meta key가 없습니다: {sorted(missing)}")


# meta target dict를 분석용 Target으로 변환한다.
# 입력: split, scene_id, target dict
# 반환: Target
def target_from_meta(split, scene_id, target):
    validate_target_schema(target)
    bin_d, bin_r = target["bin_position"]
    return Target(
        split=split,
        scene_id=scene_id,
        target_id=target["id"],
        class_id=int(target["class_id"]),
        family=int(target["family"]),
        bin_d=int(bin_d),
        bin_r=int(bin_r),
        bounce_type=target.get("bounce_type", "real"),
        parent=target.get("parent", ""),
        amp=float(target.get("amp", 0.0)),
    )


# train/val/test 전체 meta에서 target 위치/속성을 로드한다.
# 입력: meta_dir
# 반환: split -> scene_id -> Target 리스트
def load_all_targets(meta_dir) -> dict[str, dict[str, list[Target]]]:
    all_targets = {}
    for split in SPLITS:
        split_dir = Path(meta_dir) / split
        scenes = {}
        for meta_path in sorted(split_dir.glob("*.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            scene_id = meta["scene_id"]
            scenes[scene_id] = [target_from_meta(split, scene_id, target) for target in meta["targets"]]
        all_targets[split] = scenes
    return all_targets


# V4.4 missed ghost 39개를 scene_id/target_id로 로드한다.
# 입력: csv_path
# 반환: MissedGhostRef 리스트
def load_missed_ghost_refs(csv_path) -> list[MissedGhostRef]:
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"V4.4 missed ghost CSV를 찾을 수 없습니다: {csv_path}")
    with Path(csv_path).open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    refs = [MissedGhostRef(scene_id=row["scene_id"], target_id=row["target_id"]) for row in rows]
    if len(refs) != 39:
        raise ValueError(f"V4.4 missed ghost는 39개여야 합니다. 현재 {len(refs)}개입니다.")
    return refs


# 두 target의 RD bin 유클리드 거리를 계산한다.
# 입력: target_a, target_b
# 반환: bin distance
def target_distance(target_a, target_b):
    return float(math.hypot(target_a.bin_d - target_b.bin_d, target_a.bin_r - target_b.bin_r))


# 두 target이 ghost-parent 관계인지 판별한다.
# 입력: target_a, target_b
# 반환: bool
def is_ghost_parent_pair(target_a, target_b):
    return (
        target_a.class_id == 1
        and target_a.parent == target_b.target_id
        or target_b.class_id == 1
        and target_b.parent == target_a.target_id
    )


# scene 내 모든 target pair 거리를 계산한다.
# 입력: scene_targets
# 반환: PairDistance 리스트
def compute_pair_distances(scene_targets) -> list[PairDistance]:
    pairs = []
    for target_a, target_b in combinations(scene_targets, 2):
        pairs.append(
            PairDistance(
                split=target_a.split,
                scene_id=target_a.scene_id,
                target_a=target_a.target_id,
                target_b=target_b.target_id,
                distance_bins=target_distance(target_a, target_b),
                is_ghost_parent_pair=is_ghost_parent_pair(target_a, target_b),
            )
        )
    return pairs


# target 중심과 box size로 가정 bbox를 만든다.
# 입력: target, box_size
# 반환: x1, y1, x2, y2
def assumed_bbox(target, box_size):
    half = box_size / 2.0
    x1 = max(0.0, target.bin_d - half)
    y1 = max(0.0, target.bin_r - half)
    x2 = min(float(BIN_SIZE), target.bin_d + half)
    y2 = min(float(BIN_SIZE), target.bin_r + half)
    return x1, y1, x2, y2


# 두 bbox의 실제 IoU를 계산한다.
# 입력: bbox_a, bbox_b
# 반환: IoU
def compute_iou(bbox_a, bbox_b):
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, bbox_a[2] - bbox_a[0]) * max(0.0, bbox_a[3] - bbox_a[1])
    area_b = max(0.0, bbox_b[2] - bbox_b[0]) * max(0.0, bbox_b[3] - bbox_b[1])
    denom = area_a + area_b - inter_area
    return float(inter_area / denom) if denom > 0.0 else 0.0


# missed ghost와 최근접 target 간 bbox IoU가 0.5 아래로 떨어지는지 계산한다.
# 입력: missed_ref, scene_targets, box_size
# 반환: bool
def is_missed_ghost_resolved(missed_ref, scene_targets, box_size):
    target_by_id = {target.target_id: target for target in scene_targets}
    ghost = target_by_id[missed_ref.target_id]
    nearest = min(
        (target for target in scene_targets if target.target_id != ghost.target_id),
        key=lambda target: target_distance(ghost, target),
    )
    iou = compute_iou(assumed_bbox(ghost, box_size), assumed_bbox(nearest, box_size))
    return iou < SEVERE_IOU_THRESHOLD


# 해당 box size에서 conflict pair / severe conflict / missed ghost 해소 개수를 계산한다.
# 입력: scene_targets, missed_ghosts, box_size
# 반환: BoxSizeResult
def evaluate_box_size(scene_targets, missed_ghosts, box_size) -> BoxSizeResult:
    all_scenes = [targets for split_scenes in scene_targets.values() for targets in split_scenes.values()]
    all_pairs = [(a, b) for targets in all_scenes for a, b in combinations(targets, 2)]
    conflict_count = 0
    severe_count = 0

    for target_a, target_b in all_pairs:
        iou = compute_iou(assumed_bbox(target_a, box_size), assumed_bbox(target_b, box_size))
        if iou > CONFLICT_IOU_THRESHOLD:
            conflict_count += 1
        if iou >= SEVERE_IOU_THRESHOLD:
            severe_count += 1

    test_scenes = scene_targets["test"]
    resolved_count = 0
    for missed_ref in missed_ghosts:
        resolved_count += int(is_missed_ghost_resolved(missed_ref, test_scenes[missed_ref.scene_id], box_size))

    total_pairs = len(all_pairs)
    total_missed = len(missed_ghosts)
    return BoxSizeResult(
        box_size=box_size,
        conflict_pair_count=conflict_count,
        conflict_pair_ratio=conflict_count / total_pairs if total_pairs else 0.0,
        severe_conflict_pair_count=severe_count,
        severe_conflict_pair_ratio=severe_count / total_pairs if total_pairs else 0.0,
        missed_ghost_resolved_count=resolved_count,
        missed_ghost_resolved_ratio=resolved_count / total_missed if total_missed else 0.0,
    )


# 거리 histogram 구간을 반환한다.
# 입력: distance
# 반환: 구간 이름
def distance_bucket(distance):
    if distance < 1.0:
        return "[0-1)"
    if distance < 2.0:
        return "[1-2)"
    if distance < 3.0:
        return "[2-3)"
    if distance < 4.0:
        return "[3-4)"
    if distance < 6.0:
        return "[4-6)"
    return "[6+)"


# split별 전체 pair와 ghost-parent pair histogram을 만든다.
# 입력: pair_distances
# 반환: csv 저장용 row 리스트, summary dict
def build_distance_distribution(pair_distances):
    rows = []
    summary = {}
    split_keys = ["all"] + SPLITS
    for split in split_keys:
        selected = pair_distances if split == "all" else [pair for pair in pair_distances if pair.split == split]
        for pair_type, pairs in [
            ("all_pairs", selected),
            ("ghost_parent_pairs", [pair for pair in selected if pair.is_ghost_parent_pair]),
        ]:
            hist = {bucket: 0 for bucket in DISTANCE_BUCKETS}
            for pair in pairs:
                hist[distance_bucket(pair.distance_bins)] += 1
            row = {"split": split, "pair_type": pair_type, "pair_count": len(pairs), **hist}
            rows.append(row)
            summary[f"{split}_{pair_type}"] = row
    return rows, summary


# 추천 규칙대로 추천 size와 근거 문장을 산출한다.
# 입력: box_size_results
# 반환: 추천 size, 근거 문장
def recommend_box_size(box_size_results) -> tuple[int, str]:
    eligible = [result for result in box_size_results if result.box_size != 1]
    best_resolved = max(result.missed_ghost_resolved_count for result in eligible)
    candidates = [result for result in eligible if result.missed_ghost_resolved_count == best_resolved]
    recommended = min(candidates, key=lambda result: result.severe_conflict_pair_ratio)

    by_size = {result.box_size: result for result in box_size_results}
    reason_parts = [
        f"{recommended.box_size}x{recommended.box_size}는 box_size=1을 제외한 후보 중 missed ghost conflict 해소 상한이 {recommended.missed_ghost_resolved_count}/39로 가장 높고 severe conflict ratio가 {recommended.severe_conflict_pair_ratio:.4f}입니다."
    ]
    if by_size[1].missed_ghost_resolved_count >= recommended.missed_ghost_resolved_count:
        reason_parts.append("1x1이 수치상 더 유리하더라도 YOLO 학습 안정성 위험 때문에 추천에서 제외했습니다.")
    if by_size[3].severe_conflict_pair_count > 0:
        reason_parts.append(
            f"3x3에서도 severe conflict가 {by_size[3].severe_conflict_pair_count}쌍 남아 2x2를 함께 비교 후보로 둬야 합니다."
        )
    reason_parts.append("missed_ghost_resolved_count는 실제 회수량이 아니라 라벨 정의상 분리 가능해지는 상한입니다.")
    return recommended.box_size, " ".join(reason_parts)


# 결과를 CSV와 JSON으로 저장한다.
# 입력: out_dir, distance_rows, box_size_results, summary
# 반환: 없음
def save_results(out_dir, distance_rows, box_size_results, summary) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    distance_path = out_dir / "pair_distance_distribution.csv"
    distance_fields = ["split", "pair_type", "pair_count"] + DISTANCE_BUCKETS
    with distance_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=distance_fields)
        writer.writeheader()
        writer.writerows(distance_rows)

    sweep_path = out_dir / "bbox_sweep_results.csv"
    with sweep_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(box_size_results[0]).keys()))
        writer.writeheader()
        for result in box_size_results:
            writer.writerow(asdict(result))

    summary_path = out_dir / "bbox_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# 콘솔 표와 추천을 출력한다.
# 입력: distance_rows, box_size_results, summary
# 반환: 없음
def print_diagnosis(distance_rows, box_size_results, summary) -> None:
    print("=" * 92)
    print("GA-YOLO V4.5 bbox size sweep diagnosis")
    print("=" * 92)
    print("[Distance histogram]")
    for row in distance_rows:
        if row["split"] in ["all", "test"]:
            buckets = " ".join(f"{bucket}:{row[bucket]}" for bucket in DISTANCE_BUCKETS)
            print(f"  {row['split']:5s} {row['pair_type']:18s} n={row['pair_count']:5d} | {buckets}")

    print("")
    print("[BBox sweep]")
    print("box | conflict_count ratio   severe_count ratio   missed_resolved ratio")
    for result in box_size_results:
        print(
            f"{result.box_size:3d} | "
            f"{result.conflict_pair_count:14d} {result.conflict_pair_ratio:6.4f} "
            f"{result.severe_conflict_pair_count:13d} {result.severe_conflict_pair_ratio:6.4f} "
            f"{result.missed_ghost_resolved_count:15d} {result.missed_ghost_resolved_ratio:6.3f}"
        )

    print("")
    print(f"recommended_box_size: {summary['recommended_box_size']}")
    print(f"recommendation_reason: {summary['recommendation_reason']}")
    print("=" * 92)


# V4.5 bbox size 진단을 실행한다.
# 입력: 없음
# 반환: 없음
def main():
    args = parse_args()
    meta_base = resolve_meta_base(args.meta_dir)
    scene_targets = load_all_targets(meta_base)
    missed_ghosts = load_missed_ghost_refs(MISSED_GHOST_CSV)

    pair_distances = []
    for split_scenes in scene_targets.values():
        for targets in split_scenes.values():
            pair_distances.extend(compute_pair_distances(targets))

    distance_rows, distance_summary = build_distance_distribution(pair_distances)
    box_size_results = [
        evaluate_box_size(scene_targets, missed_ghosts, box_size) for box_size in CANDIDATE_SIZES
    ]
    recommended_size, reason = recommend_box_size(box_size_results)
    summary = {
        "meta_base": str(meta_base),
        "missed_ghost_csv": str(MISSED_GHOST_CSV),
        "candidate_sizes": CANDIDATE_SIZES,
        "conflict_iou": "> 0",
        "severe_conflict_iou": SEVERE_IOU_THRESHOLD,
        "distance_distribution": distance_summary,
        "bbox_sweep_results": [asdict(result) for result in box_size_results],
        "recommended_box_size": recommended_size,
        "recommendation_reason": reason,
    }

    save_results(OUT_DIR, distance_rows, box_size_results, summary)
    print_diagnosis(distance_rows, box_size_results, summary)


if __name__ == "__main__":
    main()
