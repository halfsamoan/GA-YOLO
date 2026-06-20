# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-05-26 (V5.1)
# Dependency: ga_yolo_v4_7_noisy_stress/weights/best.pt, Carrada 폴더
# Description: GA-YOLO V5.1 - V4.7 모델 CARRADA domain generalization 재평가
#              V5.0 결과(near-real 6/20)와 동일 프레임 비교가 핵심
# ================================================================================

import json
import os
import random

import cv2
import numpy as np
from ultralytics import YOLO


CARRADA_ROOT = "/home/kmin/RD_YOLO_GHOST/carrada/Carrada"
MODEL_PATH = "/home/kmin/RD_YOLO_GHOST/runs/detect/ga_yolo_v4_7_noisy_stress/weights/best.pt"
OUTPUT_DIR = "/home/kmin/RD_YOLO_GHOST/eval_v5_1"
V50_DIR = "/home/kmin/RD_YOLO_GHOST/eval_v5_0_light"
V50_BASELINE = 6

# CARRADA와 GA-YOLO는 클래스 의미가 다르므로 같은 dict로 섞어 쓰지 않는다.
GA_YOLO_NAMES = {0: "real", 1: "ghost"}
CARRADA_NAMES = {1: "pedestrian", 2: "cyclist", 3: "car"}

RANDOM_SEED = 42
NUM_SEQUENCES = 5
FRAMES_PER_SEQUENCE = 4
EXPECTED_FRAME_COUNT = 20
INPUT_SIZE = 256
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CENTER_DISTANCE_THRESHOLD = 8.0

GT_COLOR = (0, 255, 0)
PRED_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)


# 필수 폴더가 없으면 평가 대상이 달라지므로 시작 시 명확히 중단한다.
# 입력: path (확인할 폴더 경로)
# 반환: 없음
def require_directory(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {path}")


# 모델이나 RD 파일이 빠진 상태에서 조용히 실패하지 않도록 파일 존재를 확인한다.
# 입력: path (확인할 파일 경로)
# 반환: 없음
def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")


# 실제 RD map과 bbox JSON이 모두 있는 폴더만 CARRADA 평가 시퀀스로 인정한다.
# 입력: carrada_root (CARRADA 루트)
# 반환: 유효 시퀀스명 리스트
def find_valid_sequences(carrada_root):
    require_directory(carrada_root)
    valid_sequences = []

    for name in sorted(os.listdir(carrada_root)):
        sequence_dir = os.path.join(carrada_root, name)
        rd_dir = os.path.join(sequence_dir, "range_doppler_numpy")
        box_path = os.path.join(sequence_dir, "annotations", "box", "range_doppler_light.json")
        if os.path.isdir(sequence_dir) and os.path.isdir(rd_dir) and os.path.isfile(box_path):
            valid_sequences.append(name)

    if len(valid_sequences) < NUM_SEQUENCES:
        raise RuntimeError(f"유효 시퀀스가 부족합니다: {len(valid_sequences)} < {NUM_SEQUENCES}")
    return valid_sequences


# V5.0 결과 이미지 파일명에서 동일 비교 대상 프레임을 복원한다.
# 입력: v50_dir (V5.0 결과 폴더)
# 반환: (sequence_name, frame_id) 리스트 또는 None
def load_v50_frame_list(v50_dir):
    if not os.path.isdir(v50_dir):
        return None

    frame_pairs = []
    for filename in sorted(os.listdir(v50_dir)):
        if not filename.lower().endswith(".png"):
            continue
        stem = os.path.splitext(filename)[0]
        if "_" not in stem:
            continue
        sequence_name, frame_id = stem.rsplit("_", 1)
        if not sequence_name or not frame_id:
            continue
        frame_pairs.append((sequence_name, frame_id))

    if len(frame_pairs) == EXPECTED_FRAME_COUNT:
        print("V5.0 파일 기반 프레임 목록 로드 성공")
        return frame_pairs

    return None


# V5.0 결과 폴더가 없을 때만 seed=42 방식으로 프레임을 다시 선택한다.
# 입력: 없음
# 반환: (sequence_name, frame_id) 리스트
def fallback_sample_frame_list():
    print("fallback: seed=42 랜덤 선택")
    valid_sequences = find_valid_sequences(CARRADA_ROOT)
    random.seed(RANDOM_SEED)
    selected_sequences = random.sample(valid_sequences, NUM_SEQUENCES)

    frame_pairs = []
    for sequence_name in selected_sequences:
        sequence_dir = os.path.join(CARRADA_ROOT, sequence_name)
        box_data = load_box_json(sequence_dir)
        selected_frames = sample_frames(sequence_dir, box_data)
        for frame_id in selected_frames:
            frame_pairs.append((sequence_name, frame_id))

    return frame_pairs


# V5.1 평가 대상 20프레임을 V5.0 결과와 동일하게 고정한다.
# 입력: 없음
# 반환: (sequence_name, frame_id) 리스트
def load_eval_frame_list():
    frame_pairs = load_v50_frame_list(V50_DIR)
    if frame_pairs is not None:
        return frame_pairs
    return fallback_sample_frame_list()


# CARRADA RD bbox JSON을 읽어 프레임별 GT를 가져온다.
# 입력: sequence_dir (시퀀스 폴더)
# 반환: JSON dict
def load_box_json(sequence_dir):
    box_path = os.path.join(sequence_dir, "annotations", "box", "range_doppler_light.json")
    require_file(box_path)
    with open(box_path, "r", encoding="utf-8") as file:
        return json.load(file)


# fallback 선택에서 GT가 있고 npy가 있는 프레임만 뽑아 V5.0 방식과 맞춘다.
# 입력: sequence_dir, box_data
# 반환: 프레임 id 리스트
def sample_frames(sequence_dir, box_data):
    rd_dir = os.path.join(sequence_dir, "range_doppler_numpy")
    require_directory(rd_dir)

    candidates = []
    for frame_id in sorted(box_data.keys()):
        rd_path = os.path.join(rd_dir, f"{frame_id}.npy")
        if os.path.isfile(rd_path) and box_data.get(frame_id, {}).get("boxes"):
            candidates.append(frame_id)

    if len(candidates) < FRAMES_PER_SEQUENCE:
        raise RuntimeError(f"프레임 후보가 부족합니다: {sequence_dir}, {len(candidates)}개")

    random.seed(RANDOM_SEED)
    return random.sample(candidates, FRAMES_PER_SEQUENCE)


# CARRADA bbox는 [range1, doppler1, range2, doppler2]이므로 이미지 x/y로 바꿔야 한다.
# 입력: box (CARRADA bbox), rd_shape (원본 RD shape)
# 반환: 256x256 기준 x1, y1, x2, y2
def scale_carrada_box_to_input(box, rd_shape):
    if len(box) != 4:
        raise ValueError(f"bbox 길이가 4가 아닙니다: {box}")

    height, width = rd_shape[:2]
    range1, doppler1, range2, doppler2 = [float(value) for value in box]
    x_scale = INPUT_SIZE / float(width)
    y_scale = INPUT_SIZE / float(height)

    x1 = int(round(doppler1 * x_scale))
    y1 = int(round(range1 * y_scale))
    x2 = int(round(doppler2 * x_scale))
    y2 = int(round(range2 * y_scale))

    x1 = int(np.clip(x1, 0, INPUT_SIZE - 1))
    y1 = int(np.clip(y1, 0, INPUT_SIZE - 1))
    x2 = int(np.clip(x2, 0, INPUT_SIZE - 1))
    y2 = int(np.clip(y2, 0, INPUT_SIZE - 1))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


# 해당 프레임의 CARRADA GT bbox를 모델 입력 좌표계로 불러온다.
# 입력: box_data, frame_id, rd_shape
# 반환: GT bbox dict 리스트
def load_gt_boxes_for_frame(box_data, frame_id, rd_shape):
    frame_item = box_data.get(frame_id, {})
    boxes = frame_item.get("boxes", [])
    labels = frame_item.get("labels", [])
    gt_boxes = []

    for box, label in zip(boxes, labels):
        label_id = int(label)
        gt_boxes.append(
            {
                "class_id": label_id,
                "class_name": CARRADA_NAMES.get(label_id, f"class{label_id}"),
                "xyxy": scale_carrada_box_to_input(box, rd_shape),
            }
        )

    return gt_boxes


# CARRADA RD map을 V4.7 모델 입력 크기와 채널 수에 맞춘다.
# 입력: rd_path (.npy 경로)
# 반환: BGR 이미지, 원본 RD shape
def load_and_preprocess_rd_map(rd_path):
    require_file(rd_path)
    rd_map = np.load(rd_path)
    if rd_map.ndim != 2:
        raise ValueError(f"RD map은 2D 배열이어야 합니다: {rd_path}, shape={rd_map.shape}")

    rd_min = float(np.min(rd_map))
    rd_max = float(np.max(rd_map))
    if rd_max - rd_min < 1e-12:
        normalized = np.zeros_like(rd_map, dtype=np.uint8)
    else:
        normalized = ((rd_map - rd_min) / (rd_max - rd_min) * 255.0).astype(np.uint8)

    resized = cv2.resize(normalized, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR), rd_map.shape


# 두 bbox가 겹치는지 확인해 real 예측이 GT 근처에 반응했는지 본다.
# 입력: box1, box2 (x1, y1, x2, y2)
# 반환: IoU 값
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    denom = area1 + area2 - inter_area
    if denom <= 0:
        return 0.0
    return float(inter_area / denom)


# 작은 bbox에서 IoU가 0이 되는 경우를 보조적으로 분리해 확인한다.
# 입력: box1, box2 (x1, y1, x2, y2)
# 반환: 중심거리(px)
def compute_center_distance(box1, box2):
    cx1 = (box1[0] + box1[2]) / 2.0
    cy1 = (box1[1] + box1[3]) / 2.0
    cx2 = (box2[0] + box2[2]) / 2.0
    cy2 = (box2[1] + box2[3]) / 2.0
    return float(np.hypot(cx1 - cx2, cy1 - cy2))


# YOLO 결과에서 class별 예측 개수와 bbox를 추출한다.
# 입력: result (ultralytics 결과)
# 반환: predictions, pred_real_count, pred_ghost_count
def parse_predictions(result):
    predictions = []
    pred_real_count = 0
    pred_ghost_count = 0

    if result.boxes is None or len(result.boxes) == 0:
        return predictions, pred_real_count, pred_ghost_count

    xyxy_boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    conf_scores = result.boxes.conf.cpu().numpy()

    for xyxy, class_id, conf_score in zip(xyxy_boxes, class_ids, conf_scores):
        class_id = int(class_id)
        predictions.append(
            {
                "class_id": class_id,
                "class_name": GA_YOLO_NAMES.get(class_id, f"class{class_id}"),
                "conf": float(conf_score),
                "xyxy": tuple(int(round(float(value))) for value in xyxy),
            }
        )
        if class_id == 0:
            pred_real_count += 1
        elif class_id == 1:
            pred_ghost_count += 1

    return predictions, pred_real_count, pred_ghost_count


# real 예측이 GT와 겹치거나 매우 가까운지 확인한다.
# 입력: gt_boxes, predictions
# 반환: IoU hit 여부, center hit 여부
def judge_near_real_frame(gt_boxes, predictions):
    real_predictions = [pred for pred in predictions if pred["class_id"] == 0]
    iou_hit = False
    center_hit = False

    for gt_box in gt_boxes:
        for pred in real_predictions:
            iou = compute_iou(gt_box["xyxy"], pred["xyxy"])
            distance = compute_center_distance(gt_box["xyxy"], pred["xyxy"])
            if iou > 0.0:
                iou_hit = True
            if iou <= 0.0 and distance <= CENTER_DISTANCE_THRESHOLD:
                center_hit = True

    return iou_hit, center_hit


# RD map 위에서 텍스트가 묻히지 않도록 배경 박스와 함께 라벨을 그린다.
# 입력: image, text, x, y, color
# 반환: 없음
def draw_label(image, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    label_y = max(y - 6, text_height + 6)

    bg_x1 = int(np.clip(x, 0, INPUT_SIZE - 1))
    bg_y1 = int(np.clip(label_y - text_height - baseline - 4, 0, INPUT_SIZE - 1))
    bg_x2 = int(np.clip(x + text_width + 6, 0, INPUT_SIZE - 1))
    bg_y2 = int(np.clip(label_y + baseline, 0, INPUT_SIZE - 1))
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.putText(image, text, (bg_x1 + 3, max(bg_y2 - baseline - 2, 10)), font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)


# GT와 예측을 한 이미지에 그려 V4.7 모델의 실제 RD 반응을 정성 확인한다.
# 입력: image, gt_boxes, predictions
# 반환: overlay 이미지
def draw_overlay(image, gt_boxes, predictions):
    overlay = image.copy()

    for gt_box in gt_boxes:
        x1, y1, x2, y2 = gt_box["xyxy"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), GT_COLOR, 2)
        draw_label(overlay, f"GT {gt_box['class_name']}", x1, y1, GT_COLOR)

    for pred in predictions:
        x1, y1, x2, y2 = pred["xyxy"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), PRED_COLOR, 2)
        draw_label(overlay, f"P {pred['class_name']} {pred['conf']:.2f}", x1, y1, PRED_COLOR)

    return overlay


# 한 프레임을 평가하고 overlay 이미지를 저장한다.
# 입력: model, sequence_name, frame_id, box_cache
# 반환: 프레임 결과 dict
def evaluate_frame(model, sequence_name, frame_id, box_cache):
    sequence_dir = os.path.join(CARRADA_ROOT, sequence_name)
    rd_path = os.path.join(sequence_dir, "range_doppler_numpy", f"{frame_id}.npy")
    image, rd_shape = load_and_preprocess_rd_map(rd_path)

    if sequence_name not in box_cache:
        box_cache[sequence_name] = load_box_json(sequence_dir)
    gt_boxes = load_gt_boxes_for_frame(box_cache[sequence_name], frame_id, rd_shape)

    # CARRADA에는 ghost GT가 없으므로 ghost 예측은 오탐 사례로만 기록한다.
    results = model.predict(image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=INPUT_SIZE, verbose=False)
    predictions, pred_real_count, pred_ghost_count = parse_predictions(results[0])
    iou_hit, center_hit = judge_near_real_frame(gt_boxes, predictions)

    overlay = draw_overlay(image, gt_boxes, predictions)
    output_path = os.path.join(OUTPUT_DIR, f"{sequence_name}_{frame_id}.png")
    cv2.imwrite(output_path, overlay)

    print(
        f"{sequence_name} / {frame_id} / "
        f"GT {len(gt_boxes)}개 / "
        f"Pred real {pred_real_count}개 / "
        f"Pred ghost {pred_ghost_count}개 / "
        f"near-real IoU={iou_hit}, center={center_hit}"
    )

    return {
        "sequence": sequence_name,
        "frame_id": frame_id,
        "gt_count": len(gt_boxes),
        "pred_real_count": pred_real_count,
        "pred_ghost_count": pred_ghost_count,
        "near_real_iou": iou_hit,
        "near_real_center": center_hit,
    }


# V5.0 기준과 V5.1 결과를 같은 프레임 수 기준으로 비교해 출력한다.
# 입력: frame_results
# 반환: 없음
def print_summary(frame_results):
    total_frames = len(frame_results)
    total_pred_real = sum(item["pred_real_count"] for item in frame_results)
    total_pred_ghost = sum(item["pred_ghost_count"] for item in frame_results)
    near_real_iou = sum(1 for item in frame_results if item["near_real_iou"])
    near_real_center = sum(1 for item in frame_results if item["near_real_center"])

    print("=" * 80)
    print("GA-YOLO V5.1 CARRADA 정성평가 요약")
    print("=" * 80)
    print(f"총 처리 프레임 수                : {total_frames}")
    print(f"총 Pred real 수                  : {total_pred_real}")
    print(f"총 Pred ghost 수                 : {total_pred_ghost}")
    print(f"near-real frame 수 (IoU 기준)    : {near_real_iou}")
    print(f"near-real frame 수 (center 보조) : {near_real_center}")
    print("")
    print(f"V5.0 baseline : {V50_BASELINE}/20")
    print(f"V5.1 result   : {near_real_iou}/20")
    print(f"변화           : {near_real_iou - V50_BASELINE:+d} frames")
    print("주의: CARRADA에는 ghost GT가 없으므로 ghost precision/recall은 계산하지 않음")
    print("V5.1 정성평가 완료")
    print("=" * 80)


# V4.7 모델로 V5.0 동일 프레임 CARRADA 재평가를 수행한다.
# 입력: 없음
# 반환: 없음
def main():
    require_directory(CARRADA_ROOT)
    require_file(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frame_pairs = load_eval_frame_list()
    if len(frame_pairs) != EXPECTED_FRAME_COUNT:
        raise RuntimeError(f"평가 프레임 수가 20장이 아닙니다: {len(frame_pairs)}")

    print("=" * 80)
    print("V5.1 평가 대상 프레임")
    print("=" * 80)
    for sequence_name, frame_id in frame_pairs:
        print(f"{sequence_name} / {frame_id}")
    print("=" * 80)

    model = YOLO(MODEL_PATH)
    box_cache = {}
    frame_results = []
    for sequence_name, frame_id in frame_pairs:
        frame_results.append(evaluate_frame(model, sequence_name, frame_id, box_cache))

    print_summary(frame_results)


if __name__ == "__main__":
    main()
