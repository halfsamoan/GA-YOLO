# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-05-24 (V5.0-light)
# Dependency: best.pt, Carrada 폴더
# Description: GA-YOLO V5.0-light - CARRADA RD map domain generalization 정성평가
# ================================================================================

import json
import os
import random

import cv2
import numpy as np
from ultralytics import YOLO


CARRADA_ROOT = "/home/kmin/RD_YOLO_GHOST/carrada/Carrada"
MODEL_PATH = "/home/kmin/RD_YOLO_GHOST/runs/detect/ga_yolo_v4_1_A_vir_n/weights/best.pt"
OUTPUT_DIR = "/home/kmin/RD_YOLO_GHOST/eval_v5_0_light"

# 두 데이터셋의 클래스 정의가 다르므로 평가/시각화에서 절대 섞지 않는다.
GA_YOLO_NAMES = {0: "real", 1: "ghost"}
CARRADA_NAMES = {1: "pedestrian", 2: "cyclist", 3: "car"}

RANDOM_SEED = 42
NUM_SEQUENCES = 5
FRAMES_PER_SEQUENCE = 4
INPUT_SIZE = 256
RD_HEIGHT = 256
RD_WIDTH = 64
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CENTER_DISTANCE_THRESHOLD = 8.0

GT_COLOR = (0, 255, 0)
PRED_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)


# 필수 폴더가 없으면 이후 오류 원인을 추적하기 어렵기 때문에 초기에 확인한다.
# 입력: path (확인할 폴더 경로)
# 반환: 없음
def require_directory(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {path}")


# 필수 파일이 없으면 평가 조건이 달라지므로 실행 전에 명확히 중단한다.
# 입력: path (확인할 파일 경로)
# 반환: 없음
def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")


# 유효 시퀀스만 골라 실제 RD map과 RD bbox가 함께 있는 장면만 평가한다.
# 입력: carrada_root (CARRADA 루트 경로)
# 반환: 유효 시퀀스명 리스트
def find_valid_sequences(carrada_root):
    require_directory(carrada_root)

    valid_sequences = []
    for name in sorted(os.listdir(carrada_root)):
        sequence_dir = os.path.join(carrada_root, name)
        rd_dir = os.path.join(sequence_dir, "range_doppler_numpy")
        box_json_path = os.path.join(sequence_dir, "annotations", "box", "range_doppler_light.json")

        if os.path.isdir(sequence_dir) and os.path.isdir(rd_dir) and os.path.isfile(box_json_path):
            valid_sequences.append(name)

    if len(valid_sequences) < NUM_SEQUENCES:
        raise RuntimeError(f"유효 시퀀스가 부족합니다: {len(valid_sequences)} < {NUM_SEQUENCES}")

    return valid_sequences


# 고정 seed로 시퀀스를 뽑아 이후 결과가 다시 실행해도 같게 만든다.
# 입력: sequence_names (유효 시퀀스명 리스트)
# 반환: 선택된 시퀀스명 리스트
def sample_sequences(sequence_names):
    random.seed(RANDOM_SEED)
    return random.sample(sequence_names, NUM_SEQUENCES)


# JSON을 읽어 프레임별 GT bbox 정보를 가져온다.
# 입력: sequence_dir (시퀀스 폴더 경로)
# 반환: range_doppler_light.json dict
def load_box_json(sequence_dir):
    box_json_path = os.path.join(sequence_dir, "annotations", "box", "range_doppler_light.json")
    require_file(box_json_path)

    with open(box_json_path, "r", encoding="utf-8") as file:
        return json.load(file)


# 실제 JSON 키 구조를 콘솔에 남겨 CARRADA 라벨 해석을 추적 가능하게 한다.
# 입력: sequence_name (시퀀스명), box_data (bbox JSON dict)
# 반환: 없음
def print_box_json_structure(sequence_name, box_data):
    print("=" * 80)
    print("CARRADA bbox JSON 구조 확인")
    print("=" * 80)
    print(f"시퀀스: {sequence_name}")
    print(f"프레임 키 수: {len(box_data)}")

    for frame_id in sorted(box_data.keys())[:2]:
        frame_item = box_data.get(frame_id, {})
        print(f"frame {frame_id}: keys={list(frame_item.keys())}")
        print(f"  boxes : {frame_item.get('boxes', [])[:2]}")
        print(f"  labels: {frame_item.get('labels', [])[:2]}")
    print("=" * 80)


# CARRADA RD bbox를 원본 크기에서 모델 입력 크기로 변환한다.
# 입력: box (CARRADA bbox 리스트), rd_shape (원본 RD shape)
# 반환: 256x256 기준 x1, y1, x2, y2
def scale_carrada_box_to_input(box, rd_shape):
    if len(box) != 4:
        raise ValueError(f"bbox 길이가 4가 아닙니다: {box}")

    height, width = rd_shape[:2]
    a, b, c, d = [float(value) for value in box]

    # CARRADA range_doppler_light.json은 [range1, doppler1, range2, doppler2] 순서다.
    # range 값이 64보다 작은 근거리 프레임도 있으므로 자동 감지로 바꾸면 GT가 뒤틀릴 수 있다.
    y1_raw, x1_raw, y2_raw, x2_raw = a, b, c, d

    x_scale = INPUT_SIZE / float(width)
    y_scale = INPUT_SIZE / float(height)
    x1 = int(round(x1_raw * x_scale))
    y1 = int(round(y1_raw * y_scale))
    x2 = int(round(x2_raw * x_scale))
    y2 = int(round(y2_raw * y_scale))

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
# 입력: box_data (bbox JSON dict), frame_id (프레임 번호), rd_shape (원본 RD shape)
# 반환: GT bbox dict 리스트
def load_gt_boxes_for_frame(box_data, frame_id, rd_shape):
    frame_item = box_data.get(frame_id, {})
    boxes = frame_item.get("boxes", [])
    labels = frame_item.get("labels", [])

    if not boxes:
        return []

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


# 각 시퀀스에서 bbox와 npy가 모두 있는 프레임만 뽑아 GT 없는 장면 과다 선택을 피한다.
# 입력: sequence_dir (시퀀스 폴더 경로), box_data (bbox JSON dict)
# 반환: 선택된 프레임 id 리스트
def sample_frames(sequence_dir, box_data):
    rd_dir = os.path.join(sequence_dir, "range_doppler_numpy")
    require_directory(rd_dir)

    candidate_frames = []
    for frame_id in sorted(box_data.keys()):
        rd_path = os.path.join(rd_dir, f"{frame_id}.npy")
        frame_item = box_data.get(frame_id, {})
        if os.path.isfile(rd_path) and frame_item.get("boxes"):
            candidate_frames.append(frame_id)

    if len(candidate_frames) < FRAMES_PER_SEQUENCE:
        raise RuntimeError(f"프레임 후보가 부족합니다: {sequence_dir}, {len(candidate_frames)}개")

    random.seed(RANDOM_SEED)
    return random.sample(candidate_frames, FRAMES_PER_SEQUENCE)


# CARRADA RD map을 GA-YOLO 학습 입력 형식과 같은 256x256 3채널 이미지로 바꾼다.
# 입력: rd_path (.npy 파일 경로)
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
    bgr_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return bgr_image, rd_map.shape


# 두 bbox의 IoU를 계산해 GT 근처 real 예측 여부를 보수적으로 판단한다.
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


# 중심 거리 기준은 작은 bbox에서 IoU가 0이 되는 경우를 보조적으로 보기 위해 둔다.
# 입력: box1, box2 (x1, y1, x2, y2)
# 반환: 중심 거리(px)
def compute_center_distance(box1, box2):
    cx1 = (box1[0] + box1[2]) / 2.0
    cy1 = (box1[1] + box1[3]) / 2.0
    cx2 = (box2[0] + box2[2]) / 2.0
    cy2 = (box2[1] + box2[3]) / 2.0
    return float(np.hypot(cx1 - cx2, cy1 - cy2))


# 예측 결과에서 real/ghost bbox와 개수를 분리한다.
# 입력: result (ultralytics 결과 객체)
# 반환: 예측 bbox dict 리스트, real 개수, ghost 개수
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
        box = tuple(int(round(float(value))) for value in xyxy)
        class_name = GA_YOLO_NAMES.get(int(class_id), f"class{int(class_id)}")
        predictions.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "conf": float(conf_score),
                "xyxy": box,
            }
        )

        if int(class_id) == 0:
            pred_real_count += 1
        elif int(class_id) == 1:
            pred_ghost_count += 1

    return predictions, pred_real_count, pred_ghost_count


# real 예측이 CARRADA GT 근처에 있는지 IoU와 중심거리 기준으로 나눠 판단한다.
# 입력: gt_boxes (GT 리스트), predictions (예측 리스트)
# 반환: IoU 기준 여부, 중심거리 보조 기준 여부
def judge_near_real_frame(gt_boxes, predictions):
    real_predictions = [pred for pred in predictions if pred["class_id"] == 0]
    iou_hit = False
    center_hit = False

    for gt_box in gt_boxes:
        gt_xyxy = gt_box["xyxy"]
        for pred in real_predictions:
            pred_xyxy = pred["xyxy"]
            iou = compute_iou(gt_xyxy, pred_xyxy)
            distance = compute_center_distance(gt_xyxy, pred_xyxy)

            if iou > 0.0:
                iou_hit = True
            if iou <= 0.0 and distance <= CENTER_DISTANCE_THRESHOLD:
                center_hit = True

    return iou_hit, center_hit


# 박스 라벨을 배경과 함께 그려 RD map 위에서도 텍스트가 읽히게 한다.
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


# GT와 예측 bbox를 한 이미지에 함께 그려 domain generalization을 눈으로 확인한다.
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


# 한 프레임을 처리해 저장 이미지와 집계 정보를 만든다.
# 입력: model, sequence_name, frame_id, box_data
# 반환: 프레임별 결과 dict
def evaluate_frame(model, sequence_name, frame_id, box_data):
    sequence_dir = os.path.join(CARRADA_ROOT, sequence_name)
    rd_path = os.path.join(sequence_dir, "range_doppler_numpy", f"{frame_id}.npy")
    image, rd_shape = load_and_preprocess_rd_map(rd_path)
    gt_boxes = load_gt_boxes_for_frame(box_data, frame_id, rd_shape)

    # CARRADA에는 ghost GT가 없으므로 여기서는 real/ghost 정량 평가를 하지 않는다.
    # ghost 예측은 실제 RD domain에서의 ghost-like 오탐 사례로만 기록한다.
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
        "output_path": output_path,
    }


# 전체 요약을 출력해 CARRADA 정성 확인 결과를 한눈에 볼 수 있게 한다.
# 입력: frame_results (프레임별 결과 리스트)
# 반환: 없음
def print_summary(frame_results):
    total_frames = len(frame_results)
    total_pred_real = sum(item["pred_real_count"] for item in frame_results)
    total_pred_ghost = sum(item["pred_ghost_count"] for item in frame_results)
    near_real_iou = sum(1 for item in frame_results if item["near_real_iou"])
    near_real_center = sum(1 for item in frame_results if item["near_real_center"])

    print("=" * 80)
    print("GA-YOLO V5.0-light CARRADA 정성평가 요약")
    print("=" * 80)
    print(f"총 처리 프레임 수                  : {total_frames}")
    print(f"총 Pred real 수                    : {total_pred_real}")
    print(f"총 Pred ghost 수                   : {total_pred_ghost}")
    print(f"near-real frame 수 (IoU 기준)      : {near_real_iou}")
    print(f"near-real frame 수 (center 보조)   : {near_real_center}")
    print("주의: CARRADA에는 ghost GT가 없으므로 ghost precision/recall은 계산하지 않음")
    print("V5.0-light 정성평가 완료")
    print("=" * 80)


# V5.0-light 전체 절차를 실행한다.
# 입력: 없음
# 반환: 없음
def main():
    require_directory(CARRADA_ROOT)
    require_file(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid_sequences = find_valid_sequences(CARRADA_ROOT)
    selected_sequences = sample_sequences(valid_sequences)

    print("=" * 80)
    print("선택된 CARRADA 시퀀스")
    print("=" * 80)
    for sequence_name in selected_sequences:
        print(sequence_name)
    print("=" * 80)

    model = YOLO(MODEL_PATH)
    frame_results = []

    for sequence_idx, sequence_name in enumerate(selected_sequences):
        sequence_dir = os.path.join(CARRADA_ROOT, sequence_name)
        box_data = load_box_json(sequence_dir)
        if sequence_idx == 0:
            print_box_json_structure(sequence_name, box_data)

        selected_frames = sample_frames(sequence_dir, box_data)
        for frame_id in selected_frames:
            frame_results.append(evaluate_frame(model, sequence_name, frame_id, box_data))

    print_summary(frame_results)


if __name__ == "__main__":
    main()
