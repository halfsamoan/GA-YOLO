# ================================================================================
# Main Author: 김경민
# Recently Modified Date: 2026-05-24 (V5.0-light)
# Dependency: Carrada 폴더
# Description: CARRADA 데이터셋 폴더 구조 및 RD map 포맷 탐색
# ================================================================================

import json
import os

import numpy as np


CARRADA_ROOT = "/home/kmin/RD_YOLO_GHOST/carrada/Carrada"
JSON_CANDIDATES = ["labels.json", "rd_points.json", "points.json"]
ANNOTATION_SAMPLE_COUNT = 5


# 폴더가 실제로 존재하는지 먼저 확인해 이후 탐색 오류를 명확하게 만든다.
# 입력: path (확인할 폴더 경로)
# 반환: 없음
def require_directory(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {path}")


# CARRADA 루트 아래의 시퀀스 폴더만 골라 출력한다.
# 입력: carrada_root (CARRADA 루트 경로)
# 반환: 정렬된 시퀀스 폴더명 리스트
def list_sequences(carrada_root):
    require_directory(carrada_root)

    sequence_names = []
    for name in sorted(os.listdir(carrada_root)):
        path = os.path.join(carrada_root, name)
        if not os.path.isdir(path):
            continue

        has_sequence_file = os.path.isfile(os.path.join(path, "labels.json"))
        has_rd_folder = os.path.isdir(os.path.join(path, "range_doppler_numpy"))
        if has_sequence_file or has_rd_folder:
            sequence_names.append(name)

    print("=" * 80)
    print("CARRADA 시퀀스 목록")
    print("=" * 80)
    if not sequence_names:
        print("시퀀스 폴더가 없습니다.")
    for idx, sequence_name in enumerate(sequence_names, start=1):
        print(f"{idx:02d}. {sequence_name}")
    print("")

    return sequence_names


# 첫 번째 시퀀스 내부 구조를 확인해 이후 전처리 입력 위치를 정한다.
# 입력: sequence_dir (시퀀스 폴더 경로)
# 반환: 없음
def print_sequence_contents(sequence_dir):
    require_directory(sequence_dir)

    print("=" * 80)
    print(f"첫 번째 시퀀스 내부 목록: {sequence_dir}")
    print("=" * 80)
    for name in sorted(os.listdir(sequence_dir)):
        path = os.path.join(sequence_dir, name)
        kind = "DIR " if os.path.isdir(path) else "FILE"
        print(f"[{kind}] {name}")
    print("")


# RD map npy 파일을 일부만 읽어 실제 입력 배열 포맷을 확인한다.
# 입력: sequence_dir (시퀀스 폴더 경로), max_files (확인할 파일 수)
# 반환: shape 문자열 리스트
def inspect_range_doppler_numpy(sequence_dir, max_files=3):
    rd_dir = os.path.join(sequence_dir, "range_doppler_numpy")
    shape_summaries = []

    print("=" * 80)
    print("range_doppler_numpy 포맷 확인")
    print("=" * 80)

    if not os.path.isdir(rd_dir):
        print(f"range_doppler_numpy 폴더가 없습니다: {rd_dir}")
        print("")
        return shape_summaries

    npy_files = sorted(filename for filename in os.listdir(rd_dir) if filename.endswith(".npy"))
    if not npy_files:
        print(f".npy 파일이 없습니다: {rd_dir}")
        print("")
        return shape_summaries

    for filename in npy_files[:max_files]:
        path = os.path.join(rd_dir, filename)
        rd_map = np.load(path)
        shape_summaries.append(str(rd_map.shape))
        print(f"파일명: {filename}")
        print(f"  shape: {rd_map.shape}")
        print(f"  dtype: {rd_map.dtype}")
        print(f"  min  : {float(np.min(rd_map)):.6f}")
        print(f"  max  : {float(np.max(rd_map)):.6f}")
        print(f"  mean : {float(np.mean(rd_map)):.6f}")
    print("")

    return shape_summaries


# annotations 폴더 내부를 출력해 bbox/mask/sparse 라벨의 저장 방식을 확인한다.
# 입력: sequence_dir (시퀀스 폴더 경로)
# 반환: 없음
# annotations 폴더 내부를 요약해 라벨 종류와 프레임 개수를 확인한다.
# 입력: sequence_dir (시퀀스 폴더 경로)
# 반환: 없음
def inspect_annotations(sequence_dir):
    annotation_dir = os.path.join(sequence_dir, "annotations")

    print("=" * 80)
    print("annotations 폴더 구조")
    print("=" * 80)

    if not os.path.isdir(annotation_dir):
        print(f"annotations 폴더가 없습니다: {annotation_dir}")
        print("")
        return

    annotation_types = sorted(
        name
        for name in os.listdir(annotation_dir)
        if os.path.isdir(os.path.join(annotation_dir, name))
    )
    if not annotation_types:
        print("annotations 하위 폴더가 없습니다.")
        print("")
        return

    for annotation_type in annotation_types:
        type_dir = os.path.join(annotation_dir, annotation_type)
        frame_dirs = sorted(
            name
            for name in os.listdir(type_dir)
            if os.path.isdir(os.path.join(type_dir, name))
        )
        files = sorted(
            name
            for name in os.listdir(type_dir)
            if os.path.isfile(os.path.join(type_dir, name))
        )

        print(f"[DIR] annotations/{annotation_type}")
        print(f"  frame dir count: {len(frame_dirs)}")
        print(f"  file count     : {len(files)}")

        for frame_name in frame_dirs[:ANNOTATION_SAMPLE_COUNT]:
            frame_dir = os.path.join(type_dir, frame_name)
            frame_files = sorted(os.listdir(frame_dir))
            print(f"  [DIR] {annotation_type}/{frame_name}")
            for file_name in frame_files:
                print(f"    [FILE] {file_name}")

        if len(frame_dirs) > ANNOTATION_SAMPLE_COUNT:
            print(f"  ... frame dir {len(frame_dirs) - ANNOTATION_SAMPLE_COUNT}개 생략")
    print("")


# 시퀀스 루트의 JSON을 열어 CARRADA 클래스/프레임 키 구조를 확인한다.
# 입력: sequence_dir (시퀀스 폴더 경로)
# 반환: 읽은 JSON 파일명 또는 None
def inspect_sequence_json(sequence_dir):
    print("=" * 80)
    print("시퀀스 JSON 키 구조 확인")
    print("=" * 80)

    json_path = None
    for candidate in JSON_CANDIDATES:
        candidate_path = os.path.join(sequence_dir, candidate)
        if os.path.isfile(candidate_path):
            json_path = candidate_path
            break

    if json_path is None:
        print("labels.json, rd_points.json, points.json 중 존재하는 JSON이 없습니다.")
        print("")
        return None

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"JSON 파일: {json_path}")
    print_json_structure(data)
    print("")
    return os.path.basename(json_path)


# JSON 전체를 출력하지 않고 키와 타입 중심으로 요약해 구조 파악에 집중한다.
# 입력: data (JSON 객체), depth (현재 깊이), max_depth (출력할 최대 깊이)
# 반환: 없음
def print_json_structure(data, depth=0, max_depth=3):
    indent = "  " * depth

    if depth > max_depth:
        print(f"{indent}...")
        return

    if isinstance(data, dict):
        print(f"{indent}dict: {len(data)} keys")
        for idx, (key, value) in enumerate(data.items()):
            if idx >= 10:
                print(f"{indent}... 키 {len(data) - 10}개 생략")
                break
            print(f"{indent}- {key}: {type(value).__name__}")
            print_json_structure(value, depth + 1, max_depth)
    elif isinstance(data, list):
        print(f"{indent}list: {len(data)} items")
        if data:
            print_json_structure(data[0], depth + 1, max_depth)
    else:
        print(f"{indent}{repr(data)}")


# 탐색 결과를 한 번에 실행해 다음 V5.0-light 전처리 입력을 결정한다.
# 입력: 없음
# 반환: 없음
def main():
    sequence_names = list_sequences(CARRADA_ROOT)
    if not sequence_names:
        print("CARRADA 구조 탐색 완료")
        print("RD map shape 요약: 확인 불가")
        return

    first_sequence_dir = os.path.join(CARRADA_ROOT, sequence_names[0])
    print_sequence_contents(first_sequence_dir)
    shape_summaries = inspect_range_doppler_numpy(first_sequence_dir)
    inspect_annotations(first_sequence_dir)
    inspect_sequence_json(first_sequence_dir)

    print("=" * 80)
    print("CARRADA 구조 탐색 완료")
    if shape_summaries:
        print(f"RD map shape 요약: {', '.join(shape_summaries)}")
    else:
        print("RD map shape 요약: 첫 번째 시퀀스에서 확인 불가")
    print("=" * 80)


if __name__ == "__main__":
    main()
