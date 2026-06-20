# GA-YOLO 기술 레퍼런스 · Technical Context

> 메인 문서는 저장소 루트의 [`README.md`](../README.md)입니다. 이 문서는 파라미터·버전 히스토리·연구 원칙을 정리한 부록입니다.

---

## 1. 연구 개요

- **주제:** FMCW 자동차 레이더 Range-Doppler(RD) map에서 real / ghost 표적 직접 분류
- **모델명:** Ghost-Aware YOLO (GA-YOLO)
- **핵심 신규성:** 기존 연구가 point cloud · tracking 단계에서 ghost를 처리하는 것과 달리, **RD map 영상 단계에서 직접** 분류
- **데이터 전략:** 공개 ghost 라벨 데이터셋이 없으므로 77 GHz FMCW 레이더를 물리 시뮬레이션해 라벨 포함 데이터셋 자동 생성

---

## 2. FMCW 레이더 파라미터 (확정값)

| 기호 | 값 | 단위 | 설명 |
|---|---|---|---|
| `fc` | 77e9 | Hz | 반송파 주파수 (77 GHz 자동차 레이더) |
| `BW` | 150e6 | Hz | chirp 대역폭 |
| `Tc` | 50e-6 | s | chirp 지속시간 |
| `N_samples` | 256 | — | chirp당 ADC 샘플 수 (fast-time) |
| `N_chirps` | 128 | — | chirp 개수 (slow-time) |
| `S` | BW/Tc = 3e12 | Hz/s | chirp slope |
| `fs` | N_samples/Tc = 5.12e6 | Hz | 샘플링 주파수 |
| `λ` | c/fc ≈ 3.9 | mm | 파장 |

**성능 지표**

| 항목 | 값 |
|---|---|
| R_max (최대 거리) | 256 m |
| range_res (거리 해상도) | 1.0 m |
| v_max (최대 속도) | ≈ 19.5 m/s |
| v_res (속도 해상도) | ≈ 0.3 m/s |

---

## 3. 버전 히스토리 · Roadmap

| 버전 | 상태 | 내용 |
|---|---|---|
| V1.0 | ✅ | 단일 chirp + 정지 표적 → Range FFT (거리 오차 0.000 m 검증) |
| V1.1 | ✅ | 이동 표적 + 128 chirp → RD map 생성 |
| V1.2 | ✅ | 다중 표적 + 2D CA-CFAR |
| V2.0 | ✅ | 가드레일 + Mirror ghost (4-bounce) |
| V2.1 | ✅ | Multipath ghost (3-bounce) |
| V2.2 | ✅ | Speckle / clutter noise + zero-Doppler notch filter |
| V2.3 | ✅ | CARRADA 실데이터 노이즈 통계 추출 (시뮬레이터 보정용) |
| V3.0 | ✅ | YOLO 학습용 데이터셋 자동 생성 (2,000장) |
| V3.1 | ✅ | CARRADA 통계 기반 domain randomization 데이터셋 |
| V3.2 | ✅ | negative clutter scene 추가 (FP 제어) |
| V4.0 | ✅ | YOLOv8n 학습 (baseline mAP@50 = 0.968) |
| V4.1 | ✅ | Ablation: colormap(viridis/gray) × 모델(n/s) |
| V4.2–V4.6 | ✅ | 정성평가 · NMS sweep · bbox 크기 sweep · 오류 분석 |
| V4.7 | ❌ | noisy-stress 재학습 → False Positive 폭증 (실패로 기록) |
| V4.8 | 🔲 | negative scene 포함 재학습 (FP fix) — 실데이터 재검증 대기 |
| V5.0 | ⚠️ | CARRADA 정성평가 → 도메인 갭 발견 (20프레임 중 6프레임 near-real) |
| V5.1 | ✅ | 동일 프레임 기준 재평가 |
| V5.2 | 🔲 | V4.8 모델 CARRADA 재평가 (예정) |

---

## 4. Ghost 생성 원리

**Mirror ghost (4-bounce)**
- 경로: 레이더 → 가드레일 → 차량 → 가드레일 → 레이더
- 특징: range가 real보다 멀고, Doppler는 real과 유사

**Multipath ghost (3-bounce)**
- 경로: 레이더 → 차량 → 가드레일 → 레이더
- 특징: real에서 range offset만큼 늘어난 위치에 출현

---

## 5. 클래스 정의

```
0 = real_target   (실제 표적)
1 = ghost_target  (유령 표적)
```

> CARRADA 데이터셋은 클래스 의미가 다릅니다(1=pedestrian, 2=cyclist, 3=car). 두 체계를 절대 같은 매핑으로 혼용하지 않습니다.

---

## 6. 코드 작성 규칙

모든 소스 파일 상단에 표준 헤더를 둡니다.

```python
# ================================================================================
# Main Author: 김경민
# Recently Modified Date: YYYY-MM-DD (VX.X)
# Dependency: [의존 파일명 또는 None]
# Description: [이 파일이 하는 일을 한국어로 한 줄 요약]
# ================================================================================
```

- 함수 주석: 한 줄 요약 / 입력(단위·타입) / 반환
- 인라인 주석: 모든 주석 한국어, "무엇을"이 아니라 "왜"를 설명
- 버전 규칙: 최초 V1.0 → 기능 추가 V1.1, V1.2 → 대규모 수정 V2.0

---

## 7. 연구 윤리 · Research Integrity

실험 결과를 좋게 보이게 하기 위해 가중치·임계값·평가 기준·라벨·샘플 선택·통계 집계 방식을 임의로 조작하지 않습니다. 특히 다음을 금지합니다.

- 사전 명시 없는 class/sample/loss weight를 결과 개선 목적으로 임의 적용
- 실패 샘플 제외 또는 특정 scene만 골라 평가하는 cherry-picking
- conf / IoU / NMS / near-miss 기준을 결과가 좋아지는 방향으로 사후 변경
- GT label · bbox 크기 · class id · 중복 제거 기준의 실험 후 임의 수정
- PASS/FAIL 기준 완화 또는 실패 원인을 숨기는 로그 수정

임계값·가중치를 변경해야 할 때는 **변경 전/후 값, 이유, 시점, 영향 파일, baseline 비교**를 반드시 함께 기록합니다. 분석에서는 추정과 확정을 구분하여, 실험으로 분리되지 않은 원인은 "확정" 대신 "가설 / 추가 검증 필요"로 표현합니다.

최종 보고에는 유리한 결과뿐 아니라 FP · FN · class_error · near_miss · 실패 scene을 함께 기록합니다. GA-YOLO의 목표는 수치를 예쁘게 만드는 것이 아니라 **RD map에서 real/ghost를 물리적으로 타당하게 검출**하는 것입니다.
