# 딥페이크 탐지 — Xception+LSTM vs Video Swin Transformer (한글)

> 🥉 **3등 수상** — T-SUM 데이터분석 공모전, 고려대 세종 (2023, 팀 4인)

얼굴 탐지 전처리 + 두 가지 딥러닝 아키텍처 비교로 영상 단위 딥페이크 분류
end-to-end 파이프라인 구축.

## 한눈에 보기

- **Task:** 영상 단위 이진 분류 — *진짜* vs *조작(manipulated)*
- **파이프라인:**
  1. **RetinaFace**로 프레임마다 얼굴 검출 및 정렬
  2. 두 모델 학습 및 비교:
     - **Xception + LSTM** — 프레임별 CNN 특징 + 시퀀스 LSTM
     - **Video Swin Transformer** — 3D 윈도우 셀프 어텐션
  3. **프레임 단위 + 비디오 단위(다중 얼굴 통합)** 평가 + Confusion Matrix
- **환경:** Google Colab GPU

## 폴더 구조

```
deepfake-detection/
├── README.md  /  README_KR.md  (이 파일)
├── .gitignore
├── notebooks/                              메인 노트북
│   ├── 00_data_preprocessing.ipynb         얼굴 추출 (RetinaFace)
│   ├── 01_xception_lstm.ipynb              Xception+LSTM 메인 (비디오 단위 평가)
│   └── 02_video_swin_transformer.ipynb     Video Swin Transformer (Confusion Matrix)
├── _drafts/                                초기 실험 (시행착오 보존)
│   ├── xception_lstm__early_with_viz.ipynb
│   ├── video_swin__with_training_history.ipynb
│   └── video_swin__initial_experiment.ipynb
└── checkpoints/
    └── README.md                           학습 모델(~82MB)은 외부 보관
```

## 메소드

### 1. 데이터 전처리 — `00_data_preprocessing.ipynb`
- 영상 프레임마다 **RetinaFace**로 얼굴 검출.
- 얼굴 영역 crop·정렬 후 이미지 시퀀스로 저장.
- 단일/다중 얼굴 모두 처리.

### 2. Xception + LSTM — `01_xception_lstm.ipynb`
- **프레임 특징 추출기**: ImageNet 사전학습 Xception (2048차원).
- **시계열 모델**: LSTM으로 프레임 시퀀스 모델링.
- **비디오 단위 판정**: 한 영상 내 모든 얼굴 crop 통합 평가.

### 3. Video Swin Transformer — `02_video_swin_transformer.ipynb`
- 영상 볼륨에 3D 윈도우 셀프 어텐션 직접 적용.
- 사전학습 `swin3d_tiny` 백본 파인튜닝.
- Validation 셋 평가 + **Confusion Matrix 시각화**.

## 사용 기술

`Python` · `PyTorch` · `torchvision` · `retina-face` · `OpenCV` · `gdown` · `Google Colab`

## 실행 방법

```bash
# Google Colab 권장 (전처리에 Drive mount 사용)
# 1. notebooks/00_data_preprocessing.ipynb  → 얼굴 추출
# 2. notebooks/01_xception_lstm.ipynb 또는
#    notebooks/02_video_swin_transformer.ipynb → 학습/평가
```

데이터셋(manipulated/original)은 노트북 내 `gdown`으로 Google Drive에서 받음.
데이터와 학습 weight는 repo에서 제외 (`.gitignore` 참고).

## 학습된 모델

`best_model.pth` (약 82MB)는 GitHub 용량 관리를 위해 **repo에 포함하지 않음**.
요청 시 공유 — [`checkpoints/README.md`](./checkpoints/README.md) 참고.

## 수상

🥉 **3등** — T-SUM 데이터분석 공모전, 고려대 세종 (2023, 팀 4인)

---

*작성: 이선재 (Sunjae Lee) · 고려대 세종 빅데이터전공*
*[Computer-Vision](https://github.com/Vongole25/Computer-Vision) repo의 하위 프로젝트*
