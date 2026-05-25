# 결점두 EfficientNet 분류 (한글)

> 🥈 **2등 수상** — T-SUM 데이터분석 공모전, 고려대 세종 (2023, 팀 4인)

EfficientNet-B0 기반으로 **결점두 vs 정상두**를 분류. 도메인 특화 전처리
(배경 제거 + 정사각 패딩 + 리사이즈) 파이프라인 + 하이퍼파라미터 그리드 서치 +
ROC/AUC 평가까지 end-to-end.

## 한눈에 보기

- **Task:** 정상 vs 결점 커피콩 이진 분류
- **커스텀 전처리** (`utils/img_wrapper.py`):
  - `rembg`로 배경 제거
  - **정사각 패딩** (가로세로 비율 유지, 왜곡 방지)
  - 224×224 리사이즈
- **모델:** ImageNet 사전학습 EfficientNet-B0 파인튜닝
- **학습:** 하이퍼파라미터 그리드 서치 (LR, batch size, epoch)
- **평가:** Confusion Matrix + Accuracy + ROC curve + AUC

## 폴더 구조

```
coffee-bean-classification/
├── README.md / README_KR.md (이 파일)
├── .gitignore
├── notebooks/
│   └── 01_efficientnet_training.ipynb   메인: 모델 학습 + 그리드 서치 + 평가
├── utils/
│   ├── __init__.py
│   └── img_wrapper.py                   커스텀 전처리 클래스
└── _drafts/
    ├── README.md
    ├── preprocessing_test.ipynb         ImgWrapper 미리보기 검증
    └── data_unzipper.ipynb              데이터 압축 해제 유틸
```

## 전처리 파이프라인

```python
from utils import ImgWrapper

wrapper = ImgWrapper(height=224, width=224, channels=4)
processed = wrapper(input_pil_image)
# → 배경 제거 → 정사각 패딩 → 224×224 리사이즈
```

설계 선택:
- **배경 제거:** 콩이 다양한 배경에서 촬영됨. `rembg` (U²-Net 기반)로 콩만 분리.
- **정사각 패딩 (stretch X):** 콩 비율이 제각각이라, 패딩으로 형태 정보 보존.
- **RGBA 4채널:** rembg의 투명도 마스크를 함께 활용.

## 모델 학습 (`notebooks/01_efficientnet_training.ipynb`)

- **백본:** EfficientNet-B0, ImageNet 사전학습.
- **그리드 서치:** LR / 배치 사이즈 / 에폭 조합 탐색 후 best config 선택.
- **평가:** Confusion Matrix, Accuracy, ROC, AUC.

## 사용 기술

`Python` · `PyTorch` · `torchvision` · `rembg` (onnxruntime) ·
`scikit-learn` · `seaborn` · `Google Colab`

## 실행 방법

```bash
# Google Colab 권장 (rembg 의존성)
# 1. _drafts/data_unzipper.ipynb   (선택) — 데이터 압축 해제
# 2. _drafts/preprocessing_test.ipynb (선택) — ImgWrapper 미리보기
# 3. notebooks/01_efficientnet_training.ipynb — 학습 + 평가
```

데이터셋(결점두 이미지)은 repo에 포함하지 않음 (`.gitignore` 참고).

## 수상

🥈 **2등** — T-SUM 데이터분석 공모전, 고려대 세종 (2023, 팀 4인)

---

*작성: 이선재 (Sunjae Lee) · 고려대 세종 빅데이터전공*
*[Computer-Vision](https://github.com/Vongole25/Computer-Vision) repo의 하위 프로젝트*
