# MetaTST Practice

논문 "Metadata Matters for Time Series: Informative Forecasting with Transformers"를
직접 실습해보기 위한 스타터 프로젝트입니다.

이 저장소는 다음 흐름을 빠르게 확인하는데  초점을 둡니다.

- 시계열 context window 구성
- categorical / real metadata 인코딩
- metadata-conditioned transformer forecasting
- 간단한 학습 및 checkpoint 저장

## Structure

```text
metatst-practice/
  data/
  notebooks/
  src/
    dataset.py
    metadata.py
    model.py
  train.py
  requirements.txt
```

파일 역할:

- `src/dataset.py`: 메타데이터가 예측에 도움이 되도록 synthetic 시계열을 생성합니다.
- `src/metadata.py`: categorical / real metadata를 하나의 dense vector로 인코딩합니다.
- `src/model.py`: 시계열 토큰과 metadata vector를 함께 쓰는 Transformer 예측 모델입니다.
- `train.py`: 데이터로더 생성, 학습, 검증, 테스트, checkpoint 저장을 담당합니다.
- `notebooks/requirements.txt`: 노트북에서 최소 시각화를 할 때 필요한 패키지 목록입니다.

## Quickstart

```bash
cd /Users/taehee/tsfm/metatst-practice
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py --epochs 5
```

Jupyter 커널 등록:

```bash
python -m ipykernel install --user --name metatst-practice --display-name "Python (metatst-practice)"
```

## Practice Flow

실습을 추천하는 순서는 아래와 같습니다.

1. `src/dataset.py`를 읽으면서 어떤 metadata가 어떤 시계열 패턴을 바꾸는지 확인합니다.
2. `src/metadata.py`를 읽으면서 metadata가 embedding + projection으로 변환되는 흐름을 봅니다.
3. `src/model.py`를 읽으면서 metadata vector가 시계열 토큰에 어떻게 더해지는지 확인합니다.
4. `train.py`를 실행해서 loss가 출력되고 checkpoint가 저장되는지 확인합니다.
5. 이후 `dataset.py`를 수정해 metadata를 제거하거나 추가하면서 성능 차이를 비교합니다.

## Hands-on Procedure

처음 실습할 때는 아래 순서로 진행하면 자연스럽습니다.

1. 가상환경 활성화

```bash
cd /Users/taehee/tsfm/metatst-practice
source .venv/bin/activate
```

2. 아주 짧게 학습 실행

```bash
python train.py --epochs 1 --num-series 64 --batch-size 16
```

3. 출력 확인

- `epoch=... train_loss=... val_loss=...`
- `test_loss=...`
- `best_checkpoint=artifacts/metatst_transformer.pt`

4. 코드와 결과 연결해서 보기

- `dataset.py`에서 `region`, `category`, `scale`가 어떻게 시계열 모양을 바꾸는지 확인합니다.
- `model.py`에서 metadata가 `tokens = value_tokens + position_embedding + metadata_context` 형태로 반영되는지 확인합니다.
- `train.py`에서 validation loss가 좋아질 때만 checkpoint를 저장하는지 확인합니다.

5. 첫 실험 아이디어

- `metadata_context`를 제거한 baseline을 만들어 성능을 비교합니다.
- `scale` 대신 다른 real metadata를 추가해봅니다.
- `num_series`, `context_length`, `prediction_length`를 바꿔서 학습 난이도를 비교합니다.

## Next Steps

- `SyntheticMetaTSTDataset`를 실제 CSV/Parquet 로더로 교체
- metadata feature를 더 추가해서 ablation 진행
- metadata 없는 baseline 모델과 성능 비교
- 논문 설정에 맞는 multi-horizon 평가 지표 추가
