# MetaTST Practice

논문 "Metadata Matters for Time Series: Informative Forecasting with Transformers"를
직접 실습해보기 위한 스타터 프로젝트입니다.

이 저장소는 다음 흐름을 빠르게 확인하는 데 초점을 둡니다.

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

## Next Steps

- `SyntheticMetaTSTDataset`를 실제 CSV/Parquet 로더로 교체
- metadata feature를 더 추가해서 ablation 진행
- metadata 없는 baseline 모델과 성능 비교
- 논문 설정에 맞는 multi-horizon 평가 지표 추가
