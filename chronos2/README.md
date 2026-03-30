# Chronos-2 Minimal Setup

`chronos2/`는 로컬에서 Chronos-2 시계열 파운데이션 모델을 빠르게 실험해보기 위한 최소 구성 폴더입니다.

이 폴더는 아래 목표에만 집중합니다.

- Chronos-2 로컬 추론 환경 만들기
- 합성 시계열 데이터로 zero-shot 예측 바로 실행하기
- covariate-informed forecasting 흐름 맛보기

## 구성

```text
chronos2/
  README.md
  requirements.txt
  run_forecast.py
```

- `requirements.txt`: Chronos-2 추론에 필요한 최소 패키지 목록
- `run_forecast.py`: 합성 시계열 데이터를 만들고 Chronos-2로 예측하는 실행 예제

## 빠른 시작

```bash
cd /Users/taehee/tsfm/chronos2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_forecast.py
```

처음 실행 시 Hugging Face에서 `amazon/chronos-2` 모델이 다운로드됩니다.
모델 크기는 대략 478MB 수준이라 네트워크와 디스크 여유 공간이 조금 필요합니다.

## 실행 결과

성공하면 아래 흐름이 진행됩니다.

1. CPU / CUDA / MPS 중 사용 가능한 장치를 자동 선택합니다.
2. 여러 개의 관련 시계열과 공변량(`price_index`, `promo`)을 가진 예제 데이터를 생성합니다.
3. Chronos-2가 `prediction_length`만큼 미래를 zero-shot으로 예측합니다.
4. 결과를 `outputs/predictions.parquet`에 저장합니다.

## Chronos-2 사용 방법

Chronos-2는 학습부터 다시 시작하는 모델이라기보다, 먼저 **사전학습된 모델을 불러와 zero-shot 예측을 바로 해보는 방식**이 가장 간단합니다.

핵심 코드는 아래 두 단계입니다.

```python
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
pred_df = pipeline.predict_df(
    context_df=context_df,
    future_df=future_df,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)
```

### 입력 데이터 형식

`context_df`에는 과거 데이터가 들어갑니다.

- `id`: 시계열 식별자
- `timestamp`: 시점
- `target`: 예측 대상 값
- 그 외 컬럼: 공변량으로 함께 전달 가능

예시:

```text
id        timestamp            target  price_index  promo
series_00 2024-01-01 00:00:00  100.0   100.0        1
series_00 2024-01-01 01:00:00  104.5   102.0        1
...
```

`future_df`는 미래 시점의 공변량을 줄 때 사용합니다.

- `id`
- `timestamp`
- 미래에 이미 알고 있는 covariate 컬럼
- 일반적으로 `target`은 포함하지 않음

예시:

```text
id        timestamp            price_index  promo
series_00 2024-01-05 00:00:00  104.0        0
series_00 2024-01-05 01:00:00  106.0        0
...
```

## 자주 바꾸는 옵션

```bash
python run_forecast.py --num-series 5 --context-length 168 --prediction-length 48
```

- `--num-series`: 함께 넣을 시계열 개수
- `--context-length`: 각 시계열의 과거 길이
- `--prediction-length`: 예측 길이
- `--output`: 예측 결과 저장 경로

예시:

```bash
python run_forecast.py --output outputs/chronos2_exp1.parquet
```

## 실제 데이터로 바꾸려면

합성 데이터 대신 실제 CSV/Parquet 데이터를 쓰려면 `run_forecast.py`에서
`build_example_frames()` 부분만 교체하면 됩니다.

핵심 조건은 아래와 같습니다.

- `context_df`에 `id`, `timestamp`, `target`이 있어야 함
- covariate를 쓸 경우 과거 공변량은 `context_df`에 포함
- 미래에 이미 아는 공변량은 `future_df`에 포함
- `prediction_length`와 `future_df` 길이가 맞아야 함

## 참고

- 공식 모델 카드: https://huggingface.co/amazon/chronos-2
- 공식 코드 저장소: https://github.com/amazon-science/chronos-forecasting

위 내용은 2026-03-30 기준 공개된 Chronos-2 공식 모델 카드와 `chronos-forecasting` 저장소 사용 예시를 바탕으로 정리했습니다.
