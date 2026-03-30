# Chronos-2 Minimal Setup

`chronos2/`는 로컬에서 Chronos-2 시계열 파운데이션 모델을 빠르게 실험해보기 위한 최소 구성 폴더입니다.
CLI로 바로 돌릴 수도 있고, 맥북에서 혼자 쓰기 좋은 아주 단순한 GUI로도 실행할 수 있습니다.

이 폴더는 아래 목표에만 집중합니다.

- Chronos-2 로컬 추론 환경 만들기
- 합성 시계열 데이터로 zero-shot 예측 바로 실행하기
- covariate-informed forecasting 흐름 맛보기

## 구성

```text
chronos2/
  README.md
  app.py
  chronos2_core.py
  requirements.txt
  run_forecast.py
```

- `requirements.txt`: Chronos-2 추론에 필요한 최소 패키지 목록
- `app.py`: Streamlit 기반 로컬 GUI
- `chronos2_core.py`: CLI와 GUI가 같이 쓰는 공통 예측 함수
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

## GUI 실행

배포 없이 내 맥북에서만 간단히 쓰려면 GUI 쪽이 더 편합니다.

```bash
cd /Users/taehee/tsfm/chronos2
source .venv/bin/activate
streamlit run app.py
```

브라우저가 열리면 아래 두 방식 중 하나로 실험하면 됩니다.

1. `합성 예제`
2. `파일 업로드`

### GUI에서 할 수 있는 일

- 합성 시계열 예제로 바로 예측 실행
- 미래 공변량 사용 여부를 켜고 끄며 비교 실험
- `context_df` 업로드만으로도 예측 실행
- 필요할 때만 `future_df` 파일 업로드
- `id`가 없는 단일 시계열도 자동 단일 시계열 모드로 실행
- 타임스탬프 시작/종료 구간을 인덱스로 선택
- 입력한 시작/종료 인덱스에 대응하는 실제 타임스탬프 확인
- 데이터셋 내부 평가 모드와 선택 구간 끝 미래 예측 모드 지원
- `id`, `timestamp`, `target` 컬럼 선택
- 예측 결과 표 확인
- 특정 시계열 하나를 Plotly 그래프로 더 부드럽게 확인
- 결과 CSV 다운로드

## 추천 사용 흐름

처음에는 아래 순서가 가장 편합니다.

1. GUI 실행
2. `합성 예제`로 먼저 동작 확인
3. 미래 공변량 없이 한 번 실행
4. 미래 공변량을 넣어 다시 실행
5. 실제 데이터 업로드 후 `id`, `timestamp`, `target` 확인
6. `구간 확인용 시계열`을 고르고 시작/종료 인덱스 입력
7. 인덱스에 대응하는 실제 타임스탬프를 확인
8. `데이터셋 내부 평가` 모드로 먼저 비교 실험
9. 필요할 때 `선택 구간 끝에서 미래 예측` 모드 사용

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
    context_df,
    future_df=future_df,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)
```

미래 공변량을 모르는 경우에는 `future_df` 없이도 바로 예측할 수 있습니다.

```python
pred_df = pipeline.predict_df(
    context_df,
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

`future_df`는 선택입니다. 따라서 아래 두 상황을 모두 비교할 수 있습니다.

- 미래 공변량 없이 예측
- 미래 공변량을 함께 넣고 예측

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

GUI를 쓸 때는 코드를 바꿀 필요 없이 파일 업로드로 바로 실험할 수 있습니다.
CLI를 계속 쓸 경우에는 `run_forecast.py`에서 `build_example_frames()` 부분만 교체하면 됩니다.

핵심 조건은 아래와 같습니다.

- `context_df`에 `timestamp`, `target`이 있어야 함
- `id`가 없고 단일 시계열이라면 GUI에서 자동 생성 가능
- covariate를 쓸 경우 과거 공변량은 `context_df`에 포함
- 미래에 이미 아는 공변량이 있을 때만 `future_df`에 포함
- `future_df`를 쓸 경우 `prediction_length`와 `future_df` 길이가 맞아야 함

예를 들어:

- `context_df`: 과거 실제값 + 과거 공변량
- `future_df`: 미래 시점 + 미래에 이미 아는 공변량
- 미래 공변량이 없으면 `future_df` 없이 실행 가능

형태만 맞으면 GUI에서 컬럼을 선택해서 바로 Chronos-2에 넣을 수 있습니다.

## 로컬 전용 GUI로 충분한 이유

이번 구성은 배포보다 개인 연구 생산성에 맞췄습니다.

- 서버 분리 없이 `streamlit run app.py`만 실행하면 됨
- 코드 수정 없이 파일 업로드로 실험 가능
- 파라미터를 CLI 인자 대신 화면에서 바꿀 수 있음
- 결과를 표와 그래프로 바로 확인 가능

혼자 맥북에서 반복 실험하는 목적이라면 이 정도가 가장 가볍고 유지보수 부담도 적습니다.

## 참고

- 공식 모델 카드: https://huggingface.co/amazon/chronos-2
- 공식 코드 저장소: https://github.com/amazon-science/chronos-forecasting

위 내용은 2026-03-30 기준 공개된 Chronos-2 공식 모델 카드와 `chronos-forecasting` 저장소 사용 예시를 바탕으로 정리했습니다.
