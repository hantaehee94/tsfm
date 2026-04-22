# Chronos-2 Local Lab

`TSFM/chronos2/`는 로컬에서 Chronos-2를 파일 업로드 기반으로 실험하는 작은 Streamlit 워크벤치입니다.

## 구성

```text
TSFM/chronos2/
  README.md
  app.py
  chronos2_core.py
  requirements.txt
```

- `app.py`: Streamlit UI
- `chronos2_core.py`: 모델 로딩, 입력 길이 검증, 예측 실행 공통 함수
- `requirements.txt`: 최소 의존성

## 실행

```bash
cd /Users/taehee/tsfm/TSFM/chronos2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

처음 실행 시 `amazon/chronos-2` 모델이 다운로드됩니다.

## 입력 형식

과거 데이터:

```text
id, timestamp, target, covariates...
```

미래 공변량:

```text
id, timestamp, known_covariates...
```

## 현재 앱에서 할 수 있는 일

- 파일 업로드 기반 단일 예측
- `id`, `timestamp`, `target` 컬럼 매핑
- 단일 시계열 자동 ID 생성
- 분석 구간 인덱스 선택
- `과거 전용 공변량`과 `미래 known 공변량` 분리
- 데이터셋 내부 평가 / 선택 구간 끝 미래 예측
- Validation 탭에서 `요일 미사용 / 과거 요일 유 / 과거+미래 요일 유` 비교
- Validation 탭에서 시나리오별 actual vs prediction 시각 비교

## 공변량 규칙

- `과거 전용 공변량`: `context_df`에만 사용
- `미래 known 공변량`: `context_df`와 `future_df`에 함께 사용
- 미래 시점에 실제로 알 수 없는 값은 `future_df`에 넣지 않는 것이 원칙

## 요일 비교

Validation 탭에서는 요일 공변량을 다음 세 시나리오로 비교할 수 있습니다.

- `요일 미사용`
- `과거 요일 유`
- `과거+미래 요일 유`

인코딩은 `문자열`, `0~6`, `sin/cos` 중 하나를 선택합니다.

## 참고

- 공식 모델 카드: https://huggingface.co/amazon/chronos-2
- 공식 코드 저장소: https://github.com/amazon-science/chronos-forecasting
