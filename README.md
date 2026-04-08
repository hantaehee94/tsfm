# tsfm

이 저장소의 실험 프로젝트들은 이제 `TSFM/` 아래에 모아두었습니다.

```text
tsfm/
  TSFM/
    chronos2/
    metatst-practice/
```

- `TSFM/chronos2/`: Chronos-2 로컬 추론과 GUI 실험용 프로젝트
- `TSFM/metatst-practice/`: 메타데이터 조건부 시계열 Transformer 학습 연습 프로젝트

빠른 시작 예시:

```bash
cd /Users/taehee/tsfm/TSFM/metatst-practice
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py --epochs 5
```
