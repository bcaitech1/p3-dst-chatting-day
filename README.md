## Baseline model of BoostCamp2021 P-Stage DST


Open-vocab based DST model인 [TRADE](https://arxiv.org/abs/1905.08743)의 한국어 구현체입니다. (5강, 6강 내용 참고) <br>

- 기존의 GloVe, Char Embedding 대신 `monologg/koelectra-base-v3-discriminator`의 `token_embeddings`을pretrained Subword Embedding으로 사용합니다.
- 메모리를 아끼기 위해 Token Embedding (768) => Hidden Dimension (400)으로의 Projection layer가 들어 있습니다.
- 빠른 학습을 위해 `Parallel Decoding`이 구현되어 있습니다.


### 1. 필요한 라이브러리 설치

`pip install -r requirements.txt`

### 2. 모델 학습

`SM_CHANNEL_TRAIN=data/train_dataset SM_MODEL_DIR=[model saving dir] python train.py` <br>
학습된 모델은 epoch 별로 `SM_MODEL_DIR/model-{epoch}.bin` 으로 저장됩니다.<br>
추론에 필요한 부가 정보인 configuration들도 같은 경로에 저장됩니다.<br>
Best Checkpoint Path가 학습 마지막에 표기됩니다.<br>

### 3. 추론하기

`SM_CHANNEL_EVAL=data/eval_dataset/public SM_CHANNEL_MODEL=[Model Checkpoint Path] SM_OUTPUT_DATA_DIR=[Output path] python inference.py`

### 4. 제출하기

3번 스텝 `inference.py`에서 `SM_OUTPUT_DATA_DIR`에 저장된 `predictions.json`을 제출합니다.
