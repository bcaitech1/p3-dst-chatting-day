### Baseline (TRADE)

- TRADE는 Open-vocab based DST model로서, Ontology를 기반에서 벗어난 모델입니다.
- 기존의 GloVe, Char Embedding 대신 `monologg/koelectra-base-v3-discriminator`의 `token_embeddings`을 pretrained Subword Embedding으로 사용합니다.
- 메모리를 아끼기 위해 Token Embedding (768) => Hidden Dimension (400)으로의 Projection layer가 들어 있습니다.
- 빠른 학습을 위해 `Parallel Decoding`이 구현되어 있습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/37205213/120572293-a8040780-c456-11eb-95a5-b69dd28bc801.png" width="50%" height="50%"></p>


- **Paper Review** **:** [TRADE 논문 리뷰](https://www.notion.so/TRADE-47f74ea2ed134116bc9a089591a8ee60)

```python
p3-dst-chatting-day/Baseline/
│
├── train.py
├── preprocessor.py
├── model.py
├── inference.py
├── evaluation.py
└── data_utils.py
```
