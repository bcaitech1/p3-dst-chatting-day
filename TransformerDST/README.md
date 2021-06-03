### Transformer-DST

- Transformer-DST는 Open-vocab based DST model 기반의 모델로서, Slot Generator 부분을 Transformer Decoder로 대체해 사용합니다.
- `WOS (Wizard Of Seoul) dataset`에 적합한 코드로 수정해서 사용했습니다.
- `Encoder`와 `Decoder`는 `dsksd/bert-ko-small-minimal`을 사용했습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/37205213/120572347-bd793180-c456-11eb-9580-a453c247339b.png" width="50%" height="50%"></p>


- **Paper Review :** [Transformer-DST 논문 리뷰](https://www.notion.so/Transformer-DST-14301d1c2fc3420c85f12d50ae4bf9d7)
- **구현 :** [Transformer-DST 구현](https://www.notion.so/Transformer-DST-79127751e3fe495481c72a557f1ec420)

```python
p3-dst-chatting-day/TransformerDST/
│
├── utils/ - util files
│   ├── bert_config_multilingual_uncased.json
│   ├── bert_config_base_uncased.json
│   ├── bert_ko_small_minimal.json
│   ├── ckpt_utils.py - for saving model
│   ├── data_utils.py - for controlling data
│   └── eval_utils.py
├── TransformerDSTevaluation.py
├── TransformerDSTinference.py
├── TransformerDSTmodel.py
├── TransformerDSTtrain.py
└── modeling_bert.py - custom bert
```
