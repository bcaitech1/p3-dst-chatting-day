### SOM-DST

- SOM-DST는 Open-vocab based DST model 기반의 모델로서, TRADE의 느린 학습/추론 시간을 개선한 모델입니다.
- `WOS (Wizard Of Seoul) dataset`에 적합한 코드로 수정해서 사용했습니다.
- `Encoder`로는 `dsksd/bert-ko-small-minimal`을 사용했습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/37205213/120572304-af2b1580-c456-11eb-966a-d87144a15747.png" width="50%" height="50%"></p>


- **Paper Review :** [SOM-DST 논문 리뷰](https://www.notion.so/SOM-DST-1d255651bfbd488bbe3b029e380424dd)
- **구현 :** [SOM-DST 구현](https://www.notion.so/SOM-DST-3119f6cdc48f4f78a02fd48dd83826d4)

```python
p3-dst-chatting-day/SomDST/
│
├── assets/ - config files & vocab
│   ├── bert_config_base_uncased.json
│   ├── bert_config_large_uncased.json
│   └── vocab.txt
├── utils/ - util files
│   ├── __init__.py
│   ├── ckpt_utils.py - for saving model
│   ├── data_utils.py - for controlling data
│   ├── fix_label.py
│   └── mapping.pair - convert general word
├── LICENSE
├── NOTICE
├── README.md
├── create_data.py
├── evaluation.py
├── inference.py
├── model.py
└── train.py
```
