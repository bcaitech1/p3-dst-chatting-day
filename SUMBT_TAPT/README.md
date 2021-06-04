### TAPT (Task-adaptive Pretraining)

- TAPT는 학습할 데이터에 대해 모델을 pretraining (또는 multi 학습) 하는 방법입니다.
- `WOS (Wizard Of Seoul) dataset`로 DST task를 수행하면서 초반에 (또는 동시에) MLM task를 수행합니다.
- TAPT는 `TRADE` 모델과 `SUMBT` 모델에 적용했습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/37205213/120572353-c0742200-c456-11eb-8191-23a88c207435.png" width="50%" height="50%"></p>


- **구현 :** [TAPT 적용](https://www.notion.so/TAPT-e351338941da4d50bdd749e77e16d18e)

```python
p3-dst-chatting-day/
│
├── SUMBT_TAPT/
│   ├── sumbt_tapt.py
└── └── data_utils.py
```
