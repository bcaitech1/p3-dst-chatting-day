# Chatting-Day's Dialogue State Tracking (DST)

<br>

## [목차]

* [\[목차\]](#목차)
* [\[Dialogue State Tracking 소개\]](#dialogue-state-tracking-소개)
* [\[Installation\]](#installation)
    * [Dependencies](#dependencies)
* [\[Usage\]](#usage)
* [\[File Structure\]](#file-structure)
    * [Baseline(TRADE)](#baseline-trade)
    * [CHAN-DST](#chan-dst)
    * [SOM-DST](#som-dst)
    * [Transformer-DST](#transformer-dst)
    * [TAPT (Task adaptive Pretraining)](#tapt-task-adaptive-pretraining)
    * [Serving](#serving)
* [\[Input JSON File\]](#input-json-file)
* [\[Contributors\]](#contributors)
* [\[Collaborative Tool\]](#collaborative-tool)
* [\[Reference\]](#reference)
    * [Papers](#papers)
    * [Dataset](#dataset)


<br>
<br>

## [Dialogue State Tracking 소개]


<p align="center"><img src="https://user-images.githubusercontent.com/37205213/120572270-a1759000-c456-11eb-8273-80054cb9db1f.png"></p>


- **대화 상태 추적(Dialogue State Tracking)** 은 목적 지향형 대화(Task-Oriented Dialogue)의 중요한 하위 테스크 중 하나입니다.
- 유저와의 대화에서 미리 시나리오에 의해 정의된 정보인 Slot과 매 턴마다 그에 속할 수 있는 Value의 집합인, **대화 상태 (Dialogue State)를 매 턴마다 추론하는 테스크**입니다.
- 시스템은 유저의 목적(Goal)을 파악해야만 하고, 보통 이 Goal은 **(Slot, Value) 페어의 집합**으로 표현될 수 있습니다.
- 예를들어, 숙소를 예약하는 시나리오의 경우 "숙소의 종류", "숙소의 가격대"가 Slot의 타입이 될 수 있고, 이에 속할 수 있는 Value로 각각 ("호텔", "모텔", "에어비앤비", ...), ("저렴", "적당", "비싼", ...) 등을 가질 수 있습니다.

<br>
<br>

## [Installation]



### Dependencies

- torch==1.7.0+cu101
- transformers==3.5.1
- wandb
- pytorch-transformers==1.0.0
- wget==3.2
- pytorch-pretrained-bert

```python
pip install -r requirements.txt
```

<br>
<br>

## [Usage]



모델을 사용하기 위해서는 `run.sh` 를 실행시킵니다.

```bash
$ p3-dst-chatting-day/run.sh
```

총 4가지의 모델을 선택할 수 있습니다.

- **SomDST**
- **ChanDST**
- **TransformerDST**
- **TRADE_TAPT**

<br>
<br>

## [File Structure]



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

<br>
<br>

### CHAN-DST

- CHAN-DST  Open-vocab based DST model로서, SOTA (2020, MULTIWOZ 2.1) 모델입니다.
- `WOS (Wizard Of Seoul) dataset`에 적합한 코드로 수정해서 사용했습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/59716219/117595672-6b592f00-b17c-11eb-8e10-712d63db4c0d.png" width="50%" height="50%"></p>

- **Paper Review :** [Chan-DST 논문 리뷰](https://www.notion.so/Chan-DST-aa6f7444a6a64b4c83842a4a4f333720)
- **구현 :** [Chan-DST 구현](https://www.notion.so/Chan-DST-9a5ac0c6f2c545319d78958715aa20cc)

```python
p3-dst-chatting-day/ChanDST/
│
├── main_chan.py - for train
├── preprocessor.py
├── model_chan.py
├── inference.py
├── evaluation.py
└── data_utils.py
```

<br>
<br>

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

<br>
<br>

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

<br>
<br>

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
│   └── data_utils.py
├── TRADE_TAPT/
│   ├── data_utils.py
│   ├── eval_utils.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── model.py
│   ├── preprocessor.py
└── └── train.py
```

<br>
<br>

### Serving

- Serving은 AI 모델을 실제 서비스에서 사용할 수 있게 Server에 올리는 작업입니다.
- Serving은 Pytorch에서 제공하는 `torchserve`를 사용했습니다.
- Django Chat App 서버와 연결(`REST API`)해 챗봇과 유사하게 작동합니다.

<p align="center"><img src="https://user-images.githubusercontent.com/37205213/120572360-c407a900-c456-11eb-825b-99ec490d7906.png" width="50%" height="50%"></p>


- **TorchServe 기본 :** [TorchServe 사용하기](https://www.notion.so/TorchServe-364c7808daa94abe96590c7655b8eb63)
- **모델 Serving :** [TRADE 모델 Serving하기](https://www.notion.so/TRADE-Serving-b191f5725b9a4daf84d30bf569acfe9c)
- **Django와 연동 :** [Django Chat App](https://www.notion.so/Django-Chat-App-f3d09a4abdcf42b58500b06fa59e77e1)

```python
p3-dst-chatting-day/
│
├── Serving/ - License from Pytorch
│   ├── ...
│   ├── dst_trade/
│   │   ├── DST_custom_handler.py - 모델이 동작하는 방식을 명시
│   │   ├── README.md
│   │   ├── config.properties - serving management config
│   │   ├── data_utils.py
│   │   ├── exp_config.json
│   │   ├── model.py
│   │   ├── preprocessor.py
│   │   ├── requirements.txt - 아카이브 제작 때 dependencies 사용 가능
│   │   ├── slot_meta.json
│   │   └── test.json - model serving 후 test해볼 수 있는 json 파일
│   └── ...
├── django_server/
│   ├── chat/
│   │   ├── migrations/ - migration 관리 폴더
│   │   │   ├── ...
│   │   ├── templates/chat/ - Html 등 frontend 부분
│   │   │   ├── ...
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── consumers.py - 사용자들 간의 message 처리
│   │   ├── models.py
│   │   ├── routing.py - routing pattern
│   │   ├── tests.py
│   │   ├── urls.py
│   │   └── views.py - view templates
│   ├── project_dst/
│   │   └── ...
└── └── manage.py - django의 manage 파일
```

<br>
<br>

## [Input JSON File]



Input으로 들어가는 JSON File은 `WOS (Wizard Of Seoul)`의 양식을 따릅니다.

- `dialogue_idx` **:** 대화 고유의 index를 나타냅니다.
- `domain` : 대화 전체 turn에서 나오는 domain들입니다.
- `dialgoue`
    - `role` : user 또는 system으로 구성됩니다.
    - `text` : user 또는 system의 발화입니다.
    - `state` : 추적해야할 domain-slot-value을 의미합니다.

```json
[
	{
    "dialogue_idx": "snowy-hat-1111:관광_식당_11",
    "domains": [
      "관광",
      "식당"
    ],
    "dialogue": [
      {
        "role": "user",
        "text": "서울 중앙에 있는 박물관을 찾아주세요",
        "state": [
          "관광-종류-박물관",
          "관광-지역-서울 중앙"
        ]
      },
      {
        "role": "sys",
        "text": "안녕하세요. 문화역서울 284은 어떠신가요? 평점도 4점으로 방문객들에게 좋은 평가를 받고 있습니다."
      },
      {
        "role": "user",
        "text": "좋네요 거기 평점은 말해주셨구 전화번호가 어떻게되나요?",
        "state": [
          "관광-종류-박물관",
          "관광-지역-서울 중앙",
          "관광-이름-문화역서울 284"
        ]
      },
      {
        "role": "sys",
        "text": "전화번호는 983880764입니다. 더 필요하신 게 있으실까요?"
      },
      {
        "role": "user",
        "text": "네 관광지와 같은 지역의 한식당을 가고싶은데요 야외석이 있어야되요",
        "state": [
          "관광-종류-박물관",
          "관광-지역-서울 중앙",
          "관광-이름-문화역서울 284",
          "식당-지역-서울 중앙",
          "식당-종류-한식당",
          "식당-야외석 유무-yes"
        ]
      },
      {
        "role": "sys",
        "text": "생각하고 계신 가격대가 있으신가요?"
      },
      {
        "role": "user",
        "text": "음.. 저렴한 가격대에 있나요?",
        "state": [
          "관광-종류-박물관",
          "관광-지역-서울 중앙",
          "관광-이름-문화역서울 284",
          "식당-가격대-저렴",
          "식당-지역-서울 중앙",
          "식당-종류-한식당",
          "식당-야외석 유무-yes"
        ]
      },
      {
        "role": "sys",
        "text": "죄송하지만 저렴한 가격대에는 없으시네요."
      },

			..생략
      
      {
        "role": "sys",
        "text": "감사합니다. 즐거운 여행되세요."
      }
    ]
  },
   
  ..생략

]
```

<br>
<br>

## [Contributors]



- **정희석** ([Heeseok-Jeong](https://github.com/Heeseok-Jeong))
- **신문종** ([moon-jong](https://github.com/moon-jong))
- **이창우** ([Changwoomon](https://github.com/changwoomon))
- **안유진** ([dkswndms4782](https://github.com/dkswndms4782))
- **선재우** ([JAEWOOSUN](https://github.com/JAEWOOSUN))

<br>
<br>

## [Collaborative Tool]

Chatting Day 피어들의 `Ground Rule`, `실험노트`, `피어세션` 등 한달 간의 행보를 확인하시려면 다음 링크를 클릭하세요. 
- **LINK** : https://www.notion.so/DST-7-ChattingDay-dba744b4c9c141f59ec797d3f8b13289

<br>
<br>

## [Reference]



### Papers

- [Transferable Multi-Domain State Generator for Task-Oriented
Dialogue Systems(wu et al., arXiv 2019)](https://arxiv.org/pdf/1905.08743v2.pdf)
- [SUMBT: Slot-Utterance Matching for Universal and Scalable Belief Tracking (Lee et al., arXiv 2019)](https://arxiv.org/pdf/1907.07421v1.pdf)
- [Efficient Dialogue State Tracking by Selectively Overwriting Memory (Kim et al., arXiv 2020)](https://arxiv.org/pdf/1911.03906v2.pdf)
- [A Contextual Hierarchical Attention Network with Adaptive Objective for Dialogue State Tracking (Shan et al., ACL 2020)](https://www.aclweb.org/anthology/2020.acl-main.563.pdf)
- [Jointly Optimizing State Operation Prediction and Value Generation for
Dialogue State Tracking (Zeng et al., arXiv 2021)](https://arxiv.org/pdf/2010.14061v2.pdf)

### Dataset

- WOS (Wizard Of Seoul)
