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
└── └── ...
```

<br>
<br>

### Reference

- **TorchServe 공식 Github :** [TorchServe](https://github.com/pytorch/serve)
