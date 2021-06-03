### Serving (Django Chat Application)

- Serving은 AI 모델을 실제 서비스에서 사용할 수 있게 Server에 올리는 작업입니다.
- Serving은 Pytorch에서 제공하는 `torchserve`를 사용했습니다.
- Django Chat App 서버와 연결(`REST API`)해 챗봇과 유사하게 작동합니다.

<p align="center"><img src="https://user-images.githubusercontent.com/37205213/120576284-7cd0e680-c45d-11eb-94d8-bc32b80e99f1.png" width="60%" height="60%"></p>

- **TorchServe 기본 :** [TorchServe 사용하기](https://www.notion.so/TorchServe-364c7808daa94abe96590c7655b8eb63)
- **모델 Serving :** [TRADE 모델 Serving하기](https://www.notion.so/TRADE-Serving-b191f5725b9a4daf84d30bf569acfe9c)
- **Django와 연동 :** [Django Chat App](https://www.notion.so/Django-Chat-App-f3d09a4abdcf42b58500b06fa59e77e1)

```python
p3-dst-chatting-day/
│
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
