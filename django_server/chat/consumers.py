from channels.generic.websocket import AsyncWebsocketConsumer
import json
import requests

EXPERIMENT_DOMAINS = ["관광", "숙소", "식당", "지하철", "택시"]

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'chat_%s' % self.room_name
        self.receive_cnt = 0

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)

        if text_data_json['utter'] == "question":
            message = text_data_json['message']
            # Send message to room group
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'utter' : "question",
                    'type': 'chat_message',
                    'message': message
                }
            )

        elif text_data_json['utter'] == "answer":
            message = text_data_json['message']
            messageAll = text_data_json['messageAll']
            # Send message to room group
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'utter': "answer",
                    'type': 'chat_message',
                    'message': message,
                    'messageAll': messageAll
                }
            )



    # Receive message from room group
    async def chat_message(self, event):
        if event['utter'] == "question":
            message = event['message']

            # Send message to WebSocket
            await self.send(text_data=json.dumps({
                'send': self.channel_name,
                'message': message
            }))

        elif event['utter'] == "answer":

            message = event['message']
            messageAll = event['messageAll']

            messageAll = messageAll.split('\n')

            dialogue_idx = str(self.channel_name)+":숙소_관광_"+str(self.receive_cnt)

            domains = EXPERIMENT_DOMAINS

            dialogue = []
            for val in messageAll:
                if val == '': continue
                text = val.split(" : ")
                role = text[0]
                text = ''.join(text[1:])
                _dialogue = {"role":role, "text":text}
                dialogue.append(_dialogue)

            dialogue.append({"role":"user", "text":message})

            _json = [{"dialogue_idx":dialogue_idx, "domains":domains, "dialogue":dialogue}]

            self.receive_cnt+=1

            values = await self.request_post(_json)

            # Send message to WebSocket
            await self.send(text_data=json.dumps({
                'send': "sys",
                'message': values
            }))


    async def request_post(self, data):

        r = requests.post("http://ec2-18-221-229-123.us-east-2.compute.amazonaws.com:8080/predictions/TRADE", data=json.dumps(data))

        values = {}
        for key, value in r.json().items():
            for domain_slot_value in value:
                domain_slot = domain_slot_value.split("-")
                _domain_slot = domain_slot[0] +"-"+domain_slot[1]
                _value = domain_slot[2]

                values[_domain_slot] = _value


        return values
