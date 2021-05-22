from abc import ABC
import json
import logging
import os
import argparse

import torch
from transformers import BertTokenizer
from data_utils import (WOSDataset, get_examples_from_dialogues)
from torch.utils.data import DataLoader, SequentialSampler
from model import TRADE
from preprocessor import TRADEPreprocessor

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TRADEHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TRADEHandler, self).__init__()
        self.initialized = False

        self.config, self.slot_meta = self.load_json_data(
            "./exp_config.json",
            "./slot_meta.json"
        )

    def load_json_data(self, exp_config_path, slot_meta_path):

        config = json.load(open(exp_config_path, "r"))
        config = argparse.Namespace(**config)

        slot_meta = json.load(open(slot_meta_path, "r"))

        return config, slot_meta

    def initialize(self, ctx):

        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        self.tokenizer = BertTokenizer.from_pretrained(self.config.model_name_or_path)
        self.processor = TRADEPreprocessor(self.slot_meta, self.tokenizer)

        tokenized_slot_meta = []
        for slot in self.slot_meta:
            tokenized_slot_meta.append(
                self.tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
            )


        self.model = TRADE(self.config, tokenized_slot_meta)
        ckpt = torch.load(model_pt_path, map_location="cpu")

        self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        print("Model is loaded")


        self.initialized = True

    def preprocess(self, requests):
        """ Very basic preprocessing code - only tokenizes.
            Extend with your own preprocessing steps as needed.
        """
        input_batch = []
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode('utf-8')

            input_text = json.loads(input_text)
            input_batch.extend(input_text)

        eval_examples = get_examples_from_dialogues(
            input_batch, user_first=False, dialogue_level=False
        )
        eval_features = self.processor.convert_examples_to_features(eval_examples)
        eval_data = WOSDataset(eval_features)
        eval_sampler = SequentialSampler(eval_data)
        eval_loader = DataLoader(
            eval_data,
            batch_size=1,
            sampler=eval_sampler,
            collate_fn=self.processor.collate_fn,
        )

        return eval_loader

    def postprocess_state(self, state):
        for i, s in enumerate(state):
            s = s.replace(" : ", ":")
            state[i] = s.replace(" , ", ", ")
        return state

    def inference(self, inputs):
        self.model.eval()
        output_lst = []
        predictions = {}
        for batch in inputs:
            input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
                b.to(self.device) if not isinstance(b, list) else b for b in batch
            ]

            with torch.no_grad():
                o, g = self.model(input_ids, segment_ids, input_masks, 9)

                _, generated_ids = o.max(-1)
                _, gated_ids = g.max(-1)

            for guid, gate, gen in zip(guids, gated_ids.tolist(), generated_ids.tolist()):
                prediction = self.processor.recover_state(gate, gen)
                prediction = self.postprocess_state(prediction)
                predictions[guid] = prediction

        output_lst.append(predictions)
        return output_lst

    # def inference(self, inputs):
    #     """
    #     Predict the class of a text using a trained transformer model.
    #     """
    #     # NOTE: This makes the assumption that your model expects text to be tokenized
    #     # with "input_ids" and "token_type_ids" - which is true for some popular transformer models, e.g. bert.
    #     # If your transformer model expects different tokenization, adapt this code to suit
    #     # its expected input format.
    #     prediction = self.model(
    #         inputs['input_ids'].to(self.device),
    #         token_type_ids=inputs['token_type_ids'].to(self.device)
    #     )[0].argmax().item()
    #     logger.info("Model predicted: '%s'", prediction)
    #
    #     if self.mapping:
    #         prediction = self.mapping[str(prediction)]
    #
    #     return [prediction]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TRADEHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)
        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        print(e)
        raise e