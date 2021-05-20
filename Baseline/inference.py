import argparse
import os
import json

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer

from data_utils import (WOSDataset, get_examples_from_dialogues)
from model import TRADE
from preprocessor import TRADEPreprocessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def postprocess_state(state):
    for i, s in enumerate(state):
        s = s.replace(" : ", ":")
        state[i] = s.replace(" , ", ", ")
    return state


def inference(model, eval_loader, processor, device):
    model.eval()
    predictions = {}
    for batch in tqdm(eval_loader):
        input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
            b.to(device) if not isinstance(b, list) else b for b in batch
        ]

        with torch.no_grad():
            o, g = model(input_ids, segment_ids, input_masks, 9)

            _, generated_ids = o.max(-1)
            _, gated_ids = g.max(-1)

        for guid, gate, gen in zip(guids, gated_ids.tolist(), generated_ids.tolist()):
            prediction = processor.recover_state(gate, gen)
            prediction = postprocess_state(prediction)
            predictions[guid] = prediction
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/opt/ml/input/data/eval_dataset')
    parser.add_argument("--model_dir", type=str, default='/opt/ml/code/results/exp2/model-1.bin')
    parser.add_argument("--output_dir", type=str, default='/opt/ml/code/prediction')
    parser.add_argument("--eval_batch_size", type=int, default=32)
    args = parser.parse_args()
    # args.data_dir = os.environ['SM_CHANNEL_EVAL']
    # args.model_dir = os.environ['SM_CHANNEL_MODEL']
    # args.output_dir = os.environ['SM_OUTPUT_DATA_DIR']
    
    model_dir_path = os.path.dirname(args.model_dir)
    eval_data = json.load(open(f"{args.data_dir}/eval_dials.json", "r"))
    config = json.load(open(f"{model_dir_path}/exp_config.json", "r"))
    config = argparse.Namespace(**config)
    slot_meta = json.load(open(f"{model_dir_path}/slot_meta.json", "r"))

    tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)
    processor = TRADEPreprocessor(slot_meta, tokenizer)

    eval_examples = get_examples_from_dialogues(
        eval_data, user_first=False, dialogue_level=False
    )

    # Extracting Featrues
    eval_features = processor.convert_examples_to_features(eval_examples)
    eval_data = WOSDataset(eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_loader = DataLoader(
        eval_data,
        batch_size=args.eval_batch_size,
        sampler=eval_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# eval:", len(eval_data))

    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )

    model = TRADE(config, tokenized_slot_meta)
    ckpt = torch.load(args.model_dir, map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    print("Model is loaded")

    predictions = inference(model, eval_loader, processor, device)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    model_name = args.model_dir.split('/')
    model_name = args.output_dir + '/' + model_name[-2] + '-' + model_name[-1].split('.')[0]
    json.dump(
        predictions,
        open(f"{model_name}-output.csv", "w"),
        indent=2,
        ensure_ascii=False,
    )
