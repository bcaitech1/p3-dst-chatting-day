"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

from utils.data_utils import prepare_dataset_eval, make_turn_label
from utils.data_utils import make_slot_meta, domain2id, OP_SET, postprocessing
from transformers import BertTokenizer, BertConfig

from model import SomDST
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import numpy as np
import os
import time
import argparse
import json
from copy import deepcopy
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    ontology = json.load(open(os.path.join(args.data_root, args.ontology_data)))
    slot_meta, _ = make_slot_meta(ontology)
    tokenizer = BertTokenizer.from_pretrained("dsksd/bert-ko-small-minimal")

    out_file = '/opt/ml/code/p3-dst-chatting-day/SomDST/pickles/test_data_raw.pkl'
    if os.path.exists(out_file):
        print("Pickles are exist!")
        with open(out_file, 'rb') as f:
            test_data_raw = pickle.load(f)
        # with open(out_path+'/test_data.pkl', 'rb') as f:
        #     test_data = pickle.load(f)
        print("Pickles brought!")
    else:
        print("Pickles are not exist!")
        test_data_raw = prepare_dataset_eval(data_path=args.test_data_path,
                                        tokenizer=tokenizer,
                                        slot_meta=slot_meta,
                                        n_history=args.n_history,
                                        max_seq_length=args.max_seq_length,
                                        op_code=args.op_code)

        # test_data = WosDataset(train_data_raw,
        #                             tokenizer,
        #                             slot_meta,
        #                             args.max_seq_length,
        #                             rng,
        #                             ontology,
        #                             args.word_dropout,
        #                             args.shuffle_state,
        #                             args.shuffle_p)

        with open(out_file, 'wb') as f:
            pickle.dump(test_data_raw, f)
        # with open(out_path+'/test_data.pkl', 'wb') as f:
        #     pickle.dump(test_data, f)
        print("Pickles saved!")

    print("# test examples %d" % len(test_data_raw))


    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = 0.1
    op2id = OP_SET[args.op_code]
    model = SomDST(model_config, len(op2id), len(domain2id), op2id['update'])
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)

    model.eval()
    model.to(device)
    print("Model is loaded")

    inference_model(model, test_data_raw, tokenizer, slot_meta, args.op_code)


def inference_model(model, test_data, tokenizer, slot_meta, op_code='4'):
    model.eval()
    op2id = OP_SET[op_code]
    id2op = {v: k for k, v in op2id.items()}
    id2domain = {v: k for k, v in domain2id.items()}

    results = {}
    last_dialog_state = {}
    wall_times = []
    for di, i in enumerate(test_data):
        if (di+1) % 1000 == 0:
            print(f"{di+1}'s test data is been inferencing")

        if i.turn_id == 0:
            last_dialog_state = {}

        i.last_dialog_state = deepcopy(last_dialog_state)
        # print(di, last_dialog_state)
        i.make_instance(tokenizer, word_dropout=0.)

        input_ids = torch.LongTensor([i.input_id]).to(device)
        segment_ids = torch.LongTensor([i.segment_id]).to(device)
        state_position_ids = torch.LongTensor([i.slot_position]).to(device)
        input_mask = torch.LongTensor([i.input_mask]).to(device)
        # print(f"input_id : {input_ids}")
        # print(f"segment_id : {segment_ids}")
        # print(f"slot_position : {state_position_ids}")
        # print(f"input_mask : {input_mask}")

        start = time.perf_counter()
        MAX_LENGTH = 9
        with torch.no_grad():
            _, s, g = model(input_ids=input_ids,
                            token_type_ids=segment_ids,
                            state_positions=state_position_ids,
                            attention_mask=input_mask,
                            max_value=MAX_LENGTH,
                            )
        # print(s.shape, s)
        # print(g.shape, g)
        _, op_ids = s.view(-1, len(op2id)).max(-1)
        # print(f"op_ids : {op_ids}")

        if g.size(1) > 0:
            generated = g.squeeze(0).max(-1)[1].tolist()
        else:
            generated = []
        # print(f"g.shape : {g.shape}, before_generated : {generated}")

        pred_ops = [id2op[a] for a in op_ids.tolist()]
        # print(pred_ops)
        
        generated, last_dialog_state = postprocessing(slot_meta, pred_ops, last_dialog_state,
                                                      generated, tokenizer, op_code)


        end = time.perf_counter()
        wall_times.append(end - start)
        pred_state = []
        for k, v in last_dialog_state.items():
            try:
                v = v.split('[UNK]')[0].strip()
            except:
                v = v
            # print(v)
            pred_state.append('-'.join([k, v]))

        key = str(i.id) + '-' +str(i.turn_id)
        results[key] = pred_state
        # print(f"{di}, {key} : {pred_state}, generated : {generated}, {len(generated)}")

        # # postprocess to results
        # for k, v in results.items():
        #     temp = []
        #     for vv in v:
        #         value = 
        #         try:
        #             temp.append([vv.split('[UNK]')[0].strip()])
        #         except:
        #             # print(vv)
        #             if value:
        #                 temp.append([vv])
        #     # print(temp)
        #     results[k] = temp
        #     print(temp)

    output_path = '/opt/ml/code/p3-dst-chatting-day/SomDST/predictions/'
    os.makedirs(output_path, exist_ok=True)
    output_file = args.model_ckpt_path.split('/')[-1].split('.')[0] + '_outputs.csv'
    with open(output_path + output_file, 'w', encoding='UTF-8') as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"{output_path + output_file} is saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='/opt/ml/input/data/eval_dataset', type=str)
    parser.add_argument("--test_data", default='eval_dials.json', type=str)
    parser.add_argument("--ontology_data", default='ontology.json', type=str)
    parser.add_argument("--vocab_path", default='./assets/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='./assets/bert_config_base_uncased.json', type=str)
    parser.add_argument("--model_ckpt_path", default='/opt/ml/outputs/model_40.bin', type=str)
    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--op_code", default="4", type=str)

    args = parser.parse_args()
    args.test_data_path = os.path.join(args.data_root, args.test_data)
    main(args)
