import sys

from TransformerDSTmodel import TransformerDST
from pytorch_transformers import AdamW, WarmupLinearSchedule, BertConfig
from transformers import BertTokenizer
from utils.data_utils import prepare_dataset_for_inference, MultiWozDataset
from utils.data_utils import make_slot_meta, domain2id, OP_SET, make_turn_label, postprocessing
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy
from utils.ckpt_utils import download_ckpt, convert_ckpt_compatible
from TransformerDSTevaluation import model_evaluation

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import argparse
import random
import os
import json
import time


def main(args):

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        print("### mkdir {:}".format(args.save_dir))

    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)

    n_gpu = 0
    if torch.cuda.is_available() and args.use_cpu:
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda')
        print("### Device: {:}".format(device))
    else:
        print("### Use CPU (Debugging)")
        device = torch.device("cpu")

    if args.random_seed < 0:
        print("### Pick a random seed")
        args.random_seed = random.sample(list(range(1, 100000)), 1)[0]

    print("### Random Seed: {:}".format(args.random_seed))
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    rng = random.Random(args.random_seed)
    torch.manual_seed(args.random_seed)

    if n_gpu > 0:
        if args.random_seed >= 0:
            torch.cuda.manual_seed(args.random_seed)
            torch.cuda.manual_seed_all(args.random_seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    ontology = json.load(open(args.ontology_data))
    slot_meta, ontology = make_slot_meta(ontology)
    op2id = OP_SET[args.op_code]
    print(op2id)

    tokenizer = BertTokenizer.from_pretrained(args.bert_config)

    special_tokens = ['[SLOT]', '[NULL]']
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    test_path = os.path.join(args.data_root_test, "test.pt")

    if not os.path.exists(test_path):
        test_data_raw = prepare_dataset_for_inference(data_path=args.test_data_path,
                                                      data_list=None,
                                                      tokenizer=tokenizer,
                                                      slot_meta=slot_meta,
                                                      n_history=args.n_history,
                                                      max_seq_length=args.max_seq_length,
                                                      op_code=args.op_code)
        torch.save(test_data_raw, test_path)
    else:
        test_data_raw = torch.load(test_path)

    print("# test examples %d" % len(test_data_raw))

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = 0.
    model_config.attention_probs_dropout_prob = 0.
    model_config.hidden_dropout_prob = 0.

    type_vocab_size = 4
    dec_config = args
    model = TransformerDST(model_config, dec_config, len(op2id), len(domain2id),
                           op2id['update'],
                           tokenizer.convert_tokens_to_ids(['[MASK]'])[0],
                           tokenizer.convert_tokens_to_ids(['[SEP]'])[0],
                           tokenizer.convert_tokens_to_ids(['[PAD]'])[0],
                           tokenizer.convert_tokens_to_ids(['-'])[0],
                           type_vocab_size, args.exclude_domain)


    state_dict = torch.load(args.bert_ckpt_path, map_location='cpu')
    _k = 'bert.embeddings.token_type_embeddings.weight'
    print("config.type_vocab_size != state_dict[bert.embeddings.token_type_embeddings.weight] ({0} != {1})".format(
        type_vocab_size, state_dict[_k].shape[0]))
    state_dict[_k] = state_dict[_k].repeat(int(type_vocab_size / state_dict[_k].shape[0]), 1)
    state_dict[_k].data[2, :].copy_(state_dict[_k].data[0, :])
    state_dict[_k].data[3, :].copy_(state_dict[_k].data[0, :])
    model.bert.load_state_dict(state_dict, strict=False)

    # re-initialize added special tokens ([SLOT], [NULL], [EOS])
    model.bert.embeddings.word_embeddings.weight.data[1].normal_(mean=0.0, std=0.02)
    model.bert.embeddings.word_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)
    model.bert.embeddings.word_embeddings.weight.data[3].normal_(mean=0.0, std=0.02)

    # re-initialize seg-2, seg-3
    model.bert.embeddings.token_type_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)
    model.bert.embeddings.token_type_embeddings.weight.data[3].normal_(mean=0.0, std=0.02)
    model.bert.resize_token_embeddings(len(tokenizer))

    test_epochs = [int(e) for e in args.load_epoch.strip().lower().split('-')]
    for best_epoch in test_epochs:
        print("### Epoch {:}...".format(best_epoch))
        sys.stdout.flush()
        ckpt_path = os.path.join('/opt/ml/code/transformer_dst', args.save_dir, 'model.e{:}.bin'.format(best_epoch))
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt)
        model.to(device)

        eval_res = model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                                    use_full_slot=args.use_full_slot, use_dt_only=args.use_dt_only,
                                    no_dial=args.no_dial, n_gpu=n_gpu,
                                    is_gt_op=False, is_gt_p_state=False, is_gt_gen=False, submission=True)

        print("### Epoch {:} Test Score : ".format(best_epoch), eval_res)
        print('\n' * 2)
        sys.stdout.flush()


if __name__ == "__main__":

    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser()

    # Using only [CLS]
    parser.add_argument("--use_cls_only", type=bool, default=False, help='Using only [CLS]')

    # w/o re-using dialogue
    parser.add_argument("--no_dial", type=bool, default=False, help='w/o re-using dialogue')

    # Using only D_t in generation
    parser.add_argument("--use_dt_only", type=bool, default=True, help='Using only D_t in generation')

    # By default, "decoder" only attend on a specific [SLOT] position.
    # If using this option, the "decoder" can access to this group of "[SLOT] domain slot - value".
    # NEW: exclude "- value"
    parser.add_argument("--use_full_slot", type=bool, default=False, help='By default, "decoder" only attend on a specific [SLOT] position.')

    parser.add_argument("--only_pred_op", type=bool, default=False)

    parser.add_argument("--use_one_optim", type=bool, default=True)  # I use one optim

    # Required parameters
    parser.add_argument("--data_root", default='/opt/ml/input/data/train_dataset', type=str)
    parser.add_argument("--data_root_test", default='/opt/ml/input/data/eval_dataset', type=str)
    parser.add_argument("--train_data", default='train_dials.json', type=str)
    parser.add_argument("--test_data", default='eval_dials.json', type=str)
    parser.add_argument("--load_epoch", required=True, type=str, help="example: '10-11-12' ")

    parser.add_argument("--ontology_data", default='/opt/ml/input/data/train_dataset/ontology.json', type=str)
    parser.add_argument("--vocab_path", default='assets/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default="./utils/bert_ko_small_minimal.json", type=str)
    parser.add_argument("--bert_config", default='dsksd/bert-ko-small-minimal', type=str)
    parser.add_argument("--bert_ckpt_path", default='./assets/dsksd/bert-ko-small-minimal-pytorch_model.bin', type=str)
    parser.add_argument("--save_dir", default='outputs', type=str)

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=3e-5, type=float)  # my Transformer-AR uses 3e-5
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=30, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)

    parser.add_argument("--op_code", default="4", type=str)
    parser.add_argument("--slot_token", default="[SLOT]", type=str)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float)
    parser.add_argument("--decoder_teacher_forcing", default=1, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)
    parser.add_argument("--shuffle_p", default=0.5, type=float)
    parser.add_argument("--shuffle_state", default=False, type=bool)

    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--msg", default=None, type=str)
    parser.add_argument("--exclude_domain", default=False)

    # generator
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument("--min_len", default=1, type=int)
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")
    parser.add_argument('--forbid_duplicate_ngrams', type=bool, default=True)
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Ignore the word during forbid_duplicate_ngrams")
    parser.add_argument('--ngram_size', type=int, default=2)
    parser.add_argument('--use_cpu', type=bool, default=True)


    args = parser.parse_args()
    args.train_data_path = os.path.join(args.data_root, args.train_data)
    args.test_data_path = os.path.join(args.data_root_test, args.test_data)
    args.ontology_data = os.path.join(args.data_root, args.ontology_data)

    print(args)
    main(args)