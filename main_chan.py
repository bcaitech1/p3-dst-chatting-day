import csv
import os
import logging
import argparse
import random
import collections
from tqdm import tqdm, trange
import json
from typing import List, Optional, Union
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from tensorboardX import SummaryWriter

from model_chan import BeliefTracker
from data_utils import CHANExample, load_chan_dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def eval_all_accs(pred_slot, labels, accuracies):

    def _eval_acc(_pred_slot, _labels):
        slot_dim = _labels.size(-1)
        accuracy = (_pred_slot == _labels).view(-1, slot_dim)
        num_turn = torch.sum(_labels[:, :, 0].view(-1) > -1, 0).float()
        num_data = torch.sum(_labels > -1).float()
        # joint accuracy
        joint_acc = sum(torch.sum(accuracy, 1) / slot_dim).float()
        # slot accuracy
        slot_acc = torch.sum(accuracy).float()
        return joint_acc, slot_acc, num_turn, num_data

    # 7 domains
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot, labels)
    accuracies['joint7'] += joint_acc
    accuracies['slot7'] += slot_acc
    accuracies['num_turn'] += num_turn
    accuracies['num_slot7'] += num_data

    # restaurant domain
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot[:,:,18:25], labels[:,:,18:25])
    accuracies['joint_rest'] += joint_acc
    accuracies['slot_rest'] += slot_acc
    accuracies['num_slot_rest'] += num_data

    pred_slot5 = torch.cat((pred_slot[:,:,0:3], pred_slot[:,:,8:]), 2)
    label_slot5 = torch.cat((labels[:,:,0:3], labels[:,:,8:]), 2)

    # 5 domains (excluding bus and hotel domain)
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot5, label_slot5)
    accuracies['joint5'] += joint_acc
    accuracies['slot5'] += slot_acc
    accuracies['num_slot5'] += num_data

    return accuracies

def save_configure(args, num_labels, ontology):
    with open(os.path.join(args.output_dir, "config.json"),'w') as outfile:
        data = { "hidden_dim": args.hidden_dim,
                "num_rnn_layers": args.num_rnn_layers,
                "zero_init_rnn": args.zero_init_rnn,
                "max_seq_length": args.max_seq_length,
                "max_label_length": args.max_label_length,
                "num_labels": num_labels,
                "attn_head": args.attn_head,
                 "distance_metric": args.distance_metric,
                 "fix_utterance_encoder": args.fix_utterance_encoder,
                 "task_name": args.task_name,
                 "bert_dir": args.bert_dir,
                 "bert_model": args.bert_model,
                 "do_lower_case": args.do_lower_case,
                 "ontology": ontology}
        json.dump(data, outfile, indent=4)

# -------------------------------------
def chan_dst_examples(data_dir, dataset):
    ontology = json.load(open(f"{data_dir}/ontology.json"))
    slot_meta = json.load(open(f"{data_dir}/slot_meta.json"))
    train_dials = dataset

    ontology_label = {}
    for idx, i in enumerate(ontology.keys()):
        ontology_label[i] = idx

    examples = []
    for dials in train_dials:
        dial_idx = dials["dialogue_idx"].split(":")[0]
        dial_idx = dial_idx.split('-')
        dial_idx = '_'.join(dial_idx)
        guid = "%s-%s" % ("train", dial_idx)
        turn = 0
        label = ["none" for i in range(45)]
        update = [0 for i in range(45)]
        for idx, dial in enumerate(dials["dialogue"]):
            if dial['role'] == 'user':
                continue
            text_a = dials['dialogue'][idx - 1]['text']
            text_b = dial['text']
            guid_tmp = f"{guid}-{turn}"
            turn += 1
            for state in dials['dialogue'][idx - 1]['state']:
                d, s, v = state.split('-')
                if label[ontology_label[f"{d}-{s}"]] == 'none':
                    label[ontology_label[f"{d}-{s}"]] = v
                    update[ontology_label[f"{d}-{s}"]] = 1
                else:
                    update[ontology_label[f"{d}-{s}"]] = 0
            examples.append(
                deepcopy(CHANExample(guid=guid_tmp, text_a=text_a, text_b=text_b, label=label, update=update)))
    return examples

def chan_dst_test():
    ontology = json.load(open("/opt/ml/input/train_dataset/ontology.json"))
    train_dials = json.load(open("/opt/ml/input/eval_dataset/eval_dials.json"))

    ontology_label = {}
    for idx, i in enumerate(ontology.keys()):
        ontology_label[i] = idx

    examples = []
    for dials in train_dials:
        dial_idx = dials["dialogue_idx"].split(":")[0]
        dial_idx = dial_idx.split('-')
        dial_idx = '_'.join(dial_idx)
        guid = "%s-%s" % ("train", dial_idx)
        turn = 0
        label = ["none" for i in range(45)]
        update = [0 for i in range(45)]
        for idx, dial in enumerate(dials["dialogue"]):
            if dial['role'] == 'user':
                continue
            text_a = dials['dialogue'][idx - 1]['text']
            text_b = dial['text']
            guid_tmp = f"{guid}-{turn}"
            turn += 1
            for state in dials['dialogue'][idx - 1]['state']:
                d, s, v = state.split('-')
                if label[ontology_label[f"{d}-{s}"]] == 'none':
                    label[ontology_label[f"{d}-{s}"]] = v
                    update[ontology_label[f"{d}-{s}"]] = 1
                else:
                    update[ontology_label[f"{d}-{s}"]] = 0
            examples.append(
                deepcopy(CHANExample(guid=guid_tmp, text_a=text_a, text_b=text_b, label=label, update=update)))
    return examples

def make_num_label(data_dir):
    ontology = json.load(open(f"{data_dir}/ontology.json"))
    num_label = []
    for i in ontology.keys():
        num_label.append(len(ontology[i]))
    return num_label

def make_label_list(data_dir):
    ontology = json.load(open(f"{data_dir}/ontology.json"))
    label_list = [ontology[slot] for slot in ontology.keys()]
    return label_list

def get_target_slot(data_dir):
    ontology = json.load(open(f"{data_dir}/ontology.json"))
    return list(ontology.keys())

# ------------------------------------------

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_len, label_id, update):
        self.input_ids = input_ids
        self.input_len = input_len
        self.label_id = label_id
        self.update = update

class SUMBTDataset(Dataset):
    def __init__(self, examples, label_list, tokenizer, max_seq_length=64, max_turn_length=22):
        self.examples = examples
        self.label_list = label_list
        self.tokenizer = tokenizer
        label_map = [{label: i for i, label in enumerate(labels)} for labels in label_list]
        slot_dim = len(label_list)

        self.all_features = []
        prev_dialogue_idx = None

        max_turn = 0
        for (ex_index, example) in enumerate(examples):
            if max_turn < int(example.guid.split('-')[2]):
                max_turn = int(example.guid.split('-')[2])
        max_turn_length = min(max_turn+1, max_turn_length)
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_a)]
            tokens_b = None
            if example.text_b:
                tokens_b = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_b)]
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            input_len = [len(tokens), 0]

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                input_len[1] = len(tokens_b) + 1

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Zero-pad up to the sequence length.
            #padding = [0] * (max_seq_length - len(input_ids))
            #input_ids += padding
            #assert len(input_ids) == max_seq_length

            label_id = []
            label_info = 'label: '
            for i, label in enumerate(example.label):
                label_id.append(label_map[i][label])
                label_info += '%s (id = %d) ' % (label, label_map[i][label])

            curr_dialogue_idx = example.guid.split('-')[1]
            curr_turn_idx = int(example.guid.split('-')[2])

            if prev_dialogue_idx is not None and prev_dialogue_idx != curr_dialogue_idx:
                self.all_features.append(features)
                features = []

            if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
                features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_len=input_len,
                                  label_id=label_id,
                                  update=list(map(int, example.update))))

            prev_dialogue_idx = curr_dialogue_idx
            prev_turn_idx = curr_turn_idx

        self.all_features.append(features)

    def __len__(self):
        return len(self.all_features)

    def __getitem__(self, index):
        input_ids = [f.input_ids for f in self.all_features[index]]
        max_len = max([len(i) for i in input_ids])
        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i] + [0] * (max_len-len(input_ids[i]))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_len= torch.tensor([f.input_len for f in self.all_features[index]], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in self.all_features[index]], dtype=torch.long)
        update = torch.tensor([f.update for f in self.all_features[index]], dtype=torch.long)
        return input_ids, input_len, label_ids, update

def collate_fn(batch):
    def padding(seq, pad_token):
        max_len = max([i.size(0) for i in seq])
        max_dim = max([i.size(1) for i in seq])
        result = torch.ones((len(seq), max_len, max_dim)).long() * pad_token
        for i in range(len(seq)):
            result[i, :seq[i].size(0), :seq[i].size(1)] = seq[i]
        return result

    input_ids_list, input_len_list, label_ids_list, update_list = [], [], [], []
    for i in batch:
        input_ids_list.append(i[0])
        input_len_list.append(i[1])
        label_ids_list.append(i[2])
        update_list.append(i[3])

    input_ids = padding(input_ids_list, torch.LongTensor([0]))
    input_len = padding(input_len_list, torch.LongTensor([0]))
    label_ids = padding(label_ids_list, torch.LongTensor([-1]))
    update = padding(update_list, torch.LongTensor([-1])).float()
    return input_ids, input_len, label_ids, update


def get_label_embedding(labels, max_seq_length, tokenizer, device) -> object:
    features = []
    for label in labels:
        label_tokens = ["[CLS]"] + tokenizer.tokenize(label) + ["[SEP]"]
        label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
        label_len = len(label_token_ids)

        label_padding = [0] * (max_seq_length - len(label_token_ids))
        label_token_ids += label_padding
        assert len(label_token_ids) == max_seq_length

        features.append((label_token_ids, label_len))

    all_label_token_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to(device)
    all_label_len = torch.tensor([f[1] for f in features], dtype=torch.long).to(device)

    return all_label_token_ids, all_label_len


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default='/opt/ml/input/data/train_dataset',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_dir", default='bert-base-uncased',
                        type=str,
                        help="The directory of the pretrained BERT model")
    parser.add_argument("--task_name",
                        default='bert-gru-sumbt',
                        type=str,
                        help="The name of the task to train: bert-gru-sumbt, bert-lstm-sumbt"
                             "bert-label-embedding, bert-gru-label-embedding, bert-lstm-label-embedding")
    parser.add_argument("--output_dir",
                        default='/opt/ml/code/p3-dst-chatting-day/results',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--target_slot",
                        default='all',
                        type=str,
                        help="Target slot idx to train model. ex. 'all', '0:1:2', or an excluding slot name 'attraction'" )
    parser.add_argument("--tf_dir",
                        default='tensorboard',
                        type=str,
                        help="Tensorboard directory")
    parser.add_argument("--model",
                        default='BeliefTracker',
                        type=str,
                        help="model file name" )
    parser.add_argument("--mt_drop", type=float, default=0.)
    parser.add_argument("--fix_utterance_encoder",
                        default=None,
                        action='store_true',
                        help="Do not train BERT utterance encoder")

    ## Other parameters
    parser.add_argument("--window", default=1, type=int)
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_label_length",
                        default=60,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_turn_length",
                        default=22,
                        type=int,
                        help="The maximum total input turn length. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=100,
                        help="hidden dimension used in belief tracker")
    parser.add_argument('--num_rnn_layers',
                        type=int,
                        default=1,
                        help="number of RNN layers")
    parser.add_argument('--zero_init_rnn',
                        action='store_true',
                        help="set initial hidden of rnns zero")
    parser.add_argument('--attn_head',
                        type=int,
                        default=4,
                        help="the number of heads in multi-headed attention")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_eval_best_acc",
                        action='store_true')
    parser.add_argument("--combine",
                        type=str)
    parser.add_argument("--curr",
                        type=str,
                        default="attn")
    parser.add_argument("--do_lower_case",
                        default= True,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--distance_metric",
                        type=str,
                        default="cosine",
                        help="The metric for distance between label embeddings: cosine, euclidean.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total dialog batch size for training.")
    parser.add_argument("--dev_batch_size",
                        default=1,
                        type=int,
                        help="Total dialog batch size for validation.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total dialog batch size for evaluation.")
    parser.add_argument("--lamb",
                        default=0.5,
                        type=float)
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for BertAdam.")
    parser.add_argument("--num_train_epochs",
                        default=300,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--patience",
                        default=10.0,
                        type=float,
                        help="The number of epochs to allow no further improvement.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--do_not_use_tensorboard",
                        action='store_true',
                        default=0,
                        help="Whether to run eval on the test set.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.do_train and not args.do_eval and not args.do_analyze:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # Tensorboard logging
    if not args.do_not_use_tensorboard:
        tb_file_name = args.output_dir.split('/')[1]
        summary_writer = SummaryWriter("./%s/%s" % (args.tf_dir, tb_file_name))
    else:
        summary_writer = None

    # Logger
    fileHandler = logging.FileHandler(os.path.join(args.output_dir, "%s.txt"%(tb_file_name)))
    logger.addHandler(fileHandler)
    logger.info(args)

    # CUDA setup
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    ###############################################################################
    # Load data
    ###############################################################################

    # Get Processor
    label_list = make_label_list(args.data_dir)
    num_labels = make_num_label(args.data_dir)# number of slot-values in each slot-type

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    num_train_steps = None
    accumulation = False

    if args.do_train:
        train_dataset, dev_dataset = load_chan_dataset(f"{args.data_dir}/train_dials.json")
        train_examples = chan_dst_examples(args.data_dir, train_dataset)
        dev_examples = chan_dst_examples(args.data_dir, dev_dataset)

        ## Training utterances
        train_dataset = SUMBTDataset(train_examples, label_list, tokenizer, max_seq_length=args.max_seq_length, max_turn_length=args.max_turn_length)

        num_train_batches = len(train_dataset)
        num_train_steps = int(num_train_batches / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)


        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True, num_workers=16, collate_fn=lambda x: collate_fn(x))

        ## Dev utterances
        # all_input_ids_dev, all_input_len_dev, all_label_ids_dev, all_update_dev = convert_examples_to_features(
        #    dev_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)
        dev_dataset = SUMBTDataset(dev_examples, label_list, tokenizer, max_seq_length=args.max_seq_length, max_turn_length=args.max_turn_length)

        logger.info("***** Running validation *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)

        dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.dev_batch_size, drop_last=False, collate_fn=lambda x: collate_fn(x))

    logger.info("Loaded data!")

    ###############################################################################
    # Build the models
    ###############################################################################

    # Prepare model
    model = BeliefTracker(args, num_labels, device)

    if args.fp16:
        model.half()
    model.to(device)
    # save_configure(args, num_labels, processor.ontology)

    ## Get slot-value embeddings
    label_token_ids, label_len = [], []
    for labels in label_list:
        token_ids, lens = get_label_embedding(labels, args.max_label_length, tokenizer, device)
        label_token_ids.append(token_ids)
        label_len.append(lens)

    ## Get domain-slot-type embeddings
    slot_token_ids, slot_len = \
        get_label_embedding(get_target_slot(args.data_dir), args.max_label_length, tokenizer, device)

    ## Initialize slot and value embeddings
    model.initialize_slot_value_lookup(label_token_ids, slot_token_ids)

    # Data parallelize when use multi-gpus
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                #{'params': [p for n, p in param_optimizer if ('bert' in n) and (not any(nd in n for nd in no_decay))],
                #    'weight_decay': 0.01, 'lr': 4e-5},
                #{'params': [p for n, p in param_optimizer if ('bert' in n) and (any(nd in n for nd in no_decay))],
                #    'weight_decay': 0.0, 'lr': 4e-5},
                #{'params': [p for n, p in param_optimizer if ('bert' not in n) and (any(nd in n for nd in no_decay))],
                #    'weight_decay': 0.0, 'lr': 1e-3},
                #{'params': [p for n, p in param_optimizer if ('bert' not in n) and (not any(nd in n for nd in no_decay))],
                #    'weight_decay': 0.01, 'lr': 1e-3},
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.learning_rate},
            ]
            return optimizer_grouped_parameters
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)


        t_total = num_train_steps

        if args.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=t_total)
        logger.info(optimizer)

    ###############################################################################
    # Training code
    ###############################################################################

    if args.do_train:
        logger.info("Training...")

        global_step = 0
        last_update = None
        best_loss = None
        best_acc = None
        train_mt = True

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            # Train
            model.train()
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0

            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_len, label_ids, update = batch
                print(label_ids)

                # Forward
                if n_gpu == 1:
                    loss, loss_slot, acc, acc_slot, _, tup = model(input_ids, input_len, label_ids, update, n_gpu, mt=train_mt)
                else:
                    loss, _, acc, acc_slot, _, tup_1, tup_2, tup_3 = model(input_ids, input_len, label_ids, update, n_gpu, mt=train_mt)
                    tup = (tup_1.mean(), tup_2.mean(), tup_3.mean())

                    # average to multi-gpus
                    loss = loss.mean()
                    acc = acc.mean()
                    acc_slot = acc_slot.mean(0)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Backward
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify lealrning rate with special warm up BERT uses
                    #lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    #if summary_writer is not None:
                    #    summary_writer.add_scalar("Train/LearningRate", lr_this_step, global_step)
                    #for param_group in optimizer.param_groups:
                    #    param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Perform evaluation on validation dataset
            badcase_list = []
            model.eval()
            dev_loss = 0
            dev_acc = 0
            dev_loss_slot, dev_acc_slot = None, None
            dev_tup = [0, 0, 0]
            nb_dev_examples, nb_dev_steps = 0, 0

            for step, batch in enumerate(dev_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_len, label_ids, update = batch
                if input_ids.dim() == 2:
                    input_ids = input_ids.unsqueeze(0)
                    input_len = input_len.unsqueeze(0)
                    label_ids = label_ids.unsuqeeze(0)
                    update = update.unsqueeze(0)

                with torch.no_grad():
                    loss, loss_slot, acc, acc_slot, pred_slot, tup = model(input_ids, input_len, label_ids, update, n_gpu)

                badcase = (pred_slot != label_ids) * (label_ids > -1)
                badcase_1 = badcase.sum(-1).nonzero()
                for b in badcase_1:
                    b_idx = b.cpu().numpy().tolist()
                    sent = ' '.join(tokenizer.convert_ids_to_tokens(input_ids[b_idx[0], b_idx[1]].cpu().numpy().tolist()))
                    pred = ' '.join([label_list[i][j] for i, j in enumerate(pred_slot[b_idx[0], b_idx[1]].cpu().numpy().tolist())])
                    gold = ' '.join([label_list[i][j] for i, j in enumerate(label_ids[b_idx[0], b_idx[1]].cpu().numpy().tolist())])
                    badcase_list.append(f"{sent}\t{pred}\t{gold}\n")
                num_valid_turn = torch.sum(label_ids[:,:,0].view(-1) > -1, 0).item()
                dev_loss += loss.item() * num_valid_turn
                dev_acc += acc.item() * num_valid_turn
                dev_tup[0] += tup[0] * num_valid_turn
                dev_tup[1] += tup[1] * num_valid_turn
                dev_tup[2] += tup[2] * num_valid_turn

                if n_gpu == 1:
                    if dev_loss_slot is None:
                        dev_loss_slot = [ l * num_valid_turn for l in loss_slot]
                        dev_acc_slot = acc_slot * num_valid_turn
                    else:
                        for i, l in enumerate(loss_slot):
                            dev_loss_slot[i] = dev_loss_slot[i] + l * num_valid_turn
                        dev_acc_slot += acc_slot * num_valid_turn

                nb_dev_examples += num_valid_turn

            dev_loss = dev_loss / nb_dev_examples
            dev_acc = dev_acc / nb_dev_examples
            dev_tup = list(map(lambda x: x / nb_dev_examples, dev_tup))
            #train_mt = dev_tup[2] < 0.95

            if n_gpu == 1:
                dev_acc_slot = dev_acc_slot / nb_dev_examples

            dev_loss = round(dev_loss, 6)
            if last_update is None or dev_acc > best_acc:
                # Save a trained model
                output_model_file = os.path.join(args.output_dir, "acc.best")
                if args.do_train:
                    if n_gpu == 1:
                        torch.save(model.state_dict(), output_model_file)
                    else:
                        torch.save(model.module.state_dict(), output_model_file)
                best_acc = dev_acc
            if last_update is None or dev_tup[0] < best_loss:
                badcase_fp = open(os.path.join(args.output_dir, "dev_badcase.txt"), "w", encoding="utf-8")
                for line in badcase_list:
                    badcase_fp.write(line)
                badcase_fp.close()
                # Save a trained model
                output_model_file = os.path.join(args.output_dir, "loss.best")
                if args.do_train:
                    if n_gpu == 1:
                        torch.save(model.state_dict(), output_model_file)
                    else:
                        torch.save(model.module.state_dict(), output_model_file)

                last_update = epoch
                best_loss = dev_tup[0]

            logger.info("*** Epoch=%d, Dev Loss=%.6f, Dev Acc=%.6f, Best Loss=%.6f, Best Acc=%.6f ***" % (epoch, dev_tup[0], dev_acc, best_loss, best_acc))

            if last_update + args.patience <= epoch:
                break


    ###############################################################################
    # Evaluation
    ###############################################################################
    # Load a trained model that you have fine-tuned
    # if args.do_eval_best_acc:
    #     output_model_file = os.path.join(args.output_dir, "acc.best")
    # else:
    #     output_model_file = os.path.join(args.output_dir, "loss.best")
    # model = BeliefTracker(args, num_labels, device)
    #
    # if args.local_rank != -1:
    #     try:
    #         from apex.parallel import DistributedDataParallel as DDP
    #     except ImportError:
    #         raise ImportError(
    #             "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    #
    #     model = DDP(model)
    # elif n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    #
    # ptr_model = torch.load(output_model_file)
    #
    # if n_gpu == 1:
    #     state = model.state_dict()
    #     state.update(ptr_model)
    #     model.load_state_dict(state)
    # else:
    #     print("Evaluate using only one device!")
    #     model.module.load_state_dict(ptr_model)
    # # in the case that slot and values are different between the training and evaluation
    #
    # model.to(device)
    #
    # # Evaluation
    # if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #
    #     eval_examples = chan_dst_test()
    #     #all_input_ids, all_input_len, all_label_ids, all_update = convert_examples_to_features(
    #     #    eval_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)
    #     #all_input_ids, all_input_len, all_label_ids = all_input_ids.to(device), all_input_len.to(device), all_label_ids.to(device)
    #     #all_update = all_update.to(device)
    #     logger.info("***** Running evaluation *****")
    #     logger.info("  Num examples = %d", len(eval_examples))
    #     logger.info("  Batch size = %d", args.eval_batch_size)
    #
    #     #eval_data = TensorDataset(all_input_ids, all_input_len, all_label_ids, all_update)
    #     eval_dataset = SUMBTDataset(eval_examples, label_list, tokenizer, max_seq_length=args.max_seq_length, max_turn_length=args.max_turn_length)
    #
    #     # Run prediction for full data
    #     eval_sampler = SequentialSampler(eval_dataset)
    #     eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=lambda x: collate_fn(x))
    #
    #     model.eval()
    #     eval_loss, eval_accuracy = 0, 0
    #     eval_update_acc = 0
    #     eval_loss_slot, eval_acc_slot = None, None
    #     nb_eval_steps, nb_eval_examples = 0, 0
    #
    #     accuracies = {'joint7':0, 'slot7':0, 'joint5':0, 'slot5':0, 'joint_rest':0, 'slot_rest':0,
    #                   'num_turn':0, 'num_slot7':0, 'num_slot5':0, 'num_slot_rest':0}
    #
    #     for batch in tqdm(eval_dataloader, desc="Evaluating"):
    #         batch = tuple(t.to(device) for t in batch)
    #         input_ids, input_len, label_ids, update = batch
    #         if input_ids.dim() == 2:
    #             input_ids = input_ids.unsqueeze(0)
    #             input_len = input_len.unsqueeze(0)
    #             label_ids = label_ids.unsuqeeze(0)
    #             update = update.unsqueeze(0)
    #
    #         with torch.no_grad():
    #             if n_gpu == 1:
    #                 loss, loss_slot, acc, acc_slot, pred_slot, tup = model(input_ids, input_len, label_ids, update, n_gpu)
    #             else:
    #                 loss, _, acc, acc_slot, pred_slot, tup_1, tup_2, tup_3 = model(input_ids, input_len, label_ids, update, n_gpu)
    #                 tup = (tup_1.mean(), tup_2.mean(), tup_3.mean())
    #                 nbatch = label_ids.size(0)
    #                 nslot = pred_slot.size(3)
    #                 pred_slot = pred_slot.view(nbatch, -1, nslot)
    #
    #         accuracies = eval_all_accs(pred_slot, label_ids, accuracies)
    #
    #         nb_eval_ex = (label_ids[:,:,0].view(-1) != -1).sum().item()
    #         nb_eval_examples += nb_eval_ex
    #         nb_eval_steps += 1
    #         eval_update_acc += tup[2] * nb_eval_ex
    #
    #         if n_gpu == 1:
    #             eval_loss += loss.item() * nb_eval_ex
    #             eval_accuracy += acc.item() * nb_eval_ex
    #             if eval_loss_slot is None:
    #                 eval_loss_slot = [ l * nb_eval_ex for l in loss_slot]
    #                 eval_acc_slot = acc_slot * nb_eval_ex
    #             else:
    #                 for i, l in enumerate(loss_slot):
    #                     eval_loss_slot[i] = eval_loss_slot[i] + l * nb_eval_ex
    #                 eval_acc_slot += acc_slot * nb_eval_ex
    #         else:
    #             eval_loss += sum(loss) * nb_eval_ex
    #             eval_accuracy += sum(acc) * nb_eval_ex
    #
    #     eval_update_acc = eval_update_acc / nb_eval_examples
    #     eval_loss = eval_loss / nb_eval_examples
    #     eval_accuracy = eval_accuracy / nb_eval_examples
    #     if n_gpu == 1:
    #         eval_acc_slot = eval_acc_slot / nb_eval_examples
    #
    #     loss = tr_loss / nb_tr_steps if args.do_train else None
    #
    #     if n_gpu == 1:
    #         result = {'eval_loss': eval_loss,
    #                   'eval_accuracy': eval_accuracy,
    #                   'loss': loss,
    #                   'eval_loss_slot':'\t'.join([ str(val/ nb_eval_examples) for val in eval_loss_slot]),
    #                   'eval_acc_slot':'\t'.join([ str((val).item()) for val in eval_acc_slot])
    #                     }
    #     else:
    #         result = {'eval_loss': eval_loss,
    #                   'eval_accuracy': eval_accuracy,
    #                   'loss': loss
    #                   }
    #
    #     out_file_name = 'eval_results'
    #     if args.target_slot=='all':
    #         out_file_name += '_all'
    #     output_eval_file = os.path.join(args.output_dir, "%s.txt" % out_file_name)
    #
    #     if n_gpu == 1:
    #         with open(output_eval_file, "w") as writer:
    #             logger.info("***** Eval results *****")
    #             for key in sorted(result.keys()):
    #                 logger.info("  %s = %s", key, str(result[key]))
    #                 writer.write("%s = %s\n" % (key, str(result[key])))
    #
    #     out_file_name = 'eval_all_accuracies'
    #     with open(os.path.join(args.output_dir, "%s.txt" % out_file_name), 'w') as f:
    #         f.write('joint acc (7 domain) : slot acc (7 domain) : joint acc (5 domain): slot acc (5 domain): joint restaurant : slot acc restaurant \n')
    #         f.write('%.5f : %.5f : %.5f : %.5f : %.5f : %.5f \n' % (
    #             (accuracies['joint7']/accuracies['num_turn']).item(),
    #             (accuracies['slot7']/accuracies['num_slot7']).item(),
    #             (accuracies['joint5']/accuracies['num_turn']).item(),
    #             (accuracies['slot5'] / accuracies['num_slot5']).item(),
    #             (accuracies['joint_rest']/accuracies['num_turn']).item(),
    #             (accuracies['slot_rest'] / accuracies['num_slot_rest']).item()
    #         ))

