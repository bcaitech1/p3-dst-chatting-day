import torch

from tqdm.auto import tqdm
import numpy as np
from data_utils import DSTPreprocessor, OpenVocabDSTFeature, convert_state_dict
from tqdm.auto import tqdm

class SomDSTPreprocessor(DSTPreprocessor):
    def __init__():
        ...

class TRADEPreprocessor(DSTPreprocessor):
    def __init__(
            self,
            slot_meta,
            src_tokenizer,
            trg_tokenizer=None,
            ontology=None,
            max_seq_length=512,
            word_drop = None
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.word_drop = word_drop ### word_dropout
        self.gating2id = {"none": 0, "dontcare": 1, "yes": 2, "no": 3, "ptr": 4}
        self.id2gating = {v: k for k, v in self.gating2id.items()}
        self.max_seq_length = max_seq_length

    def _convert_example_to_feature(self, example):
        dialogue_context = " [SEP] ".join(example.context_turns + example.current_turn)

        input_id = self.src_tokenizer.encode(dialogue_context, add_special_tokens=False)
        max_length = self.max_seq_length - 2
        if len(input_id) > max_length:
            gap = len(input_id) - max_length
            input_id = input_id[gap:]

        input_id = (
            [self.src_tokenizer.cls_token_id]
            + input_id
            + [self.src_tokenizer.sep_token_id]
        )
        segment_id = [0] * len(input_id)

        target_ids = []
        gating_id = []
        if not example.label:
            example.label = []

        state = convert_state_dict(example.label)
        for slot in self.slot_meta:
            value = state.get(slot, "none")
            target_id = self.trg_tokenizer.encode(value, add_special_tokens=False) + [
                self.trg_tokenizer.sep_token_id
            ]
            target_ids.append(target_id)
            gating_id.append(self.gating2id.get(value, self.gating2id["ptr"]))
        target_ids = self.pad_ids(target_ids, self.trg_tokenizer.pad_token_id)
        return OpenVocabDSTFeature(
            example.guid, input_id, segment_id, gating_id, target_ids
        )

    def convert_examples_to_features(self, examples):
        return list(map(self._convert_example_to_feature, tqdm(examples)))

    def recover_state(self, gate_list, gen_list):
        assert len(gate_list) == len(self.slot_meta)
        assert len(gen_list) == len(self.slot_meta)

        recovered = []
        for slot, gate, value in zip(self.slot_meta, gate_list, gen_list):
            if self.id2gating[gate] == "none":
                continue

            if self.id2gating[gate] in ["dontcare", "yes", "no"]:
                recovered.append("%s-%s" % (slot, self.id2gating[gate]))
                continue

            token_id_list = []
            for id_ in value:
                if id_ in self.trg_tokenizer.all_special_ids:
                    break

                token_id_list.append(id_)
            value = self.trg_tokenizer.decode(token_id_list, skip_special_tokens=True)

            if value == "none":
                continue

            recovered.append("%s-%s" % (slot, value))
        return recovered

    def collate_fn(self, batch):
        guids = [b.guid for b in batch]
        if self.word_drop > 0.0: ### word_dropout
            input_ids = []
            for b in batch:
                drop_mask = (
                    np.array(
                        self.src_tokenizer.get_special_tokens_mask(b.input_id,
                                                                   already_has_special_tokens=True)
                    ) == 0
                ).astype(int)
                word_drop = np.random.binomial(drop_mask, self.word_drop)
                input_id = [
                    token_id if word_drop[i] == 0 else self.src_tokenizer.unk_token_id
                    for i, token_id in enumerate(b.input_id)
                ]
                input_ids.append(input_id)
            input_ids = torch.LongTensor(
                self.pad_ids([b for b in input_ids],
                             self.src_tokenizer.pad_token_id,
                             max_length=512)
            )
        else:
            input_ids = torch.LongTensor(
                self.pad_ids([b.input_id for b in batch], self.src_tokenizer.pad_token_id)
            )
        segment_ids = torch.LongTensor(
            self.pad_ids([b.segment_id for b in batch], self.src_tokenizer.pad_token_id)
        )
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)

        gating_ids = torch.LongTensor([b.gating_id for b in batch])
        target_ids = self.pad_id_of_matrix(
            [torch.LongTensor(b.target_ids) for b in batch],
            self.trg_tokenizer.pad_token_id,
        )
        return input_ids, segment_ids, input_masks, gating_ids, target_ids, guids
