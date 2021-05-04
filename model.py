import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraModel


def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss


class TRADE(nn.Module):
    def __init__(self, config, tokenized_slot_meta, pad_idx=0):
        super(TRADE, self).__init__()
        self.encoder = GRUEncoder(
            config.vocab_size,
            config.hidden_size,
            1,
            config.hidden_dropout_prob,
            config.proj_dim,
            pad_idx,
        )

        self.decoder = SlotGenerator(
            config.vocab_size,
            config.hidden_size,
            config.hidden_dropout_prob,
            config.n_gate,
            config.proj_dim,
            pad_idx,
        )

        self.decoder.set_slot_idx(tokenized_slot_meta)
        self.tie_weight()
        
    def set_subword_embedding(self, model_name_or_path):
        model = ElectraModel.from_pretrained(model_name_or_path)
        self.encoder.embed.weight = model.embeddings.word_embeddings.weight
        self.tie_weight()

    def tie_weight(self):
        self.decoder.embed.weight = self.encoder.embed.weight
        if self.decoder.proj_layer:
            self.decoder.proj_layer.weight = self.encoder.proj_layer.weight

    def forward(
        self, input_ids, token_type_ids, attention_mask=None, max_len=10, teacher=None
    ):

        encoder_outputs, pooled_output = self.encoder(input_ids=input_ids)
        all_point_outputs, all_gate_outputs = self.decoder(
            input_ids,
            encoder_outputs,
            pooled_output.unsqueeze(0),
            attention_mask,
            max_len,
            teacher,
        )

        return all_point_outputs, all_gate_outputs


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, dropout, proj_dim=None, pad_idx=0):
        super(GRUEncoder, self).__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        if proj_dim:
            self.proj_layer = nn.Linear(d_model, proj_dim, bias=False)
        else:
            self.proj_layer = None

        self.d_model = proj_dim if proj_dim else d_model
        self.gru = nn.GRU(
            self.d_model,
            self.d_model,
            n_layer,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        mask = input_ids.eq(self.pad_idx).unsqueeze(-1)
        x = self.embed(input_ids)
        if self.proj_layer:
            x = self.proj_layer(x)
        x = self.dropout(x)
        o, h = self.gru(x)
        o = o.masked_fill(mask, 0.0)
        output = o[:, :, : self.d_model] + o[:, :, self.d_model :]
        hidden = h[0] + h[1]  # n_layer 고려
        return output, hidden


class SlotGenerator(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, dropout, n_gate, proj_dim=None, pad_idx=0
    ):
        super(SlotGenerator, self).__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_idx
        )  # shared with encoder

        if proj_dim:
            self.proj_layer = nn.Linear(hidden_size, proj_dim, bias=False)
        else:
            self.proj_layer = None
        self.hidden_size = proj_dim if proj_dim else hidden_size

        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size, 1, dropout=dropout, batch_first=True
        )
        self.n_gate = n_gate
        self.dropout = nn.Dropout(dropout)
        self.w_gen = nn.Linear(self.hidden_size * 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.w_gate = nn.Linear(self.hidden_size, n_gate)

    def set_slot_idx(self, slot_vocab_idx):
        whole = []
        max_length = max(map(len, slot_vocab_idx))
        for idx in slot_vocab_idx:
            if len(idx) < max_length:
                gap = max_length - len(idx)
                idx.extend([self.pad_idx] * gap)
            whole.append(idx)
        self.slot_embed_idx = whole  # torch.LongTensor(whole)

    def embedding(self, x):
        x = self.embed(x)
        if self.proj_layer:
            x = self.proj_layer(x)
        return x

    def forward(
        self, input_ids, encoder_output, hidden, input_masks, max_len, teacher=None
    ):
        input_masks = input_masks.ne(1)
        # J, slot_meta : key : [domain, slot] ex> LongTensor([1,2])
        # J,2
        batch_size = encoder_output.size(0)
        slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device)  ##
        slot_e = torch.sum(self.embedding(slot), 1)  # J,d
        J = slot_e.size(0)

        all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size).to(
            input_ids.device
        )
        
        # Parallel Decoding
        w = slot_e.repeat(batch_size, 1).unsqueeze(1)
        hidden = hidden.repeat_interleave(J, dim=1)
        encoder_output = encoder_output.repeat_interleave(J, dim=0)
        input_ids = input_ids.repeat_interleave(J, dim=0)
        input_masks = input_masks.repeat_interleave(J, dim=0)
        for k in range(max_len):
            w = self.dropout(w)
            _, hidden = self.gru(w, hidden)  # 1,B,D

            # B,T,D * B,D,1 => B,T
            attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
            attn_e = attn_e.squeeze(-1).masked_fill(input_masks, -1e9)
            attn_history = F.softmax(attn_e, -1)  # B,T

            if self.proj_layer:
                hidden_proj = torch.matmul(hidden, self.proj_layer.weight)
            else:
                hidden_proj = hidden

            # B,D * D,V => B,V
            attn_v = torch.matmul(
                hidden_proj.squeeze(0), self.embed.weight.transpose(0, 1)
            )  # B,V
            attn_vocab = F.softmax(attn_v, -1)

            # B,1,T * B,T,D => B,1,D
            context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D
            p_gen = self.sigmoid(
                self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1))
            )  # B,1
            p_gen = p_gen.squeeze(-1)

            p_context_ptr = torch.zeros_like(attn_vocab).to(input_ids.device)
            p_context_ptr.scatter_add_(1, input_ids, attn_history)  # copy B,V
            p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
            _, w_idx = p_final.max(-1)

            if teacher is not None:
                w = self.embedding(teacher[:, :, k]).transpose(0, 1).reshape(batch_size * J, 1, -1)
            else:
                w = self.embedding(w_idx).unsqueeze(1)  # B,1,D
            if k == 0:
                gated_logit = self.w_gate(context.squeeze(1))  # B,3
                all_gate_outputs = gated_logit.view(batch_size, J, self.n_gate)
            all_point_outputs[:, :, k, :] = p_final.view(batch_size, J, self.vocab_size)

        return all_point_outputs, all_gate_outputs
