"""ST-LLM-plus adapts the GPT2+LoRA+graph-attention baseline to the current flow runner."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import GPT2Model

from base.model import BaseModel


class TemporalProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super().__init__()
        self.position_emb = nn.Parameter(torch.empty(1, input_dim, 1, seq_len))
        self.proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, seq_len))
        nn.init.xavier_uniform_(self.position_emb)

    def forward(self, x):
        return self.proj(x + self.position_emb)


@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class PartiallyFrozenGraphAttention(nn.Module):
    def __init__(self, pretrained_model="gpt2", gpt_layers=6, unfreeze_layers=1, lora_rank=16, lora_dropout=0.1):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(
            pretrained_model,
            attn_implementation="eager",
            output_attentions=False,
            output_hidden_states=False,
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.unfreeze_layers = unfreeze_layers

        self.gpt2 = get_peft_model(
            self.gpt2,
            LoraConfig(
                r=lora_rank,
                lora_alpha=32,
                lora_dropout=lora_dropout,
                target_modules=["q_attn", "c_attn"],
                bias="none",
            ),
        )
        self.dropout = nn.Dropout(lora_dropout)
        self._configure_trainable_layers(gpt_layers)

    @property
    def hidden_size(self):
        return self.gpt2.config.hidden_size

    def _configure_trainable_layers(self, gpt_layers):
        frozen_until = max(gpt_layers - self.unfreeze_layers, 0)
        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < frozen_until:
                    param.requires_grad = "ln" in name
                else:
                    param.requires_grad = "mlp" not in name

    def custom_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        adjacency_matrix: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, dict]:
        del token_type_ids, encoder_hidden_states, encoder_attention_mask, adjacency_matrix

        output_attentions = output_attentions if output_attentions is not None else self.gpt2.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.gpt2.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else self.gpt2.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.gpt2.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.gpt2.wte(input_ids)
        hidden_states = inputs_embeds + self.gpt2.wpe(position_ids)

        all_self_attentions = () if output_attentions else None
        presents = () if use_cache else None

        for i, (block, layer_past) in enumerate(zip(self.gpt2.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask.to(hidden_states.device) if attention_mask is not None else None,
                head_mask=head_mask[i] if head_mask is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
            if use_cache and len(outputs) > 1:
                presents = presents + (outputs[1],)
            if output_attentions:
                attn_index = 2 if use_cache else 1
                if len(outputs) > attn_index:
                    all_self_attentions = all_self_attentions + (outputs[attn_index],)

        hidden_states = self.gpt2.ln_f(hidden_states)
        hidden_states = hidden_states.reshape((-1,) + input_shape[1:] + (hidden_states.size(-1),))

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            attentions=all_self_attentions,
        )

    def forward(self, x, adjacency_matrix):
        batch_size = x.shape[0]
        num_heads = self.gpt2.config.n_head
        attention_mask = adjacency_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
        output = self.custom_forward(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
        return self.dropout(output)


class STLLMPlus(BaseModel):
    def __init__(
        self,
        adj_mx,
        input_dim,
        output_dim,
        node_num,
        seq_len,
        horizon,
        gpt_channel=256,
        llm_layer=6,
        U=1,
        lora_rank=16,
        dropout=0.1,
        pretrained_model="gpt2",
    ):
        super().__init__(
            node_num=node_num,
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            horizon=horizon,
        )

        self.gpt = PartiallyFrozenGraphAttention(
            pretrained_model=pretrained_model,
            gpt_layers=llm_layer,
            unfreeze_layers=U,
            lora_rank=lora_rank,
            lora_dropout=dropout,
        )
        self.gpt_hidden = self.gpt.hidden_size
        self.start_conv = nn.Conv2d(self.input_dim * self.seq_len, gpt_channel, kernel_size=(1, 1))
        self.temporal_proj = TemporalProjection(self.input_dim, gpt_channel, self.seq_len)
        self.node_emb = nn.Parameter(torch.empty(self.node_num, gpt_channel))
        self.in_layer = nn.Conv2d(gpt_channel * 3, self.gpt_hidden, kernel_size=(1, 1))
        self.dropout = nn.Dropout(dropout)
        self.regression_layer = nn.Conv2d(self.gpt_hidden, self.horizon * self.output_dim, kernel_size=(1, 1))

        adj_tensor = self._prepare_adj(adj_mx)
        self.register_buffer("adj_mx", adj_tensor)
        nn.init.xavier_uniform_(self.node_emb)

    @staticmethod
    def _prepare_adj(adj_mx):
        adj = np.asarray(adj_mx, dtype=np.float32)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"Expected square adjacency matrix, got shape {adj.shape}")
        adj = np.maximum(adj, adj.T)
        adj = adj + np.eye(adj.shape[0], dtype=np.float32)
        max_val = float(adj.max())
        if max_val > 0:
            adj = adj / max_val
        return torch.tensor(adj, dtype=torch.float32)

    def forward(self, history_data, label=None):
        del label
        data = history_data.permute(0, 3, 2, 1).contiguous()
        batch_size, _, node_num, _ = data.shape

        temporal_context = self.temporal_proj(data)
        flattened = data.transpose(1, 2).reshape(batch_size, node_num, -1).transpose(1, 2).unsqueeze(-1)
        input_context = self.start_conv(flattened)

        node_context = (
            self.node_emb.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )

        gpt_input = torch.cat([input_context, temporal_context, node_context], dim=1)
        gpt_input = F.leaky_relu(self.in_layer(gpt_input))
        gpt_input = self.dropout(gpt_input).permute(0, 2, 1, 3).squeeze(-1)

        outputs = self.gpt(gpt_input, self.adj_mx)
        outputs = outputs.permute(0, 2, 1).unsqueeze(-1)
        outputs = self.regression_layer(outputs)
        outputs = outputs.squeeze(-1).permute(0, 2, 1)
        outputs = outputs.reshape(batch_size, node_num, self.horizon, self.output_dim)
        return outputs.permute(0, 2, 1, 3)
