"""st_llm_plus adapts the GPT2+LoRA+graph-attention baseline to the current flow runner."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import GPT2Model
from models.base import BaseModel
from engine.recipe import ModelRecipe
from data.loader import load_adj_from_numpy


class TemporalProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super().__init__()
        self.position_emb = nn.Parameter(torch.empty(1, input_dim, 1, seq_len))
        self.proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, seq_len))
        nn.init.xavier_uniform_(self.position_emb)

    def forward(self, x):
        return self.proj(x + self.position_emb)


class PartiallyFrozenGraphAttention(nn.Module):
    def __init__(
        self,
        pretrained_model="gpt2",
        gpt_layers=6,
        unfreeze_layers=1,
        lora_rank=16,
        lora_dropout=0.1,
    ):
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
                fan_in_fan_out=True,
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

    def custom_forward(self, inputs_embeds, adjacency_matrix=None):
        """Apply GPT-2 blocks using the Transformers 5 tensor-return API."""
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.gpt2.wpe(positions.unsqueeze(0))
        total_layers = len(self.gpt2.h)
        for index, block in enumerate(self.gpt2.h):
            mask = adjacency_matrix if index >= total_layers - self.unfreeze_layers else None
            hidden_states = block(
                hidden_states,
                attention_mask=mask.to(hidden_states.device) if mask is not None else None,
                use_cache=False,
            )
        return self.gpt2.ln_f(hidden_states)

    def forward(self, x, adjacency_matrix):
        batch_size = x.shape[0]
        num_heads = self.gpt2.config.n_head
        graph_mask = adjacency_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
        graph_mask = graph_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
        # Pass the graph mask via adjacency_matrix so custom_forward applies it
        # only to the last U layers; the rest run with standard (None) attention.
        output = self.custom_forward(inputs_embeds=x, adjacency_matrix=graph_mask)
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
        self.regression_layer = nn.Conv2d(
            self.gpt_hidden, self.horizon * self.output_dim, kernel_size=(1, 1)
        )

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
        flattened = (
            data.transpose(1, 2).reshape(batch_size, node_num, -1).transpose(1, 2).unsqueeze(-1)
        )
        input_context = self.start_conv(flattened)

        node_context = (
            self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
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


def setup(config, data_path, adj_path, node_num, device, logger):
    del config, data_path, node_num, device, logger
    return {"adj_mx": load_adj_from_numpy(adj_path)}


def build_model(config, node_num, **ctx):
    from config.paths import load_paths

    pretrained = config.model.params.pretrained_model
    local = load_paths().models / pretrained
    pretrained = str(local) if local.is_dir() else pretrained
    return STLLMPlus(
        adj_mx=ctx["adj_mx"],
        node_num=node_num,
        input_dim=config.data.input_dim,
        output_dim=config.data.output_dim,
        seq_len=config.data.seq_len,
        horizon=config.data.horizon,
        gpt_channel=config.model.params.gpt_channel,
        llm_layer=config.model.params.llm_layer,
        U=config.model.params.U,
        lora_rank=config.model.params.lora_rank,
        dropout=config.model.params.dropout,
        pretrained_model=pretrained,
    )


def get_recipe():
    return ModelRecipe(build_model=build_model, setup=setup)
