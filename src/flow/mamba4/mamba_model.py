import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

# ============================================================================
# Basic Components (Aligned with STLLM2)
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm - Similar to Mistral/LLaMA2"""
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class GeGLU(nn.Module):
    """GeGLU Activation - Gated GELU"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(GeGLU, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.gelu(self.w1(x)) * self.w3(x)))


# ============================================================================
# Spatial Modeling (Aligned with STLLM2)
# ============================================================================

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention - Similar to Mistral
    Used for Spatial Dimension (Nodes)
    """
    def __init__(self, d_model, num_heads, window_size=4, dropout=0.1):
        super(SlidingWindowAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def _create_sliding_window_mask(self, seq_len, device):
        mask = torch.ones(seq_len, seq_len, device=device) * float('-inf')
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 0
        return mask
    
    def forward(self, x):
        # x: (Batch, Seq, D) - In our case (B*T, N, D)
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        window_mask = self._create_sliding_window_mask(seq_len, x.device)
        attn = attn + window_mask.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


# ============================================================================
# FFN Modeling (Aligned with STLLM2)
# ============================================================================

class Expert(nn.Module):
    """Single Expert for MoE"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Expert, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) - Similar to Mixtral
    """
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2, dropout=0.1):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            expert_mask = (top_k_indices == i).any(dim=-1)
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = expert(expert_input)
                
                expert_weights = torch.where(
                    top_k_indices == i,
                    top_k_probs,
                    torch.zeros_like(top_k_probs)
                ).sum(dim=-1)[expert_mask]
                
                output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)
        
        return output.view(batch_size, seq_len, d_model)


# ============================================================================
# Temporal Modeling: Mamba (Replaces STLLM2's GQA + ALiBi)
# ============================================================================

class TemporalMamba(nn.Module):
    """
    Temporal Mamba
    Replaces GroupedQueryAttention (GQA) from STLLM2 for O(T) efficiency.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super(TemporalMamba, self).__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is not installed.")
            
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, T, D)
        out = self.mamba(x)
        return self.dropout(out)


# ============================================================================
# STMamba Block
# ============================================================================

class STMambaBlock(nn.Module):
    """
    ST-Mamba Block
    Structure aligned with STLLM2Block:
    1. Temporal: Mamba (Replaces GQA)
    2. Spatial: SlidingWindowAttention (Same as STLLM2)
    3. FFN: Mixture of Experts (Same as STLLM2)
    """
    def __init__(
        self, 
        d_model, 
        num_heads,
        d_ff,
        num_experts=4, 
        top_k=2,
        window_size=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    ):
        super(STMambaBlock, self).__init__()
        
        # 1. Temporal Mamba
        self.temporal_norm = RMSNorm(d_model)
        self.temporal_mamba = TemporalMamba(d_model, d_state, d_conv, expand, dropout)
        
        # 2. Spatial Sliding Window Attention
        self.spatial_norm = RMSNorm(d_model)
        self.spatial_attn = SlidingWindowAttention(d_model, num_heads, window_size, dropout)
        
        # 3. MoE FFN
        self.ffn_norm = RMSNorm(d_model)
        self.moe = MixtureOfExperts(d_model, d_ff, num_experts, top_k, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, node_num, d_model = x.shape
        
        # 1. Temporal Mamba
        # Reshape: (B, T, N, D) -> (B*N, T, D)
        x_reshape = x.permute(0, 2, 1, 3).contiguous().view(batch_size * node_num, seq_len, d_model)
        x_norm = self.temporal_norm(x_reshape)
        temporal_out = self.temporal_mamba(x_norm)
        x_reshape = x_reshape + self.dropout(temporal_out)
        x = x_reshape.view(batch_size, node_num, seq_len, d_model).permute(0, 2, 1, 3)
        
        # 2. Spatial Sliding Window Attention
        # Reshape: (B, T, N, D) -> (B*T, N, D)
        x_reshape = x.contiguous().view(batch_size * seq_len, node_num, d_model)
        x_norm = self.spatial_norm(x_reshape)
        spatial_out = self.spatial_attn(x_norm)
        x_reshape = x_reshape + self.dropout(spatial_out)
        x = x_reshape.view(batch_size, seq_len, node_num, d_model)
        
        # 3. MoE FFN
        x_reshape = x.view(batch_size * seq_len, node_num, d_model)
        x_norm = self.ffn_norm(x_reshape)
        moe_out = self.moe(x_norm)
        x_reshape = x_reshape + moe_out
        x = x_reshape.view(batch_size, seq_len, node_num, d_model)
        
        return x


# ============================================================================
# Main Model
# ============================================================================

class UNetMamba(BaseModel):
    """
    Mamba4: STLLM2-Style Spatio-Temporal Mamba
    """
    def __init__(
        self,
        d_model=96,
        num_heads=8,
        d_ff=256,
        num_layers=3,
        num_experts=4,
        top_k=2,
        window_size=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.3,
        **args
    ):
        super(UNetMamba, self).__init__(**args)
        
        self.d_model = d_model
        
        # Input Embedding
        self.input_embedding = nn.Linear(self.input_dim, d_model)
        
        # Time Embedding (Kept from STLLM2)
        self.time_embedding = nn.Parameter(torch.randn(1, 512, 1, d_model) * 0.02)
        
        # Blocks
        self.blocks = nn.ModuleList([
            STMambaBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=top_k,
                window_size=window_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output Head (Kept from STLLM2)
        self.output_norm = RMSNorm(d_model)
        self.output_gate = GeGLU(d_model, d_model * 2, dropout)
        self.output_proj = nn.Linear(d_model, self.output_dim * self.horizon)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, label=None):
        batch_size, seq_len, node_num, input_dim = x.shape

        # Input Embedding
        x = self.input_embedding(x)
        
        # Add Time Embedding
        x = x + self.time_embedding[:, :seq_len, :, :]
        
        # Blocks
        for block in self.blocks:
            x = block(x)
        
        # Output Head
        x = self.output_norm(x)
        x = x[:, -1, :, :]  # Take last time step
        x = self.output_gate(x)
        x = self.output_proj(x)
        
        x = x.view(batch_size, node_num, self.horizon, self.output_dim)
        x = x.permute(0, 2, 1, 3)

        return x
