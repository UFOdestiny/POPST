import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base.model import BaseModel
from mamba_ssm import Mamba

class RMSNorm(nn.Module):
    """RMSNorm - 类似Mistral/LLaMA2中使用的归一化方法"""
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class SlidingWindowAttention(nn.Module):
    """
    滑动窗口注意力 - 来自Mistral
    限制注意力范围以提高效率
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
        """创建滑动窗口掩码"""
        mask = torch.ones(seq_len, seq_len, device=device) * float('-inf')
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 0
        return mask
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 应用滑动窗口掩码
        window_mask = self._create_sliding_window_mask(seq_len, x.device)
        attn = attn + window_mask.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)

class Expert(nn.Module):
    """MoE中的单个专家"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Expert, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.gelu(self.w1(x))))

class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) - 来自Mixtral
    使用稀疏激活的专家网络
    """
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2, dropout=0.1):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 路由网络
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (batch * seq, d_model)
        
        # 计算路由分数
        router_logits = self.router(x_flat)  # (batch * seq, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 选择top-k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # 归一化
        
        # 计算专家输出
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # 找到选择了这个专家的token
            expert_mask = (top_k_indices == i).any(dim=-1)
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = expert(expert_input)
                
                # 获取这个专家的权重
                expert_weights = torch.where(
                    top_k_indices == i,
                    top_k_probs,
                    torch.zeros_like(top_k_probs)
                ).sum(dim=-1)[expert_mask]
                
                output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)
        
        return output.view(batch_size, seq_len, d_model)

class GeGLU(nn.Module):
    """GeGLU激活函数 - GELU门控变体"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(GeGLU, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.gelu(self.w1(x)) * self.w3(x)))

class BidirectionalMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.fwd_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        fwd = self.fwd_mamba(x)
        bwd = self.bwd_mamba(x.flip([1])).flip([1])
        out = torch.cat([fwd, bwd], dim=-1)
        out = self.out_proj(out)
        return self.dropout(out)

class Mamba5Block(nn.Module):
    """
    Mamba5 Block
    Temporal: Bidirectional Mamba
    Spatial: Sliding Window Attention
    FFN: MoE
    """
    def __init__(self, d_model, num_heads, d_ff, 
                 num_experts=4, top_k=2, window_size=4, dropout=0.1, 
                 d_state=16, d_conv=4, expand=2):
        super(Mamba5Block, self).__init__()
        
        # 时序混合: Bidirectional Mamba
        self.temporal_norm = RMSNorm(d_model)
        self.temporal_mixer = BidirectionalMamba(d_model, d_state, d_conv, expand, dropout)
        
        # 空间混合: 滑动窗口注意力 (保留STLLM2的设计)
        self.spatial_norm = RMSNorm(d_model)
        self.spatial_attn = SlidingWindowAttention(
            d_model, num_heads, window_size, dropout
        )
        
        # MoE前馈网络
        self.ffn_norm = RMSNorm(d_model)
        self.moe = MixtureOfExperts(d_model, d_ff, num_experts, top_k, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, node_num, d_model = x.shape
        
        # Temporal Mamba (per node)
        x_reshape = x.permute(0, 2, 1, 3).contiguous().view(batch_size * node_num, seq_len, d_model)
        x_norm = self.temporal_norm(x_reshape)
        temporal_out = self.temporal_mixer(x_norm)
        x_reshape = x_reshape + temporal_out
        x = x_reshape.view(batch_size, node_num, seq_len, d_model).permute(0, 2, 1, 3)
        
        # Spatial Sliding Window Attention (per timestamp)
        x_reshape = x.contiguous().view(batch_size * seq_len, node_num, d_model)
        x_norm = self.spatial_norm(x_reshape)
        spatial_out = self.spatial_attn(x_norm)
        x_reshape = x_reshape + self.dropout(spatial_out)
        x = x_reshape.view(batch_size, seq_len, node_num, d_model)
        
        # MoE前馈网络
        x_reshape = x.view(batch_size * seq_len, node_num, d_model)
        x_norm = self.ffn_norm(x_reshape)
        moe_out = self.moe(x_norm)
        x_reshape = x_reshape + moe_out
        x = x_reshape.view(batch_size, seq_len, node_num, d_model)
        
        return x

class Mamba5(BaseModel):
    """
    Mamba5: 基于STLLM2架构的Mamba改进版
    特点:
    1. 使用双向Mamba替代Temporal GQA
    2. 保留Spatial Sliding Window Attention
    3. 保留MoE FFN
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
        dropout=0.1,
        d_state=16,
        d_conv=4,
        expand=2,
        **args
    ):
        super(Mamba5, self).__init__(**args)
        
        self.d_model = d_model
        
        # 输入嵌入
        self.input_embedding = nn.Linear(self.input_dim, d_model)
        
        # 可学习的时间嵌入 (保留STLLM2设计)
        self.time_embedding = nn.Parameter(torch.randn(1, 512, 1, d_model) * 0.02)
        
        # Mamba5 Blocks
        self.blocks = nn.ModuleList([
            Mamba5Block(
                d_model, num_heads, d_ff,
                num_experts, top_k, window_size, dropout,
                d_state, d_conv, expand
            )
            for _ in range(num_layers)
        ])
        
        # 输出层
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
        
        # 输入嵌入
        x = self.input_embedding(x)
        
        # 添加时间嵌入
        x = x + self.time_embedding[:, :seq_len, :, :]
        
        # 通过Blocks
        for block in self.blocks:
            x = block(x)
        
        # 输出
        x = self.output_norm(x)
        x = x[:, -1, :, :]  # 取最后时间步
        x = self.output_gate(x)
        x = self.output_proj(x)
        
        x = x.view(batch_size, node_num, self.horizon, self.output_dim)
        x = x.permute(0, 2, 1, 3)
        
        return x
