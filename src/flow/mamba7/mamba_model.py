"""
Mamba7: STLLM2-Style Spatio-Temporal Mamba
基于STLLM2架构重新设计的Mamba时空预测模型

核心改进 (相比原STLLM2):
1. 时序建模: GQA+ALiBi → Mamba SSM (线性复杂度, 更好的长序列建模)
2. 空间建模: SlidingWindowAttention → 图卷积+自适应邻接 (利用真实图拓扑)
3. 前馈网络: MoE → GeGLU+MoE混合 (保留稀疏激活优势)
4. 归一化: RMSNorm (与STLLM2一致)
5. 整体架构: 保持STLLM2的Block堆叠设计

架构示意图:
┌─────────────────────────────────────────────────────────────────┐
│  Input: (B, T, N, F)                                            │
│      ↓                                                          │
│  [Input Embedding + Learnable Time Embedding]                   │
│      ↓                                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  STMambaBlock × num_layers                                │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ 1. Temporal Mamba (时序) - Bi-directional Mamba     │  │  │
│  │  │    RMSNorm → Mamba(forward) + Mamba(backward)       │  │  │
│  │  ├─────────────────────────────────────────────────────┤  │  │
│  │  │ 2. Spatial Graph Conv (空间) - 图卷积+自适应邻接    │  │  │
│  │  │    RMSNorm → GraphConv(A_base + A_adaptive)         │  │  │
│  │  ├─────────────────────────────────────────────────────┤  │  │
│  │  │ 3. MoE FFN (特征变换) - 稀疏专家网络                │  │  │
│  │  │    RMSNorm → Router → Top-K Experts                 │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│      ↓                                                          │
│  [Output: RMSNorm → GeGLU → Linear]                             │
│      ↓                                                          │
│  Output: (B, H, N, F)                                           │
└─────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


# ============================================================================
# 基础组件 (来自STLLM2)
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm - 类似Mistral/LLaMA2中使用的归一化方法"""
    
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class GeGLU(nn.Module):
    """GeGLU激活函数 - GELU门控变体, 来自STLLM2"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(GeGLU, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.gelu(self.w1(x)) * self.w3(x)))


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
    Mixture of Experts (MoE) - 来自Mixtral/STLLM2
    使用稀疏激活的专家网络
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
# 时序建模: Mamba替换GQA+ALiBi
# ============================================================================

class BidirectionalMamba(nn.Module):
    """
    双向Mamba - 替换STLLM2中的GQA+ALiBi
    
    优势:
    1. O(T) 线性复杂度 vs O(T²) 注意力
    2. 选择性状态空间,自动学习关注重要信息
    3. 双向扫描捕获前后文依赖
    """
    
    def __init__(self, d_model, dropout=0.1):
        super(BidirectionalMamba, self).__init__()
        self.mamba_forward = Mamba(d_model=d_model)
        self.mamba_backward = Mamba(d_model=d_model)
        self.merge = nn.Linear(d_model * 2, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (B, T, D)
        """
        # 前向Mamba
        out_forward = self.mamba_forward(x)
        
        # 后向Mamba (翻转序列)
        x_backward = torch.flip(x, dims=[1])
        out_backward = self.mamba_backward(x_backward)
        out_backward = torch.flip(out_backward, dims=[1])
        
        # 融合双向输出
        out = torch.cat([out_forward, out_backward], dim=-1)
        out = self.merge(out)
        
        return self.dropout(out)


class TemporalMambaBlock(nn.Module):
    """
    时序Mamba块 - 替换STLLM2的Temporal GQA
    结构: RMSNorm → BidirectionalMamba → 残差
    """
    
    def __init__(self, d_model, dropout=0.1):
        super(TemporalMambaBlock, self).__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = BidirectionalMamba(d_model, dropout)
    
    def forward(self, x):
        """x: (B, T, D)"""
        return x + self.mamba(self.norm(x))


# ============================================================================
# 空间建模: 图卷积替换SlidingWindowAttention
# ============================================================================

class AdaptiveGraphConv(nn.Module):
    """
    自适应图卷积 - 替换STLLM2的SlidingWindowAttention
    
    改进点:
    1. 利用真实图拓扑 (邻接矩阵) 而非简单滑动窗口
    2. 结合先验图结构和学习到的自适应依赖
    3. 多头图注意力增强表达能力
    
    A_combined = α × A_base + (1-α) × A_adaptive
    其中 A_adaptive = softmax(E @ E^T)
    """
    
    def __init__(self, node_num, d_model, embed_dim=16, base_adj=None, dropout=0.1):
        super(AdaptiveGraphConv, self).__init__()
        self.node_num = node_num
        self.d_model = d_model
        
        # 节点嵌入用于自适应邻接矩阵
        self.node_embed = nn.Parameter(torch.randn(node_num, embed_dim) * 0.02)
        
        # 消息传递投影
        self.msg_proj = nn.Linear(d_model, d_model, bias=False)
        self.update_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 可学习的融合系数
        self.alpha = nn.Parameter(torch.tensor(0.7))
        
        self.dropout = nn.Dropout(dropout)
        
        # 处理基础邻接矩阵
        if base_adj is not None:
            adj = base_adj.clone().detach().float()
            adj = adj + torch.eye(node_num, device=adj.device, dtype=adj.dtype)
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            adj = adj / deg
            self.register_buffer("base_adj", adj)
        else:
            self.register_buffer("base_adj", None)
    
    def forward(self, x):
        """
        x: (B*T, N, D)
        return: (B*T, N, D)
        """
        # 计算自适应邻接矩阵
        adaptive_adj = torch.mm(self.node_embed, self.node_embed.t())
        adaptive_adj = F.softmax(adaptive_adj / (self.node_embed.size(1) ** 0.5), dim=-1)
        
        # 融合邻接矩阵
        alpha = torch.sigmoid(self.alpha)
        if self.base_adj is not None:
            adj = alpha * self.base_adj + (1 - alpha) * adaptive_adj
        else:
            adj = adaptive_adj
        
        # 消息传递
        msg = self.msg_proj(x)  # (B*T, N, D)
        agg = torch.matmul(adj, msg)  # (B*T, N, D)
        out = self.update_proj(agg)  # (B*T, N, D)
        
        return self.dropout(out)


class SpatialGraphBlock(nn.Module):
    """
    空间图卷积块 - 替换STLLM2的Spatial SlidingWindowAttention
    结构: RMSNorm → AdaptiveGraphConv → 残差
    """
    
    def __init__(self, node_num, d_model, embed_dim=16, base_adj=None, dropout=0.1):
        super(SpatialGraphBlock, self).__init__()
        self.norm = RMSNorm(d_model)
        self.graph_conv = AdaptiveGraphConv(node_num, d_model, embed_dim, base_adj, dropout)
    
    def forward(self, x):
        """x: (B*T, N, D)"""
        return x + self.graph_conv(self.norm(x))


# ============================================================================
# STMamba Block: 完整的时空Mamba块
# ============================================================================

class STMambaBlock(nn.Module):
    """
    ST-Mamba块 - 对应STLLM2Block的Mamba版本
    
    结构对比:
    STLLM2Block:                          STMambaBlock:
    ├── Temporal GQA + ALiBi        →    ├── Temporal Mamba (双向)
    ├── Spatial SlidingWindow       →    ├── Spatial GraphConv (自适应图)
    └── MoE FFN                     →    └── MoE FFN (保留)
    
    所有子模块采用 Pre-norm + 残差 设计
    """
    
    def __init__(
        self, 
        d_model, 
        node_num,
        d_ff=256,
        num_experts=4, 
        top_k=2,
        embed_dim=16,
        base_adj=None,
        dropout=0.1
    ):
        super(STMambaBlock, self).__init__()
        
        # 1. 时序Mamba (替换GQA+ALiBi)
        self.temporal_norm = RMSNorm(d_model)
        self.temporal_mamba = BidirectionalMamba(d_model, dropout)
        
        # 2. 空间图卷积 (替换SlidingWindowAttention)
        self.spatial_norm = RMSNorm(d_model)
        self.spatial_graph = AdaptiveGraphConv(node_num, d_model, embed_dim, base_adj, dropout)
        
        # 3. MoE前馈网络 (保留STLLM2设计)
        self.ffn_norm = RMSNorm(d_model)
        self.moe = MixtureOfExperts(d_model, d_ff, num_experts, top_k, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (B, T, N, D)
        """
        batch_size, seq_len, node_num, d_model = x.shape
        
        # 1. 时序Mamba处理 (每个节点独立处理时序)
        x_reshape = x.permute(0, 2, 1, 3).contiguous().view(batch_size * node_num, seq_len, d_model)
        x_norm = self.temporal_norm(x_reshape)
        temporal_out = self.temporal_mamba(x_norm)
        x_reshape = x_reshape + self.dropout(temporal_out)
        x = x_reshape.view(batch_size, node_num, seq_len, d_model).permute(0, 2, 1, 3)
        
        # 2. 空间图卷积处理 (每个时间步独立处理空间)
        x_reshape = x.contiguous().view(batch_size * seq_len, node_num, d_model)
        x_norm = self.spatial_norm(x_reshape)
        spatial_out = self.spatial_graph(x_norm)
        x_reshape = x_reshape + self.dropout(spatial_out)
        x = x_reshape.view(batch_size, seq_len, node_num, d_model)
        
        # 3. MoE前馈网络
        x_reshape = x.view(batch_size * seq_len, node_num, d_model)
        x_norm = self.ffn_norm(x_reshape)
        moe_out = self.moe(x_norm)
        x_reshape = x_reshape + moe_out
        x = x_reshape.view(batch_size, seq_len, node_num, d_model)
        
        return x


# ============================================================================
# 主模型: UNetMamba (Mamba7)
# ============================================================================

class UNetMamba(BaseModel):
    """
    Mamba7: STLLM2-Style Spatio-Temporal Mamba
    
    基于STLLM2架构,使用Mamba替换Attention的时空预测模型
    
    与STLLM2的对应关系:
    ┌─────────────────────────────────────────────────────────────────┐
    │  STLLM2                         │  Mamba7 (本模型)              │
    ├─────────────────────────────────┼───────────────────────────────┤
    │  Input Embedding                │  Input Embedding              │
    │  + Time Embedding (可学习)       │  + Time Embedding (可学习)    │
    ├─────────────────────────────────┼───────────────────────────────┤
    │  STLLM2Block × N                │  STMambaBlock × N             │
    │  ├─ GQA + ALiBi (时序)          │  ├─ BiMamba (时序)            │
    │  ├─ SlidingWindow (空间)        │  ├─ GraphConv (空间)          │
    │  └─ MoE FFN                     │  └─ MoE FFN                   │
    ├─────────────────────────────────┼───────────────────────────────┤
    │  RMSNorm → GeGLU → Linear       │  RMSNorm → GeGLU → Linear     │
    └─────────────────────────────────┴───────────────────────────────┘
    
    主要改进:
    1. 时序建模: Mamba提供O(T)复杂度,更好的长序列建模
    2. 空间建模: 图卷积利用真实拓扑,优于滑动窗口
    3. 参数效率: Mamba参数更少,推理更快
    """
    
    def __init__(
        self,
        d_model,
        num_layers,
        feature,
        adj=None,
        graph_embed_dim=16,
        dropout=0.1,
        d_ff=256,
        num_experts=4,
        top_k=2,
        num_graph_layers=1,  # 保留兼容性,但不再使用
        gate_init=-2.0,      # 保留兼容性,但不再使用
        **args
    ):
        super(UNetMamba, self).__init__(**args)
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.feature = feature

        # 输入嵌入 (与STLLM2一致)
        self.input_embedding = nn.Linear(self.feature, d_model)
        
        # 可学习的时间嵌入 (与STLLM2一致)
        self.time_embedding = nn.Parameter(torch.randn(1, 512, 1, d_model) * 0.02)
        
        # ST-Mamba blocks (对应STLLM2的STLLM2Block)
        self.blocks = nn.ModuleList([
            STMambaBlock(
                d_model=d_model,
                node_num=self.node_num,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=top_k,
                embed_dim=graph_embed_dim,
                base_adj=adj,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 输出层 (与STLLM2一致)
        self.output_norm = RMSNorm(d_model)
        self.output_gate = GeGLU(d_model, d_model * 2, dropout)
        self.output_proj = nn.Linear(d_model, self.output_dim * self.horizon)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重 (与STLLM2一致)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, label=None):  # (B, T, N, F)
        batch_size, seq_len, node_num, input_dim = x.shape

        # 输入嵌入
        x = self.input_embedding(x)  # (B, T, N, d_model)
        
        # 添加时间嵌入
        x = x + self.time_embedding[:, :seq_len, :, :]
        
        # 通过ST-Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # 输出处理 (与STLLM2一致)
        x = self.output_norm(x)
        x = x[:, -1, :, :]  # 取最后时间步 (B, N, D)
        x = self.output_gate(x)
        x = self.output_proj(x)  # (B, N, output_dim * horizon)
        
        # Reshape输出
        x = x.view(batch_size, node_num, self.horizon, self.output_dim)
        x = x.permute(0, 2, 1, 3)  # (B, H, N, F)

        return x
