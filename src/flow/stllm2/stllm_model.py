import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base.model import BaseModel


class RMSNorm(nn.Module):
    """RMSNorm - 类似Mistral/LLaMA2中使用的归一化方法"""
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class ALiBiPositionalBias(nn.Module):
    """
    ALiBi (Attention with Linear Biases) - 来自BLOOM模型
    不同于RoPE，使用线性偏置来编码位置信息
    """
    def __init__(self, num_heads, max_seq_len=512):
        super(ALiBiPositionalBias, self).__init__()
        self.num_heads = num_heads
        
        # 计算每个头的斜率
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        
        # 预计算位置偏置
        self._build_alibi_tensor(max_seq_len)
    
    def _get_slopes(self, n):
        """生成ALiBi斜率"""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if math.log2(n).is_integer():
            return torch.tensor(get_slopes_power_of_2(n))
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
            return torch.tensor(slopes + extra_slopes)
    
    def _build_alibi_tensor(self, max_seq_len):
        positions = torch.arange(max_seq_len)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)
        relative_positions = relative_positions.unsqueeze(0)  # (1, seq, seq)
        alibi = self.slopes.unsqueeze(1).unsqueeze(1) * relative_positions  # (heads, seq, seq)
        self.register_buffer('alibi', alibi)
    
    def forward(self, seq_len):
        return self.alibi[:, :seq_len, :seq_len]


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - 来自LLaMA2/Mistral
    使用较少的KV头来减少参数量和计算量
    """
    def __init__(self, d_model, num_heads, num_kv_heads, max_seq_len=512, dropout=0.1):
        super(GroupedQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        
        self.alibi = ALiBiPositionalBias(num_heads, max_seq_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 扩展KV头以匹配Q头
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 添加ALiBi位置偏置
        alibi_bias = self.alibi(seq_len).to(x.device)
        attn = attn + alibi_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


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


class STLLM2Block(nn.Module):
    """
    ST-LLM2基础块 - 使用GQA、滑动窗口注意力和MoE
    
    与STLLM的区别：
    1. 使用ALiBi替代RoPE
    2. 使用Grouped Query Attention减少KV头
    3. 使用滑动窗口注意力处理空间关系
    4. 使用Mixture of Experts替代单一FFN
    """
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff, 
                 num_experts=4, top_k=2, max_seq_len=512, window_size=4, dropout=0.1):
        super(STLLM2Block, self).__init__()
        
        # 时序注意力 - GQA with ALiBi
        self.temporal_norm = RMSNorm(d_model)
        self.temporal_attn = GroupedQueryAttention(
            d_model, num_heads, num_kv_heads, max_seq_len, dropout
        )
        
        # 空间注意力 - 滑动窗口
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
        
        # 时序GQA注意力
        x_reshape = x.permute(0, 2, 1, 3).contiguous().view(batch_size * node_num, seq_len, d_model)
        x_norm = self.temporal_norm(x_reshape)
        temporal_out = self.temporal_attn(x_norm)
        x_reshape = x_reshape + self.dropout(temporal_out)
        x = x_reshape.view(batch_size, node_num, seq_len, d_model).permute(0, 2, 1, 3)
        
        # 空间滑动窗口注意力
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


class STLLM(BaseModel):
    """
    ST-LLM2: 时空大语言模型风格的预测模型 V2
    
    与STLLM的主要区别：
    1. ALiBi位置编码（替代RoPE）
    2. Grouped Query Attention (GQA) - 减少KV头数量
    3. 滑动窗口注意力 - 用于空间建模
    4. Mixture of Experts (MoE) - 稀疏激活的专家网络
    5. GeGLU激活函数
    
    灵感来源：Mistral、Mixtral、BLOOM
    参数量约1M
    """
    def __init__(
        self,
        d_model=96,
        num_heads=8,
        num_kv_heads=2,
        d_ff=256,
        num_layers=3,
        num_experts=4,
        top_k=2,
        window_size=4,
        dropout=0.3,
        **args
    ):
        super(STLLM, self).__init__(**args)
        
        self.d_model = d_model
        
        # 输入嵌入
        self.input_embedding = nn.Linear(self.input_dim, d_model)
        
        # 可学习的时间嵌入
        self.time_embedding = nn.Parameter(torch.randn(1, 512, 1, d_model) * 0.02)
        
        # ST-LLM2 blocks
        self.blocks = nn.ModuleList([
            STLLM2Block(
                d_model, num_heads, num_kv_heads, d_ff,
                num_experts, top_k, self.seq_len, window_size, dropout
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
        
        # 通过ST-LLM2 blocks
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


class STLLM2Light(BaseModel):
    """
    轻量版ST-LLM2
    
    使用简化的MoE和GQA结构
    """
    def __init__(
        self,
        d_model=64,
        num_heads=4,
        num_kv_heads=2,
        d_ff=128,
        num_layers=2,
        num_experts=2,
        top_k=1,
        dropout=0.1,
        **args
    ):
        super(STLLM2Light, self).__init__(**args)
        
        self.d_model = d_model
        
        # 输入嵌入
        self.input_embedding = nn.Linear(self.input_dim, d_model)
        
        # ST-LLM2 blocks
        self.blocks = nn.ModuleList([
            STLLM2Block(
                d_model, num_heads, num_kv_heads, d_ff,
                num_experts, top_k, self.seq_len, 3, dropout
            )
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_norm = RMSNorm(d_model)
        self.output_fc1 = nn.Linear(d_model * self.seq_len, d_model)
        self.output_fc2 = nn.Linear(d_model, self.output_dim * self.horizon)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, label=None):
        batch_size, seq_len, node_num, input_dim = x.shape
        
        x = self.input_embedding(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_norm(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, node_num, -1)
        
        x = F.gelu(self.output_fc1(x))
        x = self.output_fc2(x)
        
        x = x.view(batch_size, node_num, self.horizon, self.output_dim)
        x = x.permute(0, 2, 1, 3)
        
        return x
