import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base.model import BaseModel


class RMSNorm(nn.Module):
    """RMSNorm - 类似LLaMA中使用的归一化方法"""
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码 (RoPE) - 类似LLaMA中使用的位置编码"""
    def __init__(self, dim, max_seq_len=512, base=10000):
        super(RotaryPositionalEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0))
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        return self.cos_cached[:, :seq_len, :], self.sin_cached[:, :seq_len, :]


def rotate_half(x):
    """旋转半个维度"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SpatialAttention(nn.Module):
    """空间注意力模块 - 用于捕获节点间的空间依赖"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(SpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, node_num, d_model)
        """
        batch_size, node_num, d_model = x.shape
        
        q = self.q_proj(x).view(batch_size, node_num, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, node_num, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, node_num, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, node_num, d_model)
        return self.out_proj(out)


class TemporalAttention(nn.Module):
    """时序注意力模块 - 带有旋转位置编码"""
    def __init__(self, d_model, num_heads, max_seq_len=512, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用旋转位置编码
        cos, sin = self.rope(x, seq_len)
        cos = cos.unsqueeze(1)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU激活函数 - 类似LLaMA中的FFN"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(SwiGLU, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class STLLMBlock(nn.Module):
    """ST-LLM基础块 - 结合时空注意力和FFN"""
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=512, dropout=0.1):
        super(STLLMBlock, self).__init__()
        
        # 时序注意力
        self.temporal_norm = RMSNorm(d_model)
        self.temporal_attn = TemporalAttention(d_model, num_heads, max_seq_len, dropout)
        
        # 空间注意力
        self.spatial_norm = RMSNorm(d_model)
        self.spatial_attn = SpatialAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, node_num, d_model)
        """
        batch_size, seq_len, node_num, d_model = x.shape
        
        # 时序注意力 - 对每个节点独立处理
        x_reshape = x.permute(0, 2, 1, 3).contiguous().view(batch_size * node_num, seq_len, d_model)
        x_norm = self.temporal_norm(x_reshape)
        temporal_out = self.temporal_attn(x_norm)
        x_reshape = x_reshape + self.dropout(temporal_out)
        x = x_reshape.view(batch_size, node_num, seq_len, d_model).permute(0, 2, 1, 3)
        
        # 空间注意力 - 对每个时间步独立处理
        x_reshape = x.contiguous().view(batch_size * seq_len, node_num, d_model)
        x_norm = self.spatial_norm(x_reshape)
        spatial_out = self.spatial_attn(x_norm)
        x_reshape = x_reshape + self.dropout(spatial_out)
        x = x_reshape.view(batch_size, seq_len, node_num, d_model)
        
        # 前馈网络
        x_reshape = x.view(batch_size * seq_len * node_num, d_model)
        x_norm = self.ffn_norm(x_reshape)
        ffn_out = self.ffn(x_norm)
        x_reshape = x_reshape + ffn_out
        x = x_reshape.view(batch_size, seq_len, node_num, d_model)
        
        return x


class STLLM(BaseModel):
    """
    ST-LLM: 时空大语言模型风格的预测模型
    
    模拟LLM的结构特点：
    1. RMSNorm代替LayerNorm
    2. 旋转位置编码 (RoPE)
    3. SwiGLU激活函数
    4. 时空分离的注意力机制
    
    参数量约1M
    """
    def __init__(
        self,
        d_model=128,
        num_heads=8,
        d_ff=384,
        num_layers=4,
        dropout=0.1,
        **args
    ):
        super(STLLM, self).__init__(**args)
        
        self.d_model = d_model
        
        # 输入嵌入
        self.input_embedding = nn.Linear(self.input_dim, d_model)
        
        # 节点嵌入
        self.node_embedding = nn.Embedding(self.node_num, d_model)
        
        # ST-LLM blocks
        self.blocks = nn.ModuleList([
            STLLMBlock(d_model, num_heads, d_ff, self.seq_len, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.output_dim * self.horizon)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, label=None):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, node_num, input_dim)
            label: 标签（可选）
            
        Returns:
            output: 预测结果，形状为 (batch, horizon, node_num, output_dim)
        """
        batch_size, seq_len, node_num, input_dim = x.shape
        
        # 输入嵌入
        x = self.input_embedding(x)  # (batch, seq_len, node_num, d_model)
        
        # 添加节点嵌入
        node_ids = torch.arange(node_num, device=x.device)
        node_emb = self.node_embedding(node_ids)  # (node_num, d_model)
        x = x + node_emb.unsqueeze(0).unsqueeze(0)  # 广播添加
        
        # 通过ST-LLM blocks
        for block in self.blocks:
            x = block(x)
        
        # 输出投影
        x = self.output_norm(x)
        
        # 取最后一个时间步
        x = x[:, -1, :, :]  # (batch, node_num, d_model)
        
        # 预测
        x = self.output_proj(x)  # (batch, node_num, output_dim * horizon)
        x = x.view(batch_size, node_num, self.horizon, self.output_dim)
        x = x.permute(0, 2, 1, 3)  # (batch, horizon, node_num, output_dim)
        
        return x


class STLLMLight(BaseModel):
    """
    轻量版ST-LLM
    
    适合较小的数据集，参数量更少
    """
    def __init__(
        self,
        d_model=64,
        num_heads=4,
        d_ff=192,
        num_layers=3,
        dropout=0.1,
        **args
    ):
        super(STLLMLight, self).__init__(**args)
        
        self.d_model = d_model
        
        # 输入嵌入
        self.input_embedding = nn.Linear(self.input_dim, d_model)
        
        # ST-LLM blocks
        self.blocks = nn.ModuleList([
            STLLMBlock(d_model, num_heads, d_ff, self.seq_len, dropout)
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
        
        # 输入嵌入
        x = self.input_embedding(x)
        
        # 通过ST-LLM blocks
        for block in self.blocks:
            x = block(x)
        
        # 输出
        x = self.output_norm(x)
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, node_num, seq_len, d_model)
        x = x.view(batch_size, node_num, -1)  # (batch, node_num, seq_len * d_model)
        
        x = F.gelu(self.output_fc1(x))
        x = self.output_fc2(x)  # (batch, node_num, output_dim * horizon)
        
        x = x.view(batch_size, node_num, self.horizon, self.output_dim)
        x = x.permute(0, 2, 1, 3)
        
        return x
