"""
Mamba6: ASTGCN + Mamba
优化版本: 将ASTGCN的Temporal Attention和Time Conv替换为Mamba，保留Spatial部分。
结合ASTGCN的空间建模能力和Mamba的时间建模能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base.model import BaseModel
from mamba_ssm import Mamba
from utils.graph_algo import calculate_scaled_laplacian

class Spatial_Attention_layer(nn.Module):
    def __init__(self, in_channels, node_num, seq_len):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(seq_len))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, seq_len))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(1, node_num, node_num))
        self.Vs = nn.Parameter(torch.FloatTensor(node_num, node_num))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1.unsqueeze(0))
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W3.unsqueeze(0))
        nn.init.xavier_uniform_(self.bs)
        nn.init.xavier_uniform_(self.Vs)

    def forward(self, x):
        # x: (B, N, F, T)
        # lhs = (xW1)W2 -> ((B,N,F,T) * (T)) * (F,T)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2) # (B,N,F) * (F,T) -> (B,N,T)
        
        # rhs = W3x -> (F) * (B,N,F,T) -> (B,N,T) (matmul contracts F)
        rhs = torch.matmul(self.W3, x) # (B, N, T)
        rhs = rhs.transpose(-1, -2) # (B, T, N)
        
        product = torch.matmul(lhs, rhs) # (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        S_normalized = F.softmax(S, dim=1)
        return S_normalized

class cheb_conv_withSAt(nn.Module):
    def __init__(self, order, cheb_polynomials, in_channels, out_channels):
        super(cheb_conv_withSAt, self).__init__()
        self.order = order
        # cheb_polynomials will be moved to device in forward or handled by caller
        self.cheb_polynomials = cheb_polynomials 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(order)])
        self.reset_parameters()
    
    def reset_parameters(self):
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta)

    def forward(self, x, spatial_attention):
        # x: (B, N, F, T)
        bs, node_num, in_channels, seq_len = x.shape
        outputs = []
        
        device = x.device
        
        for time_step in range(seq_len):
            graph_signal = x[:, :, :, time_step] # (B, N, F)
            
            output = torch.zeros(bs, node_num, self.out_channels, device=device)
            for k in range(self.order):
                T_k = self.cheb_polynomials[k].to(device) # (N, N)
                T_k_with_at = T_k.mul(spatial_attention) # (B, N, N)
                theta_k = self.Theta[k] # (F_in, F_out)
                
                # rhs = (B, N, N) * (B, N, F) -> (B, N, F)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
                
                # output += (B, N, F) * (F, F_out) -> (B, N, F_out)
                output = output + rhs.matmul(theta_k)
            outputs.append(output.unsqueeze(-1)) # (B, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1)) # (B, N, F_out, T)

class MambaBlock(nn.Module):
    """Mamba block to replace Temporal Convolution"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, N, F, T)
        B, N, F, T = x.shape
        
        # Prepare for Mamba: (Batch, Seq, Dim)
        # We treat (B*N) as batch, T as sequence, F as dim
        x_in = x.permute(0, 1, 3, 2).reshape(B * N, T, F) # (B*N, T, F)
        
        # Mamba
        x_mamba = self.mamba(self.norm(x_in))
        
        x_out = x_mamba + x_in
        x_out = self.dropout(x_out)
        
        # Reshape back
        x_out = x_out.reshape(B, N, T, F).permute(0, 1, 3, 2) # (B, N, F, T)
        return x_out

class MambaASTGCN_block(nn.Module):
    def __init__(self, in_channels, out_channels, order, cheb_polynomials, seq_len, dropout=0.1):
        super(MambaASTGCN_block, self).__init__()
        self.SAt = Spatial_Attention_layer(in_channels, cheb_polynomials[0].shape[0], seq_len)
        self.cheb_conv = cheb_conv_withSAt(order, cheb_polynomials, in_channels, out_channels)
        self.mamba = MambaBlock(out_channels, dropout)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x: (B, N, F, T)
        
        # Spatial Attention
        spatial_At = self.SAt(x)
        
        # GCN
        x_gcn = self.cheb_conv(x, spatial_At) # (B, N, F_out, T)
        
        # Mamba (Temporal)
        x_mamba = self.mamba(x_gcn) # (B, N, F_out, T)
        
        # Residual
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3)) # Conv2d expects (B, F, N, T)
        x_residual = x_residual.permute(0, 2, 1, 3) # Back to (B, N, F, T)
        
        # Combine
        # (B, N, F, T) -> (B, T, N, F) for LayerNorm on F
        out = x_residual + x_mamba
        out = out.permute(0, 3, 1, 2) # (B, T, N, F)
        out = self.ln(out)
        out = F.relu(out)
        out = out.permute(0, 2, 3, 1) # (B, N, F, T)
        
        return out

class UNetMamba(BaseModel):
    """
    Renamed to UNetMamba to keep compatibility with main.py, 
    but effectively implements MambaASTGCN.
    """
    def __init__(self, node_num, input_dim, output_dim, seq_len, horizon, num_layers, d_model, feature, dropout=0.1, adj_mx=None, **kwargs):
        super().__init__(node_num, input_dim, output_dim, seq_len, horizon)
        
        # Compute Chebyshev polynomials
        if adj_mx is not None:
            L_tilde = calculate_scaled_laplacian(adj_mx)
            # Ensure L_tilde is dense or handle sparse
            if hasattr(L_tilde, 'toarray'):
                L_tilde = L_tilde.toarray()
            
            cheb_polynomials = [torch.eye(node_num), torch.from_numpy(L_tilde).float()]
            for i in range(2, 4): # Order 3 (0, 1, 2). Need T0, T1, T2.
                # T_k = 2 * L * T_{k-1} - T_{k-2}.
                cheb_polynomials.append(2 * torch.from_numpy(L_tilde).float() @ cheb_polynomials[i-1] - cheb_polynomials[i-2])
        else:
            cheb_polynomials = [torch.eye(node_num) for _ in range(3)]
        
        # Model Components
        self.blocks = nn.ModuleList()
        
        # First block: feature -> d_model
        self.blocks.append(MambaASTGCN_block(
            in_channels=feature, 
            out_channels=d_model, 
            order=3, 
            cheb_polynomials=cheb_polynomials, 
            seq_len=seq_len,
            dropout=dropout
        ))
        
        # Subsequent blocks: d_model -> d_model
        for _ in range(num_layers - 1):
            self.blocks.append(MambaASTGCN_block(
                in_channels=d_model, 
                out_channels=d_model, 
                order=3, 
                cheb_polynomials=cheb_polynomials, 
                seq_len=seq_len, 
                dropout=dropout
            ))
            
        # Final Layer
        self.final_conv = nn.Conv2d(
            in_channels=seq_len,
            out_channels=horizon * output_dim,
            kernel_size=(1, d_model)
        )

    def forward(self, x):
        # x: (B, T, N, F)
        x = x.permute(0, 2, 3, 1) # (B, N, F, T)
        
        for block in self.blocks:
            x = block(x)
            
        # Output layer
        # x: (B, N, F, T)
        x = x.permute(0, 3, 1, 2) # (B, T, N, F)
        
        # Conv2d: (B, Cin, H, W). H=N, W=F.
        x = self.final_conv(x) # (B, Horizon*Output, N, 1)
        
        x = x.squeeze(-1) # (B, Horizon*Output, N)
        x = x.permute(0, 2, 1) # (B, N, Horizon*Output)
        x = x.reshape(x.shape[0], x.shape[1], self.horizon, self.output_dim) # (B, N, Horizon, Output)
        x = x.permute(0, 2, 1, 3) # (B, Horizon, N, Output)
        
        return x
