import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_causal_mask: bool = False,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_causal_mask = use_causal_mask
        
        # 线性投影层
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query: [B, T_q, D]
            key: [B, T_k, D]
            value: [B, T_v, D]
            mask: [B, T_q, T_k] 或 [T_q, T_k]
            
        返回:
            output: [B, T_q, D]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # 线性投影并重塑为多头
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用因果掩码
        if self.use_causal_mask:
            causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=query.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # 应用外部掩码
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)
        
        # 重塑并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.hidden_dim
        )
        output = self.out_proj(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer编码块"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
        use_causal_mask: bool = False,
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_causal_mask=use_causal_mask,
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播"""
        # 自注意力 + 残差连接
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


