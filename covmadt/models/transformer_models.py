import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple

from .base_models import BaseModule
from .attention_modules import TransformerBlock, MultiHeadAttention


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, hidden_dim: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-np.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, hidden_dim]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: [B, T, D] 输入序列
            
        返回:
            x: [B, T, D] 添加位置编码后的序列
        """
        # 确保 x 和 pe 的 dtype 匹配
        if x.dtype != self.pe.dtype:
            x = x.to(self.pe.dtype)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AgentEmbedding(nn.Module):
    """智能体嵌入"""
    
    def __init__(self, n_agents: int, hidden_dim: int):
        super().__init__()
        self.agent_embedding = nn.Embedding(n_agents, hidden_dim)
        
    def forward(self, agent_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            agent_ids: [B, T] 智能体ID
            
        返回:
            embeddings: [B, T, D] 智能体嵌入
        """
        return self.agent_embedding(agent_ids)


class MultiAgentTransformer(BaseModule):
    """
    多智能体Transformer模型
    
    用于处理多智能体轨迹序列，输出策略和价值
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
        max_seq_len: int = 100,
        use_causal_mask: bool = True,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # 输入投影层
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # 智能体嵌入
        self.agent_embedding = AgentEmbedding(n_agents, hidden_dim)
        
        # Transformer编码层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
                use_causal_mask=use_causal_mask,
            )
            for _ in range(num_layers)
        ])
        
        # 输出层
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def encode(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        编码输入序列
        
        参数:
            states: [B, T, S] 或 [B, S] 状态序列
            actions: [B, T, A] 或 [B, A] 动作序列（可选）
            agent_ids: [B, T] 或 [B] 智能体ID（可选）
            rewards: [B, T, 1] 或 [B, 1] 奖励序列（可选）
            mask: [B, T] 序列掩码（可选）
            
        返回:
            encoded: [B, T, D] 或 [B, D] 编码后的序列
        """
        # 确保 states 是 float32 类型
        if states.dtype != torch.float32:
            states = states.float()
        
        # 处理 2 维输入（单个时间步）
        if states.dim() == 2:
            states = states.unsqueeze(1)  # [B, 1, S]
            if actions is not None and actions.dim() == 2:
                actions = actions.unsqueeze(1)  # [B, 1, A]
            if agent_ids is not None and agent_ids.dim() == 1:
                agent_ids = agent_ids.unsqueeze(1)  # [B, 1]
            if rewards is not None and rewards.dim() == 2:
                rewards = rewards.unsqueeze(1)  # [B, 1, 1]
            was_2d = True
        else:
            was_2d = False
        
        batch_size, seq_len, _ = states.shape
        
        # 再次确保 states 是 float32（在调用 state_proj 之前）
        if states.dtype != torch.float32:
            states = states.float()
        
        # 投影状态
        x = self.state_proj(states)  # [B, T, D]
        
        # 添加智能体嵌入
        if agent_ids is not None:
            agent_emb = self.agent_embedding(agent_ids)  # [B, T, D]
            x = x + agent_emb
        
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # 通过Transformer块
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # 如果输入是 2 维的，返回时也保持 2 维
        if was_2d:
            x = x.squeeze(1)  # [B, D]
        
        return x
    
    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
            states: [B, T, S] 或 [B, S] 状态序列
            actions: [B, T, A] 或 [B, A] 动作序列（可选）
            agent_ids: [B, T] 或 [B] 智能体ID（可选）
            rewards: [B, T, 1] 或 [B, 1] 奖励序列（可选）
            mask: [B, T] 序列掩码（可选）
            return_all: 是否返回所有中间结果
            
        返回:
            dict包含:
                - logits: [B, T, A] 或 [B, A] 动作logits
                - values: [B, T] 或 [B] 状态价值
                - encoded: [B, T, D] 或 [B, D] 编码序列（如果return_all=True）
        """
        # 确保所有输入都是 float32
        if states.dtype != torch.float32:
            states = states.float()
        if actions is not None and actions.dtype != torch.float32:
            actions = actions.float()
        if rewards is not None and rewards.dtype != torch.float32:
            rewards = rewards.float()
        
        # 检查输入维度
        was_2d = states.dim() == 2
        
        # 编码序列
        encoded = self.encode(states, actions, agent_ids, rewards, mask)
        
        # 如果 encoded 是 2 维的，需要添加时间维度
        if encoded.dim() == 2:
            encoded = encoded.unsqueeze(1)  # [B, 1, D]
        
        # 策略输出
        logits = self.policy_head(encoded)  # [B, T, A] 或 [B, 1, A]
        
        # 价值输出
        values = self.value_head(encoded)  # [B, T, 1] 或 [B, 1, 1]
        
        # 如果输入是 2 维的，输出也应该是 2 维的
        if was_2d:
            logits = logits.squeeze(1)  # [B, A]
            values = values.squeeze(1).squeeze(-1)  # [B]
            if return_all:
                encoded = encoded.squeeze(1)  # [B, D]
        else:
            values = values.squeeze(-1)  # [B, T]
        
        result = {
            "logits": logits,
            "values": values,
        }
        
        if return_all:
            result["encoded"] = encoded
        
        return result
    
    def predict_action(
        self,
        states: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测动作
        
        参数:
            states: [B, T, S] 或 [B, S] 状态
            agent_ids: [B, T] 或 [B] 智能体ID（可选）
            mask: [B, A] 动作掩码（可选）
            deterministic: 是否使用确定性策略
            
        返回:
            actions: [B, T] 或 [B] 动作
            log_probs: [B, T] 或 [B] 对数概率
        """
        # 确保状态维度正确
        if states.dim() == 2:
            states = states.unsqueeze(1)  # [B, 1, S]
            if agent_ids is not None and agent_ids.dim() == 1:
                agent_ids = agent_ids.unsqueeze(1)  # [B, 1]
        
        # 前向传播
        outputs = self.forward(states, agent_ids=agent_ids)
        logits = outputs["logits"]
        
        # 如果只有单个时间步，取最后一个
        if logits.shape[1] == 1:
            logits = logits.squeeze(1)  # [B, A]
        else:
            logits = logits[:, -1, :]  # [B, A] 取最后一个时间步
        
        # 应用动作掩码
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))
        
        # 采样动作
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            action_dist = torch.distributions.Categorical(logits=logits)
            actions = action_dist.sample()
        
        # 计算对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)
        
        return actions, log_probs


