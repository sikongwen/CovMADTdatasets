"""
Transformer Critic网络
使用Transformer架构进行价值估计，适合处理序列数据和多智能体场景
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np

from models.base_models import BaseModule
from models.transformer_models import PositionalEncoding, AgentEmbedding
from models.attention_modules import TransformerBlock


class TransformerCritic(BaseModule):
    """
    Transformer Critic网络
    
    使用Transformer架构来估计状态价值，特别适合：
    - 序列数据（轨迹）
    - 多智能体场景
    - 需要捕捉长期依赖的情况
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
        max_seq_len: int = 100,
        use_causal_mask: bool = False,  # Critic通常不需要因果掩码
        use_action: bool = True,  # 是否使用动作作为输入
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.use_action = use_action
        
        # 输入投影层
        # 如果使用动作，输入维度是 state_dim + action_dim
        # 否则只使用 state_dim
        input_dim = state_dim + action_dim if use_action else state_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # 智能体嵌入（多智能体场景）
        if n_agents > 1:
            self.agent_embedding = AgentEmbedding(n_agents, hidden_dim)
        else:
            self.agent_embedding = None
        
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
        
        # 价值输出头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
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
    
    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        empirical_dist: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            states: [B, T, S] 或 [B, S] 状态序列
            actions: [B, T, A] 或 [B, A] 动作序列（可选，如果use_action=True则需要）
            empirical_dist: [B, T, K, S] 经验分布（可选，为了接口兼容性保留，但不使用）
            rewards: [B, T, 1] 或 [B, 1] 奖励序列（可选，为了接口兼容性保留，但不使用）
            agent_ids: [B, T] 或 [B] 智能体ID（可选，多智能体场景）
            mask: [B, T] 序列掩码（可选）
            
        返回:
            values: [B, T, 1] 或 [B, 1] 状态价值
        """
        # 确保输入是 float32
        if states.dtype != torch.float32:
            states = states.float()
        
        # 处理维度：确保是3D [B, T, S]
        original_shape = states.shape
        if states.dim() == 2:
            # [B, S] -> [B, 1, S]
            states = states.unsqueeze(1)
            was_2d = True
        else:
            was_2d = False
        
        batch_size, seq_len, state_dim = states.shape
        
        # 处理动作输入
        if self.use_action:
            if actions is None:
                # 如果没有提供动作，使用零动作（或者可以抛出错误）
                # 这里使用零动作作为默认值
                if states.dim() == 3:
                    actions = torch.zeros(
                        batch_size, seq_len, self.action_dim,
                        device=states.device, dtype=torch.float32
                    )
                else:
                    actions = torch.zeros(
                        batch_size, self.action_dim,
                        device=states.device, dtype=torch.float32
                    )
            
            # 处理动作维度 - 确保最终是 [B, T, A]
            original_actions_dtype = actions.dtype
            original_actions_shape = actions.shape
            
            # 如果是离散动作索引（long类型），先转换为one-hot
            if actions.dtype == torch.long:
                if actions.dim() == 1:
                    # [B] -> [B, A]
                    actions = F.one_hot(actions, num_classes=self.action_dim).float()
                    # [B, A] -> [B, 1, A]
                    actions = actions.unsqueeze(1)
                elif actions.dim() == 2:
                    # [B, T] -> [B, T, A]
                    actions = F.one_hot(actions, num_classes=self.action_dim).float()
                # 如果已经是3D，假设已经是one-hot格式
            else:
                # 连续动作，确保是 float32
                if actions.dtype != torch.float32:
                    actions = actions.float()
                
                # 处理维度
                if actions.dim() == 1:
                    # [B] -> [B, 1, A]
                    if actions.shape[0] == batch_size:
                        # 可能是 [B] 的连续动作，需要扩展
                        actions = actions.unsqueeze(1).unsqueeze(-1)  # [B, 1, 1]
                        if actions.shape[-1] != self.action_dim:
                            actions = actions.expand(-1, -1, self.action_dim)
                    else:
                        # 可能是其他情况
                        actions = actions.unsqueeze(0).unsqueeze(-1)
                        if actions.shape[-1] != self.action_dim:
                            actions = actions.expand(batch_size, -1, self.action_dim)
                elif actions.dim() == 2:
                    # [B, T] 或 [B, A]
                    if actions.shape[1] == seq_len:
                        # [B, T] - 需要扩展为 [B, T, A]
                        if actions.shape[-1] == 1:
                            actions = actions.unsqueeze(-1).expand(-1, -1, self.action_dim)
                        else:
                            # 已经是 [B, T, A]？检查
                            if actions.shape[-1] != self.action_dim:
                                actions = actions.unsqueeze(-1).expand(-1, -1, self.action_dim)
                    elif actions.shape[1] == self.action_dim:
                        # [B, A] -> [B, 1, A]
                        actions = actions.unsqueeze(1)
                    else:
                        # 其他情况，尝试扩展
                        actions = actions.unsqueeze(1)
                elif actions.dim() == 3:
                    # 已经是 [B, T, A] 或 [B, T, 1]
                    if actions.shape[-1] == 1 and self.action_dim > 1:
                        # [B, T, 1] -> [B, T, A]
                        actions = actions.expand(-1, -1, self.action_dim)
                    # 否则假设已经是 [B, T, A]
            
            # 确保最终形状是 [B, T, A]
            if actions.dim() == 2:
                # 如果还是2D，可能是 [B, A]，需要扩展
                if actions.shape[1] == self.action_dim:
                    actions = actions.unsqueeze(1)  # [B, 1, A]
                    # 如果 seq_len > 1，需要扩展
                    if seq_len > 1:
                        actions = actions.expand(-1, seq_len, -1)
                else:
                    # 其他情况，报错
                    raise ValueError(f"无法处理动作形状: {original_actions_shape}, 期望 [B, T, A] 或 [B, T] (离散索引)")
            
            # 最终检查
            if actions.shape[0] != batch_size or actions.shape[1] != seq_len or actions.shape[2] != self.action_dim:
                raise ValueError(
                    f"动作形状不匹配: 得到 {actions.shape}, 期望 [{batch_size}, {seq_len}, {self.action_dim}]"
                )
            
            # 拼接状态和动作
            x = torch.cat([states, actions], dim=-1)  # [B, T, S+A]
        else:
            x = states  # [B, T, S]
        
        # 输入投影
        x = self.input_proj(x)  # [B, T, hidden_dim]
        
        # 添加位置编码
        x = self.pos_encoding(x)  # [B, T, hidden_dim]
        
        # 添加智能体嵌入（多智能体场景）
        if self.agent_embedding is not None and agent_ids is not None:
            # 处理 agent_ids 维度
            if agent_ids.dim() == 1:
                agent_ids = agent_ids.unsqueeze(1)  # [B, 1]
            elif agent_ids.dim() == 2 and agent_ids.shape[1] != seq_len:
                # 如果维度不匹配，可能需要扩展
                if agent_ids.shape[1] == 1:
                    agent_ids = agent_ids.expand(-1, seq_len)
            
            agent_emb = self.agent_embedding(agent_ids)  # [B, T, hidden_dim]
            x = x + agent_emb
        
        # Transformer编码
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=mask)  # [B, T, hidden_dim]
        
        # 价值输出
        values = self.value_head(x)  # [B, T, 1]
        
        # 恢复原始维度
        if was_2d:
            values = values.squeeze(1)  # [B, 1]
        
        return values

