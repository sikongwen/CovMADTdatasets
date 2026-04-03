"""
标准Critic网络（普通价值网络）
用于替代MFVICritic，提供更简单的价值估计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from models.base_models import BaseModule


class StandardCritic(BaseModule):
    """
    标准Critic网络
    
    简单的多层感知机，用于估计状态价值
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        gamma: float = 0.99,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        
        # 构建价值网络
        layers = []
        # 输入层：状态维度
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        
        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
        
        # 输出层：价值（标量）
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.value_net = nn.Sequential(*layers)
    
    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        empirical_dist: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            states: [..., S] 状态
            actions: [..., A] 动作（可选，为了接口兼容性保留，但不使用）
            empirical_dist: [..., K, S] 经验分布（可选，为了接口兼容性保留，但不使用）
            rewards: [..., 1] 奖励（可选，为了接口兼容性保留，但不使用）
            
        返回:
            values: [..., 1] 状态价值
        """
        # 确保输入是 float32
        if states.dtype != torch.float32:
            states = states.float()
        
        # 处理维度：如果是3D [B, T, S]，展平为2D [B*T, S]
        original_shape = states.shape
        if states.dim() == 3:
            batch_size, seq_len, state_dim = states.shape
            states = states.view(-1, state_dim)
            was_3d = True
        else:
            was_3d = False
        
        # 计算价值
        values = self.value_net(states)
        
        # 恢复原始维度
        if was_3d:
            values = values.view(batch_size, seq_len, 1)
        
        return values


