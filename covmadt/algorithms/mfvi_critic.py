import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from models.base_models import BaseModule
from models.rkhs_models import RKHSEmbedding


class MFVICritic(BaseModule):
    """
    均值场价值迭代Critic
    
    使用RKHS嵌入和均值场近似来估计价值函数
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        kernel_type: str = "rbf",
        gamma: float = 0.99,
        num_iterations: int = 10,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.num_iterations = num_iterations
        
        # RKHS嵌入用于状态转移建模
        self.rkhs_embedding = RKHSEmbedding(
            state_dim=state_dim,
            action_dim=action_dim,
            embedding_dim=hidden_dim,
            kernel_type=kernel_type,
            use_neural_features=True,
        )
        
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # 奖励预测器（可选）
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        empirical_dist: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            states: [..., S] 状态
            actions: [..., A] 动作
            empirical_dist: [..., K, S] 经验分布（可选）
            rewards: [..., 1] 奖励（可选，如果提供则用于训练）
            
        返回:
            values: [..., 1] 状态价值
        """
        # 计算RKHS嵌入
        rkhs_output = self.rkhs_embedding(
            states=states,
            actions=actions,
            empirical_dist=empirical_dist,
            return_embeddings=True,
        )
        
        embeddings = rkhs_output["embeddings"]
        
        # 确保所有输入都是 float32
        if states.dtype != torch.float32:
            states = states.float()
        if embeddings.dtype != torch.float32:
            embeddings = embeddings.float()
        
        # 拼接状态和嵌入
        state_emb = torch.cat([states, embeddings], dim=-1)
        
        # 再次确保 state_emb 是 float32
        if state_emb.dtype != torch.float32:
            state_emb = state_emb.float()
        
        # 计算价值
        values = self.value_net(state_emb)
        
        return values
    
    def value_iteration(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        empirical_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        执行均值场价值迭代
        
        参数:
            states: [B, S] 状态
            actions: [B, A] 动作
            empirical_dist: [B, K, S] 经验分布（可选）
            
        返回:
            values: [B, 1] 收敛后的价值
        """
        # 初始化价值
        values = self.forward(states, actions, empirical_dist)
        
        # 迭代更新
        for _ in range(self.num_iterations):
            # 预测奖励
            sa_pairs = torch.cat([states, actions], dim=-1)
            rewards = self.reward_predictor(sa_pairs)
            
            # 预测下一个状态
            rkhs_output = self.rkhs_embedding(
                states=states,
                actions=actions,
                empirical_dist=empirical_dist,
            )
            next_states_pred = rkhs_output["next_state_pred"]
            
            # 预测下一个状态的价值
            # 这里简化处理，使用当前动作
            next_values = self.forward(next_states_pred, actions, empirical_dist)
            
            # Bellman更新
            values = rewards + self.gamma * next_values
        
        return values


