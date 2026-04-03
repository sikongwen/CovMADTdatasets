"""
SAFARI (peSsimistic meAn-Field vAlue iteRatIon) 算法实现

Pessimism Meets Invariance: Provably Efficient Offline Mean-Field Multi-Agent RL

主要特点:
1. Mean-Field方法：处理大量同质智能体
2. RKHS均值嵌入：近似值函数，避免状态-动作空间指数增长
3. 不确定性量化：引入悲观惩罚，避免虚假相关性
4. 不变性原理：利用环境对称性提高效率
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import math
import copy

from models.base_models import BaseModule
from models.rkhs_models import RKHSEmbedding, RBFKernel


class MeanFieldValueNetwork(BaseModule):
    """
    均值场价值网络
    
    使用RKHS均值嵌入来近似值函数，避免状态-动作空间的指数增长
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        kernel_type: str = "rbf",
        kernel_bandwidth: float = 1.0,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # RKHS嵌入模块
        self.rkhs_embedding = RKHSEmbedding(
            state_dim=state_dim,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            kernel_type=kernel_type,
            kernel_params={"bandwidth": kernel_bandwidth},
            use_neural_features=True,
        )
        
        # 价值网络：输入是状态和均值嵌入
        layers = []
        input_dim = state_dim + embedding_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.value_net = nn.Sequential(*layers)
        
        # 不确定性量化网络（用于悲观惩罚）
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # 确保不确定性为正
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        empirical_dist: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
            states: [B, T, S] 或 [B, S] 状态
            actions: [B, T, A] 或 [B, A] 动作（one-hot编码）
            empirical_dist: [B, T, K, S] 经验分布样本（可选）
            return_uncertainty: 是否返回不确定性
            
        返回:
            dict包含价值估计和不确定性
        """
        # 确保输入是float32
        if states.dtype != torch.float32:
            states = states.float()
        if actions.dtype != torch.float32:
            actions = actions.float()
        
        # 处理维度
        original_shape = states.shape
        if states.dim() == 2:
            states = states.unsqueeze(1)  # [B, 1, S]
            actions = actions.unsqueeze(1)  # [B, 1, A]
            was_2d = True
        else:
            was_2d = False
        
        batch_size, seq_len, _ = states.shape
        
        # 计算均值嵌入
        states_flat = states.view(-1, self.state_dim)  # [B*T, S]
        actions_flat = actions.view(-1, self.action_dim)  # [B*T, A]
        
        # 使用RKHS嵌入计算均值嵌入
        if empirical_dist is not None:
            empirical_flat = empirical_dist.view(-1, empirical_dist.shape[-2], self.state_dim)
            # 计算均值嵌入
            mean_embeddings = []
            for i in range(batch_size * seq_len):
                sa_pair = torch.cat([states_flat[i:i+1], actions_flat[i:i+1]], dim=-1)
                emp = empirical_flat[i] if i < empirical_flat.shape[0] else empirical_flat[0]
                # 使用核函数计算均值嵌入
                embedding = self.rkhs_embedding.compute_mean_embedding(
                    states_flat[i:i+1],
                    actions_flat[i:i+1],
                    emp.unsqueeze(0),
                )
                mean_embeddings.append(embedding)
            mean_embeddings = torch.cat(mean_embeddings, dim=0)  # [B*T, embedding_dim]
        else:
            # 如果没有经验分布，直接使用特征网络
            mean_embeddings = self.rkhs_embedding.feature_net(
                torch.cat([states_flat, actions_flat], dim=-1)
            )  # [B*T, embedding_dim]
        
        # 拼接状态和均值嵌入
        value_input = torch.cat([states_flat, mean_embeddings], dim=-1)  # [B*T, S + embedding_dim]
        
        # 计算价值
        values_flat = self.value_net(value_input)  # [B*T, 1]
        values = values_flat.view(batch_size, seq_len, 1)  # [B, T, 1]
        
        result = {"values": values}
        
        # 计算不确定性（用于悲观惩罚）
        if return_uncertainty:
            uncertainty_flat = self.uncertainty_net(value_input)  # [B*T, 1]
            uncertainty = uncertainty_flat.view(batch_size, seq_len, 1)  # [B, T, 1]
            result["uncertainty"] = uncertainty
        
        # 恢复原始维度
        if was_2d:
            result["values"] = result["values"].squeeze(1)  # [B, 1]
            if return_uncertainty:
                result["uncertainty"] = result["uncertainty"].squeeze(1)
        
        return result


class SAFARIPolicy(BaseModule):
    """
    SAFARI策略网络
    
    使用Mean-Field方法，输出动作概率分布
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.policy_net = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            states: [..., S] 状态
            
        返回:
            logits: [..., A] 动作logits
        """
        if states.dtype != torch.float32:
            states = states.float()
        
        return self.policy_net(states)


class SAFARI(nn.Module):
    """
    SAFARI (peSsimistic meAn-Field vAlue iteRatIon) 算法
    
    主要特点:
    1. Mean-Field价值网络：使用RKHS均值嵌入
    2. 不确定性量化：悲观惩罚避免分布外高估
    3. 不变性原理：利用环境对称性
    4. 离线学习：保守性约束
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int = 1,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        num_layers: int = 3,
        kernel_type: str = "rbf",
        kernel_bandwidth: float = 1.0,
        gamma: float = 0.99,
        beta: float = 1.0,  # 悲观惩罚系数
        alpha: float = 0.1,  # 保守性系数
        device: str = "cuda",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.beta = beta  # 不确定性惩罚系数
        self.alpha = alpha  # 保守性系数
        self.device = device
        
        # Mean-Field价值网络
        self.value_net = MeanFieldValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            kernel_type=kernel_type,
            kernel_bandwidth=kernel_bandwidth,
            num_layers=num_layers,
        )
        
        # 目标价值网络
        self.target_value_net = copy.deepcopy(self.value_net)
        
        # 策略网络
        self.policy_net = SAFARIPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        
        # 参考策略（行为克隆）
        self.reference_policy = SAFARIPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        
        # 移动到设备
        self.to(device)
    
    def predict_action(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        预测动作
        
        参数:
            states: [B, T, S] 或 [B, S] 状态
            deterministic: 是否使用确定性策略
            mask: [B, T, A] 动作掩码（可选）
            
        返回:
            actions: [B, T] 或 [B] 动作索引
            log_probs: [B, T] 或 [B] 对数概率
            info: 额外信息
        """
        # 处理维度
        was_2d = states.dim() == 2
        if was_2d:
            states = states.unsqueeze(1)  # [B, 1, S]
        
        batch_size, seq_len, _ = states.shape
        
        # 获取策略logits
        states_flat = states.view(-1, self.state_dim)  # [B*T, S]
        logits_flat = self.policy_net(states_flat)  # [B*T, A]
        
        # 应用掩码
        if mask is not None:
            mask_flat = mask.view(-1, self.action_dim)
            logits_flat = logits_flat.masked_fill(~mask_flat.bool(), float('-inf'))
        
        # 计算动作概率
        probs_flat = F.softmax(logits_flat, dim=-1)  # [B*T, A]
        log_probs_flat = F.log_softmax(logits_flat, dim=-1)  # [B*T, A]
        
        # 选择动作
        if deterministic:
            actions_flat = torch.argmax(probs_flat, dim=-1)  # [B*T]
        else:
            actions_flat = torch.multinomial(probs_flat, 1).squeeze(-1)  # [B*T]
        
        # 获取选中动作的对数概率
        selected_log_probs_flat = torch.gather(
            log_probs_flat, -1, actions_flat.unsqueeze(-1)
        ).squeeze(-1)  # [B*T]
        
        # 恢复维度
        actions = actions_flat.view(batch_size, seq_len)  # [B, T]
        log_probs = selected_log_probs_flat.view(batch_size, seq_len)  # [B, T]
        
        if was_2d:
            actions = actions.squeeze(1)  # [B]
            log_probs = log_probs.squeeze(1)  # [B]
        
        info = {
            "probs": probs_flat.view(batch_size, seq_len, self.action_dim),
            "logits": logits_flat.view(batch_size, seq_len, self.action_dim),
        }
        
        return actions, log_probs, info
    
    def compute_value(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        empirical_dist: Optional[torch.Tensor] = None,
        use_target: bool = False,
        return_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        计算价值
        
        参数:
            states: [B, T, S] 状态
            actions: [B, T] 动作索引（需要转换为one-hot）
            empirical_dist: [B, T, K, S] 经验分布样本（可选）
            use_target: 是否使用目标网络
            return_uncertainty: 是否返回不确定性
            
        返回:
            dict包含价值估计和不确定性
        """
        # 转换动作为one-hot编码
        if actions.dim() == 2:
            actions_onehot = F.one_hot(actions.long(), self.action_dim).float()  # [B, T, A]
        else:
            actions_onehot = F.one_hot(actions.long(), self.action_dim).float()  # [B, A]
            actions_onehot = actions_onehot.unsqueeze(1)  # [B, 1, A]
        
        value_net = self.target_value_net if use_target else self.value_net
        
        return value_net(
            states,
            actions_onehot,
            empirical_dist=empirical_dist,
            return_uncertainty=return_uncertainty,
        )
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        empirical_dist: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算SAFARI损失
        
        参数:
            states: [B, T, S] 状态
            actions: [B, T] 动作索引
            rewards: [B, T, 1] 奖励
            next_states: [B, T, S] 下一个状态
            dones: [B, T, 1] 终止标志（可选）
            empirical_dist: [B, T, K, S] 经验分布样本（可选）
            
        返回:
            dict包含损失项
        """
        batch_size, seq_len, _ = states.shape
        
        # 转换动作为one-hot编码
        actions_onehot = F.one_hot(actions.long(), self.action_dim).float()  # [B, T, A]
        
        # 计算当前状态的价值和不确定性
        value_outputs = self.compute_value(
            states,
            actions,
            empirical_dist=empirical_dist,
            use_target=False,
            return_uncertainty=True,
        )
        values = value_outputs["values"]  # [B, T, 1]
        uncertainties = value_outputs["uncertainty"]  # [B, T, 1]
        
        # 计算目标价值（使用目标网络）
        with torch.no_grad():
            # 获取下一个状态的动作（使用当前策略）
            next_actions, _, _ = self.predict_action(next_states, deterministic=False)
            next_value_outputs = self.compute_value(
                next_states,
                next_actions,
                empirical_dist=empirical_dist,
                use_target=True,
                return_uncertainty=False,
            )
            next_values = next_value_outputs["values"]  # [B, T, 1]
            
            # 计算TD目标
            if dones is not None:
                dones_float = dones.float() if dones.dtype != torch.float32 else dones
                targets = rewards + self.gamma * next_values * (1 - dones_float)
            else:
                targets = rewards + self.gamma * next_values
        
        # 价值损失（带悲观惩罚）
        # 悲观价值 = 价值估计 - beta * 不确定性
        pessimistic_values = values - self.beta * uncertainties
        value_loss = F.mse_loss(pessimistic_values, targets)
        
        # 策略损失：最大化悲观价值
        policy_logits = self.policy_net(states.view(-1, self.state_dim))  # [B*T, A]
        policy_logits = policy_logits.view(batch_size, seq_len, self.action_dim)  # [B, T, A]
        policy_probs = F.softmax(policy_logits, dim=-1)  # [B, T, A]
        
        # 计算所有动作的价值
        all_actions_onehot = torch.eye(self.action_dim, device=states.device).unsqueeze(0).unsqueeze(0)  # [1, 1, A, A]
        all_actions_onehot = all_actions_onehot.expand(batch_size, seq_len, -1, -1)  # [B, T, A, A]
        all_actions_onehot = all_actions_onehot.reshape(-1, self.action_dim)  # [B*T*A, A]
        
        states_expanded = states.unsqueeze(2).expand(-1, -1, self.action_dim, -1)  # [B, T, A, S]
        states_expanded = states_expanded.reshape(-1, self.state_dim)  # [B*T*A, S]
        
        all_value_outputs = self.value_net(
            states_expanded.unsqueeze(1),
            all_actions_onehot.unsqueeze(1),
            empirical_dist=empirical_dist,
            return_uncertainty=True,
        )
        all_values = all_value_outputs["values"].squeeze(1)  # [B*T*A, 1]
        all_uncertainties = all_value_outputs["uncertainty"].squeeze(1)  # [B*T*A, 1]
        
        all_pessimistic_values = all_values - self.beta * all_uncertainties  # [B*T*A, 1]
        all_pessimistic_values = all_pessimistic_values.view(batch_size, seq_len, self.action_dim)  # [B, T, A]
        
        # 策略损失：最大化期望悲观价值
        expected_pessimistic_value = (policy_probs * all_pessimistic_values).sum(dim=-1, keepdim=True)  # [B, T, 1]
        policy_loss = -expected_pessimistic_value.mean()
        
        # 保守性损失：KL散度（当前策略 vs 参考策略）
        reference_logits = self.reference_policy(states.view(-1, self.state_dim))  # [B*T, A]
        reference_logits = reference_logits.view(batch_size, seq_len, self.action_dim)  # [B, T, A]
        reference_probs = F.softmax(reference_logits, dim=-1)  # [B, T, A]
        
        kl_div = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            reference_probs,
            reduction='none',
        ).sum(dim=-1, keepdim=True)  # [B, T, 1]
        conservative_loss = self.alpha * kl_div.mean()
        
        # 总损失
        total_loss = value_loss + policy_loss + conservative_loss
        
        return {
            "loss": total_loss,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "conservative_loss": conservative_loss,
            "value_mean": values.mean(),
            "uncertainty_mean": uncertainties.mean(),
            "pessimistic_value_mean": pessimistic_values.mean(),
            "target_mean": targets.mean(),
        }
    
    def update_target_networks(self, tau: float = 0.005):
        """
        软更新目标网络
        
        参数:
            tau: 更新系数
        """
        for param, target_param in zip(
            self.value_net.parameters(),
            self.target_value_net.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
















