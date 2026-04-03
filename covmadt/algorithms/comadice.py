"""
ComaDICE (Conservative Offline Multi-Agent Distribution Correction Estimation) 算法实现

ComaDICE是一种离线多智能体强化学习算法，通过重要性采样权重来纠正离线数据分布，
并使用保守性约束避免分布外动作的高估。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import math
import copy

from models.base_models import BaseModule


class ComaDICEPolicy(BaseModule):
    """
    ComaDICE策略网络
    
    使用MLP网络作为策略网络，输出动作概率分布
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
        
        # 初始化权重
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


class ComaDICECritic(BaseModule):
    """
    ComaDICE价值网络
    
    估计状态-动作价值函数
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
        
        # 状态-动作联合输入
        input_dim = state_dim + action_dim
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.value_net = nn.Sequential(*layers)
        
        # 初始化权重
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
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            states: [..., S] 状态
            actions: [..., A] 动作（one-hot编码或索引）
            
        返回:
            values: [..., 1] 状态-动作价值
        """
        if states.dtype != torch.float32:
            states = states.float()
        
        # 处理动作：如果是索引，转换为one-hot
        original_states_dim = states.dim()
        original_actions_dim = actions.dim()
        
        # 如果states是3维[B, T, S]，确保actions也是3维[B, T, A]
        if original_states_dim == 3:
            batch_size, seq_len, state_dim = states.shape
            
            # 处理actions维度
            if original_actions_dim == 2:  # [B, T]
                # 动作是索引，转换为one-hot
                actions_onehot = F.one_hot(actions.long(), self.action_dim).float()  # [B, T, A]
            elif original_actions_dim == 3:
                if actions.shape[-1] == 1:  # [B, T, 1] - 索引
                    actions_onehot = F.one_hot(actions.long().squeeze(-1), self.action_dim).float()  # [B, T, A]
                elif actions.shape[-1] == self.action_dim:  # [B, T, A] - 已经是one-hot
                    actions_onehot = actions.float()
                else:
                    # 尝试转换为one-hot
                    actions_onehot = F.one_hot(actions.long(), self.action_dim).float()
            elif original_actions_dim == 4:
                # 如果是4维 [B, T, A, A]，取第一个A维度
                if actions.shape[2] == self.action_dim and actions.shape[3] == self.action_dim:
                    actions_onehot = actions[:, :, 0, :].float()  # [B, T, A]
                else:
                    actions_onehot = actions.reshape(batch_size, seq_len, -1)
                    if actions_onehot.shape[-1] != self.action_dim:
                        actions_onehot = actions_onehot[:, :, :self.action_dim]
            else:
                # 其他情况，展平处理
                actions_flat = actions.view(-1)
                actions_onehot_flat = F.one_hot(actions_flat.long(), self.action_dim).float()
                actions_onehot = actions_onehot_flat.view(batch_size, seq_len, self.action_dim)
        else:
            # states是2维[B, S]
            if original_actions_dim == 1:  # [B] - 索引
                actions_onehot = F.one_hot(actions.long(), self.action_dim).float()  # [B, A]
            elif original_actions_dim == 2:
                if actions.shape[-1] == self.action_dim:  # [B, A] - 已经是one-hot
                    actions_onehot = actions.float()
                else:
                    actions_onehot = F.one_hot(actions.long(), self.action_dim).float()
            else:
                actions_onehot = F.one_hot(actions.long(), self.action_dim).float()
        
        # 确保维度匹配（最终都应该是[B, T, ...]或[B, ...]）
        if original_states_dim == 3:
            # 确保actions_onehot也是3维 [B, T, A]
            if actions_onehot.dim() == 2:
                # [B, A] -> [B, 1, A]
                actions_onehot = actions_onehot.unsqueeze(1)
            elif actions_onehot.dim() == 4:
                # 如果是4维，需要压缩
                if actions_onehot.shape[1] == 1:
                    actions_onehot = actions_onehot.squeeze(1)  # [B, T, A]
                elif actions_onehot.shape[2] == 1:
                    actions_onehot = actions_onehot.squeeze(2)  # [B, T, A]
                elif actions_onehot.shape[2] == self.action_dim and actions_onehot.shape[3] == self.action_dim:
                    # [B, T, A, A] -> [B, T, A] (取第一个A)
                    actions_onehot = actions_onehot[:, :, 0, :]
                else:
                    # 其他情况，尝试reshape
                    batch_size, seq_len = states.shape[:2]
                    actions_onehot = actions_onehot.reshape(batch_size, seq_len, self.action_dim)
            elif actions_onehot.dim() != 3:
                # 其他维度，尝试reshape
                batch_size, seq_len = states.shape[:2]
                actions_onehot = actions_onehot.reshape(batch_size, seq_len, self.action_dim)
        
        # 拼接状态和动作
        sa_input = torch.cat([states, actions_onehot], dim=-1)
        
        # 处理维度：如果是3D [B, T, ...]，展平为2D [B*T, ...]
        original_shape = sa_input.shape
        if sa_input.dim() == 3:
            batch_size, seq_len, input_dim = sa_input.shape
            sa_input = sa_input.view(-1, input_dim)
            was_3d = True
        else:
            was_3d = False
        
        # 计算价值
        values = self.value_net(sa_input)
        
        # 恢复原始维度
        if was_3d:
            values = values.view(batch_size, seq_len, 1)
        
        return values


class ComaDICE(nn.Module):
    """
    ComaDICE (Conservative Offline Multi-Agent Distribution Correction Estimation) 算法
    
    主要特点:
    1. 重要性采样权重：纠正离线数据分布
    2. 保守性约束：避免分布外动作的高估
    3. 多智能体支持：处理多智能体场景
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.1,  # 保守性系数
        beta: float = 0.1,   # 重要性权重正则化系数
        device: str = "cuda",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
        # 策略网络
        self.policy_net = ComaDICEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        
        # 价值网络
        self.critic_net = ComaDICECritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        
        # 目标价值网络（用于稳定训练）
        self.target_critic_net = copy.deepcopy(self.critic_net)
        
        # 行为策略网络（用于重要性采样）
        self.behavior_policy_net = ComaDICEPolicy(
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
            states: [B, S] 或 [B, T, S] 状态
            deterministic: 是否使用确定性策略
            mask: [B, A] 动作掩码（可选）
            
        返回:
            actions: 动作
            log_probs: 对数概率
            info: 附加信息
        """
        if states.dtype != torch.float32:
            states = states.float()
        
        # 确保状态维度正确
        if states.dim() == 2:
            states = states.unsqueeze(1)  # [B, 1, S]
        
        batch_size, seq_len, state_dim = states.shape
        states_flat = states.view(-1, state_dim)  # [B*T, S]
        
        # 获取策略logits
        logits = self.policy_net(states_flat)  # [B*T, A]
        logits = logits.view(batch_size, seq_len, self.action_dim)  # [B, T, A]
        
        # 应用动作掩码
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)  # [B, 1, A]
            logits = logits.masked_fill(mask == 0, -1e10)
        
        # 采样动作
        if deterministic:
            actions = torch.argmax(logits, dim=-1)  # [B, T]
        else:
            logits_flat = logits.view(-1, self.action_dim)  # [B*T, A]
            action_dist = torch.distributions.Categorical(logits=logits_flat)
            actions_flat = action_dist.sample()  # [B*T]
            actions = actions_flat.view(batch_size, seq_len)  # [B, T]
        
        # 计算对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)
        
        # 获取价值估计
        actions_onehot = F.one_hot(actions.long(), self.action_dim).float()  # [B, T, A]
        values = self.critic_net(states, actions_onehot)
        
        info = {
            "logits": logits,
            "values": values,
            "action_dist": torch.softmax(logits, dim=-1),
        }
        
        return actions, log_probs, info
    
    def compute_importance_weights(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算重要性采样权重
        
        w(s,a) = π(a|s) / π_b(a|s)
        
        参数:
            states: [B, T, S] 状态
            actions: [B, T] 动作索引
            
        返回:
            weights: [B, T, 1] 重要性权重
        """
        batch_size, seq_len, _ = states.shape
        
        # 处理 actions 的不同格式
        if actions.dim() == 3:
            # [B, T, A] - one-hot 格式，转换为索引
            if actions.shape[-1] == self.action_dim:
                # 是 one-hot 格式，转换为索引
                actions = torch.argmax(actions, dim=-1)  # [B, T]
            else:
                # [B, T, 1] -> [B, T]
                actions = actions.squeeze(-1)
        elif actions.dim() == 2:
            # [B, T] 或 [B, T*A] (one-hot展平)
            if actions.shape[-1] == self.action_dim:
                # [B, T, A] 被当作 [B, T*A] 处理，需要reshape
                actions = actions.view(batch_size, seq_len, self.action_dim)
                actions = torch.argmax(actions, dim=-1)  # [B, T]
            # 否则已经是 [B, T] 格式（索引）
        elif actions.dim() == 1:
            # [B*T] -> [B, T]
            if actions.shape[0] == batch_size * seq_len:
                actions = actions.view(batch_size, seq_len)
            else:
                raise ValueError(f"Actions shape mismatch: expected {batch_size * seq_len}, got {actions.shape[0]}")
        
        # 确保 actions 是 [B, T] 格式的索引
        if actions.dim() != 2 or actions.shape[0] != batch_size or actions.shape[1] != seq_len:
            raise ValueError(
                f"Actions shape error: expected [B={batch_size}, T={seq_len}], "
                f"got {actions.shape}"
            )
        
        states_flat = states.view(-1, self.state_dim)  # [B*T, S]
        actions_flat = actions.view(-1).long()  # [B*T]
        
        # 确保 actions_flat 的大小与 states_flat 的第一维匹配
        if actions_flat.shape[0] != states_flat.shape[0]:
            raise ValueError(
                f"Actions and states size mismatch: "
                f"actions_flat={actions_flat.shape[0]}, states_flat={states_flat.shape[0]}"
            )
        
        # 当前策略概率
        policy_logits = self.policy_net(states_flat)  # [B*T, A]
        policy_probs = F.softmax(policy_logits, dim=-1)  # [B*T, A]
        policy_action_probs = torch.gather(policy_probs, 1, actions_flat.unsqueeze(1)).squeeze(1)  # [B*T]
        
        # 行为策略概率
        behavior_logits = self.behavior_policy_net(states_flat)  # [B*T, A]
        behavior_probs = F.softmax(behavior_logits, dim=-1)  # [B*T, A]
        behavior_action_probs = torch.gather(behavior_probs, 1, actions_flat.unsqueeze(1)).squeeze(1)  # [B*T]
        
        # 重要性权重（添加小值避免除零）
        weights = policy_action_probs / (behavior_action_probs + 1e-8)  # [B*T]
        weights = weights.view(batch_size, seq_len, 1)  # [B, T, 1]
        
        return weights
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算ComaDICE损失
        
        参数:
            states: [B, T, S] 状态
            actions: [B, T] 动作索引
            rewards: [B, T, 1] 奖励
            next_states: [B, T, S] 下一个状态
            dones: [B, T, 1] 终止标志（可选）
            
        返回:
            dict包含损失项
        """
        batch_size, seq_len, _ = states.shape
        
        # 处理 actions 格式：如果是 one-hot，转换为索引
        if actions.dim() == 3 and actions.shape[-1] == self.action_dim:
            # [B, T, A] - one-hot 格式，转换为索引
            actions = torch.argmax(actions, dim=-1)  # [B, T]
        elif actions.dim() == 2 and actions.shape[-1] == self.action_dim:
            # [B, T*A] 被当作 [B, T*A] 处理，需要reshape
            actions = actions.view(batch_size, seq_len, self.action_dim)
            actions = torch.argmax(actions, dim=-1)  # [B, T]
        elif actions.dim() == 3 and actions.shape[-1] == 1:
            # [B, T, 1] -> [B, T]
            actions = actions.squeeze(-1)
        
        # 确保 actions 是 [B, T] 格式的索引
        if actions.dim() != 2 or actions.shape[0] != batch_size or actions.shape[1] != seq_len:
            raise ValueError(
                f"Actions shape error in compute_loss: expected [B={batch_size}, T={seq_len}], "
                f"got {actions.shape}"
            )
        
        # 计算重要性权重
        importance_weights = self.compute_importance_weights(states, actions)  # [B, T, 1]
        
        # 将动作转换为one-hot
        actions_onehot = F.one_hot(actions.long(), self.action_dim).float()  # [B, T, A]
        
        # 计算当前状态-动作价值
        current_values = self.critic_net(states, actions_onehot)  # [B, T, 1]
        
        # 计算目标价值（使用目标网络）
        with torch.no_grad():
            # 对下一个状态，使用当前策略采样动作（或使用期望）
            next_states_flat = next_states.view(-1, self.state_dim)  # [B*T, S]
            next_policy_logits = self.policy_net(next_states_flat)  # [B*T, A]
            next_policy_probs = F.softmax(next_policy_logits, dim=-1)  # [B*T, A]
            
            # 计算期望价值：对每个动作的价值加权求和
            # 为每个动作计算价值，然后加权平均
            next_states_expanded = next_states.unsqueeze(2).expand(-1, -1, self.action_dim, -1)  # [B, T, A, S]
            next_states_flat_all = next_states_expanded.reshape(-1, self.state_dim)  # [B*T*A, S]
            all_actions_onehot = torch.eye(self.action_dim, device=next_states.device).unsqueeze(0).unsqueeze(0)  # [1, 1, A]
            all_actions_onehot = all_actions_onehot.expand(batch_size, seq_len, -1, -1)  # [B, T, A, A]
            all_actions_flat = all_actions_onehot.reshape(-1, self.action_dim)  # [B*T*A, A]
            
            all_next_values_flat = self.target_critic_net(next_states_flat_all, all_actions_flat)  # [B*T*A, 1]
            all_next_values = all_next_values_flat.view(batch_size, seq_len, self.action_dim)  # [B, T, A]
            
            # 加权平均（使用策略概率）
            next_policy_probs_reshaped = next_policy_probs.view(batch_size, seq_len, self.action_dim)  # [B, T, A]
            next_values = (next_policy_probs_reshaped * all_next_values).sum(dim=-1, keepdim=True)  # [B, T, 1]
            
            # 计算TD目标
            if dones is not None:
                dones_float = dones.float() if dones.dtype != torch.float32 else dones
                targets = rewards + self.gamma * next_values * (1 - dones_float)
            else:
                targets = rewards + self.gamma * next_values
        
        # 加权TD损失（使用重要性权重）
        td_errors = targets - current_values  # [B, T, 1]
        weighted_td_loss = (importance_weights * td_errors ** 2).mean()
        
        # 保守性损失：惩罚分布外动作的高估
        # 简化版本：只对当前动作的价值进行惩罚
        # 行为策略概率
        states_flat_for_behavior = states.view(-1, self.state_dim)  # [B*T, S]
        behavior_logits = self.behavior_policy_net(states_flat_for_behavior)  # [B*T, A]
        behavior_probs = F.softmax(behavior_logits, dim=-1)  # [B*T, A]
        behavior_probs = behavior_probs.view(batch_size, seq_len, self.action_dim)  # [B, T, A]
        
        # 获取当前动作的行为策略概率
        actions_expanded = actions.unsqueeze(-1)  # [B, T, 1]
        behavior_action_probs = torch.gather(behavior_probs, -1, actions_expanded)  # [B, T, 1]
        
        # 保守性损失：对低概率动作的价值进行惩罚
        # 如果行为策略概率低，则惩罚价值
        conservative_penalty = (1.0 - behavior_action_probs) * current_values  # [B, T, 1]
        conservative_loss = self.alpha * conservative_penalty.mean()
        
        # 重要性权重正则化
        weight_reg = self.beta * (importance_weights - 1.0).pow(2).mean()
        
        # 策略损失：最大化加权价值
        policy_logits = self.policy_net(states.view(-1, self.state_dim))  # [B*T, A]
        policy_logits = policy_logits.view(batch_size, seq_len, self.action_dim)  # [B, T, A]
        policy_probs = F.softmax(policy_logits, dim=-1)  # [B, T, A]
        
        # 计算策略梯度（使用加权价值）
        weighted_values = importance_weights * current_values  # [B, T, 1]
        action_log_probs = F.log_softmax(policy_logits, dim=-1)  # [B, T, A]
        selected_log_probs = torch.gather(action_log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)  # [B, T]
        policy_loss = -(selected_log_probs.unsqueeze(-1) * weighted_values).mean()
        
        # 总损失
        total_loss = weighted_td_loss + conservative_loss + weight_reg + policy_loss
        
        return {
            "loss": total_loss,
            "td_loss": weighted_td_loss,
            "conservative_loss": conservative_loss,
            "weight_reg": weight_reg,
            "policy_loss": policy_loss,
            "importance_weights_mean": importance_weights.mean(),
            "value_mean": current_values.mean(),
            "target_mean": targets.mean(),
        }
    
    def update_target_networks(self, tau: float = 0.005):
        """
        软更新目标网络
        
        参数:
            tau: 软更新系数
        """
        for target_param, param in zip(self.target_critic_net.parameters(), 
                                      self.critic_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def update_behavior_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        num_epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
    ):
        """
        更新行为策略网络（行为克隆）
        
        参数:
            states: [N, S] 状态
            actions: [N] 动作索引
            num_epochs: 训练轮数
            batch_size: 批量大小
            lr: 学习率
        """
        optimizer = torch.optim.Adam(self.behavior_policy_net.parameters(), lr=lr)
        
        dataset_size = states.shape[0]
        num_batches = (dataset_size + batch_size - 1) // batch_size
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            # 随机打乱数据
            indices = torch.randperm(dataset_size)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices].long()
                
                # 前向传播
                logits = self.behavior_policy_net(batch_states)
                
                # 计算损失
                loss = F.cross_entropy(
                    logits.view(-1, self.action_dim),
                    batch_actions.view(-1),
                )
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            if epoch % 5 == 0:
                print(f"Behavior Policy Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        compute_loss: bool = False,
    ) -> Dict[str, Any]:
        """
        前向传播
        
        参数:
            states: [B, T, S] 状态
            actions: [B, T] 动作（可选）
            rewards: [B, T, 1] 奖励（可选）
            next_states: [B, T, S] 下一个状态（可选）
            dones: [B, T, 1] 终止标志（可选）
            compute_loss: 是否计算损失
            
        返回:
            dict包含所有输出
        """
        # 获取策略logits
        states_flat = states.view(-1, self.state_dim)
        logits_flat = self.policy_net(states_flat)
        logits = logits_flat.view(states.shape[0], states.shape[1], self.action_dim)
        
        results = {
            "logits": logits,
        }
        
        # 如果有动作，计算价值
        if actions is not None:
            # 处理 actions 格式：如果是 one-hot，转换为索引
            actions_for_value = actions
            if actions.dim() == 3 and actions.shape[-1] == self.action_dim:
                # [B, T, A] - one-hot 格式，转换为索引
                actions_for_value = torch.argmax(actions, dim=-1)  # [B, T]
            elif actions.dim() == 2 and actions.shape[-1] == self.action_dim:
                # [B, T*A] 被当作 [B, T*A] 处理，需要reshape
                batch_size, _ = actions.shape
                seq_len = states.shape[1]
                actions_for_value = actions.view(batch_size, seq_len, self.action_dim)
                actions_for_value = torch.argmax(actions_for_value, dim=-1)  # [B, T]
            
            actions_onehot = F.one_hot(actions_for_value.long(), self.action_dim).float()
            values = self.critic_net(states, actions_onehot)
            results["values"] = values
        
        # 计算损失（如果需要）
        if compute_loss and rewards is not None and actions is not None and next_states is not None:
            loss_dict = self.compute_loss(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
            )
            results["loss_dict"] = loss_dict
        
        return results
    
    def get_num_params(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'policy_net_state_dict': self.policy_net.state_dict(),
            'critic_net_state_dict': self.critic_net.state_dict(),
            'behavior_policy_net_state_dict': self.behavior_policy_net.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'n_agents': self.n_agents,
                'hidden_dim': self.hidden_dim,
                'gamma': self.gamma,
                'alpha': self.alpha,
                'beta': self.beta,
            }
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if 'policy_net_state_dict' in checkpoint:
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        
        if 'critic_net_state_dict' in checkpoint:
            self.critic_net.load_state_dict(checkpoint['critic_net_state_dict'])
        
        if 'behavior_policy_net_state_dict' in checkpoint:
            self.behavior_policy_net.load_state_dict(checkpoint['behavior_policy_net_state_dict'])
        
        print(f"Checkpoint loaded from {path}")

