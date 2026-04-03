"""
OMIGA (Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization) 算法实现

OMIGA是一种离线多智能体强化学习算法，通过隐式的全局到局部值正则化来优化策略。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import math
import copy

from models.base_models import BaseModule


class OMIGAPolicy(BaseModule):
    """
    OMIGA策略网络
    
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


class GlobalValueNetwork(BaseModule):
    """
    全局价值网络
    
    估计全局状态价值 V_g(s)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
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
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            states: [..., S] 状态
            
        返回:
            values: [..., 1] 全局状态价值
        """
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


class LocalValueNetwork(BaseModule):
    """
    局部价值网络
    
    估计局部状态-动作价值 Q_l(s,a)
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
            values: [..., 1] 局部状态-动作价值
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
                    # [B, T, A, A] -> [B, T, A] (取第一个动作)
                    actions_onehot = actions[:, :, 0, :].float()
                else:
                    # 其他4维情况，尝试reshape
                    actions_onehot = actions.reshape(batch_size, seq_len, -1)
                    if actions_onehot.shape[-1] != self.action_dim:
                        # 如果reshape后维度不对，取前action_dim个维度
                        actions_onehot = actions_onehot[:, :, :self.action_dim]
            else:
                # 其他情况，尝试reshape
                try:
                    actions_onehot = actions.reshape(batch_size, seq_len, -1)
                    if actions_onehot.shape[-1] != self.action_dim:
                        # 如果reshape后维度不对，尝试转换为one-hot
                        actions_flat = actions.reshape(-1)
                        actions_onehot_flat = F.one_hot(actions_flat.long(), self.action_dim).float()
                        actions_onehot = actions_onehot_flat.view(batch_size, seq_len, self.action_dim)
                except:
                    # 如果reshape失败，尝试转换为one-hot
                    actions_flat = actions.reshape(-1)
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
                # 如果是4维 [B, T, 1, A] 或 [B, 1, T, A]，需要压缩
                if actions_onehot.shape[1] == 1:
                    actions_onehot = actions_onehot.squeeze(1)  # [B, T, A]
                elif actions_onehot.shape[2] == 1:
                    actions_onehot = actions_onehot.squeeze(2)  # [B, T, A]
                else:
                    # 如果是 [B, T, A, A]，取第一个A维度
                    actions_onehot = actions_onehot[:, :, 0, :]  # [B, T, A]
            elif actions_onehot.dim() != 3:
                # 其他维度，尝试reshape
                batch_size, seq_len = states.shape[:2]
                actions_onehot = actions_onehot.view(batch_size, seq_len, self.action_dim)
        
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


class OMIGA(nn.Module):
    """
    OMIGA (Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization) 算法
    
    主要特点:
    1. 全局价值网络：估计全局状态价值 V_g(s)
    2. 局部价值网络：估计局部状态-动作价值 Q_l(s,a)
    3. 隐式正则化：通过全局到局部的值正则化来优化策略
    4. 多智能体支持：处理多智能体场景
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        gamma: float = 0.99,
        lambda_reg: float = 0.1,  # 正则化系数
        alpha: float = 0.1,  # 保守性系数
        device: str = "cuda",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.device = device
        
        # 策略网络
        self.policy_net = OMIGAPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        
        # 全局价值网络
        self.global_value_net = GlobalValueNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        
        # 局部价值网络
        self.local_value_net = LocalValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        
        # 目标网络（用于稳定训练）
        self.target_global_value_net = copy.deepcopy(self.global_value_net)
        self.target_local_value_net = copy.deepcopy(self.local_value_net)
        
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
        local_values = self.local_value_net(states, actions_onehot)
        global_values = self.global_value_net(states)
        
        info = {
            "logits": logits,
            "local_values": local_values,
            "global_values": global_values,
            "action_dist": torch.softmax(logits, dim=-1),
        }
        
        return actions, log_probs, info
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算OMIGA损失
        
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
        
        # 确保actions维度正确 [B, T]
        if actions.dim() == 1:
            # [B] -> [B, 1]
            actions = actions.unsqueeze(1)
        elif actions.dim() == 3:
            # [B, T, 1] -> [B, T]
            if actions.shape[-1] == 1:
                actions = actions.squeeze(-1)
            else:
                # 如果是one-hot [B, T, A]，转换为索引
                actions = actions.argmax(dim=-1)
        elif actions.dim() > 3:
            # 其他维度，尝试reshape
            actions = actions.view(batch_size, seq_len)
        
        # 确保actions的batch和seq维度匹配
        if actions.shape[0] != batch_size or actions.shape[1] != seq_len:
            actions = actions.view(batch_size, seq_len)
        
        # 将动作转换为one-hot
        actions_onehot = F.one_hot(actions.long(), self.action_dim).float()  # [B, T, A]
        
        # 计算全局和局部价值
        global_values = self.global_value_net(states)  # [B, T, 1]
        local_values = self.local_value_net(states, actions_onehot)  # [B, T, 1]
        
        # 计算目标价值（使用目标网络）
        with torch.no_grad():
            # 全局价值目标
            next_global_values = self.target_global_value_net(next_states)  # [B, T, 1]
            
            # 局部价值目标：对下一个状态，使用当前策略采样动作
            next_states_flat = next_states.view(-1, self.state_dim)  # [B*T, S]
            next_policy_logits = self.policy_net(next_states_flat)  # [B*T, A]
            next_policy_probs = F.softmax(next_policy_logits, dim=-1)  # [B*T, A]
            
            # 优化：只计算top-k动作的局部价值（而不是所有动作），大幅减少计算量
            k = min(5, self.action_dim)  # 使用更小的k值（5而不是所有40个动作）
            next_policy_probs_reshaped = next_policy_probs.view(batch_size, seq_len, self.action_dim)  # [B, T, A]
            next_topk_values, next_topk_indices = torch.topk(next_policy_probs_reshaped, k, dim=-1)  # [B, T, k]
            
            # 计算top-k动作的局部价值
            next_states_expanded = next_states.unsqueeze(2).expand(-1, -1, k, -1)  # [B, T, k, S]
            next_states_flat_topk = next_states_expanded.reshape(-1, self.state_dim)  # [B*T*k, S]
            
            # 创建top-k动作的one-hot编码
            next_topk_indices_flat = next_topk_indices.view(-1)  # [B*T*k]
            next_topk_actions_onehot = F.one_hot(next_topk_indices_flat, self.action_dim).float()  # [B*T*k, A]
            
            # 只计算top-k动作的局部价值
            next_topk_local_values_flat = self.target_local_value_net(next_states_flat_topk, next_topk_actions_onehot)  # [B*T*k, 1]
            next_topk_local_values = next_topk_local_values_flat.view(batch_size, seq_len, k)  # [B, T, k]
            
            # 加权平均（使用top-k动作的概率）
            next_topk_probs_normalized = next_topk_values / (next_topk_values.sum(dim=-1, keepdim=True) + 1e-8)  # [B, T, k]
            next_local_values = (next_topk_probs_normalized * next_topk_local_values).sum(dim=-1, keepdim=True)  # [B, T, 1]
            
            # 计算TD目标
            if dones is not None:
                dones_float = dones.float() if dones.dtype != torch.float32 else dones
                global_targets = rewards + self.gamma * next_global_values * (1 - dones_float)
                local_targets = rewards + self.gamma * next_local_values * (1 - dones_float)
            else:
                global_targets = rewards + self.gamma * next_global_values
                local_targets = rewards + self.gamma * next_local_values
        
        # 全局价值损失
        global_value_loss = F.mse_loss(global_values, global_targets)
        
        # 局部价值损失
        local_value_loss = F.mse_loss(local_values, local_targets)
        
        # 隐式正则化损失：全局到局部的值正则化
        # 正则化项：使局部价值与全局价值保持一致
        regularization_loss = self.lambda_reg * F.mse_loss(
            local_values,
            global_values.expand_as(local_values)
        )
        
        # 保守性损失：惩罚分布外动作的高估
        # 优化：只对top-k动作计算，减少计算量
        policy_logits = self.policy_net(states.view(-1, self.state_dim))  # [B*T, A]
        policy_logits = policy_logits.view(batch_size, seq_len, self.action_dim)  # [B, T, A]
        policy_probs = F.softmax(policy_logits, dim=-1)  # [B, T, A]
        
        # 优化：只计算top-k动作的局部价值（k=min(5, action_dim)，减少计算量）
        k = min(5, self.action_dim)
        topk_values, topk_indices = torch.topk(policy_probs, k, dim=-1)  # [B, T, k]
        
        # 计算top-k动作的局部价值
        states_expanded = states.unsqueeze(2).expand(-1, -1, k, -1)  # [B, T, k, S]
        states_flat_topk = states_expanded.reshape(-1, self.state_dim)  # [B*T*k, S]
        
        # 创建top-k动作的one-hot编码
        topk_indices_flat = topk_indices.view(-1)  # [B*T*k]
        topk_actions_onehot = F.one_hot(topk_indices_flat, self.action_dim).float()  # [B*T*k, A]
        
        # 只计算top-k动作的局部价值
        topk_local_values_flat = self.local_value_net(states_flat_topk, topk_actions_onehot)  # [B*T*k, 1]
        topk_local_values = topk_local_values_flat.view(batch_size, seq_len, k)  # [B, T, k]
        
        # 使用top-k动作的加权平均来近似期望值
        topk_probs_normalized = topk_values / (topk_values.sum(dim=-1, keepdim=True) + 1e-8)  # [B, T, k]
        expected_local_value_topk = (topk_probs_normalized * topk_local_values).sum(dim=-1, keepdim=True)  # [B, T, 1]
        
        # 保守性损失：惩罚高估（使用top-k的最大值）
        conservative_loss = self.alpha * (topk_local_values.max(dim=-1, keepdim=True)[0] - expected_local_value_topk).mean()
        
        # 策略损失：最大化加权局部价值
        action_log_probs = F.log_softmax(policy_logits, dim=-1)  # [B, T, A]
        selected_log_probs = torch.gather(action_log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)  # [B, T]
        policy_loss = -(selected_log_probs.unsqueeze(-1) * local_values).mean()
        
        # 总损失
        total_loss = (
            global_value_loss +
            local_value_loss +
            regularization_loss +
            conservative_loss +
            policy_loss
        )
        
        return {
            "loss": total_loss,
            "global_value_loss": global_value_loss,
            "local_value_loss": local_value_loss,
            "regularization_loss": regularization_loss,
            "conservative_loss": conservative_loss,
            "policy_loss": policy_loss,
            "global_value_mean": global_values.mean(),
            "local_value_mean": local_values.mean(),
            "target_mean": local_targets.mean(),
        }
    
    def update_target_networks(self, tau: float = 0.005):
        """
        软更新目标网络
        
        参数:
            tau: 软更新系数
        """
        # 更新全局价值网络
        for target_param, param in zip(self.target_global_value_net.parameters(), 
                                      self.global_value_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # 更新局部价值网络
        for target_param, param in zip(self.target_local_value_net.parameters(), 
                                      self.local_value_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
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
            # 确保actions是正确的维度
            if actions.dim() == 2 and states.dim() == 3:
                # actions是[B, T]，转换为one-hot后是[B, T, A]
                actions_onehot = F.one_hot(actions.long(), self.action_dim).float()
            elif actions.dim() == 3 and actions.shape[-1] == self.action_dim:
                # actions已经是one-hot [B, T, A]
                actions_onehot = actions.float()
            else:
                # 其他情况，尝试转换
                actions_onehot = F.one_hot(actions.long().squeeze(-1) if actions.dim() == 3 and actions.shape[-1] == 1 else actions.long(), self.action_dim).float()
                if states.dim() == 3 and actions_onehot.dim() == 2:
                    actions_onehot = actions_onehot.unsqueeze(1)
            
            local_values = self.local_value_net(states, actions_onehot)
            global_values = self.global_value_net(states)
            results["local_values"] = local_values
            results["global_values"] = global_values
        
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
            'global_value_net_state_dict': self.global_value_net.state_dict(),
            'local_value_net_state_dict': self.local_value_net.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'n_agents': self.n_agents,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'gamma': self.gamma,
                'lambda_reg': self.lambda_reg,
                'alpha': self.alpha,
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
        
        if 'global_value_net_state_dict' in checkpoint:
            self.global_value_net.load_state_dict(checkpoint['global_value_net_state_dict'])
        
        if 'local_value_net_state_dict' in checkpoint:
            self.local_value_net.load_state_dict(checkpoint['local_value_net_state_dict'])
        
        print(f"Checkpoint loaded from {path}")

