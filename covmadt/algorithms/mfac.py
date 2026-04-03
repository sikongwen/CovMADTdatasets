"""
Mean-Field Actor-Critic (MFAC) 算法实现

适用于大量同质智能体的在线多智能体强化学习

主要特点:
1. Mean-Field方法：避免状态-动作空间指数增长
2. 在线学习：直接与环境交互学习
3. 内存高效：共享参数，适合大量智能体
4. 可扩展：智能体数量可扩展到数百个
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import copy

from models.base_models import BaseModule


class MeanFieldActor(BaseModule):
    """
    Mean-Field Actor网络
    
    所有智能体共享同一个策略网络
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_encoder: bool = False,
        encoded_dim: int = 512,
        discrete: bool = True,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_encoder = use_encoder
        self.discrete = discrete
        
        # 观察编码器（用于大观察空间）
        if use_encoder and state_dim > 1000:
            # 使用单层编码器以节省内存（164520 -> 256）
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, encoded_dim),
                nn.ReLU(),
            )
            effective_state_dim = encoded_dim
        else:
            self.encoder = None
            effective_state_dim = state_dim
        
        # 策略网络
        layers = []
        layers.append(nn.Linear(effective_state_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
        
        if discrete:
            # 离散动作：输出动作logits
            layers.append(nn.Linear(hidden_dim, action_dim))
        else:
            # 连续动作：输出动作均值（使用tanh限制在[-1, 1]）
            layers.append(nn.Linear(hidden_dim, action_dim))
            layers.append(nn.Tanh())
        self.policy_net = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            states: [B, S] 或 [B, T, S] 状态
            
        返回:
            logits/actions: [B, A] 或 [B, T, A] 
                - 离散动作：logits
                - 连续动作：动作值（已通过tanh限制在[-1, 1]）
        """
        if states.dtype != torch.float32:
            states = states.float()
        
        # 处理维度
        was_3d = states.dim() == 3
        if was_3d:
            batch_size, seq_len, state_dim = states.shape
            states = states.view(-1, state_dim)  # [B*T, S]
        
        # 编码（如果需要）
        if self.encoder is not None:
            states = self.encoder(states)
        
        # 策略网络
        output = self.policy_net(states)  # [B*T, A] 或 [B, A]
        
        if was_3d:
            output = output.view(batch_size, seq_len, self.action_dim)  # [B, T, A]
        
        return output
    
    def get_action_and_log_prob(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
        action_std: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取动作和对数概率
        
        参数:
            states: [B, S] 状态
            deterministic: 是否使用确定性策略
            action_std: 连续动作的标准差（仅用于连续动作）
            
        返回:
            actions: [B] 或 [B, A] 动作（离散为索引，连续为值）
            log_probs: [B] 对数概率
        """
        if self.discrete:
            logits = self.forward(states)  # [B, A]
            probs = F.softmax(logits, dim=-1)  # [B, A]
            log_probs = F.log_softmax(logits, dim=-1)  # [B, A]
            
            if deterministic:
                actions = torch.argmax(probs, dim=-1)  # [B]
            else:
                actions = torch.multinomial(probs, 1).squeeze(-1)  # [B]
            
            # 获取选中动作的对数概率
            selected_log_probs = torch.gather(
                log_probs, -1, actions.unsqueeze(-1)
            ).squeeze(-1)  # [B]
        else:
            # 连续动作：输出动作均值
            action_mean = self.forward(states)  # [B, A]
            
            if deterministic:
                actions = action_mean  # [B, A]
                # 确定性策略的对数概率为0（实际上应该使用delta分布）
                selected_log_probs = torch.zeros(action_mean.shape[0], device=action_mean.device)
            else:
                # 添加噪声（使用正态分布）
                action_std_tensor = torch.ones_like(action_mean) * action_std
                dist = torch.distributions.Normal(action_mean, action_std_tensor)
                actions = dist.sample()  # [B, A]
                actions = torch.clamp(actions, -1.0, 1.0)  # 限制在[-1, 1]
                selected_log_probs = dist.log_prob(actions).sum(dim=-1)  # [B]
        
        return actions, selected_log_probs


class MeanFieldCritic(BaseModule):
    """
    Mean-Field Critic网络
    
    使用Mean-Field近似来估计价值函数
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_encoder: bool = False,
        encoded_dim: int = 512,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_encoder = use_encoder
        
        # 观察编码器（用于大观察空间）
        if use_encoder and state_dim > 1000:
            # 使用单层编码器以节省内存
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, encoded_dim),
                nn.ReLU(),
            )
            effective_state_dim = encoded_dim
        else:
            self.encoder = None
            effective_state_dim = state_dim
        
        # Mean-Field输入：状态 + 动作分布（均值嵌入）
        # 动作分布用one-hot编码的期望来近似
        input_dim = effective_state_dim + action_dim
        
        # 价值网络
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
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(
        self,
        states: torch.Tensor,
        action_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            states: [B, S] 或 [B, T, S] 状态
            action_probs: [B, A] 或 [B, T, A] 动作概率分布（Mean-Field近似）
            
        返回:
            values: [B, 1] 或 [B, T, 1] 价值估计
        """
        if states.dtype != torch.float32:
            states = states.float()
        if action_probs.dtype != torch.float32:
            action_probs = action_probs.float()
        
        # 处理维度
        was_3d = states.dim() == 3
        if was_3d:
            batch_size, seq_len, state_dim = states.shape
            states = states.view(-1, state_dim)  # [B*T, S]
            action_probs = action_probs.view(-1, self.action_dim)  # [B*T, A]
        
        # 编码（如果需要）
        if self.encoder is not None:
            states = self.encoder(states)
        
        # 拼接状态和动作分布
        value_input = torch.cat([states, action_probs], dim=-1)  # [B*T, S + A]
        
        # 价值网络
        values = self.value_net(value_input)  # [B*T, 1]
        
        if was_3d:
            values = values.view(batch_size, seq_len, 1)  # [B, T, 1]
        
        return values


class MFAC(nn.Module):
    """
    Mean-Field Actor-Critic (MFAC) 算法
    
    适用于大量同质智能体的在线多智能体强化学习
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,  # GAE参数
        clip_ratio: float = 0.2,  # PPO风格的clip
        value_coef: float = 0.5,  # 价值损失系数
        entropy_coef: float = 0.01,  # 熵正则化系数
        use_encoder: bool = False,
        encoded_dim: int = 512,
        discrete: bool = True,  # 是否离散动作
        action_std: float = 0.1,  # 连续动作的标准差
        device: str = "cuda",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.discrete = discrete
        self.action_std = action_std
        self.device = device
        
        # 共享的Actor网络（所有智能体使用同一个网络）
        self.actor = MeanFieldActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_encoder=use_encoder,
            encoded_dim=encoded_dim,
            discrete=discrete,
        )
        
        # 共享的Critic网络
        self.critic = MeanFieldCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_encoder=use_encoder,
            encoded_dim=encoded_dim,
        )
        
        # 目标网络（用于稳定训练）
        self.target_critic = copy.deepcopy(self.critic)
        
        # 移动到设备
        self.to(device)
    
    def select_actions(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        为所有智能体选择动作
        
        参数:
            states: [B, S] 或 [n_agents, S] 状态（单个智能体的状态）
            deterministic: 是否使用确定性策略
            
        返回:
            actions: [B] 或 [B, A] 或 [n_agents] 或 [n_agents, A] 动作
            log_probs: [B] 或 [n_agents] 对数概率
            info: 额外信息
        """
        actions, log_probs = self.actor.get_action_and_log_prob(
            states,
            deterministic=deterministic,
            action_std=self.action_std,
        )
        
        # 计算动作概率/均值（用于Critic）
        if self.discrete:
            logits = self.actor(states)
            probs = F.softmax(logits, dim=-1)
            info = {
                "probs": probs,
                "logits": logits,
            }
        else:
            action_mean = self.actor(states)
            info = {
                "action_mean": action_mean,
            }
        
        return actions, log_probs, info
    
    def compute_values(
        self,
        states: torch.Tensor,
        action_probs: torch.Tensor,
        use_target: bool = False,
    ) -> torch.Tensor:
        """
        计算价值
        
        参数:
            states: [B, S] 或 [B, T, S] 状态
            action_probs: [B, A] 或 [B, T, A] 动作概率分布
            use_target: 是否使用目标网络
            
        返回:
            values: [B, 1] 或 [B, T, 1] 价值估计
        """
        critic_net = self.target_critic if use_target else self.critic
        return critic_net(states, action_probs)
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算MFAC损失
        
        参数:
            states: [B, T, S] 状态序列
            actions: [B, T] 动作序列
            rewards: [B, T] 奖励序列
            next_states: [B, T, S] 下一个状态序列
            dones: [B, T] 终止标志序列
            old_log_probs: [B, T] 旧的对数概率（用于PPO clip）
            
        返回:
            dict包含损失项
        """
        batch_size, seq_len, _ = states.shape
        
        # 获取当前策略的动作概率/均值和对数概率
        if self.discrete:
            logits = self.actor(states)  # [B, T, A]
            probs = F.softmax(logits, dim=-1)  # [B, T, A]
            log_probs = F.log_softmax(logits, dim=-1)  # [B, T, A]
            
            # 获取选中动作的对数概率
            actions_onehot = F.one_hot(actions.long(), self.action_dim).float()  # [B, T, A]
            selected_log_probs = (log_probs * actions_onehot).sum(dim=-1)  # [B, T]
        else:
            # 连续动作
            action_mean = self.actor(states)  # [B, T, A]
            action_std_tensor = torch.ones_like(action_mean) * self.action_std
            dist = torch.distributions.Normal(action_mean, action_std_tensor)
            selected_log_probs = dist.log_prob(actions).sum(dim=-1)  # [B, T]
            probs = action_mean  # 用于Critic（使用均值）
            log_probs = None  # 连续动作不需要log_probs
        
        # 计算价值（使用Mean-Field动作分布）
        # 对于离散动作，probs是概率分布；对于连续动作，probs是动作均值
        values = self.compute_values(states, probs, use_target=False)  # [B, T, 1]
        values = values.squeeze(-1)  # [B, T]
        
        # 计算目标价值（使用目标网络）
        with torch.no_grad():
            if self.discrete:
                next_logits = self.actor(next_states)  # [B, T, A]
                next_probs = F.softmax(next_logits, dim=-1)  # [B, T, A]
            else:
                next_probs = self.actor(next_states)  # [B, T, A] (动作均值)
            next_values = self.compute_values(next_states, next_probs, use_target=True)  # [B, T, 1]
            next_values = next_values.squeeze(-1)  # [B, T]
            
            # 计算TD目标
            dones_float = dones.float()
            targets = rewards + self.gamma * next_values * (1 - dones_float)  # [B, T]
        
        # 计算优势（GAE）
        advantages = targets - values  # [B, T]
        
        # 价值损失
        value_loss = F.mse_loss(values, targets)
        
        # 策略损失（PPO风格）
        ratio = torch.exp(selected_log_probs - old_log_probs)  # [B, T]
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # 熵正则化（鼓励探索）
        if self.discrete:
            entropy = -(probs * log_probs).sum(dim=-1).mean()  # 标量
        else:
            # 连续动作的熵（正态分布的熵）
            action_std_tensor = torch.tensor(self.action_std, device=states.device, dtype=torch.float32)
            entropy = 0.5 * torch.log(2 * torch.tensor(np.pi, device=states.device) * torch.tensor(np.e, device=states.device) * action_std_tensor**2)
        
        # 总损失
        total_loss = (
            policy_loss +
            self.value_coef * value_loss -
            self.entropy_coef * entropy
        )
        
        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "advantages_mean": advantages.mean(),
            "values_mean": values.mean(),
            "targets_mean": targets.mean(),
        }
    
    def update_target_network(self, tau: float = 0.005):
        """软更新目标网络"""
        for param, target_param in zip(
            self.critic.parameters(),
            self.target_critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

