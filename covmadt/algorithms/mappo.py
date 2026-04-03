"""
MAPPO (Multi-Agent Proximal Policy Optimization) 算法实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import copy


class Actor(nn.Module):
    """Actor网络（策略网络）- 每个智能体独立"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        discrete: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discrete = discrete
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        if discrete:
            # 离散动作：输出动作概率分布
            self.action_head = nn.Linear(hidden_dim // 2, action_dim)
        else:
            # 连续动作：输出动作均值
            self.action_head = nn.Linear(hidden_dim // 2, action_dim)
            self.tanh = nn.Tanh()
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """前向传播，返回动作概率分布或动作均值"""
        x = self.net(obs)
        if self.discrete:
            return F.softmax(self.action_head(x), dim=-1)
        else:
            return self.tanh(self.action_head(x))
    
    def get_action_and_log_prob(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作和对数概率
        
        Args:
            obs: 观察
            action_mask: 动作掩码（可选）
            deterministic: 是否使用确定性策略（用于评估）
        """
        probs = self.forward(obs)
        
        # 应用动作掩码（如果提供）
        if action_mask is not None:
            probs = probs * action_mask
            probs_sum = probs.sum(dim=-1, keepdim=True)
            # 确保至少有一个合法动作
            if (probs_sum < 1e-8).any():
                # 如果没有合法动作，使用掩码本身作为概率
                probs = action_mask / (action_mask.sum(dim=-1, keepdim=True) + 1e-8)
            else:
                probs = probs / probs_sum
        
        if deterministic:
            # 确定性策略：选择概率最大的动作
            action = torch.argmax(probs, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
        else:
            # 随机策略：从分布中采样
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def get_log_prob(self, obs: torch.Tensor, actions: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取给定动作的对数概率"""
        probs = self.forward(obs)
        
        # 应用动作掩码（如果提供）
        if action_mask is not None:
            probs = probs * action_mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(actions)
        
        return log_prob
    
    def get_entropy(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算策略熵"""
        probs = self.forward(obs)
        
        # 应用动作掩码（如果提供）
        if action_mask is not None:
            probs = probs * action_mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        dist = torch.distributions.Categorical(probs)
        return dist.entropy()


class Critic(nn.Module):
    """Critic网络（价值网络）- 使用全局状态"""
    
    def __init__(
        self,
        global_obs_dim: int,  # 所有智能体观察的拼接
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.global_obs_dim = global_obs_dim
        
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """前向传播，返回价值估计"""
        return self.net(global_obs).squeeze(-1)


class MAPPO:
    """MAPPO算法"""
    
    def __init__(
        self,
        obs_dim: int,  # 单个智能体的观察维度
        action_dim: int,  # 单个智能体的动作维度
        n_agents: int,  # 智能体数量
        hidden_dim: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        discrete: bool = True,
        device: str = "cuda",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.discrete = discrete
        self.device = device
        
        # 全局观察维度（所有智能体观察的拼接）
        self.global_obs_dim = obs_dim * n_agents
        
        # 为每个智能体创建Actor和Critic
        self.actors = nn.ModuleList([
            Actor(obs_dim, action_dim, hidden_dim, discrete).to(device)
            for _ in range(n_agents)
        ])
        
        # 所有智能体共享一个Critic（使用全局观察）
        self.critic = Critic(self.global_obs_dim, hidden_dim).to(device)
        
        # 优化器
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 训练步数
        self.train_step = 0
    
    def select_actions(self, obs_dict: Dict[str, np.ndarray], action_masks: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, int]:
        """
        为所有智能体选择动作
        
        参数:
            obs_dict: {agent_id: observation}
            action_masks: {agent_id: action_mask} (可选)
            
        返回:
            {agent_id: action}
        """
        actions = {}
        # 确保按固定顺序处理智能体
        agent_ids = sorted(obs_dict.keys())
        
        for i, agent_id in enumerate(agent_ids):
            obs = obs_dict[agent_id]
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            action_mask = None
            if action_masks is not None and agent_id in action_masks:
                mask = action_masks[agent_id]
                # 确保掩码是有效的
                if mask.sum() > 0:
                    action_mask = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                else:
                    # 如果没有合法动作，创建全1掩码（不应该发生）
                    action_mask = torch.ones(self.action_dim).unsqueeze(0).to(self.device)
            
            self.actors[i].eval()
            with torch.no_grad():
                action, _ = self.actors[i].get_action_and_log_prob(obs_tensor, action_mask)
                action_val = action.item()
                
                # 验证动作是否合法
                if action_masks is not None and agent_id in action_masks:
                    if action_masks[agent_id][action_val] == 0:
                        # 如果动作不合法，从合法动作中随机选择
                        valid_actions = np.where(action_masks[agent_id] > 0)[0]
                        if len(valid_actions) > 0:
                            action_val = int(np.random.choice(valid_actions))
                        else:
                            action_val = 0  # 默认动作
                
                actions[agent_id] = action_val
        
        return actions
    
    def compute_gae(
        self,
        rewards: torch.Tensor,  # [T, n_agents]
        values: torch.Tensor,    # [T, n_agents]
        dones: torch.Tensor,     # [T, n_agents]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GAE优势
        
        返回:
            advantages: [T, n_agents]
            returns: [T, n_agents]
        """
        T, n_agents = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        last_gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            last_gae = advantages[t]
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(
        self,
        obs_batch: List[Dict[str, np.ndarray]],  # [T, {agent_id: obs}]
        action_batch: List[Dict[str, int]],       # [T, {agent_id: action}]
        reward_batch: List[Dict[str, float]],    # [T, {agent_id: reward}]
        done_batch: List[Dict[str, bool]],       # [T, {agent_id: done}]
        old_log_probs_batch: List[Dict[str, float]],  # [T, {agent_id: log_prob}]
        global_obs_batch: List[np.ndarray],      # [T, global_obs]
        action_masks_batch: Optional[List[Dict[str, np.ndarray]]] = None,  # [T, {agent_id: mask}]
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        更新策略和价值网络
        
        参数:
            obs_batch: 观察序列 [T, {agent_id: obs}]
            action_batch: 动作序列 [T, {agent_id: action}]
            reward_batch: 奖励序列 [T, {agent_id: reward}]
            done_batch: 完成标志序列 [T, {agent_id: done}]
            old_log_probs_batch: 旧对数概率序列 [T, {agent_id: log_prob}]
            global_obs_batch: 全局观察序列 [T, global_obs]
            action_masks_batch: 动作掩码序列 [T, {agent_id: mask}] (可选)
            n_epochs: PPO更新轮数
            batch_size: 批量大小
            
        返回:
            损失统计信息
        """
        T = len(obs_batch)
        # 获取所有智能体ID（确保顺序一致）
        agent_ids = sorted(obs_batch[0].keys())
        
        # 验证智能体数量
        if len(agent_ids) != self.n_agents:
            raise ValueError(f"智能体数量不匹配: 期望 {self.n_agents}, 实际 {len(agent_ids)}")
        
        # 转换为tensor格式 [T, n_agents]
        obs_tensor = torch.zeros(T, self.n_agents, self.obs_dim).to(self.device)
        action_tensor = torch.zeros(T, self.n_agents, dtype=torch.long).to(self.device)
        reward_tensor = torch.zeros(T, self.n_agents).to(self.device)
        done_tensor = torch.zeros(T, self.n_agents).to(self.device)
        old_log_prob_tensor = torch.zeros(T, self.n_agents).to(self.device)
        global_obs_tensor = torch.zeros(T, self.global_obs_dim).to(self.device)
        
        for t in range(T):
            for i, agent_id in enumerate(agent_ids):
                # 确保所有字典都有这个agent_id
                if agent_id not in obs_batch[t]:
                    raise KeyError(f"时间步 {t} 缺少智能体 {agent_id} 的观察")
                if agent_id not in action_batch[t]:
                    raise KeyError(f"时间步 {t} 缺少智能体 {agent_id} 的动作")
                if agent_id not in reward_batch[t]:
                    raise KeyError(f"时间步 {t} 缺少智能体 {agent_id} 的奖励")
                if agent_id not in done_batch[t]:
                    raise KeyError(f"时间步 {t} 缺少智能体 {agent_id} 的done标志")
                if agent_id not in old_log_probs_batch[t]:
                    raise KeyError(f"时间步 {t} 缺少智能体 {agent_id} 的old_log_prob")
                
                obs_tensor[t, i] = torch.FloatTensor(obs_batch[t][agent_id]).to(self.device)
                action_tensor[t, i] = action_batch[t][agent_id]
                reward_tensor[t, i] = reward_batch[t][agent_id]
                done_tensor[t, i] = 1.0 if done_batch[t][agent_id] else 0.0
                old_log_prob_tensor[t, i] = old_log_probs_batch[t][agent_id]
            global_obs_tensor[t] = torch.FloatTensor(global_obs_batch[t]).to(self.device)
        
        # 计算价值估计
        self.critic.eval()
        with torch.no_grad():
            values = self.critic(global_obs_tensor)  # [T]
            # 扩展为 [T, n_agents]（所有智能体共享相同的全局价值）
            values = values.unsqueeze(-1).expand(-1, self.n_agents)
        
        # 计算GAE优势
        advantages, returns = self.compute_gae(reward_tensor, values, done_tensor)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 准备动作掩码（如果提供）
        action_mask_tensor = None
        if action_masks_batch is not None and any(m is not None for m in action_masks_batch):
            action_mask_tensor = torch.zeros(T, self.n_agents, self.action_dim).to(self.device)
            for t in range(T):
                if action_masks_batch[t] is not None:
                    for i, agent_id in enumerate(agent_ids):
                        if agent_id in action_masks_batch[t]:
                            action_mask_tensor[t, i] = torch.FloatTensor(action_masks_batch[t][agent_id]).to(self.device)
        
        # 创建数据集索引
        dataset_size = T * self.n_agents
        indices = np.arange(dataset_size)
        
        # 训练多个epoch
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # 从batch_indices中提取时间步和智能体索引
                time_indices = batch_indices // self.n_agents
                agent_indices = batch_indices % self.n_agents
                
                # 收集batch数据
                batch_obs = obs_tensor[time_indices, agent_indices]  # [B, obs_dim]
                batch_action = action_tensor[time_indices, agent_indices]  # [B]
                batch_old_log_prob = old_log_prob_tensor[time_indices, agent_indices]  # [B]
                batch_advantages = advantages[time_indices, agent_indices]  # [B]
                batch_returns = returns[time_indices, agent_indices]  # [B]
                batch_values = values[time_indices, agent_indices]  # [B]
                batch_global_obs = global_obs_tensor[time_indices]  # [B, global_obs_dim]
                
                batch_action_mask = None
                if action_mask_tensor is not None:
                    batch_action_mask = action_mask_tensor[time_indices, agent_indices]  # [B, action_dim]
                
                # 按智能体分组更新
                for agent_idx in range(self.n_agents):
                    agent_mask = agent_indices == agent_idx
                    if not agent_mask.any():
                        continue
                    
                    agent_obs = batch_obs[agent_mask]
                    agent_action = batch_action[agent_mask]
                    agent_old_log_prob = batch_old_log_prob[agent_mask]
                    agent_advantages = batch_advantages[agent_mask]
                    agent_action_mask = None
                    if batch_action_mask is not None:
                        agent_action_mask = batch_action_mask[agent_mask]
                    
                    # 计算新策略的对数概率
                    self.actors[agent_idx].train()
                    new_log_prob = self.actors[agent_idx].get_log_prob(agent_obs, agent_action, agent_action_mask)
                    
                    # 计算重要性采样比率
                    ratio = torch.exp(new_log_prob - agent_old_log_prob)
                    
                    # PPO裁剪策略损失
                    policy_loss_1 = ratio * agent_advantages
                    policy_loss_2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * agent_advantages
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                    
                    # 熵正则化
                    entropy = self.actors[agent_idx].get_entropy(agent_obs, agent_action_mask).mean()
                    
                    # 总策略损失
                    actor_loss = policy_loss - self.entropy_coef * entropy
                    
                    # 更新Actor
                    self.actor_optimizers[agent_idx].zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), self.max_grad_norm)
                    self.actor_optimizers[agent_idx].step()
                    
                    total_policy_loss += policy_loss.item()
                    total_entropy += entropy.item()
                
                # 更新Critic（使用全局观察）
                agent_global_obs = batch_global_obs  # [B, global_obs_dim]
                agent_returns = batch_returns  # [B] - 使用所有智能体的returns的平均值
                agent_values = batch_values  # [B]
                
                self.critic.train()
                new_values = self.critic(agent_global_obs)  # [B]
                
                # 价值损失（带裁剪）
                value_pred_clipped = agent_values + torch.clamp(
                    new_values - agent_values, -self.value_clip, self.value_clip
                )
                value_loss_1 = (new_values - agent_returns) ** 2
                value_loss_2 = (value_pred_clipped - agent_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                
                # 更新Critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_value_loss += value_loss.item()
                n_updates += 1
        
        self.train_step += 1
        
        if n_updates > 0:
            return {
                "policy_loss": total_policy_loss / n_updates / self.n_agents,
                "value_loss": total_value_loss / n_updates,
                "entropy": total_entropy / n_updates / self.n_agents,
            }
        else:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
            }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict(),
            'config': {
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'n_agents': self.n_agents,
                'hidden_dim': self.hidden_dim,
                'discrete': self.discrete,
            }
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        # 使用weights_only=False以支持包含numpy对象的检查点（PyTorch 2.6+兼容性）
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        for i, actor_state in enumerate(checkpoint['actors']):
            self.actors[i].load_state_dict(actor_state)
        self.critic.load_state_dict(checkpoint['critic'])

