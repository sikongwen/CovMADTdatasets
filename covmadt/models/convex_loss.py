import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import math

from .base_models import BaseModule


class OccupancyMeasure:
    """占用度量计算类"""
    
    @staticmethod
    def compute_discounted_occupancy(
        states: torch.Tensor,
        actions: torch.Tensor,
        policy_probs: torch.Tensor,
        gamma: float = 0.99,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        计算折扣占用度量
        
        μ(s,a) = (1-γ) Σ_{t=0}^∞ γ^t P(s_t=s, a_t=a | π)
        
        简化实现：使用经验轨迹
        """
        batch_size, seq_len, state_dim = states.shape
        
        # 处理actions的形状：可能是[B, T]（离散索引）或[B, T, A]（one-hot）
        if actions.dim() == 2:
            # actions是离散索引[B, T]，需要转换为one-hot
            # 从actions的最大值推断action_dim（加1因为索引从0开始）
            max_action = int(actions.max().item())
            action_dim = max_action + 1
            # 确保action_dim至少为1
            action_dim = max(action_dim, 1)
            actions_onehot = torch.zeros(batch_size, seq_len, action_dim, 
                                        device=actions.device, dtype=torch.float32)
            actions_onehot.scatter_(-1, actions.unsqueeze(-1).long(), 1.0)
            actions = actions_onehot
        else:
            _, _, action_dim = actions.shape
        
        # 计算折扣因子
        discounts = torch.tensor([gamma**t for t in range(seq_len)], 
                                device=states.device)
        discounts = discounts.view(1, seq_len, 1, 1)  # [1, seq_len, 1, 1]
        
        # 扩展为状态-动作网格
        # 这里简化处理，实际应根据具体状态-动作空间离散化
        
        # 经验占用度量（简化：假设状态已离散化）
        occupancy = torch.zeros(batch_size, seq_len, state_dim, action_dim, 
                              device=states.device)
        
        # 为每个时间步添加折扣贡献
        # 注意：对于连续状态空间，这里使用简化实现
        # 将状态和动作投影到低维空间进行近似
        for t in range(seq_len):
            # 对于连续状态，使用简化的占用度量计算
            # 将状态和动作的乘积作为占用度量的近似
            state_t = states[:, t]  # [B, state_dim]
            action_t = actions[:, t]  # [B, action_dim]
            
            # 计算状态-动作对的占用（简化：使用外积）
            # occupancy[:, t, :, :] = state_t.unsqueeze(-1) * action_t.unsqueeze(-2)
            # 但由于内存限制，这里使用更简化的方法
            # 只计算每个时间步的平均占用
            state_norm = state_t.norm(dim=-1, keepdim=True)  # [B, 1]
            action_norm = action_t.norm(dim=-1, keepdim=True)  # [B, 1]
            
            # 简化的占用度量：使用状态和动作的范数
            occupancy[:, t, :, :] = (state_norm.unsqueeze(-1) * action_norm.unsqueeze(-2) * 
                                    discounts[:, t].unsqueeze(-1).unsqueeze(-1))
        
        # 归一化
        if normalize:
            occupancy_sum = occupancy.sum(dim=(-2, -1), keepdim=True)
            occupancy = occupancy / (occupancy_sum + 1e-8)
        
        return occupancy
    
    @staticmethod
    def state_occupancy_from_policy(
        states: torch.Tensor,
        policy_probs: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """
        从策略计算状态占用度量
        
        μ^s(s) = (1-γ) Σ_{t=0}^∞ γ^t P(s_t=s | π)
        """
        batch_size, seq_len, state_dim = states.shape
        
        # 计算折扣因子
        discounts = torch.tensor([gamma**t for t in range(seq_len)], 
                                device=states.device)
        discounts = discounts.view(1, seq_len, 1)  # [1, seq_len, 1]
        
        # 经验状态访问频率
        state_occupancy = torch.zeros(batch_size, seq_len, state_dim, 
                                     device=states.device)
        
        # 这里简化处理
        for t in range(seq_len):
            state_idx = states[:, t].long() % state_dim
            batch_idx = torch.arange(batch_size, device=states.device)
            
            state_occupancy[batch_idx, t, state_idx] += discounts[:, t]
        
        # 归一化
        state_occupancy_sum = state_occupancy.sum(dim=-1, keepdim=True)
        state_occupancy = state_occupancy / (state_occupancy_sum + 1e-8)
        
        return state_occupancy


class ConvexRegularizationLoss(BaseModule):
    """
    凸正则化损失模块
    
    实现论文中的效用函数:
    u(μ) = r^T μ - τ d_KL(μ || μ_ref)
    """
    
    def __init__(
        self,
        tau: float = 0.1,
        alpha: float = 0.1,
        epsilon: float = 1e-8,
        use_occupancy_measure: bool = True,
    ):
        super().__init__()
        
        self.tau = tau
        self.alpha = alpha  # 梯度正则化系数
        self.epsilon = epsilon
        self.use_occupancy_measure = use_occupancy_measure
        
        # 日志
        self.register_buffer("log_tau", torch.tensor(math.log(tau)))
        
    def kl_divergence(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算KL散度 d_KL(p || q)
        
        参数:
            p: 分布P [..., K]
            q: 分布Q [..., K]
            mask: 掩码 [..., K]（可选）
            
        返回:
            kl: KL散度 [...]
        """
        # 确保数值稳定性
        p = torch.clamp(p, self.epsilon, 1.0)
        q = torch.clamp(q, self.epsilon, 1.0)
        
        # 计算KL散度
        log_ratio = torch.log(p) - torch.log(q)
        kl = torch.sum(p * log_ratio, dim=-1)
        
        # 应用掩码
        if mask is not None:
            kl = kl * mask
        
        return kl
    
    def js_divergence(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算Jensen-Shannon散度"""
        m = 0.5 * (p + q)
        js = 0.5 * self.kl_divergence(p, m, mask) + 0.5 * self.kl_divergence(q, m, mask)
        return js
    
    def compute_utility(
        self,
        reward: torch.Tensor,
        occupancy: torch.Tensor,
        ref_occupancy: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算凸效用函数
        
        u(μ) = r^T μ - τ d_KL(μ || μ_ref)
        
        参数:
            reward: 奖励 [..., 1] 或 [..., S, A]
            occupancy: 当前占用度量 [..., S, A]
            ref_occupancy: 参考占用度量 [..., S, A]
            mask: 掩码 [..., S, A]（可选）
            normalize: 是否归一化
            
        返回:
            utility: 效用值 [...]
            reward_term: 奖励项 [...]
            kl_term: KL散度项 [...]
        """
        batch_dims = occupancy.shape[:-2]
        
        # 确保维度匹配
        if reward.dim() == occupancy.dim() - 1:
            # reward: [..., 1] -> 扩展为 [..., S, A]
            # 确保reward是4维 [B, T, 1, 1]，然后expand到 [B, T, S, A]
            if reward.dim() == 3:  # [B, T, 1]
                reward = reward.unsqueeze(-1)  # [B, T, 1, 1]
            elif reward.dim() > 4:
                # 如果维度过多，先squeeze到4维
                while reward.dim() > 4:
                    reward = reward.squeeze(-1)
            reward = reward.expand(*batch_dims, occupancy.shape[-2], occupancy.shape[-1])
        
        # 归一化占用度量
        if normalize:
            occupancy_sum = occupancy.sum(dim=(-2, -1), keepdim=True)
            occupancy = occupancy / (occupancy_sum + self.epsilon)
            
            ref_sum = ref_occupancy.sum(dim=(-2, -1), keepdim=True)
            ref_occupancy = ref_occupancy / (ref_sum + self.epsilon)
        
        # 计算奖励项: r^T μ
        reward_term = torch.sum(reward * occupancy, dim=(-2, -1))
        
        # 计算KL散度项: d_KL(μ || μ_ref)
        kl_term = self.kl_divergence(
            occupancy.view(-1, occupancy.shape[-2] * occupancy.shape[-1]),
            ref_occupancy.view(-1, ref_occupancy.shape[-2] * ref_occupancy.shape[-1]),
            mask.view(-1, mask.shape[-2] * mask.shape[-1]) if mask is not None else None
        )
        kl_term = kl_term.view(*batch_dims)
        
        # 凸效用函数
        utility = reward_term - self.tau * kl_term
        
        return utility, reward_term, kl_term
    
    def compute_exploitability(
        self,
        policy_gradients: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算利用性上界
        
        ε ≤ τ log(|A|) + √2 ||Π(∇u)||
        
        参数:
            policy_gradients: 策略梯度 [..., A]
            action_mask: 动作掩码 [..., A]（可选）
            action_dim: 动作维度（如果未提供掩码）
            
        返回:
            exploitability_bound: 利用性上界 [...]
            projected_grad_norm: 投影梯度范数 [...]
        """
        if action_dim is None and action_mask is None:
            action_dim = policy_gradients.shape[-1]
        
        # 应用动作掩码
        if action_mask is not None:
            masked_gradients = policy_gradients * action_mask
        else:
            masked_gradients = policy_gradients
        
        # 计算投影梯度范数
        grad_norm = torch.norm(masked_gradients, dim=-1)
        
        # 利用性上界
        log_action_dim = math.log(action_dim if action_dim is not None else 
                                 action_mask.shape[-1])
        exploitability_bound = self.tau * log_action_dim + math.sqrt(2) * grad_norm
        
        return exploitability_bound, grad_norm
    
    def compute_policy_utility(
        self,
        policy_logits: torch.Tensor,
        ref_policy_logits: torch.Tensor,
        rewards: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        基于策略分布的效用计算（简化版）
        
        参数:
            policy_logits: 当前策略logits [B, T, A]
            ref_policy_logits: 参考策略logits [B, T, A]
            rewards: 奖励 [B, T, 1]
            states: 状态 [B, T, S]（可选）
            actions: 动作 [B, T]（可选）
            gamma: 折扣因子
            action_mask: 动作掩码 [B, T, A]（可选）
            
        返回:
            dict包含各种损失项
        """
        # 处理 2 维和 3 维输入
        if policy_logits.dim() == 2:
            # [B, A] -> [B, 1, A]
            policy_logits = policy_logits.unsqueeze(1)
            ref_policy_logits = ref_policy_logits.unsqueeze(1)
            if rewards.dim() == 2:
                rewards = rewards.unsqueeze(1)
            elif rewards.dim() == 1:
                rewards = rewards.unsqueeze(0).unsqueeze(-1)  # [B] -> [B, 1, 1]
            if states is not None and states.dim() == 2:
                states = states.unsqueeze(1)
            if actions is not None:
                if actions.dim() == 1:
                    # [B] -> [B, 1]
                    actions = actions.unsqueeze(1)
                elif actions.dim() == 2:
                    if actions.shape[-1] == 1:
                        # [B, 1] 索引格式
                        pass  # 已经是 [B, 1]，不需要修改
                    elif actions.shape[-1] > 1 and actions.shape[-1] < 100:
                        # 可能是 [B, num_agents] 或其他格式，转换为索引
                        # 假设是每个智能体的动作索引，取第一个或平均
                        if actions.shape[-1] <= 4:  # 可能是智能体数量
                            actions = actions[:, 0:1]  # 取第一个智能体的动作
                        else:
                            # 其他情况，尝试转换为索引
                            actions = actions[:, 0:1]  # 默认取第一个
                elif actions.dim() == 3:
                    # [B, T, A] one-hot，转换为索引
                    actions = torch.argmax(actions, dim=-1)
                    if actions.shape[1] == 1:
                        pass  # 已经是 [B, 1]
                    else:
                        actions = actions[:, 0:1]  # 取第一个时间步
            was_2d = True
        else:
            was_2d = False
        
        batch_size, seq_len, action_dim = policy_logits.shape
        
        # 计算策略分布
        policy_probs = F.softmax(policy_logits, dim=-1)
        ref_policy_probs = F.softmax(ref_policy_logits, dim=-1)
        
        # 应用动作掩码
        if action_mask is not None:
            policy_probs = policy_probs * action_mask
            ref_policy_probs = ref_policy_probs * action_mask
            # 重新归一化
            policy_probs = policy_probs / (policy_probs.sum(dim=-1, keepdim=True) + self.epsilon)
            ref_policy_probs = ref_policy_probs / (ref_policy_probs.sum(dim=-1, keepdim=True) + self.epsilon)
        
        if self.use_occupancy_measure and states is not None:
            # 使用占用度量
            occupancy = OccupancyMeasure.compute_discounted_occupancy(
                states, actions, policy_probs, gamma
            )
            ref_occupancy = OccupancyMeasure.compute_discounted_occupancy(
                states, actions, ref_policy_probs, gamma
            )
            
            # 扩展奖励维度以匹配占用度量
            # rewards形状: [B, T, 1]，需要扩展到 [B, T, S, A]
            # 确保reward_expanded是4维 [B, T, 1, 1]，然后expand到 [B, T, S, A]
            if rewards.dim() == 3:  # [B, T, 1]
                reward_expanded = rewards.unsqueeze(-1)  # [B, T, 1, 1]
            else:
                # 如果维度不对，先reshape
                reward_expanded = rewards.view(batch_size, seq_len, 1).unsqueeze(-1)  # [B, T, 1, 1]
            
            # 确保是4维
            while reward_expanded.dim() > 4:
                reward_expanded = reward_expanded.squeeze(-1)
            while reward_expanded.dim() < 4:
                reward_expanded = reward_expanded.unsqueeze(-1)
            
            reward_expanded = reward_expanded.expand(
                batch_size, seq_len, occupancy.shape[-2], occupancy.shape[-1]
            )
            
            # 计算效用
            utility, reward_term, kl_term = self.compute_utility(
                reward_expanded, occupancy, ref_occupancy, action_mask
            )
            
            # 平均效用
            utility = utility.mean()
            reward_term = reward_term.mean()
            kl_term = kl_term.mean()
            
        else:
            # 简化：直接使用策略分布的期望奖励和KL散度
            
            # 折扣因子
            discounts = torch.tensor([gamma**t for t in range(seq_len)], 
                                    device=rewards.device)
            discounts = discounts.view(1, seq_len, 1)
            
            # 期望奖励
            if actions is not None:
                # 使用动作的log概率
                log_probs = F.log_softmax(policy_logits, dim=-1)
                # 处理 actions 的维度
                if actions.dim() == 1:
                    # [B] -> [B, 1]
                    actions = actions.unsqueeze(1)
                elif actions.dim() == 2:
                    # [B, T] 或 [B, 1]，确保是 [B, T]
                    if actions.shape[1] != seq_len:
                        if actions.shape[1] == 1:
                            actions = actions.expand(-1, seq_len)
                        else:
                            # 如果维度不匹配，取最后一个时间步或扩展
                            if actions.shape[1] < seq_len:
                                # 扩展到最后
                                last_action = actions[:, -1:]
                                actions = torch.cat([actions, last_action.expand(-1, seq_len - actions.shape[1])], dim=1)
                            else:
                                # 截取
                                actions = actions[:, :seq_len]
                elif actions.dim() == 3:
                    # [B, T, A] one-hot，转换为索引
                    actions = torch.argmax(actions, dim=-1)
                
                # 确保 actions 是 long 类型且维度正确
                if actions.dtype != torch.long:
                    actions = actions.long()
                
                # 确保 actions 是 [B, T] 格式
                if actions.dim() == 2 and actions.shape[1] == seq_len:
                    action_log_probs = torch.gather(log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)  # [B, T]
                    # 确保维度匹配：discounts [1, T, 1], rewards [B, T, 1], action_log_probs [B, T]
                    action_log_probs_exp = action_log_probs.exp().unsqueeze(-1)  # [B, T, 1]
                    reward_term = torch.sum(discounts * rewards * action_log_probs_exp) / batch_size
                else:
                    # 如果维度仍然不匹配，使用策略分布的期望
                    policy_probs_mean = policy_probs.mean(dim=-1).unsqueeze(-1)  # [B, T, 1]
                    reward_term = torch.sum(discounts * rewards * policy_probs_mean) / batch_size
            else:
                # 使用策略分布的期望
                policy_probs_mean = policy_probs.mean(dim=-1).unsqueeze(-1)  # [B, T, 1]
                reward_term = torch.sum(discounts * rewards * policy_probs_mean) / batch_size
            
            # KL散度
            kl_term = self.kl_divergence(
                policy_probs.view(-1, action_dim),
                ref_policy_probs.view(-1, action_dim)
            ).mean()
            
            # 效用
            utility = reward_term - self.tau * kl_term
        
        # 计算策略梯度（用于利用性上界）
        # 检查policy_logits是否需要梯度
        if policy_logits.requires_grad and utility.requires_grad:
            try:
                policy_gradients = torch.autograd.grad(
                    utility,
                    policy_logits,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if policy_gradients is None:
                    # 如果梯度为None，创建零梯度
                    policy_gradients = torch.zeros_like(policy_logits)
            except RuntimeError:
                # 如果计算梯度失败，使用零梯度
                policy_gradients = torch.zeros_like(policy_logits)
        else:
            # 如果不需要梯度，使用零梯度
            policy_gradients = torch.zeros_like(policy_logits)
        
        # 计算利用性上界
        if was_2d:
            # 如果是 2 维输入，policy_gradients 现在是 [B, 1, A]，需要压缩
            policy_gradients_mean = policy_gradients.mean(dim=0).squeeze(0)  # [A]
            action_mask_mean = action_mask.mean(dim=0).squeeze(0) if action_mask is not None else None
        else:
            policy_gradients_mean = policy_gradients.mean(dim=0).mean(dim=0)  # 平均梯度
            action_mask_mean = action_mask.mean(dim=0).mean(dim=0) if action_mask is not None else None
        
        exploitability_bound, grad_norm = self.compute_exploitability(
            policy_gradients_mean,
            action_mask_mean,
            action_dim
        )
        
        # 计算总损失
        loss = -utility + self.alpha * exploitability_bound
        
        return {
            "loss": loss,
            "utility": utility,
            "reward_term": reward_term,
            "kl_term": kl_term,
            "exploitability_bound": exploitability_bound,
            "grad_norm": grad_norm,
        }
    
    def forward(
        self,
        policy_logits: torch.Tensor,
        ref_policy_logits: torch.Tensor,
        rewards: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        返回:
            dict包含所有损失项
        """
        return self.compute_policy_utility(
            policy_logits, ref_policy_logits, rewards, states, actions, gamma, action_mask
        )


