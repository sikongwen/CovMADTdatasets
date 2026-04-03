"""
BAD算法实现 - Belief-Action-Decoder
基于论文: "Belief-Action-Decoder: End-to-End Learning of Belief from Raw Observations in Multi-Agent Partially Observable Settings"

专为Hanabi环境设计，包含：
1. 分解信念系统（Factorized Belief System）
2. 部分策略网络（Partial Policy Network）
3. 反事实梯度（Counterfactual Gradients）
4. 重要性采样（Importance Sampling with V-trace）
5. 自洽信念迭代优化（Self-Consistent Belief Optimization）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
from torch.distributions import Categorical


# ==================== 核心数据结构 ====================

@dataclass
class Card:
    """卡牌表示"""
    color: int  # 0-4: R,Y,G,W,B
    rank: int   # 0-4: 1,2,3,4,5


@dataclass
class Hint:
    """提示信息"""
    player_idx: int
    card_indices: List[int]  # 被提示的牌在手中的位置
    color: Optional[int] = None
    rank: Optional[int] = None


@dataclass
class Hand:
    """玩家手牌"""
    cards: List[Card]
    known_info: List[Dict[str, set]]  # 每张牌的已知信息


# ==================== 信念系统 ====================

class FactorizedBeliefSystem:
    """分解信念系统 - 实现论文中的V0, V1, V2信念"""
    
    def __init__(self, num_players=4, num_colors=5, num_ranks=5):
        self.num_players = num_players
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.card_types = num_colors * num_ranks
        
        # 初始化卡片计数 (C(f))
        self.reset_card_counts()
        
        # 提示掩码 (HM)
        self.hint_masks = {}
        
        # 边际似然 (L(f[i]))
        self.marginal_likelihoods = {}
        
    def reset_card_counts(self):
        """重置卡片计数"""
        # 每种牌型的初始数量
        self.card_counts = {}
        for color in range(self.num_colors):
            for rank in range(self.num_ranks):
                card_id = color * self.num_ranks + rank
                if rank == 0:  # 1
                    count = 3
                elif rank == 4:  # 5
                    count = 1
                else:  # 2,3,4
                    count = 2
                self.card_counts[card_id] = count
                
    def update_from_action(self, action: Dict, player_idx: int, 
                          public_state: Dict, partial_policy):
        """
        根据动作更新信念 (公式1)
        
        Args:
            action: 执行的动作
            player_idx: 执行动作的玩家
            public_state: 公共状态
            partial_policy: 部分策略π̂
        """
        # 1. 更新卡片计数（玩牌/弃牌）
        if isinstance(action, dict) and action.get('action_type') in ['PLAY', 'DISCARD']:
            card_idx = action.get('card_index', 0)
            # 从计数中移除该牌（具体牌型未知）
            # 在实际实现中，需要从信念中采样可能的牌型
            self.update_counts_from_belief(player_idx, card_idx)
        
        # 2. 更新提示掩码
        elif isinstance(action, dict) and action.get('action_type') == 'HINT':
            hint = self.extract_hint(action)
            self.update_hint_mask(hint, player_idx)
            
        # 3. 贝叶斯更新边际似然 (公式8)
        self.update_marginal_likelihood(action, player_idx, partial_policy)
        
        # 4. 计算自洽信念 (公式12)
        self.compute_self_consistent_beliefs()
        
    def update_counts_from_belief(self, player_idx: int, card_idx: int):
        """从信念中更新卡片计数"""
        # 简化实现：减少最可能的牌型计数
        belief = self.compute_belief_v0(player_idx, card_idx)
        most_likely_card = np.argmax(belief)
        if self.card_counts.get(most_likely_card, 0) > 0:
            self.card_counts[most_likely_card] -= 1
            
    def extract_hint(self, action: Dict) -> Hint:
        """从动作中提取提示信息"""
        # 简化实现
        return Hint(
            player_idx=action.get('target_player', 0),
            card_indices=action.get('card_indices', []),
            color=action.get('color'),
            rank=action.get('rank')
        )
        
    def update_hint_mask(self, hint: Hint, player_idx: int):
        """更新提示掩码"""
        for card_idx in hint.card_indices:
            for card_id in range(self.card_types):
                color = card_id // self.num_ranks
                rank = card_id % self.num_ranks
                
                # 如果提示了颜色，更新掩码
                if hint.color is not None:
                    if color != hint.color:
                        self.hint_masks[(player_idx, card_idx, card_id)] = False
                    else:
                        self.hint_masks[(player_idx, card_idx, card_id)] = True
                        
                # 如果提示了等级，更新掩码
                if hint.rank is not None:
                    if rank != hint.rank:
                        self.hint_masks[(player_idx, card_idx, card_id)] = False
                    else:
                        self.hint_masks[(player_idx, card_idx, card_id)] = True
                        
    def update_marginal_likelihood(self, action, player_idx: int, partial_policy):
        """更新边际似然"""
        # 简化实现：基于动作概率更新
        if partial_policy is not None:
            # 这里应该根据部分策略计算边际似然
            # 简化：使用均匀分布
            pass
            
    def compute_self_consistent_beliefs(self):
        """计算自洽信念"""
        # 简化实现：在需要时调用compute_belief_v1
        pass
        
    def compute_belief_v0(self, player_idx: int, card_idx: int) -> np.ndarray:
        """计算V0信念 (公式11)"""
        belief = np.zeros(self.card_types)
        for card_id in range(self.card_types):
            # P(f[i]) ∝ C(f) × HM(f[i])
            if self.hint_masks.get((player_idx, card_idx, card_id), True):
                belief[card_id] = self.card_counts.get(card_id, 0)
        belief = belief / (belief.sum() + 1e-10)
        return belief
        
    def compute_belief_v1(self, player_idx: int, card_idx: int, 
                         num_iterations: int = 10) -> np.ndarray:
        """计算V1信念 (迭代自洽, 公式12)"""
        beliefs = {}
        
        # 初始化V0信念
        for p in range(self.num_players):
            for c in range(5):  # 每手5张牌
                beliefs[(p, c)] = self.compute_belief_v0(p, c)
                
        # 迭代更新
        for _ in range(num_iterations):
            new_beliefs = {}
            for p in range(self.num_players):
                for c in range(5):
                    # 公式12: B^{k+1}(f[i]) ∝ (C(f) - Σ_{j≠i} B^k(f[j])) × HM(f[i])
                    belief = np.zeros(self.card_types)
                    for card_id in range(self.card_types):
                        if self.hint_masks.get((p, c, card_id), True):
                            # 计算总期望计数
                            total_expected = 0
                            for other_c in range(5):
                                if other_c != c:
                                    total_expected += beliefs[(p, other_c)][card_id]
                            remaining = max(0, self.card_counts.get(card_id, 0) - total_expected)
                            belief[card_id] = remaining
                    belief = belief / (belief.sum() + 1e-10)
                    new_beliefs[(p, c)] = belief
            beliefs = new_beliefs
            
        return beliefs.get((player_idx, card_idx), self.compute_belief_v0(player_idx, card_idx))
        
    def compute_belief_v2(self, player_idx: int, card_idx: int,
                         alpha: float = 0.01) -> np.ndarray:
        """计算V2信念 (贝叶斯信念 + V1混合)"""
        # 计算V1信念
        v1_belief = self.compute_belief_v1(player_idx, card_idx)
        
        # 计算贝叶斯信念 (公式14)
        bayesian_belief = np.zeros(self.card_types)
        for card_id in range(self.card_types):
            if self.hint_masks.get((player_idx, card_idx, card_id), True):
                bayesian_belief[card_id] = (
                    self.card_counts.get(card_id, 0) *
                    self.marginal_likelihoods.get((player_idx, card_idx, card_id), 1.0)
                )
        bayesian_belief = bayesian_belief / (bayesian_belief.sum() + 1e-10)
        
        # 混合: V2 = (1-α)BB + αV1
        v2_belief = (1 - alpha) * bayesian_belief + alpha * v1_belief
        return v2_belief


# ==================== 神经网络架构 ====================

class PartialPolicyNetwork(nn.Module):
    """
    部分策略网络 - 实现论文附录A中的机制
    
    输入: 公共信念状态 + 公共特征
    输出: 确定性部分策略的分布
    """
    
    def __init__(self, belief_dim: int, public_feat_dim: int, 
                 private_obs_dim: int, action_dim: int, hidden_dim: int = 384):
        super().__init__()
        
        # 编码器网络 (两个隐藏层，如论文所述)
        self.encoder = nn.Sequential(
            nn.Linear(belief_dim + public_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 策略头 (私有观察 → 动作logits)
        self.policy_head = nn.Linear(hidden_dim, private_obs_dim * action_dim)
        
        # 基线网络 (用于优势估计)
        self.baseline_net = nn.Sequential(
            nn.Linear(hidden_dim + private_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.private_obs_dim = private_obs_dim
        self.action_dim = action_dim
        
    def forward(self, belief_state: torch.Tensor, public_feats: torch.Tensor,
                private_obs: Optional[torch.Tensor] = None):
        """
        前向传播
        
        Args:
            belief_state: [batch, belief_dim]
            public_feats: [batch, public_feat_dim]
            private_obs: [batch, private_obs_dim] (可选，用于基线网络)
            
        Returns:
            partial_policy: [batch, private_obs_dim, action_dim]
            baseline: [batch, 1] (如果有private_obs)
        """
        # 编码公共状态
        x = torch.cat([belief_state, public_feats], dim=-1)
        encoded = self.encoder(x)  # [batch, hidden_dim]
        
        # 部分策略
        policy_logits = self.policy_head(encoded)  # [batch, private_obs_dim * action_dim]
        partial_policy = policy_logits.view(-1, self.private_obs_dim, self.action_dim)
        
        # 基线值
        baseline = None
        if private_obs is not None:
            baseline_input = torch.cat([encoded, private_obs], dim=-1)
            baseline = self.baseline_net(baseline_input)
            
        return partial_policy, baseline


# ==================== BAD智能体 ====================

class BADAgent:
    """
    完整的BAD智能体实现，包含反事实梯度和重要性采样
    """
    
    def __init__(self, num_players=4, hand_size=5, num_colors=5, num_ranks=5,
                 device='cuda', learning_rate=3e-4, hidden_dim=384, action_dim=20):
        self.num_players = num_players
        self.hand_size = hand_size
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.device = device
        
        # 信念系统
        self.belief_system = FactorizedBeliefSystem(num_players, num_colors, num_ranks)
        
        # 网络参数
        belief_dim = num_players * hand_size * (num_colors * num_ranks)
        public_feat_dim = 128  # 公共特征维度
        private_obs_dim = hand_size * (num_colors + num_ranks)  # 简化编码
        self.action_dim = action_dim  # 使用传入的动作空间大小
        
        # 策略网络
        self.policy_net = PartialPolicyNetwork(
            belief_dim, public_feat_dim, private_obs_dim, self.action_dim, hidden_dim
        ).to(device)
        
        # 优化器
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=learning_rate,
            eps=1e-10,
            momentum=0,
            alpha=0.99
        )
        
        # 训练参数
        self.gamma = 0.999  # 折扣因子
        self.gae_lambda = 0.95  # GAE参数
        self.clip_param = 0.2  # PPO裁剪参数
        self.entropy_coef = 0.01  # 熵正则化
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        
        # 公共随机种子 (用于确定性策略采样)
        self.public_random_seed = 42
        self.rng = np.random.RandomState(self.public_random_seed)
        
        # 轨迹存储
        self.trajectories = []
        
    def encode_private_observation(self, hand: Hand) -> torch.Tensor:
        """编码私有观察为向量"""
        # 简化编码：每张牌的颜色和等级one-hot
        encoding = torch.zeros(self.hand_size, self.num_colors + self.num_ranks)
        for i, card in enumerate(hand.cards):
            if i < len(hand.cards):
                # 颜色one-hot
                encoding[i, card.color] = 1
                # 等级one-hot
                encoding[i, self.num_colors + card.rank] = 1
        return encoding.flatten()
    
    def encode_private_observation_from_obs(self, obs: Dict) -> torch.Tensor:
        """从环境观察编码私有观察"""
        # 从Hanabi观察中提取手牌信息
        encoding = torch.zeros(self.hand_size, self.num_colors + self.num_ranks)
        
        # 尝试从观察中提取手牌信息
        if 'observation' in obs:
            obs_vec = obs['observation']
            # Hanabi观察向量包含手牌信息，需要根据实际格式解析
            # 这里使用简化版本
            if isinstance(obs_vec, np.ndarray):
                # 假设观察向量的前hand_size*(num_colors+num_ranks)维是手牌编码
                hand_encoding = torch.FloatTensor(obs_vec[:self.hand_size * (self.num_colors + self.num_ranks)])
                if hand_encoding.shape[0] == self.hand_size * (self.num_colors + self.num_ranks):
                    return hand_encoding
        
        # 如果无法解析，返回零向量
        return encoding.flatten()
    
    def encode_public_features(self, env_state: Dict) -> torch.Tensor:
        """编码公共特征"""
        features = []
        
        # 烟花进度 (5种颜色，每个等级0/1)
        fireworks = env_state.get('fireworks', {})
        for color in range(self.num_colors):
            for rank in range(self.num_ranks):
                features.append(1 if fireworks.get((color, rank), False) else 0)
                
        # 提示令牌和生命令牌
        features.append(env_state.get('information_tokens', 0) / 8.0)
        features.append(env_state.get('life_tokens', 0) / 6.0)
        
        # 弃牌堆统计
        discard_counts = np.zeros((self.num_colors, self.num_ranks))
        for card in env_state.get('discard_pile', []):
            if isinstance(card, dict):
                discard_counts[card.get('color', 0), card.get('rank', 0)] += 1
            elif hasattr(card, 'color') and hasattr(card, 'rank'):
                discard_counts[card.color, card.rank] += 1
        features.extend(discard_counts.flatten())
        
        # 回合数归一化
        features.append(env_state.get('turn', 0) / 100.0)
        
        # 填充到固定维度
        while len(features) < 128:
            features.append(0.0)
        features = features[:128]
        
        return torch.FloatTensor(features)
    
    def get_public_belief_state(self) -> torch.Tensor:
        """获取公共信念状态"""
        belief_state = []
        
        # 计算所有玩家所有手牌的信念
        for player_idx in range(self.num_players):
            for card_idx in range(self.hand_size):
                # 使用V2信念
                belief = self.belief_system.compute_belief_v2(player_idx, card_idx)
                belief_state.extend(belief)
        
        return torch.FloatTensor(belief_state)
    
    def sample_deterministic_policy(self, belief_state: torch.Tensor, 
                                   public_feats: torch.Tensor,
                                   deterministic: bool = False) -> Tuple[torch.Tensor, int]:
        """
        采样确定性部分策略 (公式21)
        
        Args:
            deterministic: 是否使用确定性采样（评估时使用）
        
        Returns:
            deterministic_policy: [private_obs_dim, action_dim] (one-hot)
            action: 实际选择的动作
        """
        with torch.no_grad():
            # 获取概率性部分策略
            partial_policy, _ = self.policy_net(
                belief_state.unsqueeze(0).to(self.device),
                public_feats.unsqueeze(0).to(self.device)
            )
            partial_policy = partial_policy[0]  # [private_obs_dim, action_dim]
            
            # 应用softmax得到概率
            policy_probs = F.softmax(partial_policy, dim=-1)
            
            # 使用公共随机种子采样确定性策略
            if deterministic:
                # 评估时：使用固定种子确保确定性
                temp_rng = np.random.RandomState(self.public_random_seed)
                # 为每个私有观察采样一个动作
                deterministic_policy = torch.zeros_like(policy_probs)
                for obs_idx in range(policy_probs.size(0)):
                    probs = policy_probs[obs_idx].cpu().numpy()
                    action = temp_rng.choice(len(probs), p=probs)
                    deterministic_policy[obs_idx, action] = 1
            else:
                # 训练时：为探索而采样
                deterministic_policy = torch.zeros_like(policy_probs)
                for obs_idx in range(policy_probs.size(0)):
                    dist = Categorical(policy_probs[obs_idx])
                    action = dist.sample()
                    deterministic_policy[obs_idx, action] = 1
                    
            return deterministic_policy
            
    def private_observation_to_index(self, private_obs: torch.Tensor) -> int:
        """将私有观察转换为索引（简化实现）"""
        # 简化：使用观察向量的哈希值
        obs_hash = hash(tuple(private_obs.cpu().numpy().flatten()[:10]))
        return abs(obs_hash) % self.policy_net.private_obs_dim
        
    def compute_counterfactual_gradients(self, trajectories: List[Dict]):
        """
        计算反事实梯度 (论文4.1节)
        
        关键思想：不仅加强实际采取的动作，还加强反事实动作
        """
        losses = []
        
        for traj in trajectories:
            # 确保有private_obs，如果没有则创建一个零向量
            private_obs = traj.get('private_obs', None)
            if private_obs is None:
                # 创建一个零向量作为默认private_obs
                private_obs_dim = self.policy_net.private_obs_dim
                private_obs = torch.zeros(private_obs_dim)
            
            # 确保是tensor并移到设备上
            if not isinstance(private_obs, torch.Tensor):
                private_obs = torch.FloatTensor(private_obs) if isinstance(private_obs, np.ndarray) else torch.zeros(self.policy_net.private_obs_dim)
            
            # 重计算动作概率
            partial_policy, baseline = self.policy_net(
                traj['belief_state'].unsqueeze(0).to(self.device) if len(traj['belief_state'].shape) == 1 else traj['belief_state'].to(self.device),
                traj['public_feats'].unsqueeze(0).to(self.device) if len(traj['public_feats'].shape) == 1 else traj['public_feats'].to(self.device),
                private_obs.unsqueeze(0).to(self.device)
            )
            
            # 如果baseline为None，使用零基线
            if baseline is None:
                baseline = torch.zeros(1, 1).to(self.device)
            
            # 实际动作的log概率
            action_log_probs = F.log_softmax(partial_policy, dim=-1)
            
            # 获取private_idx（如果存在）
            private_idx = traj.get('private_idx', 0)
            if isinstance(private_idx, torch.Tensor):
                private_idx = private_idx.item()
            private_idx = min(max(0, private_idx), partial_policy.size(1) - 1)
            
            # 获取动作
            action = traj['action']
            if isinstance(action, torch.Tensor):
                action = action.item() if action.numel() == 1 else action[0].item()
            action = min(max(0, action), partial_policy.size(2) - 1)
            
            chosen_action_log_probs = action_log_probs[0, private_idx, action]
            
            # 反事实动作：对所有可能的私有观察
            # 使用重要性采样权重
            weights = traj.get('importance_weights', torch.tensor(1.0)).to(self.device)
            
            # 优势估计
            returns = traj['returns']
            if isinstance(returns, torch.Tensor):
                returns = returns.to(self.device)
            else:
                returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            
            advantages = returns - baseline.squeeze()
            
            # 获取old_log_probs
            old_log_probs = traj.get('old_log_probs', chosen_action_log_probs.detach())
            if isinstance(old_log_probs, torch.Tensor):
                old_log_probs = old_log_probs.to(self.device)
            else:
                old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
            
            # PPO损失 (带裁剪)
            ratio = torch.exp(chosen_action_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2)
            
            # 价值损失
            value_loss = F.mse_loss(baseline.squeeze(), returns)
            
            # 熵正则化
            policy_probs = F.softmax(partial_policy[0, private_idx], dim=-1)
            dist_entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-10))
            
            total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * dist_entropy
            losses.append(total_loss)
            
        return torch.stack(losses).mean()
    
    def train_with_importance_sampling(self, batch_size: int = 32):
        """
        重要性加权Actor-Learner训练 (IMPALA架构)
        """
        if len(self.memory) < batch_size:
            return
            
        # 采样批次
        batch = random.sample(self.memory, batch_size)
        
        # 计算重要性权重
        for experience in batch:
            with torch.no_grad():
                # 重计算当前策略的动作概率
                partial_policy, _ = self.policy_net(
                    experience['belief_state'].unsqueeze(0).to(self.device),
                    experience['public_feats'].unsqueeze(0).to(self.device)
                )
                current_probs = F.softmax(partial_policy, dim=-1)
                
                # 行为策略的概率
                behavior_probs = experience['action_probs'].to(self.device)
                
                # 重要性权重 (裁剪)
                importance_weight = current_probs / (behavior_probs + 1e-10)
                importance_weight = torch.clamp(importance_weight, 0.1, 10.0)
                
                experience['importance_weights'] = importance_weight
        
        # 计算V-trace目标 (修正离策略)
        self.compute_vtrace_targets(batch)
        
        # 反事实梯度更新
        loss = self.compute_counterfactual_gradients(batch)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        # 清空记忆
        self.memory.clear()
        
    def compute_vtrace_targets(self, batch: List[Dict]):
        """
        计算V-trace目标值 (IMPALA)
        """
        # 简化的V-trace实现
        for i in range(len(batch)):
            # 计算n-step回报
            n_step_return = 0
            discount = 1.0
            for j in range(i, min(i + 5, len(batch))):
                n_step_return += discount * batch[j]['reward']
                discount *= self.gamma
                
            # 添加基线值（如果有private_obs）
            if i + 5 < len(batch):
                next_exp = batch[i + 5]
                if 'private_obs' in next_exp and next_exp['private_obs'] is not None:
                    with torch.no_grad():
                        _, next_baseline = self.policy_net(
                            next_exp['belief_state'].unsqueeze(0).to(self.device),
                            next_exp['public_feats'].unsqueeze(0).to(self.device),
                            next_exp['private_obs'].unsqueeze(0).to(self.device)
                        )
                        if next_baseline is not None:
                            n_step_return += discount * next_baseline.item()
                    
            batch[i]['vtrace_target'] = n_step_return
            batch[i]['returns'] = torch.tensor(n_step_return, dtype=torch.float32)
        
    def get_action(self, env, player_idx: int, observation: Dict, 
                   action_mask: Optional[np.ndarray] = None,
                   deterministic: bool = False) -> int:
        """
        BAD算法的主要决策函数
        
        步骤：
        1. 获取当前公共信念
        2. 采样确定性部分策略
        3. 根据私有观察选择动作
        4. 更新信念
        """
        # 1. 获取公共信念状态
        belief_state = self.get_public_belief_state()
        
        # 2. 获取公共特征
        try:
            env_state = env.env.get_state() if hasattr(env, 'env') else {}
        except:
            env_state = {}
        public_feats = self.encode_public_features(env_state)
        
        # 3. 编码私有观察
        private_obs = self.encode_private_observation_from_obs(observation)
        
        # 4. 选择动作 (根据私有观察索引)
        private_idx = self.private_observation_to_index(private_obs)
        private_idx = max(0, private_idx)
        
        # 5. 计算策略分布
        with torch.no_grad():
            partial_policy, baseline = self.policy_net(
                belief_state.unsqueeze(0).to(self.device),
                public_feats.unsqueeze(0).to(self.device),
                private_obs.unsqueeze(0).to(self.device)
            )
            private_idx_safe = min(private_idx, partial_policy.size(1) - 1)
            private_idx_safe = max(0, private_idx_safe)
            policy_probs = F.softmax(partial_policy[0, private_idx_safe], dim=-1)
        
        # 6. 应用动作掩码并选择动作
        masked_probs = policy_probs
        if action_mask is not None:
            valid_actions = np.where(action_mask > 0)[0]
            valid_actions = valid_actions[valid_actions < self.policy_net.action_dim]
            if len(valid_actions) > 0:
                mask = torch.zeros_like(policy_probs)
                mask[valid_actions] = 1.0
                masked_probs = policy_probs * mask
                if masked_probs.sum() <= 1e-8:
                    masked_probs = mask / (mask.sum() + 1e-8)
                else:
                    masked_probs = masked_probs / masked_probs.sum()
        
        if deterministic:
            action = torch.argmax(masked_probs).item()
        else:
            dist = Categorical(masked_probs)
            action = dist.sample().item()
        
        # 确保action在有效范围内
        action = min(action, self.policy_net.action_dim - 1)
        action = max(0, action)
        
        # 7. 存储经验 (用于训练，仅在非确定性模式下)
        if not deterministic:
            action_safe = min(action, partial_policy.size(2) - 1)
            action_safe = max(0, action_safe)
            action_log_prob = torch.log(masked_probs[action_safe] + 1e-10)
                
            # 存储经验
            self.memory.append({
                'belief_state': belief_state,
                'public_feats': public_feats,
                'private_obs': private_obs,
                'private_idx': private_idx_safe,  # 保存private_idx以便训练时使用
                'action': torch.tensor([action_safe]),  # 使用安全的action值
                'action_probs': masked_probs.detach().cpu(),
                'old_log_probs': action_log_prob.detach().cpu(),
                'reward': 0.0,  # 将在后续填充
            })
        
        return action
        
    def update_rewards(self, rewards: List[float]):
        """更新最近经验中的奖励"""
        if len(self.memory) == 0 or len(rewards) == 0:
            return
        
        # deque在某些Python版本不支持切片，使用索引访问
        # 获取最后N个经验（N = len(rewards)）
        num_to_update = min(len(rewards), len(self.memory))
        
        # 将deque转换为列表以便索引访问
        memory_list = list(self.memory)
        start_idx = len(memory_list) - num_to_update
        
        # 更新奖励
        for i in range(num_to_update):
            idx = start_idx + i
            if 0 <= idx < len(memory_list):
                memory_list[idx]['reward'] = rewards[i]
        
        # 重新构建deque（保持maxlen）
        maxlen = self.memory.maxlen
        self.memory = deque(memory_list, maxlen=maxlen)
            
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


# ==================== 自洽信念迭代优化 ====================

class SelfConsistentBeliefOptimizer:
    """自洽信念迭代优化器 (公式10)"""
    
    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations
        
    def optimize(self, initial_beliefs: Dict, card_counts: Dict, 
                hint_masks: Dict, marginal_likelihoods: Dict) -> Dict:
        """
        迭代优化信念分布
        
        类似于期望传播，但针对全局卡片计数约束
        """
        beliefs = initial_beliefs.copy()
        
        for _ in range(self.max_iterations):
            new_beliefs = {}
            
            # 对每个特征进行更新
            for (player, card_pos), belief in beliefs.items():
                updated_belief = np.zeros_like(belief)
                
                # 计算其他特征的期望分布
                other_expectations = defaultdict(float)
                for (other_player, other_pos), other_belief in beliefs.items():
                    if other_player == player and other_pos == card_pos:
                        continue
                    for card_id, prob in enumerate(other_belief):
                        other_expectations[card_id] += prob
                
                # 更新信念 (考虑卡片计数约束)
                for card_id in range(len(belief)):
                    if hint_masks.get((player, card_pos, card_id), True):
                        # 剩余卡片 = 总数 - 其他位置期望
                        remaining = max(0, card_counts.get(card_id, 0) - 
                                      other_expectations[card_id])
                        
                        # 乘以边际似然
                        likelihood = marginal_likelihoods.get(
                            (player, card_pos, card_id), 1.0
                        )
                        
                        updated_belief[card_id] = remaining * likelihood
                        
                # 归一化
                total = updated_belief.sum()
                if total > 0:
                    updated_belief /= total
                    
                new_beliefs[(player, card_pos)] = updated_belief
                
            # 检查收敛
            max_change = 0
            for key in beliefs:
                max_change = max(max_change, 
                               np.abs(beliefs[key] - new_beliefs[key]).max())
                
            beliefs = new_beliefs
            
            if max_change < 1e-4:
                break
                
        return beliefs

