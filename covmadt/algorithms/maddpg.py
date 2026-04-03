"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) 算法实现
用于收集高质量的离线数据
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
import copy


class Actor(nn.Module):
    """Actor网络（策略网络）"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        discrete: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
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
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.net(state)
        if self.discrete:
            return F.softmax(self.action_head(x), dim=-1)
        else:
            return self.tanh(self.action_head(x))
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0, device: str = "cuda", action_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """选择动作（带epsilon探索和动作掩码）"""
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs = self.forward(state_tensor)
            
            # 应用动作掩码（如果提供）
            if action_mask is not None:
                # 将掩码转换为tensor并应用到概率分布
                mask_tensor = torch.FloatTensor(action_mask).to(device)
                probs = probs * mask_tensor
                # 重新归一化
                probs = probs / (probs.sum() + 1e-8)
            
            if random.random() < epsilon:
                # epsilon-greedy探索（只从有效动作中选择）
                if action_mask is not None:
                    valid_actions = np.where(action_mask > 0)[0]
                    if len(valid_actions) > 0:
                        action = int(np.random.choice(valid_actions))
                    else:
                        # 如果没有有效动作，选择动作0（不应该发生）
                        action = 0
                else:
                    action = random.randint(0, self.action_dim - 1)
            else:
                # 根据概率分布采样
                if action_mask is not None and probs.sum().item() < 1e-6:
                    # 如果所有动作都被掩码，从有效动作中随机选择
                    valid_actions = np.where(action_mask > 0)[0]
                    if len(valid_actions) > 0:
                        action = int(np.random.choice(valid_actions))
                    else:
                        action = 0
                else:
                    action = torch.multinomial(probs, 1).item()
                    # 验证动作是否有效
                    if action_mask is not None and action_mask[action] == 0:
                        # 如果选择的动作无效，从有效动作中随机选择
                        valid_actions = np.where(action_mask > 0)[0]
                        if len(valid_actions) > 0:
                            action = int(np.random.choice(valid_actions))
                        else:
                            action = 0
            
            return action


class Critic(nn.Module):
    """Critic网络（Q值网络）"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
        discrete: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.discrete = discrete
        
        # 输入：所有智能体的状态和动作
        input_dim = state_dim * n_agents + action_dim * n_agents
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            states: [B, n_agents, state_dim] 或 [B, state_dim * n_agents]
            actions: [B, n_agents, action_dim] 或 [B, action_dim * n_agents]
        """
        # 展平状态和动作
        if states.dim() == 3:
            states_flat = states.view(states.shape[0], -1)  # [B, n_agents * state_dim]
        else:
            states_flat = states
        
        if actions.dim() == 3:
            actions_flat = actions.view(actions.shape[0], -1)  # [B, n_agents * action_dim]
        else:
            actions_flat = actions
        
        # 拼接状态和动作
        x = torch.cat([states_flat, actions_flat], dim=-1)
        return self.net(x)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """采样批次"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )
    
    def __len__(self):
        return len(self.buffer)


class MADDPG:
    """MADDPG算法"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        single_agent_action_dim: int,
        hidden_dim: int = 128,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        discrete: bool = True,
        device: str = "cuda",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.single_agent_action_dim = single_agent_action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.discrete = discrete
        
        # 为每个智能体创建Actor和Critic
        self.actors = []
        self.actors_target = []
        self.actors_optimizer = []
        
        self.critics = []
        self.critics_target = []
        self.critics_optimizer = []
        
        # 单个智能体的状态维度
        # 对于Hanabi等环境，如果state_dim不能被n_agents整除，说明state_dim已经是单个智能体的维度
        # 否则，尝试分割（但需要处理不能整除的情况）
        if state_dim % n_agents == 0:
            single_agent_state_dim = state_dim // n_agents
        else:
            # 不能整除，说明state_dim可能是单个智能体的维度（如Hanabi的1257）
            # 或者需要特殊处理
            single_agent_state_dim = state_dim
            # 只在第一次创建时显示提示（减少重复输出）
            if not hasattr(MADDPG, '_warned_state_dim'):
                print(f"ℹ️  提示: state_dim ({state_dim}) 不能被 n_agents ({n_agents}) 整除")
                print(f"  假设 state_dim 是单个智能体的维度（适用于Hanabi等轮流行动的环境）")
                MADDPG._warned_state_dim = True
        
        for i in range(n_agents):
            # Actor
            actor = Actor(single_agent_state_dim, single_agent_action_dim, hidden_dim, discrete).to(device)
            actor_target = copy.deepcopy(actor)
            actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
            
            self.actors.append(actor)
            self.actors_target.append(actor_target)
            self.actors_optimizer.append(actor_optimizer)
            
            # Critic
            critic = Critic(single_agent_state_dim, single_agent_action_dim, n_agents, hidden_dim, discrete).to(device)
            critic_target = copy.deepcopy(critic)
            critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)
            
            self.critics.append(critic)
            self.critics_target.append(critic_target)
            self.critics_optimizer.append(critic_optimizer)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # 训练计数器
        self.train_step = 0
    
    def select_actions(
        self,
        states: np.ndarray,
        epsilon: float = 0.0,
        add_noise: bool = False,
        action_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        为所有智能体选择动作
        
        参数:
            states: [state_dim] 或 [n_agents, single_state_dim]
            epsilon: 探索率
            add_noise: 是否添加噪声（连续动作）
        
        返回:
            actions: [n_agents] 动作索引数组
        """
        # 处理状态维度
        if states.ndim == 1:
            # 展平状态
            if len(states) == self.state_dim:
                # 检查是否可以整除
                if self.state_dim % self.n_agents == 0:
                    # 可以整除，分割给每个智能体
                    single_state_dim = self.state_dim // self.n_agents
                    states_per_agent = states.reshape(self.n_agents, single_state_dim)
                else:
                    # 不能整除，说明state_dim是单个智能体的维度（如Hanabi）
                    # 对于Hanabi，每个时间步只有一个智能体行动，所以所有智能体使用相同的观察
                    states_per_agent = np.tile(states, (self.n_agents, 1))
            else:
                # 单个智能体状态，复制给所有智能体
                states_per_agent = np.tile(states, (self.n_agents, 1))
        else:
            states_per_agent = states
        
        actions = []
        for i, actor in enumerate(self.actors):
            state_i = states_per_agent[i] if states_per_agent.shape[0] > i else states_per_agent[0]
            # 对于Hanabi，每个时间步只有一个智能体行动，使用相同的action_mask
            action = actor.select_action(state_i, epsilon=epsilon, device=self.device, action_mask=action_mask)
            actions.append(action)
        
        return np.array(actions)
    
    def push_transition(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """添加经验到回放缓冲区"""
        self.replay_buffer.push(state, actions, reward, next_state, done)
    
    def update(self, batch_size: int = 64):
        """更新网络"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 转换为tensor
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # 处理状态维度
        # 检查state_dim是否能被n_agents整除
        if self.state_dim % self.n_agents == 0:
            # 可以整除，说明state_dim是所有智能体的合并维度
            single_state_dim = self.state_dim // self.n_agents
            need_split = True
        else:
            # 不能整除，说明state_dim是单个智能体的维度（如Hanabi）
            single_state_dim = self.state_dim
            need_split = False
        
        # 为每个智能体更新
        for i in range(self.n_agents):
            # 提取智能体i的状态
            if states_t.dim() == 2:
                if need_split:
                    # [B, state_dim] -> [B, n_agents, single_state_dim]
                    states_reshaped = states_t.view(batch_size, self.n_agents, single_state_dim)
                    next_states_reshaped = next_states_t.view(batch_size, self.n_agents, single_state_dim)
                    states_i = states_reshaped[:, i, :]  # [B, single_state_dim]
                    next_states_i = next_states_reshaped[:, i, :]  # [B, single_state_dim]
                else:
                    # state_dim是单个智能体的维度，所有智能体使用相同的状态（Hanabi的情况）
                    states_i = states_t  # [B, single_state_dim]
                    next_states_i = next_states_t  # [B, single_state_dim]
                    # 为了兼容后续代码，创建虚拟的reshaped变量
                    states_reshaped = states_t.unsqueeze(1).repeat(1, self.n_agents, 1)  # [B, n_agents, single_state_dim]
                    next_states_reshaped = next_states_t.unsqueeze(1).repeat(1, self.n_agents, 1)  # [B, n_agents, single_state_dim]
            else:
                states_i = states_t
                next_states_i = next_states_t
                # 为了兼容后续代码，创建虚拟的reshaped变量
                if need_split:
                    states_reshaped = states_t.view(batch_size, self.n_agents, single_state_dim)
                    next_states_reshaped = next_states_t.view(batch_size, self.n_agents, single_state_dim)
                else:
                    states_reshaped = states_t.unsqueeze(1).repeat(1, self.n_agents, 1)
                    next_states_reshaped = next_states_t.unsqueeze(1).repeat(1, self.n_agents, 1)
            
            # 获取下一个动作（使用target actor）
            with torch.no_grad():
                next_actions_probs = self.actors_target[i](next_states_i)  # [B, action_dim]
                # 对于离散动作，使用概率分布的期望或采样
                if self.discrete:
                    # 使用概率加权
                    next_actions_onehot = next_actions_probs  # [B, action_dim]
                else:
                    next_actions_onehot = next_actions_probs
            
            # 获取当前动作的one-hot编码
            if self.discrete:
                actions_onehot = F.one_hot(actions_t[:, i].long(), self.single_agent_action_dim).float()
            else:
                actions_onehot = actions_t[:, i:i+1]
            
            # 构建所有智能体的状态和动作（用于critic）
            if need_split:
                all_states = states_reshaped.view(batch_size, -1)  # [B, n_agents * single_state_dim]
            else:
                # 对于Hanabi，所有智能体使用相同的状态，需要复制
                all_states = states_i.repeat(1, self.n_agents)  # [B, n_agents * single_state_dim]
            
            all_actions = actions_onehot.unsqueeze(1)  # [B, 1, action_dim]
            # 需要为所有智能体构建动作
            all_actions_list = []
            for j in range(self.n_agents):
                if j == i:
                    all_actions_list.append(actions_onehot)
                else:
                    # 使用其他智能体的实际动作
                    if self.discrete:
                        other_action_onehot = F.one_hot(actions_t[:, j].long(), self.single_agent_action_dim).float()
                    else:
                        other_action_onehot = actions_t[:, j:j+1]
                    all_actions_list.append(other_action_onehot)
            all_actions = torch.cat(all_actions_list, dim=1)  # [B, n_agents, action_dim]
            all_actions = all_actions.view(batch_size, -1)  # [B, n_agents * action_dim]
            
            # 更新Critic
            current_q = self.critics[i](all_states, all_actions).squeeze()  # [B]
            
            # 构建下一个状态的动作
            if need_split:
                all_next_states = next_states_reshaped.view(batch_size, -1)
            else:
                # 对于Hanabi，所有智能体使用相同的下一个状态
                all_next_states = next_states_i.repeat(1, self.n_agents)  # [B, n_agents * single_state_dim]
            
            all_next_actions_list = []
            for j in range(self.n_agents):
                if j == i:
                    all_next_actions_list.append(next_actions_onehot)
                else:
                    # 对于Hanabi，所有智能体使用相同的状态
                    next_state_j = next_states_i if not need_split else next_states_reshaped[:, j, :]
                    next_actions_probs_j = self.actors_target[j](next_state_j)
                    if self.discrete:
                        all_next_actions_list.append(next_actions_probs_j)
                    else:
                        all_next_actions_list.append(next_actions_probs_j)
            all_next_actions = torch.cat(all_next_actions_list, dim=1)  # [B, n_agents, action_dim]
            all_next_actions = all_next_actions.view(batch_size, -1)  # [B, n_agents * action_dim]
            
            next_q = self.critics_target[i](all_next_states, all_next_actions).squeeze()  # [B]
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q
            
            critic_loss = F.mse_loss(current_q, target_q.detach())
            
            # 更新Critic
            self.critics_optimizer[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 1.0)
            self.critics_optimizer[i].step()
            
            # 更新Actor
            # 使用当前策略选择动作
            actions_probs = self.actors[i](states_i)  # [B, action_dim]
            
            # 构建所有智能体的动作（其他智能体使用当前策略）
            all_actions_policy_list = []
            for j in range(self.n_agents):
                if j == i:
                    all_actions_policy_list.append(actions_probs)
                else:
                    actions_probs_j = self.actors[j](states_reshaped[:, j, :])
                    all_actions_policy_list.append(actions_probs_j)
            all_actions_policy = torch.cat(all_actions_policy_list, dim=1)  # [B, n_agents, action_dim]
            all_actions_policy = all_actions_policy.view(batch_size, -1)  # [B, n_agents * action_dim]
            
            actor_loss = -self.critics[i](all_states, all_actions_policy).mean()
            
            # 更新Actor
            self.actors_optimizer[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actors_optimizer[i].step()
            
            # 软更新target网络
            for param, target_param in zip(self.actors[i].parameters(), self.actors_target[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critics[i].parameters(), self.critics_target[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.train_step += 1
    
    def save(self, path: str):
        """保存模型"""
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'actors_target': [actor.state_dict() for actor in self.actors_target],
            'critics_target': [critic.state_dict() for critic in self.critics_target],
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(checkpoint['critics'][i])
        for i, actor_target in enumerate(self.actors_target):
            actor_target.load_state_dict(checkpoint['actors_target'][i])
        for i, critic_target in enumerate(self.critics_target):
            critic_target.load_state_dict(checkpoint['critics_target'][i])

