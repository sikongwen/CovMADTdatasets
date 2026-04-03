"""
优化版MADDPG - 支持观察降维以处理大观察空间
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


class ObservationEncoder(nn.Module):
    """观察编码器：将大观察空间降维"""
    
    def __init__(self, input_dim: int, encoded_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.encoded_dim = encoded_dim
        
        # 使用多层MLP进行降维，优化内存使用
        if input_dim > 10000:
            # 对于超大观察空间，使用更激进的降维和更小的中间层
            # 直接降维到较小尺寸，避免大中间层
            hidden1 = min(512, encoded_dim * 2)  # 更小的中间层
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden1, encoded_dim),
            )
        else:
            # 对于较小的观察空间，使用简单降维
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoded_dim),
                nn.ReLU(),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Actor(nn.Module):
    """Actor网络（策略网络）- 优化版"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        discrete: bool = True,
        use_encoder: bool = False,
        encoded_dim: int = 512,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        
        # 如果观察维度很大，使用编码器
        if use_encoder and state_dim > 1000:
            self.encoder = ObservationEncoder(state_dim, encoded_dim)
            effective_state_dim = encoded_dim
        else:
            self.encoder = None
            effective_state_dim = state_dim
        
        # 对于大观察空间，使用更小的网络结构
        if effective_state_dim > 500:
            # 使用更小的网络以节省内存
            self.net = nn.Sequential(
                nn.Linear(effective_state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(effective_state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
            )
        
        if discrete:
            self.action_head = nn.Linear(hidden_dim // 2, action_dim)
        else:
            self.action_head = nn.Linear(hidden_dim // 2, action_dim)
            self.tanh = nn.Tanh()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.encoder is not None:
            state = self.encoder(state)
        x = self.net(state)
        if self.discrete:
            return F.softmax(self.action_head(x), dim=-1)
        else:
            return self.tanh(self.action_head(x))
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0, device: str = "cuda") -> np.ndarray:
        """选择动作（带epsilon探索）"""
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs = self.forward(state_tensor)
            
            if random.random() < epsilon:
                action = random.randint(0, self.action_dim - 1)
            else:
                action = torch.multinomial(probs, 1).item()
            
            return action


class Critic(nn.Module):
    """Critic网络（Q值网络）- 优化版，支持大智能体数量"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
        discrete: bool = True,
        use_encoder: bool = False,
        encoded_dim: int = 512,
        use_attention: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.discrete = discrete
        self.use_attention = use_attention and n_agents > 20  # 智能体多时使用注意力
        
        # 如果观察维度很大，使用编码器
        if use_encoder and state_dim > 1000:
            self.encoder = ObservationEncoder(state_dim, encoded_dim)
            effective_state_dim = encoded_dim
        else:
            self.encoder = None
            effective_state_dim = state_dim
        
        # 对于大量智能体，使用注意力机制或更激进的降维
        if n_agents > 20:
            # 使用注意力机制或状态聚合
            if self.use_attention:
                # 使用注意力聚合多智能体状态
                self.state_attention = nn.MultiheadAttention(
                    embed_dim=effective_state_dim,
                    num_heads=4,
                    batch_first=True
                )
                # 聚合后的状态维度
                aggregated_state_dim = effective_state_dim
            else:
                # 使用平均池化聚合状态
                aggregated_state_dim = effective_state_dim
                self.state_pool = nn.AdaptiveAvgPool1d(1)
            
            # 输入维度：聚合后的状态 + 动作
            input_dim = aggregated_state_dim + action_dim * n_agents
        else:
            # 少量智能体，使用完整状态
            input_dim = effective_state_dim * n_agents + action_dim * n_agents
        
        # 使用更小的网络结构以节省内存
        if input_dim > 10000:
            # 超大输入，使用更激进的降维和更小的网络
            first_hidden = min(256, hidden_dim * 2)  # 更小的第一层
            self.net = nn.Sequential(
                nn.Linear(input_dim, first_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(first_hidden, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            # 简化网络结构
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
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
        batch_size = states.shape[0] if states.dim() >= 2 else 1
        
        # 处理状态
        if states.dim() == 2:
            # [B, state_dim * n_agents] -> [B, n_agents, state_dim]
            single_state_dim = self.state_dim
            states_reshaped = states.view(batch_size, self.n_agents, single_state_dim)
        else:
            states_reshaped = states
            batch_size = states_reshaped.shape[0]
        
        # 编码状态
        if self.encoder is not None:
            states_flat = states_reshaped.view(-1, self.state_dim)  # [B*n_agents, state_dim]
            states_encoded = self.encoder(states_flat)  # [B*n_agents, encoded_dim]
            states_reshaped = states_encoded.view(batch_size, self.n_agents, -1)  # [B, n_agents, encoded_dim]
        
        # 处理大量智能体：使用聚合策略
        if self.n_agents > 20:
            if self.use_attention:
                # 使用注意力机制聚合状态
                # states_reshaped: [B, n_agents, encoded_dim]
                aggregated_state, _ = self.state_attention(
                    states_reshaped, states_reshaped, states_reshaped
                )  # [B, n_agents, encoded_dim]
                # 取平均作为全局状态
                aggregated_state = aggregated_state.mean(dim=1)  # [B, encoded_dim]
            else:
                # 使用平均池化
                # 转置以适配池化层: [B, encoded_dim, n_agents]
                states_transposed = states_reshaped.transpose(1, 2)
                aggregated_state = self.state_pool(states_transposed).squeeze(-1)  # [B, encoded_dim]
        else:
            # 少量智能体，展平所有状态
            aggregated_state = states_reshaped.view(batch_size, -1)
        
        # 处理动作
        if actions.dim() == 3:
            actions_flat = actions.view(batch_size, -1)
        else:
            actions_flat = actions
        
        # 拼接状态和动作
        x = torch.cat([aggregated_state, actions_flat], dim=-1)
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


class OptimizedMADDPG:
    """优化版MADDPG算法 - 支持大观察空间"""
    
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
        use_encoder: bool = True,
        encoded_dim: int = 512,
        use_attention: bool = False,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.single_agent_action_dim = single_agent_action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.discrete = discrete
        self.use_encoder = use_encoder and state_dim > 1000  # 自动启用编码器
        
        # 为每个智能体创建Actor和Critic
        self.actors = []
        self.actors_target = []
        self.actors_optimizer = []
        
        self.critics = []
        self.critics_target = []
        self.critics_optimizer = []
        
        single_agent_state_dim = state_dim
        
        print(f"创建优化版MADDPG:")
        print(f"  观察维度: {state_dim}")
        print(f"  智能体数量: {n_agents}")
        print(f"  是否使用编码器: {self.use_encoder}")
        if self.use_encoder:
            print(f"  编码后维度: {encoded_dim}")
        if n_agents > 20:
            print(f"  状态聚合: {'注意力机制' if use_attention else '平均池化'}")
        
        # 分批创建网络以节省内存
        batch_size = min(10, n_agents)  # 每次创建10个智能体的网络
        
        for i in range(n_agents):
            # Actor
            actor = Actor(
                single_agent_state_dim, 
                single_agent_action_dim, 
                hidden_dim, 
                discrete,
                use_encoder=self.use_encoder,
                encoded_dim=encoded_dim,
            ).to(device)
            # 使用load_state_dict而不是deepcopy，节省内存
            actor_target = Actor(
                single_agent_state_dim, 
                single_agent_action_dim, 
                hidden_dim, 
                discrete,
                use_encoder=self.use_encoder,
                encoded_dim=encoded_dim,
            ).to(device)
            actor_target.load_state_dict(actor.state_dict())
            actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
            
            self.actors.append(actor)
            self.actors_target.append(actor_target)
            self.actors_optimizer.append(actor_optimizer)
            
            # 每创建一批网络后清理缓存（CPU模式）
            if device == "cpu" and (i + 1) % batch_size == 0:
                import gc
                gc.collect()
            
            # Critic
            critic = Critic(
                single_agent_state_dim, 
                single_agent_action_dim, 
                n_agents, 
                hidden_dim, 
                discrete,
                use_encoder=self.use_encoder,
                encoded_dim=encoded_dim,
                use_attention=use_attention,
            ).to(device)
            # 使用load_state_dict而不是deepcopy，节省内存
            critic_target = Critic(
                single_agent_state_dim, 
                single_agent_action_dim, 
                n_agents, 
                hidden_dim, 
                discrete,
                use_encoder=self.use_encoder,
                encoded_dim=encoded_dim,
                use_attention=use_attention,
            ).to(device)
            critic_target.load_state_dict(critic.state_dict())
            critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)
            
            self.critics.append(critic)
            self.critics_target.append(critic_target)
            self.critics_optimizer.append(critic_optimizer)
            
            # 每创建一批网络后清理缓存（CPU模式）
            if device == "cpu" and (i + 1) % batch_size == 0:
                import gc
                gc.collect()
        
        # 最终清理
        if device == "cpu":
            import gc
            gc.collect()
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # 训练计数器
        self.train_step = 0
    
    def select_actions(
        self,
        states: np.ndarray,
        epsilon: float = 0.0,
        add_noise: bool = False,
    ) -> np.ndarray:
        """
        为所有智能体选择动作
        
        参数:
            states: 可以是：
                - [state_dim] 单个智能体的状态
                - [state_dim * n_agents] 所有智能体状态的拼接
                - [n_agents, state_dim] 每个智能体的状态
        """
        if states.ndim == 1:
            if len(states) == self.state_dim:
                # 单个智能体的状态，所有智能体共享相同的全局观察
                state_single = states
                # 为所有智能体使用相同的状态
                actions = []
                for i, actor in enumerate(self.actors):
                    action = actor.select_action(state_single, epsilon=epsilon, device=self.device)
                    actions.append(action)
                return np.array(actions)
            elif len(states) == self.state_dim * self.n_agents:
                # 所有智能体状态的拼接
                single_state_dim = self.state_dim
                states_per_agent = states.reshape(self.n_agents, single_state_dim)
            else:
                # 未知格式，尝试作为单个智能体状态
                states_per_agent = np.tile(states, (self.n_agents, 1))
        else:
            states_per_agent = states
        
        actions = []
        for i, actor in enumerate(self.actors):
            state_i = states_per_agent[i] if states_per_agent.shape[0] > i else states_per_agent[0]
            action = actor.select_action(state_i, epsilon=epsilon, device=self.device)
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
        # states_t可能是 [B, state_dim] 或 [B, state_dim * n_agents]
        single_state_dim = self.state_dim
        
        # 为每个智能体更新
        for i in range(self.n_agents):
            # 提取智能体i的状态
            if states_t.dim() == 2:
                if states_t.shape[1] == self.state_dim:
                    # [B, state_dim] - 单个智能体的状态，所有智能体使用相同的状态
                    states_reshaped = states_t.unsqueeze(1).expand(-1, self.n_agents, -1)  # [B, n_agents, state_dim]
                    next_states_reshaped = next_states_t.unsqueeze(1).expand(-1, self.n_agents, -1)
                elif states_t.shape[1] == self.state_dim * self.n_agents:
                    # [B, state_dim * n_agents] - 所有智能体状态的拼接
                    states_reshaped = states_t.view(batch_size, self.n_agents, single_state_dim)
                    next_states_reshaped = next_states_t.view(batch_size, self.n_agents, single_state_dim)
                else:
                    # 未知格式，尝试reshape
                    # 假设是单个智能体状态，复制给所有智能体
                    states_reshaped = states_t.unsqueeze(1).expand(-1, self.n_agents, -1)
                    next_states_reshaped = next_states_t.unsqueeze(1).expand(-1, self.n_agents, -1)
            else:
                states_reshaped = states_t
                next_states_reshaped = next_states_t
            
            states_i = states_reshaped[:, i, :]
            next_states_i = next_states_reshaped[:, i, :]
            
            # 获取下一个动作
            with torch.no_grad():
                next_actions_probs = self.actors_target[i](next_states_i)
                if self.discrete:
                    next_actions_onehot = next_actions_probs
                else:
                    next_actions_onehot = next_actions_probs
            
            # 获取当前动作的one-hot编码
            if self.discrete:
                actions_onehot = F.one_hot(actions_t[:, i].long(), self.single_agent_action_dim).float()
            else:
                actions_onehot = actions_t[:, i:i+1]
            
            # 构建所有智能体的状态和动作
            all_states = states_reshaped.view(batch_size, -1)
            all_actions_list = []
            for j in range(self.n_agents):
                if j == i:
                    all_actions_list.append(actions_onehot)
                else:
                    if self.discrete:
                        other_action_onehot = F.one_hot(actions_t[:, j].long(), self.single_agent_action_dim).float()
                    else:
                        other_action_onehot = actions_t[:, j:j+1]
                    all_actions_list.append(other_action_onehot)
            all_actions = torch.cat(all_actions_list, dim=1)
            all_actions = all_actions.view(batch_size, -1)
            
            # 更新Critic
            current_q = self.critics[i](all_states, all_actions).squeeze()
            
            # 构建下一个状态的动作
            all_next_states = next_states_reshaped.view(batch_size, -1)
            all_next_actions_list = []
            for j in range(self.n_agents):
                if j == i:
                    all_next_actions_list.append(next_actions_onehot)
                else:
                    next_actions_probs_j = self.actors_target[j](next_states_reshaped[:, j, :])
                    if self.discrete:
                        all_next_actions_list.append(next_actions_probs_j)
                    else:
                        all_next_actions_list.append(next_actions_probs_j)
            all_next_actions = torch.cat(all_next_actions_list, dim=1)
            all_next_actions = all_next_actions.view(batch_size, -1)
            
            next_q = self.critics_target[i](all_next_states, all_next_actions).squeeze()
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q
            
            critic_loss = F.mse_loss(current_q, target_q.detach())
            
            # 更新Critic
            self.critics_optimizer[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 1.0)
            self.critics_optimizer[i].step()
            
            # 更新Actor
            actions_probs = self.actors[i](states_i)
            
            all_actions_policy_list = []
            for j in range(self.n_agents):
                if j == i:
                    all_actions_policy_list.append(actions_probs)
                else:
                    actions_probs_j = self.actors[j](states_reshaped[:, j, :])
                    all_actions_policy_list.append(actions_probs_j)
            all_actions_policy = torch.cat(all_actions_policy_list, dim=1)
            all_actions_policy = all_actions_policy.view(batch_size, -1)
            
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

