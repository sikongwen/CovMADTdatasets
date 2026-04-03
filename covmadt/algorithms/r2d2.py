"""
R2D2 (Recurrent Replay Distributed DQN) 算法实现
用于收集高质量的离线数据，特别适合部分可观测环境（如Hanabi）
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


class R2D2Network(nn.Module):
    """R2D2网络（LSTM + DQN）"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # LSTM层（处理序列）
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        
        # Q值输出层
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def flatten_parameters(self):
        """展平LSTM参数以提高性能"""
        self.lstm.flatten_parameters()
    
    def forward(
        self,
        states: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        参数:
            states: [B, T, S] 或 [B, S] 状态序列
            hidden: LSTM隐藏状态 (h, c)
        
        返回:
            q_values: [B, T, A] 或 [B, A] Q值
            hidden: LSTM隐藏状态
        """
        # 展平LSTM参数以提高性能（避免警告）
        self.flatten_parameters()
        
        # 处理输入维度
        was_2d = states.dim() == 2
        if was_2d:
            states = states.unsqueeze(1)  # [B, S] -> [B, 1, S]
        
        batch_size, seq_len, state_dim = states.shape
        
        # 编码状态
        states_flat = states.view(-1, state_dim)  # [B*T, S]
        encoded = self.state_encoder(states_flat)  # [B*T, H]
        encoded = encoded.view(batch_size, seq_len, self.hidden_dim)  # [B, T, H]
        
        # LSTM处理
        lstm_out, hidden = self.lstm(encoded, hidden)  # [B, T, H]
        
        # 计算Q值
        q_values = self.q_net(lstm_out)  # [B, T, A]
        
        if was_2d:
            q_values = q_values.squeeze(1)  # [B, T, A] -> [B, A]
        
        return q_values, hidden


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.buffer = []
        self.priorities = []
        self.max_priority = 1.0
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        hidden: Optional[Tuple] = None,
        next_hidden: Optional[Tuple] = None,
    ):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done, hidden, next_hidden)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, device: str = "cuda") -> Tuple:
        """采样批次（带优先级）"""
        if len(self.buffer) < batch_size:
            return None
        
        # 计算采样概率
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # 获取样本
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones, hiddens, next_hiddens = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).to(device),
            indices,
            torch.FloatTensor(weights).to(device),
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class R2D2:
    """R2D2算法"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.99,
        n_step: int = 5,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        device: str = "cuda",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.n_step = n_step
        self.device = device
        
        # 网络
        self.q_network = R2D2Network(state_dim, action_dim, hidden_dim, num_lstm_layers).to(device)
        self.target_network = copy.deepcopy(self.q_network)
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
        
        # Epsilon-greedy探索
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # LSTM隐藏状态（用于推理）
        self.hidden = None
        
        # 训练计数器
        self.train_step = 0
        self.update_target_freq = 1000
    
    def reset_hidden(self):
        """重置LSTM隐藏状态"""
        self.hidden = None
    
    def select_action(
        self,
        state: np.ndarray,
        epsilon: Optional[float] = None,
        training: bool = True,
    ) -> int:
        """
        选择动作
        
        参数:
            state: [state_dim] 状态
            epsilon: 探索率（如果为None，使用当前epsilon）
            training: 是否在训练模式
        
        返回:
            action: 动作索引
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if training and random.random() < epsilon:
            # 随机探索
            return random.randint(0, self.action_dim - 1)
        
        # 使用Q网络选择动作（临时设置为eval模式，但不改变全局状态）
        was_training = self.q_network.training
        self.q_network.eval()
        try:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, S]
                q_values, self.hidden = self.q_network(state_tensor, self.hidden)
                action = q_values.squeeze().argmax().item()
        finally:
            # 恢复原来的训练/评估状态
            if was_training:
                self.q_network.train()
        
        return action
    
    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """添加经验到回放缓冲区"""
        # 保存当前隐藏状态
        current_hidden = self.hidden
        # 计算下一个状态的隐藏状态
        next_hidden = None
        if not done:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.device)
                _, next_hidden = self.q_network(next_state_tensor, current_hidden)
        
        self.replay_buffer.push(
            state, action, reward, next_state, done,
            current_hidden, next_hidden
        )
    
    def update(self, batch_size: int = 32):
        """更新网络"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # 确保网络处于训练模式
        self.q_network.train()
        self.target_network.eval()  # 目标网络始终为eval模式
        
        # 采样批次
        batch = self.replay_buffer.sample(batch_size, self.device)
        if batch is None:
            return
        
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # 确保states和next_states是3D [B, 1, S]（R2D2需要序列输入）
        if states.dim() == 2:
            states = states.unsqueeze(1)  # [B, S] -> [B, 1, S]
        if next_states.dim() == 2:
            next_states = next_states.unsqueeze(1)  # [B, S] -> [B, 1, S]
        
        # 计算当前Q值
        q_values, _ = self.q_network(states)  # [B, 1, A] 或 [B, A]
        if q_values.dim() == 3:
            q_values = q_values.squeeze(1)  # [B, 1, A] -> [B, A]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]
        
        # 计算目标Q值（使用n-step returns）
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states)  # [B, 1, A] 或 [B, A]
            if next_q_values.dim() == 3:
                next_q_values = next_q_values.squeeze(1)  # [B, 1, A] -> [B, A]
            next_q_values = next_q_values.max(1)[0]  # [B]
            target_q = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values
        
        # TD误差
        td_errors = target_q - q_values
        
        # 加权损失
        loss = (weights * (td_errors ** 2)).mean()
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # 更新优先级
        td_errors_np = td_errors.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors_np)
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path: str):
        """保存模型"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
        }
        torch.save(checkpoint, path)
        print(f"R2D2 model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        if 'train_step' in checkpoint:
            self.train_step = checkpoint['train_step']
        print(f"R2D2 model loaded from {path}")

