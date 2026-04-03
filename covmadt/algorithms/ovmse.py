"""
OVMSE算法实现 - 专为Hanabi环境设计
基于论文: Offline-to-Online Multi-Agent Reinforcement Learning with Offline Value Function Memory and Sequential Exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, defaultdict
import h5py
import os
from pettingzoo.classic import hanabi_v5


class QMIXNetwork(nn.Module):
    """QMIX网络架构 - 用于Hanabi的集中式训练分散式执行"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=128, num_agents=4):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        
        # 每个智能体的Q网络
        self.agent_q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            ) for _ in range(num_agents)
        ])
        
        # 混合网络 - 将个体Q值混合为联合Q值
        self.hyper_w1 = nn.Sequential(
            nn.Linear(obs_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(obs_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hyper_b1 = nn.Linear(obs_dim * num_agents, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(obs_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs, actions=None, action_masks=None):
        """
        前向传播
        Args:
            obs: [batch_size, num_agents, obs_dim] 或 [num_agents, obs_dim]
            actions: 选择的动作 [batch_size, num_agents] 或 [num_agents]
            action_masks: 动作掩码 [batch_size, num_agents, action_dim] 或 [num_agents, action_dim]
        """
        # 处理输入维度
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)  # [1, num_agents, obs_dim]
        
        batch_size, num_agents, obs_dim = obs.shape
        
        # 获取每个智能体的Q值
        agent_q_values = []
        for i in range(num_agents):
            agent_obs = obs[:, i, :]  # [batch_size, obs_dim]
            q_values = self.agent_q_networks[i](agent_obs)  # [batch_size, action_dim]
            
            # 应用动作掩码
            if action_masks is not None:
                if len(action_masks.shape) == 2:
                    mask = action_masks[i].unsqueeze(0)  # [1, action_dim]
                else:
                    mask = action_masks[:, i, :]  # [batch_size, action_dim]
                q_values = q_values + torch.log(mask + 1e-10)
            
            agent_q_values.append(q_values)
        
        # 如果提供了动作，选择对应的Q值
        if actions is not None:
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(0)  # [1, num_agents]
            
            chosen_q_values = []
            for i in range(num_agents):
                q_vals = agent_q_values[i]  # [batch_size, action_dim]
                if len(actions.shape) == 2:
                    act = actions[:, i]  # [batch_size]
                else:
                    act = actions[i]
                chosen_q = q_vals.gather(1, act.unsqueeze(1))  # [batch_size, 1]
                chosen_q_values.append(chosen_q)
            
            chosen_q_values = torch.cat(chosen_q_values, dim=1)  # [batch_size, num_agents]
            
            # 计算联合Q值
            state = obs.view(batch_size, -1)  # [batch_size, num_agents * obs_dim]
            
            # 第一层混合权重
            w1 = self.hyper_w1(state)  # [batch_size, num_agents * hidden_dim]
            w1 = w1.view(batch_size, self.num_agents, -1)  # [batch_size, num_agents, hidden_dim]
            
            b1 = self.hyper_b1(state)  # [batch_size, hidden_dim]
            b1 = b1.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # 第二层混合权重
            w2 = self.hyper_w2(state)  # [batch_size, hidden_dim]
            w2 = w2.unsqueeze(-1)  # [batch_size, hidden_dim, 1]
            
            b2 = self.hyper_b2(state)  # [batch_size, 1]
            
            # 混合计算
            hidden = F.elu(torch.bmm(chosen_q_values.unsqueeze(1), w1) + b1)  # [batch_size, 1, hidden_dim]
            q_total = torch.bmm(hidden, w2).squeeze(-1) + b2  # [batch_size, 1]
            
            return q_total.squeeze(-1), agent_q_values
        else:
            return agent_q_values
    
    def get_actions(self, obs, action_masks=None, epsilon=0.0):
        """获取每个智能体的动作"""
        batch_size, num_agents, obs_dim = obs.shape
        actions = []
        
        for i in range(num_agents):
            agent_obs = obs[:, i, :]
            q_values = self.agent_q_networks[i](agent_obs)
            
            # 应用动作掩码
            if action_masks is not None:
                mask = action_masks[:, i, :]
                q_values = q_values + torch.log(mask + 1e-10)
            
            # ε-greedy探索
            if random.random() < epsilon:
                # 随机探索，但只在合法动作中
                if action_masks is not None:
                    valid_actions = torch.where(mask[0] > 0)[0]
                    if len(valid_actions) > 0:
                        action = random.choice(valid_actions.tolist())
                    else:
                        action = random.randint(0, self.action_dim - 1)
                else:
                    action = random.randint(0, self.action_dim - 1)
            else:
                # 贪心选择
                action = q_values.argmax(dim=1).item()
            
            actions.append(action)
        
        return torch.tensor(actions, dtype=torch.long)


class OVMSE:
    """OVMSE算法主类 - 用于Hanabi环境"""
    
    def __init__(self, 
                 obs_dim, 
                 action_dim,
                 num_agents=4,
                 device='cuda',
                 hidden_dim=128,
                 learning_rate=3e-4,
                 buffer_size=100000,
                 batch_size=256,
                 gamma=0.99,
                 tau=0.005,  # 目标网络软更新参数
                 lambda_memory_start=0.5,  # 降低初始值，避免过度依赖离线数据
                 lambda_memory_end=0.05,   # 降低最终值
                 lambda_annealing_steps=50000,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_annealing_steps=50000,
                 alpha=0.1,  # CQL正则化系数（降低以避免过度保守）
                 mixing_ratio=0.1,  # 离线数据混合比例
                 use_sequential_exploration=True):
        
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.mixing_ratio = mixing_ratio
        self.use_sequential_exploration = use_sequential_exploration
        
        # 创建在线网络和目标网络
        self.online_network = QMIXNetwork(obs_dim, action_dim, hidden_dim, num_agents).to(device)
        self.target_network = QMIXNetwork(obs_dim, action_dim, hidden_dim, num_agents).to(device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        
        # 离线价值函数记忆
        self.offline_network = QMIXNetwork(obs_dim, action_dim, hidden_dim, num_agents).to(device)
        self.offline_network.eval()  # 离线网络不更新
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.online_buffer = deque(maxlen=buffer_size)
        self.offline_buffer = None  # 离线数据集
        
        # 训练参数
        self.lambda_memory_start = lambda_memory_start
        self.lambda_memory_end = lambda_memory_end
        self.lambda_annealing_steps = lambda_annealing_steps
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_annealing_steps = epsilon_annealing_steps
        
        self.batch_size = batch_size
        
        # 训练统计
        self.training_stats = {
            'total_loss': [],
            'td_loss': [],
            'ovm_loss': [],
            'cql_loss': [],
            'lambda_memory': [],
            'epsilon': []
        }
    
    def load_offline_data(self, offline_data_path, data_usage_ratio=None, random_sample=False):
        """
        加载离线数据集（H5格式）
        
        参数:
            offline_data_path: 离线数据文件路径
            data_usage_ratio: 数据集使用比例（0.0-1.0，例如0.5表示只使用50%的数据，None表示使用全部数据）
            random_sample: 是否随机采样（True时随机选择，False时使用前N%的数据）
        """
        if os.path.exists(offline_data_path):
            # 从H5文件加载数据
            with h5py.File(offline_data_path, 'r') as f:
                total_size = len(f['states'])
                
                # 确定要使用的数据索引
                if data_usage_ratio is not None and 0 < data_usage_ratio < 1.0:
                    if random_sample:
                        # 随机采样
                        import random
                        random.seed(42)  # 固定随机种子以保证可重复性
                        selected_indices = sorted(random.sample(range(total_size), int(total_size * data_usage_ratio)))
                        print(f"随机采样 {data_usage_ratio*100:.1f}% 的数据: {len(selected_indices)} 条经验")
                    else:
                        # 顺序采样：使用前N%的数据
                        usage_size = int(total_size * data_usage_ratio)
                        selected_indices = list(range(usage_size))
                        print(f"使用前 {data_usage_ratio*100:.1f}% 的数据: {len(selected_indices)} 条经验")
                else:
                    # 使用全部数据
                    selected_indices = list(range(total_size))
                    print(f"使用全部数据: {len(selected_indices)} 条经验")
                
                # 读取选中的数据
                states = f['states'][selected_indices]
                actions = f['actions'][selected_indices]
                rewards = f['rewards'][selected_indices]
                next_states = f['next_states'][selected_indices]
                
                # 检查是否有dones字段
                if 'dones' in f:
                    dones = f['dones'][selected_indices]
                else:
                    # 如果没有dones，创建一个全False数组
                    dones = np.zeros(len(states), dtype=bool)
                
                # 检查是否有action_mask字段
                has_action_mask = 'action_mask' in f
                
                # 转换为列表格式
                self.offline_buffer = []
                action_space_mismatch_count = 0
                for i in range(len(states)):
                    action = int(actions[i])
                    
                    # 检查动作空间是否匹配
                    if action >= self.action_dim:
                        # 动作空间不匹配，跳过这条经验或映射到有效范围
                        action_space_mismatch_count += 1
                        # 将动作映射到有效范围（取模）
                        action = action % self.action_dim
                    
                    transition = {
                        'obs': states[i],
                        'action': action,
                        'reward': float(rewards[i]),
                        'next_obs': next_states[i],
                        'done': bool(dones[i]),
                    }
                    
                    # 添加action_mask（如果存在）
                    if has_action_mask:
                        mask = f['action_mask'][selected_indices[i]]
                        # 如果action_mask维度不匹配，创建全1掩码
                        if len(mask) != self.action_dim:
                            transition['action_mask'] = np.ones(self.action_dim, dtype=np.float32)
                        else:
                            transition['action_mask'] = mask
                    
                    self.offline_buffer.append(transition)
                
                if action_space_mismatch_count > 0:
                    print(f"⚠️  警告: 发现 {action_space_mismatch_count} 条经验的动作空间不匹配，已自动映射到有效范围")
            
            print(f"已加载离线数据: {len(self.offline_buffer)} 条经验")
        else:
            print(f"离线数据文件不存在: {offline_data_path}")
            self.offline_buffer = []
    
    def load_offline_model(self, model_path):
        """加载离线预训练模型"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.offline_network.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载离线预训练模型: {model_path}")
        else:
            print(f"离线模型文件不存在: {model_path}")
    
    def collect_offline_data(self, num_episodes=1000, save_path='offline_data.h5'):
        """收集离线数据（保存为H5格式）"""
        print(f"收集 {num_episodes} 个episode的离线数据...")
        
        env = hanabi_v5.env(players=self.num_agents, max_life_tokens=6)
        
        # 用于存储数据的列表
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []
        action_masks_list = []
        
        for episode in range(num_episodes):
            env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 100:
                # 获取当前观察
                obs, _, _, _, info = env.last()
                if isinstance(obs, dict):
                    current_obs = obs["observation"]
                    action_mask = obs.get('action_mask', None)
                else:
                    current_obs = obs
                    action_mask = None
                
                # 随机选择动作
                if action_mask is not None:
                    valid_actions = np.where(action_mask > 0)[0]
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                    else:
                        action = np.random.randint(self.action_dim)
                else:
                    action = np.random.randint(self.action_dim)
                
                # 执行动作
                env.step(action)
                
                # 获取下一个观察和奖励（Hanabi环境使用last()方法）
                next_obs, reward, termination, truncation, info = env.last()
                done = termination or truncation
                
                if isinstance(next_obs, dict):
                    next_obs_processed = next_obs["observation"]
                else:
                    next_obs_processed = next_obs
                
                # 存储数据
                states_list.append(current_obs)
                actions_list.append(action)
                rewards_list.append(reward)
                next_states_list.append(next_obs_processed)
                dones_list.append(done)
                
                # 存储action_mask（如果存在）
                if action_mask is not None:
                    action_masks_list.append(action_mask)
                
                step_count += 1
            
            if episode % 100 == 0:
                print(f"  已收集 {episode} 个episode")
        
        env.close()
        
        # 转换为numpy数组
        states = np.array(states_list)
        actions = np.array(actions_list, dtype=np.int64)
        rewards = np.array(rewards_list, dtype=np.float32)
        next_states = np.array(next_states_list)
        dones = np.array(dones_list, dtype=bool)
        
        # 保存为H5格式
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('states', data=states)
            f.create_dataset('actions', data=actions)
            f.create_dataset('rewards', data=rewards)
            f.create_dataset('next_states', data=next_states)
            f.create_dataset('dones', data=dones)
            
            # 如果有action_mask，也保存
            if len(action_masks_list) > 0:
                action_masks = np.array(action_masks_list)
                f.create_dataset('action_mask', data=action_masks)
        
        # 加载到内存缓冲区
        self.load_offline_data(save_path)
        
        print(f"离线数据已保存到: {save_path}, 共 {len(states_list)} 条经验")
    
    def add_to_online_buffer(self, transition):
        """添加到在线缓冲区"""
        self.online_buffer.append(transition)
    
    def sample_batch(self):
        """从在线和离线缓冲区采样批次"""
        # 确定采样比例
        online_batch_size = int(self.batch_size * (1 - self.mixing_ratio))
        offline_batch_size = self.batch_size - online_batch_size
        
        # 从在线缓冲区采样
        if len(self.online_buffer) >= online_batch_size:
            online_batch = random.sample(self.online_buffer, online_batch_size)
        else:
            online_batch = list(self.online_buffer)
        
        # 从离线缓冲区采样
        if self.offline_buffer is not None and len(self.offline_buffer) >= offline_batch_size:
            offline_batch = random.sample(self.offline_buffer, offline_batch_size)
        else:
            offline_batch = []
        
        # 合并批次
        batch = online_batch + offline_batch
        
        if len(batch) == 0:
            return None
        
        # 转换为张量（先转换为numpy数组以提高效率）
        obs_array = np.array([t['obs'] for t in batch])
        obs_batch = torch.FloatTensor(obs_array).to(self.device)
        
        action_array = np.array([t['action'] for t in batch])
        action_batch = torch.LongTensor(action_array).to(self.device)
        
        reward_array = np.array([t['reward'] for t in batch])
        reward_batch = torch.FloatTensor(reward_array).to(self.device)
        
        next_obs_array = np.array([t['next_obs'] for t in batch])
        next_obs_batch = torch.FloatTensor(next_obs_array).to(self.device)
        
        done_array = np.array([t['done'] for t in batch])
        done_batch = torch.FloatTensor(done_array).to(self.device)
        
        # 处理动作掩码（先转换为numpy数组以提高效率）
        action_masks = []
        next_action_masks = []
        for t in batch:
            action_mask = t.get('action_mask', None)
            if action_mask is not None:
                action_masks.append(action_mask)
                next_action_masks.append(t.get('next_action_mask', action_mask))
            else:
                # 如果没有动作掩码，创建全1掩码
                action_masks.append(np.ones(self.action_dim, dtype=np.float32))
                next_action_masks.append(np.ones(self.action_dim, dtype=np.float32))
        
        # 先转换为numpy数组，再创建tensor
        action_mask_array = np.array(action_masks)
        action_mask_batch = torch.FloatTensor(action_mask_array).to(self.device)
        
        next_action_mask_array = np.array(next_action_masks)
        next_action_mask_batch = torch.FloatTensor(next_action_mask_array).to(self.device)
        
        # 为Hanabi环境，我们需要为所有智能体创建相同的观察
        # 由于Hanabi是部分可观察的，每个智能体的观察不同
        # 但为了简化，我们假设所有智能体使用相同的观察（实际上是不正确的）
        # 在实际应用中，应该为每个智能体存储各自的观察
        obs_batch = obs_batch.unsqueeze(1).repeat(1, self.num_agents, 1)
        next_obs_batch = next_obs_batch.unsqueeze(1).repeat(1, self.num_agents, 1)
        action_mask_batch = action_mask_batch.unsqueeze(1).repeat(1, self.num_agents, 1)
        next_action_mask_batch = next_action_mask_batch.unsqueeze(1).repeat(1, self.num_agents, 1)
        
        return {
            'obs': obs_batch,
            'actions': action_batch,
            'rewards': reward_batch,
            'next_obs': next_obs_batch,
            'dones': done_batch,
            'action_masks': action_mask_batch,
            'next_action_masks': next_action_mask_batch
        }
    
    def compute_lambda_memory(self, step):
        """计算λ_memory的退火值"""
        if step < self.lambda_annealing_steps:
            lambda_memory = self.lambda_memory_start - (self.lambda_memory_start - self.lambda_memory_end) * (step / self.lambda_annealing_steps)
        else:
            lambda_memory = self.lambda_memory_end
        return lambda_memory
    
    def compute_epsilon(self, step):
        """计算ε的退火值"""
        if step < self.epsilon_annealing_steps:
            epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (step / self.epsilon_annealing_steps)
        else:
            epsilon = self.epsilon_end
        return epsilon
    
    def get_sequential_exploration_rate(self, step, agent_id):
        """获取顺序探索率 - 分散式版本"""
        epsilon = self.compute_epsilon(step)
        if self.use_sequential_exploration:
            # 分散式顺序探索: ε_dec = ε / N
            epsilon_dec = epsilon / self.num_agents
        else:
            # 标准ε-greedy
            epsilon_dec = epsilon
        return epsilon_dec
    
    def train_step(self, step):
        """执行一步训练"""
        # 采样批次
        batch = self.sample_batch()
        if batch is None:
            return None
        
        obs = batch['obs']  # [batch_size, num_agents, obs_dim]
        actions = batch['actions']  # [batch_size]
        rewards = batch['rewards']  # [batch_size]
        next_obs = batch['next_obs']  # [batch_size, num_agents, obs_dim]
        dones = batch['dones']  # [batch_size]
        action_masks = batch['action_masks']  # [batch_size, num_agents, action_dim]
        next_action_masks = batch['next_action_masks']  # [batch_size, num_agents, action_dim]
        
        # 将动作转换为智能体动作格式
        # 在Hanabi中，每个时间步只有一个智能体行动
        # 我们将动作转换为每个智能体的动作向量
        agent_actions = torch.zeros(obs.shape[0], self.num_agents, dtype=torch.long).to(self.device)
        for i in range(obs.shape[0]):
            # 假设第一个智能体是当前行动的智能体
            agent_actions[i, 0] = actions[i]
        
        # 1. 计算在线TD目标
        with torch.no_grad():
            # 使用目标网络计算下一个状态的Q值（当actions=None时，只返回agent_q_values列表）
            next_agent_q_values = self.target_network(next_obs, None, next_action_masks)
            
            # 选择每个智能体的最大Q值
            next_max_q_values = []
            for i in range(self.num_agents):
                agent_q = next_agent_q_values[i]  # [batch_size, action_dim]
                # 应用动作掩码
                if next_action_masks is not None:
                    mask = next_action_masks[:, i, :]
                    agent_q = agent_q + torch.log(mask + 1e-10)
                max_q = agent_q.max(dim=1)[0]  # [batch_size]
                next_max_q_values.append(max_q)
            
            # 计算联合最大Q值（简单求和，符合QMIX的单调性）
            next_max_q_total = torch.stack(next_max_q_values, dim=1).sum(dim=1)  # [batch_size]
            
            # TD目标
            td_target = rewards + self.gamma * (1 - dones) * next_max_q_total
        
        # 2. 计算离线价值函数记忆
        with torch.no_grad():
            # 使用离线网络计算当前状态的Q值
            offline_q_total, _ = self.offline_network(obs, agent_actions, action_masks)
            
            # OVM目标: max(离线Q值, TD目标)
            ovm_target = torch.max(offline_q_total, td_target)
        
        # 3. 计算在线网络的Q值
        online_q_total, online_agent_q_values = self.online_network(obs, agent_actions, action_masks)
        
        # 4. 计算损失
        lambda_memory = self.compute_lambda_memory(step)
        
        # TD损失
        td_loss = F.mse_loss(online_q_total, td_target)
        
        # OVM损失
        ovm_loss = F.mse_loss(online_q_total, ovm_target)
        
        # CQL正则化损失（保守Q学习）
        cql_loss = 0
        if self.alpha > 0:
            # 对于每个智能体
            for i in range(self.num_agents):
                agent_q = online_agent_q_values[i]  # [batch_size, action_dim]
                
                # 计算期望Q值
                logsumexp_q = torch.logsumexp(agent_q, dim=1)  # [batch_size]
                dataset_q = agent_q.gather(1, agent_actions[:, i].unsqueeze(1)).squeeze(1)  # [batch_size]
                
                cql_loss += (logsumexp_q - dataset_q).mean()
            
            cql_loss = cql_loss / self.num_agents
        
        # 总损失
        total_loss = (1 - lambda_memory) * td_loss + lambda_memory * ovm_loss + self.alpha * cql_loss
        
        # 5. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 0.5)
        self.optimizer.step()
        
        # 6. 软更新目标网络（不是每次训练都更新，提高稳定性）
        # 每4步更新一次目标网络，而不是每步都更新
        if step % 4 == 0:
            self.soft_update_target_network()
        
        # 记录统计信息
        stats = {
            'total_loss': total_loss.item(),
            'td_loss': td_loss.item(),
            'ovm_loss': ovm_loss.item(),
            'cql_loss': cql_loss.item() if self.alpha > 0 else 0,
            'lambda_memory': lambda_memory,
            'epsilon': self.compute_epsilon(step)
        }
        
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        return stats
    
    def soft_update_target_network(self):
        """软更新目标网络"""
        for target_param, online_param in zip(self.target_network.parameters(), self.online_network.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)
    
    def get_action(self, obs, action_mask=None, step=0, agent_id=0, deterministic=False):
        """获取动作 - 支持顺序探索
        
        Args:
            obs: 观察
            action_mask: 动作掩码
            step: 当前步数（用于epsilon退火）
            agent_id: 智能体ID
            deterministic: 是否使用确定性策略（评估时使用，epsilon=0）
        """
        # 将观察转换为张量
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, obs_dim]
        obs_tensor = obs_tensor.repeat(1, self.num_agents, 1)  # [1, num_agents, obs_dim]
        
        if action_mask is not None:
            action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, action_dim]
            action_mask_tensor = action_mask_tensor.repeat(1, self.num_agents, 1)  # [1, num_agents, action_dim]
        else:
            action_mask_tensor = None
        
        # 获取探索率（评估时使用确定性策略，epsilon=0）
        if deterministic:
            epsilon_dec = 0.0
        else:
            epsilon_dec = self.get_sequential_exploration_rate(step, agent_id)
        
        # ε-greedy探索
        if random.random() < epsilon_dec:
            # 随机探索，但只在合法动作中
            if action_mask is not None:
                valid_actions = np.where(action_mask > 0)[0]
                if len(valid_actions) > 0:
                    action = random.choice(valid_actions)
                else:
                    action = random.randint(0, self.action_dim - 1)
            else:
                action = random.randint(0, self.action_dim - 1)
        else:
            # 贪心选择
            self.online_network.eval()
            with torch.no_grad():
                # 获取所有智能体的Q值
                agent_q_values = self.online_network(obs_tensor, None, action_mask_tensor)
                
                # 使用正确的智能体ID（而不是总是使用第一个）
                current_agent_q = agent_q_values[agent_id]  # [1, action_dim]
                
                # 应用动作掩码
                if action_mask_tensor is not None:
                    mask = action_mask_tensor[0, agent_id, :]  # [action_dim]
                    current_agent_q = current_agent_q + torch.log(mask + 1e-10)
                
                # 选择最大Q值的动作
                action = current_agent_q.argmax().item()
                
                # 验证动作是否合法
                if action_mask is not None:
                    if action >= len(action_mask) or action_mask[action] == 0:
                        # 如果动作不合法，从合法动作中选择Q值最高的
                        valid_actions = np.where(action_mask > 0)[0]
                        if len(valid_actions) > 0:
                            valid_q_values = current_agent_q[0, valid_actions].cpu().numpy()
                            best_valid_idx = np.argmax(valid_q_values)
                            action = valid_actions[best_valid_idx]
                        else:
                            action = 0
        
        return action
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'offline_network_state_dict': self.offline_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.offline_network.load_state_dict(checkpoint['offline_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
    
    # 为了兼容性，添加一些方法
    def predict_action(self, states, deterministic=False, mask=None):
        """兼容性方法：预测动作"""
        if isinstance(states, torch.Tensor):
            if states.dim() == 1:
                obs = states.cpu().numpy()
            else:
                obs = states[0].cpu().numpy()
        else:
            obs = states
        
        if isinstance(mask, torch.Tensor):
            if mask.dim() > 1:
                action_mask = mask[0].cpu().numpy()
            else:
                action_mask = mask.cpu().numpy()
        else:
            action_mask = mask
        
        # 使用get_action方法，传递deterministic参数
        # 对于评估，使用step=0和agent_id=0（单智能体场景）
        action = self.get_action(obs, action_mask, step=0, agent_id=0, deterministic=deterministic)
        action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
        log_prob = torch.tensor([0.0], dtype=torch.float32).to(self.device)
        
        return action_tensor, log_prob, {}
    
    def save_checkpoint(self, path):
        """兼容性方法：保存检查点"""
        self.save_model(path)
    
    def load_checkpoint(self, path):
        """兼容性方法：加载检查点"""
        self.load_model(path)
    
    def get_num_params(self):
        """获取参数数量"""
        return sum(p.numel() for p in self.online_network.parameters() if p.requires_grad)
    
    def eval(self):
        """设置为评估模式"""
        self.online_network.eval()
        self.target_network.eval()
        self.offline_network.eval()
    
    def train(self):
        """设置为训练模式"""
        self.online_network.train()
        self.target_network.eval()  # 目标网络始终是eval模式
        self.offline_network.eval()  # 离线网络始终是eval模式
