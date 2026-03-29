import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pettingzoo.classic import hanabi_v5
import os
import argparse
import json
from pathlib import Path

# 设置绘图库
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# =========================
# 超参数
# =========================
GAMMA = 0.99
LR = 3e-4
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.9995
TARGET_UPDATE = 200
MAX_EPISODES = 5000
BATCH_SIZE = 8
REPLAY_SIZE = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# R2D2 网络
# =========================
class R2D2Net(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, act_dim)

    def forward(self, x, hidden):
        # x: [B, T, obs_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x, hidden = self.lstm(x, hidden)
        q = self.fc_out(x)
        return q, hidden

    def init_hidden(self, batch_size):
        h = torch.zeros(1, batch_size, self.lstm.hidden_size).to(DEVICE)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size).to(DEVICE)
        return (h, c)

# =========================
# Episode Replay Buffer
# =========================
class SequenceReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, episode):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(episode)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# =========================
# ε-greedy + action mask
# =========================
def select_action(q_values, action_mask, eps):
    if random.random() < eps:
        valid = np.where(action_mask == 1)[0]
        return int(np.random.choice(valid))
    q_values = q_values.copy()
    q_values[action_mask == 0] = -1e9
    return int(np.argmax(q_values))

# =========================
# 收集一个 episode（关键：动态 hidden state）
# =========================
def collect_episode(env, net, eps):
    episode = []
    hidden_states = {}

    env.reset()

    for agent in env.agent_iter():
        obs, reward, terminated, truncated, _ = env.last()

        if terminated or truncated:
            env.step(None)
            continue

        if agent not in hidden_states:
            hidden_states[agent] = net.init_hidden(1)

        obs_vec = torch.FloatTensor(
            obs["observation"]
        ).view(1, 1, -1).to(DEVICE)

        q, hidden = net(obs_vec, hidden_states[agent])
        hidden_states[agent] = hidden

        action = select_action(
            q.squeeze(0).squeeze(0).detach().cpu().numpy(),
            obs["action_mask"],
            eps
        )

        episode.append((
            obs["observation"],
            action,
            reward
        ))

        env.step(action)

    return episode

# =========================
# 训练奖励保存和绘制
# =========================
def save_training_rewards(episode_rewards, episode_epsilons, checkpoint_dir):
    """保存训练奖励数据到JSON文件"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    rewards_data = {
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_epsilons': [float(e) for e in episode_epsilons],
        'num_episodes': len(episode_rewards),
        'mean_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'std_reward': float(np.std(episode_rewards)) if episode_rewards else 0.0,
        'max_reward': float(np.max(episode_rewards)) if episode_rewards else 0.0,
        'min_reward': float(np.min(episode_rewards)) if episode_rewards else 0.0,
    }
    
    rewards_path = os.path.join(checkpoint_dir, 'training_rewards.json')
    with open(rewards_path, 'w') as f:
        json.dump(rewards_data, f, indent=2)
    
    print(f"✓ 训练奖励数据已保存到: {rewards_path}")


def save_and_plot_rewards(episode_rewards, episode_epsilons, checkpoint_dir, current_episode, final=False):
    """保存并绘制训练奖励图表"""
    if not HAS_PLOTTING or len(episode_rewards) == 0:
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    episodes = list(range(1, len(episode_rewards) + 1))
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. Episode奖励趋势图
    ax1 = axes[0]
    ax1.plot(episodes, episode_rewards, alpha=0.6, linewidth=1, color='#2E86AB', label='Episode Reward')
    
    # 添加移动平均线
    window = min(100, len(episode_rewards) // 10)
    if window > 1 and len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        moving_episodes = episodes[window-1:]
        ax1.plot(moving_episodes, moving_avg, linewidth=2, color='#A23B72', 
                label=f'Moving Average ({window} episodes)')
    
    # 添加均值线
    mean_reward = np.mean(episode_rewards)
    ax1.axhline(mean_reward, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_reward:.2f}')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title(f'Training Rewards (Episode {current_episode})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Epsilon衰减趋势
    ax2 = axes[1]
    ax2.plot(episodes, episode_epsilons, alpha=0.7, linewidth=1.5, color='#F18F01')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
    ax2.set_title('Epsilon Decay Over Training', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    if final:
        plot_path = os.path.join(checkpoint_dir, 'training_rewards_final.png')
    else:
        plot_path = os.path.join(checkpoint_dir, 'training_rewards.png')
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if final:
        print(f"✓ 最终训练奖励图表已保存到: {plot_path}")
    else:
        print(f"✓ 训练奖励图表已更新: {plot_path}")

# =========================
# 模型保存和加载
# =========================
def save_checkpoint(net, target_net, optimizer, episode, eps, checkpoint_dir, is_best=False):
    """保存模型检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'episode': episode,
        'net_state_dict': net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'eps': eps,
        'obs_dim': net.fc1.in_features if hasattr(net, 'fc1') else None,
        'act_dim': net.fc_out.out_features if hasattr(net, 'fc_out') else None,
    }
    
    # 保存最终模型（训练结束时）
    if episode == MAX_EPISODES:
        final_path = os.path.join(checkpoint_dir, 'final_model_4.pt')
        torch.save(checkpoint, final_path)
        print(f"✓ 最终模型已保存到: {final_path}")
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model_4.pt')
        torch.save(checkpoint, best_path)
        print(f"✓ 最佳模型已保存到: {best_path}")

def load_checkpoint(checkpoint_path, net, target_net=None, optimizer=None, device=None):
    """加载模型检查点"""
    if device is None:
        device = DEVICE
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 加载网络权重
    net.load_state_dict(checkpoint['net_state_dict'])
    print(f"✓ 主网络权重已加载")
    
    # 加载目标网络权重（如果提供）
    if target_net is not None and 'target_net_state_dict' in checkpoint:
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        print(f"✓ 目标网络权重已加载")
    
    # 加载优化器状态（如果提供）
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ 优化器状态已加载")
    
    # 返回训练信息
    episode = checkpoint.get('episode', 0)
    eps = checkpoint.get('eps', EPS_START)
    
    print(f"✓ 模型加载完成: Episode {episode}, Epsilon {eps:.4f}")
    return episode, eps

# =========================
# R2D2 更新
# =========================
def train_step(net, target_net, optimizer, batch):
    total_loss = 0.0

    for episode in batch:
        # 先转换为numpy数组，再转换为tensor（避免警告）
        obs_list = [e[0] for e in episode]
        obs_array = np.array(obs_list, dtype=np.float32)
        obs = torch.FloatTensor(obs_array).unsqueeze(0).to(DEVICE)

        actions_list = [e[1] for e in episode]
        actions_array = np.array(actions_list, dtype=np.int64)
        actions = torch.LongTensor(actions_array).to(DEVICE)

        rewards_list = [e[2] for e in episode]
        rewards_array = np.array(rewards_list, dtype=np.float32)
        rewards = torch.FloatTensor(rewards_array).to(DEVICE)

        hidden = net.init_hidden(1)
        q_seq, _ = net(obs, hidden)
        q_seq = q_seq.squeeze(0)  # [1, T, act_dim] -> [T, act_dim]

        with torch.no_grad():
            target_hidden = target_net.init_hidden(1)
            tq_seq, _ = target_net(obs, target_hidden)
            max_next_q = tq_seq.max(dim=-1)[0]  # [1, T, act_dim] -> [1, T]
            max_next_q = max_next_q.squeeze(0)  # [1, T] -> [T]

        targets = rewards + GAMMA * max_next_q  # [T] + [T] -> [T]
        q_taken = q_seq.gather(1, actions.unsqueeze(1)).squeeze(1)  # [T, act_dim] -> [T]

        # 确保维度匹配
        if q_taken.dim() != targets.dim():
            if q_taken.dim() == 1 and targets.dim() == 2:
                targets = targets.squeeze(0)
            elif q_taken.dim() == 2 and targets.dim() == 1:
                q_taken = q_taken.squeeze(0)
        
        loss = F.mse_loss(q_taken, targets)
        total_loss += loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
    optimizer.step()

# =========================
# 主函数
# =========================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='R2D2训练脚本')
    parser.add_argument('--mode', type=str, default='train_r2d2', help='运行模式')
    parser.add_argument('--env_name', type=str, default='hanabi_v5', help='环境名称')
    parser.add_argument('--max_episodes', type=int, default=5000, help='最大训练episode数')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='检查点保存目录')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练（检查点路径）')
    parser.add_argument('--save_freq', type=int, default=500, help='（已弃用）不再保存定期检查点，只保留最佳和最终模型')
    args = parser.parse_args()
    
    # 使用命令行参数或默认值
    global MAX_EPISODES
    MAX_EPISODES = args.max_episodes
    checkpoint_dir = args.checkpoint_dir
    resume_path = args.resume
    save_freq = args.save_freq
    
    # 初始化环境（4个智能体，max_life=6）
    env = hanabi_v5.env(players=4, max_life_tokens=6)
    env.reset(seed=0)

    obs, _, _, _, _ = env.last()
    obs_dim = len(obs["observation"])
    act_dim = env.action_space(env.agents[0]).n

    # 初始化网络
    net = R2D2Net(obs_dim, act_dim).to(DEVICE)
    target_net = R2D2Net(obs_dim, act_dim).to(DEVICE)
    target_net.load_state_dict(net.state_dict())

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    buffer = SequenceReplayBuffer(REPLAY_SIZE)

    # 从检查点恢复训练（如果提供）
    start_episode = 1
    eps = EPS_START
    best_reward = float('-inf')
    
    if resume_path:
        print(f"从检查点恢复训练: {resume_path}")
        start_episode, eps = load_checkpoint(resume_path, net, target_net, optimizer, DEVICE)
        start_episode += 1  # 从下一个episode开始
        print(f"从 Episode {start_episode} 继续训练...")
    else:
        print(f"开始新训练: {MAX_EPISODES} episodes")
        print(f"检查点将保存到: {checkpoint_dir}")

    # 用于记录训练奖励
    episode_rewards = []
    episode_epsilons = []
    
    # 训练循环
    for ep in range(start_episode, MAX_EPISODES + 1):
        episode = collect_episode(env, net, eps)
        buffer.add(episode)

        eps = max(EPS_END, eps * EPS_DECAY)

        if len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            train_step(net, target_net, optimizer, batch)

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(net.state_dict())

        # 计算episode奖励
        total_reward = sum(e[2] for e in episode)
        
        # 记录奖励和epsilon
        episode_rewards.append(total_reward)
        episode_epsilons.append(eps)
        
        # 更新最佳奖励
        is_best = total_reward > best_reward
        if is_best:
            best_reward = total_reward

        # 保存最佳模型（当奖励更新时）
        if is_best:
            save_checkpoint(net, target_net, optimizer, ep, eps, checkpoint_dir, is_best=True)
        
        # 训练结束时保存最终模型
        if ep == MAX_EPISODES:
            save_checkpoint(net, target_net, optimizer, ep, eps, checkpoint_dir, is_best=False)
            
            # 保存和绘制最终奖励图表
            if HAS_PLOTTING:
                save_and_plot_rewards(episode_rewards, episode_epsilons, checkpoint_dir, ep, final=True)

        # 定期打印进度
        if ep % 100 == 0:
            print(
                f"[Episode {ep}/{MAX_EPISODES}] eps={eps:.3f}, "
                f"episode reward={total_reward:.2f}, "
                f"best reward={best_reward:.2f}"
            )

    # 训练结束，保存最终模型和奖励
    print(f"\n训练完成！最终模型已保存到: {checkpoint_dir}")
    
    # 保存最终奖励数据
    save_training_rewards(episode_rewards, episode_epsilons, checkpoint_dir)
    
    # 绘制最终奖励图表
    if HAS_PLOTTING:
        save_and_plot_rewards(episode_rewards, episode_epsilons, checkpoint_dir, MAX_EPISODES, final=True)
    
    env.close()

if __name__ == "__main__":
    main()
