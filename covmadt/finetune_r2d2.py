"""
在线微调R2D2模型（适配hanabi环境）
"""
import argparse
import os
import torch
import numpy as np
from pettingzoo.classic import hanabi_v5
from tqdm import tqdm
import json

from main import R2D2Net, select_action, DEVICE

# 设置绘图库
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("警告: matplotlib未安装，将跳过绘图功能")


def collect_episode_r2d2(env, net, device, epsilon=0.1):
    """在hanabi环境中收集一个episode的数据"""
    env.reset()
    episode_reward = 0
    episode_steps = 0
    
    hidden_states = {}
    
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, _ = env.last()
        done = terminated or truncated
        
        if done:
            env.step(None)
            break
        
        # 初始化或获取LSTM隐藏状态
        if agent not in hidden_states:
            hidden_states[agent] = net.init_hidden(1)
        
        # 使用模型预测动作
        obs_vec = torch.FloatTensor(
            obs["observation"]
        ).view(1, 1, -1).to(device)
        
        with torch.no_grad():
            q, hidden = net(obs_vec, hidden_states[agent])
            hidden_states[agent] = hidden
        
        # 选择动作（使用epsilon-greedy策略）
        action = select_action(
            q.squeeze(0).squeeze(0).cpu().numpy(),
            obs["action_mask"],
            eps=epsilon
        )
        
        # 执行动作
        env.step(action)
        episode_reward += reward
        episode_steps += 1
    
    return episode_reward, episode_steps


def train_step_r2d2(net, target_net, optimizer, batch, device, gamma=0.99):
    """执行一个R2D2训练步骤"""
    net.train()
    total_loss = 0.0
    
    for episode in batch:
        # 提取episode数据
        obs_list = [e[0] for e in episode]
        obs_array = np.array(obs_list, dtype=np.float32)
        obs = torch.FloatTensor(obs_array).unsqueeze(0).to(device)  # [1, T, obs_dim]
        
        actions_list = [e[1] for e in episode]
        actions_array = np.array(actions_list, dtype=np.int64)
        actions = torch.LongTensor(actions_array).to(device)  # [T]
        
        rewards_list = [e[2] for e in episode]
        rewards_array = np.array(rewards_list, dtype=np.float32)
        rewards = torch.FloatTensor(rewards_array).to(device)  # [T]
        
        # 前向传播
        hidden = net.init_hidden(1)
        q_seq, _ = net(obs, hidden)
        q_seq = q_seq.squeeze(0)  # [1, T, act_dim] -> [T, act_dim]
        
        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            target_hidden = target_net.init_hidden(1)
            tq_seq, _ = target_net(obs, target_hidden)
            max_next_q = tq_seq.max(dim=-1)[0]  # [1, T, act_dim] -> [1, T]
            max_next_q = max_next_q.squeeze(0)  # [1, T] -> [T]
            
            targets = rewards + gamma * max_next_q  # [T]
        
        # 计算Q值
        q_taken = q_seq.gather(1, actions.unsqueeze(1)).squeeze(1)  # [T, act_dim] -> [T]
        
        # 确保维度匹配
        if q_taken.dim() != targets.dim():
            if q_taken.dim() == 1 and targets.dim() == 2:
                targets = targets.squeeze(0)
            elif q_taken.dim() == 2 and targets.dim() == 1:
                q_taken = q_taken.squeeze(0)
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(q_taken, targets)
        total_loss += loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    optimizer.step()
    
    return total_loss.item() / len(batch)


def main():
    parser = argparse.ArgumentParser(description='在线微调R2D2模型')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='预训练R2D2模型检查点路径')
    parser.add_argument('--env_name', type=str, default='hanabi_v5',
                       help='环境名称')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='微调episode数量')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批量大小（episode数量）')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率（微调时通常较小）')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='探索率')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    parser.add_argument('--target_update_freq', type=int, default=200,
                       help='目标网络更新频率（episode数）')
    
    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/r2d2_finetune',
                       help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs/r2d2_finetune',
                       help='日志保存目录')
    parser.add_argument('--device', type=str, default=None,
                       help='设备（cuda/cpu，默认: 自动检测）')
    parser.add_argument('--replay_buffer_size', type=int, default=5000,
                       help='经验回放缓冲区大小（episode数量）')
    parser.add_argument('--save_freq', type=int, default=100,
                       help='（已弃用）不再保存定期检查点，只保留最佳和最终模型')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = DEVICE
    
    print("=" * 60)
    print("R2D2 在线微调")
    print("=" * 60)
    print(f"预训练模型: {args.checkpoint}")
    print(f"环境: {args.env_name}")
    print(f"设备: {device}")
    print(f"Episode数: {args.num_episodes}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"探索率: {args.epsilon}")
    print(f"折扣因子: {args.gamma}")
    print("=" * 60)
    
    # 初始化环境
    print("\n初始化环境...")
    env = hanabi_v5.env(players=4, max_life_tokens=6)  # 4个智能体，max_life=6
    env.reset(seed=0)
    
    obs, _, _, _, _ = env.last()
    obs_dim = len(obs["observation"])
    act_dim = env.action_space(env.agents[0]).n
    num_agents = len(env.agents)
    
    print(f"观察维度: {obs_dim}")
    print(f"动作维度: {act_dim}")
    print(f"智能体数量: {num_agents}")
    
    # 加载模型
    print(f"\n加载预训练R2D2模型: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件不存在: {args.checkpoint}")
    
    # 创建网络
    net = R2D2Net(obs_dim, act_dim).to(device)
    target_net = R2D2Net(obs_dim, act_dim).to(device)
    
    # 加载权重
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # 兼容不同的检查点格式
    if 'net_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['net_state_dict'])
        if 'target_net_state_dict' in checkpoint:
            target_net.load_state_dict(checkpoint['target_net_state_dict'])
        else:
            target_net.load_state_dict(checkpoint['net_state_dict'])
        episode_info = checkpoint.get('episode', 'N/A')
        print(f"✓ 使用R2D2格式检查点")
    elif 'model_state_dict' in checkpoint:
        print(f"⚠️  警告: 检查点格式不匹配，尝试使用非严格模式加载")
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        target_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        episode_info = 'N/A'
    else:
        try:
            net.load_state_dict(checkpoint, strict=False)
            target_net.load_state_dict(checkpoint, strict=False)
            episode_info = 'N/A'
        except Exception as e:
            raise ValueError(f"无法加载模型权重。错误: {e}")
    
    # 初始化目标网络
    target_net.load_state_dict(net.state_dict())
    
    print(f"✓ 模型加载成功 (训练到 Episode {episode_info})")
    
    # 创建优化器（使用较小的学习率进行微调）
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    
    # 创建经验回放缓冲区（存储episode）
    from main import SequenceReplayBuffer
    replay_buffer = SequenceReplayBuffer(capacity=args.replay_buffer_size)
    
    # 创建输出目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 用于记录训练指标
    episode_rewards = []
    episode_losses = []
    
    # 开始微调
    print("\n开始在线微调...")
    print("=" * 60)
    
    best_reward = float('-inf')
    
    for episode in tqdm(range(1, args.num_episodes + 1), desc="微调进度"):
        # 收集episode数据
        episode_data = []
        env.reset()
        episode_reward = 0
        episode_steps = 0
        
        hidden_states = {}
        
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, _ = env.last()
            done = terminated or truncated
            
            if done:
                env.step(None)
                break
            
            # 初始化或获取LSTM隐藏状态
            if agent not in hidden_states:
                hidden_states[agent] = net.init_hidden(1)
            
            # 使用模型预测动作
            obs_vec = torch.FloatTensor(
                obs["observation"]
            ).view(1, 1, -1).to(device)
            
            with torch.no_grad():
                q, hidden = net(obs_vec, hidden_states[agent])
                hidden_states[agent] = hidden
            
            # 选择动作（使用epsilon-greedy策略）
            action = select_action(
                q.squeeze(0).squeeze(0).cpu().numpy(),
                obs["action_mask"],
                eps=args.epsilon
            )
            
            # 存储经验
            episode_data.append((
                obs["observation"],
                action,
                reward
            ))
            
            # 执行动作
            env.step(action)
            episode_reward += reward
            episode_steps += 1
        
        # 存储到回放缓冲区
        replay_buffer.add(episode_data)
        episode_rewards.append(episode_reward)
        
        # 训练（如果缓冲区有足够数据）
        train_loss = 0.0
        if len(replay_buffer) >= args.batch_size:
            batch = replay_buffer.sample(args.batch_size)
            train_loss = train_step_r2d2(net, target_net, optimizer, batch, device, args.gamma)
            episode_losses.append(train_loss)
        
        # 更新目标网络
        if episode % args.target_update_freq == 0:
            target_net.load_state_dict(net.state_dict())
        
        # 记录指标
        if episode % 10 == 0:
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            avg_reward = np.mean(recent_rewards)
            print(f"\nEpisode {episode}/{args.num_episodes}:")
            print(f"  奖励: {episode_reward:.2f}, 步数: {episode_steps}")
            print(f"  最近10个episode平均奖励: {avg_reward:.2f}")
            if train_loss > 0:
                print(f"  训练损失: {train_loss:.4f}")
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(args.checkpoint_dir, "best_r2d2_online_model.pt")
            checkpoint_data = {
                'episode': episode,
                'net_state_dict': net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eps': args.epsilon,
                'obs_dim': obs_dim,
                'act_dim': act_dim,
            }
            torch.save(checkpoint_data, best_path)
            if episode % 10 == 0:
                print(f"  ✓ 保存最佳模型 (奖励: {best_reward:.2f})")
    
    env.close()
    
    # 保存最终模型
    final_path = os.path.join(args.checkpoint_dir, "final_r2d2_online_model_4.pt")
    checkpoint_data = {
        'episode': args.num_episodes,
        'net_state_dict': net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'eps': args.epsilon,
        'obs_dim': obs_dim,
        'act_dim': act_dim,
    }
    torch.save(checkpoint_data, final_path)
    print(f"\n✓ 最终模型已保存到: {final_path}")
    
    # 保存训练指标
    metrics = {
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_losses': [float(l) for l in episode_losses],
        'num_episodes': args.num_episodes,
        'best_reward': float(best_reward),
        'mean_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'std_reward': float(np.std(episode_rewards)) if episode_rewards else 0.0,
    }
    
    metrics_path = os.path.join(args.checkpoint_dir, "r2d2_finetune_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ 训练指标已保存到: {metrics_path}")
    
    # 绘制训练曲线
    if HAS_PLOTTING and len(episode_rewards) > 0:
        plot_training_curves(episode_rewards, episode_losses, args.checkpoint_dir)
    
    print("\n" + "=" * 60)
    print("在线微调完成！")
    print(f"最佳episode奖励: {best_reward:.2f}")
    print(f"平均episode奖励: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"模型保存在: {args.checkpoint_dir}")
    print(f"日志保存在: {args.log_dir}")
    print("=" * 60)


def plot_training_curves(episode_rewards, episode_losses, output_dir):
    """绘制训练曲线"""
    if not HAS_PLOTTING:
        return
    
    print("\n生成训练曲线...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    episodes = list(range(1, len(episode_rewards) + 1))
    
    # 1. Episode奖励趋势
    ax1 = axes[0]
    ax1.plot(episodes, episode_rewards, alpha=0.6, linewidth=1, color='#2E86AB', label='Episode Reward')
    
    # 添加移动平均线
    window = min(50, len(episode_rewards) // 10)
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
    ax1.set_title('R2D2 Online Fine-tuning: Episode Rewards', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 训练损失趋势
    if len(episode_losses) > 0:
        ax2 = axes[1]
        loss_episodes = list(range(1, len(episode_losses) + 1))
        ax2.plot(loss_episodes, episode_losses, alpha=0.7, linewidth=1.5, color='#F18F01')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Training Loss', fontsize=12)
        ax2.set_title('R2D2 Online Fine-tuning: Training Loss', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No Training Loss Data', 
                    ha='center', va='center', fontsize=12)
        axes[1].axis('off')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'r2d2_finetune_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 训练曲线已保存到: {plot_path}")


if __name__ == "__main__":
    main()


