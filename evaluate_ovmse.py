"""
OVMSE算法评估脚本

评估训练好的OVMSE模型性能
"""
import argparse
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict

from algorithms.ovmse import OVMSE

# 设置绘图库
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
except ImportError:
    HAS_PLOTTING = False
    print("警告: matplotlib/seaborn未安装，将跳过绘图功能")


def evaluate_episode_gym(env, model, device, deterministic=True):
    """评估Gym环境的episode"""
    obs, info = env.reset()
    episode_stats = {
        'episode_reward': 0,
        'episode_steps': 0,
        'actions_taken': [],
    }
    
    done = False
    truncated = False
    
    while not (done or truncated):
        # 预测动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            actions, log_probs, info = model.predict_action(
                states=state_tensor,
                deterministic=deterministic,
            )
            action = actions.cpu().numpy().item()
        
        # 执行动作
        next_obs, reward, done, truncated, info = env.step(action)
        
        # 记录统计
        episode_stats['episode_reward'] += reward
        episode_stats['episode_steps'] += 1
        episode_stats['actions_taken'].append(action)
        
        obs = next_obs
    
    return episode_stats


def get_game_statistics(env, cumulative_reward=0):
    """从hanabi环境获取游戏统计信息"""
    stats = {
        'final_score': 0,
        'life_tokens_lost': 0,
        'information_tokens_used': 0,
        'total_hints_given': 0,
        'total_discards': 0,
        'total_plays': 0,
        'is_perfect_game': False,
    }
    
    # 在Hanabi中，累计奖励通常等于最终得分
    stats['final_score'] = int(cumulative_reward) if cumulative_reward > 0 else 0
    stats['is_perfect_game'] = (stats['final_score'] == 25)
    
    try:
        # 尝试访问底层环境状态
        if hasattr(env, 'env') and hasattr(env.env, 'env'):
            raw_env = env.env.env
            if hasattr(raw_env, '_state'):
                state = raw_env._state
                
                # 获取烟花得分（最终得分）- 更准确的方法
                fireworks = getattr(state, 'fireworks', None)
                if fireworks is not None:
                    if isinstance(fireworks, dict):
                        score = sum(fireworks.values())
                    elif isinstance(fireworks, (list, np.ndarray)):
                        score = sum(fireworks)
                    else:
                        score = 0
                    if score > 0:
                        stats['final_score'] = score
                        stats['is_perfect_game'] = (score == 25)
                
                # 获取生命令牌信息
                life_tokens = getattr(state, 'life_tokens', None)
                max_life_tokens = getattr(raw_env, 'max_life_tokens', 6)
                if life_tokens is not None:
                    stats['life_tokens_lost'] = max(0, max_life_tokens - life_tokens)
                
                # 获取信息令牌信息
                information_tokens = getattr(state, 'information_tokens', None)
                max_information_tokens = getattr(raw_env, 'max_information_tokens', 8)
                
                # 尝试获取动作历史（用于计算提示、丢弃、出牌次数）
                if hasattr(state, 'move_history'):
                    move_history = state.move_history
                    for move in move_history:
                        if hasattr(move, 'move_type'):
                            move_type = move.move_type
                            if move_type == 0:  # Play
                                stats['total_plays'] += 1
                            elif move_type == 1:  # Discard
                                stats['total_discards'] += 1
                            elif move_type == 2:  # Reveal Color
                                stats['total_hints_given'] += 1
                            elif move_type == 3:  # Reveal Rank
                                stats['total_hints_given'] += 1
                
                # 计算信息令牌使用数
                # 方法1：使用提示动作数量（更准确，因为每次提示都会消耗1个信息令牌）
                if stats['total_hints_given'] > 0:
                    stats['information_tokens_used'] = stats['total_hints_given']
                # 方法2：如果无法从动作历史获取，使用令牌差值（作为备选）
                elif information_tokens is not None:
                    # 注意：这个计算可能不准确，因为弃牌会恢复信息令牌
                    stats['information_tokens_used'] = max(0, max_information_tokens - information_tokens)
                # 如果两种方法都失败，保持默认值0
    except Exception as e:
        # 如果无法获取详细信息，使用默认值
        pass
    
    return stats


def evaluate_episode_hanabi(env, model, device, deterministic=True):
    """评估Hanabi环境的episode"""
    env.reset()
    
    episode_stats = {
        'final_score': 0,
        'episode_reward': 0,
        'episode_steps': 0,
        'actions_taken': [],
        'information_tokens_used': 0,
        'total_hints_given': 0,
        'information_efficiency': 0.0,
        'is_perfect_game': False,
    }
    
    # 动作类型统计
    action_type_stats = {
        'play': 0,      # 打牌动作（0-9）
        'hint': 0,      # 提示动作（10-29）
        'discard': 0,   # 弃牌动作（30-39）
        'other': 0,     # 其他动作
    }
    
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        done = terminated or truncated
        
        if done:
            env.step(None)
            # 获取游戏统计信息
            final_stats = get_game_statistics(env, cumulative_reward=episode_stats['episode_reward'])
            episode_stats.update(final_stats)
            
            # 如果仍然无法获取最终得分，使用累计奖励
            if episode_stats['final_score'] == 0:
                episode_stats['final_score'] = int(episode_stats['episode_reward'])
                episode_stats['is_perfect_game'] = (episode_stats['final_score'] == 25)
            break
        
        # 获取观察和动作掩码
        state = obs["observation"]
        action_mask = obs["action_mask"]
        
        # 获取当前智能体ID
        agent_id = env.agents.index(env.agent_selection) if hasattr(env, 'agents') and hasattr(env, 'agent_selection') else 0
        
        # 预测动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(device)
            
            actions, log_probs, info = model.predict_action(
                states=state_tensor,
                mask=mask_tensor,
                deterministic=deterministic,
            )
            action = int(actions.cpu().numpy().item())
            
            # 额外验证：确保动作合法（双重保险）
            if action_mask is not None:
                if action >= len(action_mask) or action_mask[action] == 0:
                    # 如果动作不合法，从合法动作中随机选择
                    valid_actions = np.where(action_mask > 0)[0]
                    if len(valid_actions) > 0:
                        action = int(np.random.choice(valid_actions))
                    else:
                        # 如果没有合法动作（不应该发生），使用动作0
                        action = 0
        
        # 统计动作类型
        if 0 <= action <= 9:
            action_type_stats['play'] += 1
        elif 10 <= action <= 29:
            action_type_stats['hint'] += 1
        elif 30 <= action <= 39:
            action_type_stats['discard'] += 1
        else:
            action_type_stats['other'] += 1
        
        # 执行动作
        env.step(action)
        episode_stats['episode_reward'] += reward
        episode_stats['episode_steps'] += 1
        episode_stats['actions_taken'].append(action)
    
    # 如果无法从环境获取提示数量，使用动作类型统计
    if episode_stats['total_hints_given'] == 0 and action_type_stats['hint'] > 0:
        episode_stats['total_hints_given'] = action_type_stats['hint']
        episode_stats['information_tokens_used'] = action_type_stats['hint']
    
    # 计算信息效率
    if episode_stats['information_tokens_used'] > 0:
        episode_stats['information_efficiency'] = episode_stats['final_score'] / episode_stats['information_tokens_used']
    else:
        episode_stats['information_efficiency'] = episode_stats['final_score']
    
    return episode_stats


def plot_evaluation_results(all_episode_stats, output_dir):
    """绘制评估结果图表"""
    if not HAS_PLOTTING:
        print("跳过绘图（matplotlib/seaborn未安装）")
        return
    
    print("\n生成评估图表...")
    
    # 提取数据
    episodes = list(range(1, len(all_episode_stats) + 1))
    episode_rewards = [s['episode_reward'] for s in all_episode_stats]
    episode_steps = [s['episode_steps'] for s in all_episode_stats]
    
    # 如果有最终得分（Hanabi环境）
    if 'final_score' in all_episode_stats[0]:
        final_scores = [s['final_score'] for s in all_episode_stats]
    else:
        final_scores = None
    
    # 创建图表
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Episode奖励趋势图
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(episodes, episode_rewards, alpha=0.6, linewidth=1.5, color='#2E86AB')
    window = min(20, len(episode_rewards) // 5)
    if window > 1:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        moving_episodes = episodes[window-1:]
        plt.plot(moving_episodes, moving_avg, linewidth=2, color='#A23B72', label=f'Moving Avg ({window})')
        plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Episode Reward Over Episodes')
    plt.grid(True, alpha=0.3)
    
    # 2. Episode奖励分布直方图
    ax2 = plt.subplot(2, 2, 2)
    plt.hist(episode_rewards, bins=min(30, len(set(episode_rewards))), edgecolor='black', alpha=0.7, color='#2E86AB')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Episode Reward Distribution')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 3. Episode步数趋势图
    ax3 = plt.subplot(2, 2, 3)
    plt.plot(episodes, episode_steps, alpha=0.6, linewidth=1.5, color='#F18F01')
    if window > 1:
        moving_avg_steps = np.convolve(episode_steps, np.ones(window)/window, mode='valid')
        plt.plot(moving_episodes, moving_avg_steps, linewidth=2, color='#C73E1D', label=f'Moving Avg ({window})')
        plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Episode Steps')
    plt.title('Episode Steps Over Episodes')
    plt.grid(True, alpha=0.3)
    
    # 4. 最终得分（如果是Hanabi环境）
    if final_scores is not None:
        ax4 = plt.subplot(2, 2, 4)
        plt.plot(episodes, final_scores, alpha=0.6, linewidth=1.5, color='#6A994E')
        if window > 1:
            moving_avg_scores = np.convolve(final_scores, np.ones(window)/window, mode='valid')
            plt.plot(moving_episodes, moving_avg_scores, linewidth=2, color='#BC4749', label=f'Moving Avg ({window})')
            plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Final Score')
        plt.title('Final Score Over Episodes')
        plt.grid(True, alpha=0.3)
    else:
        # 如果没有最终得分，显示步数分布
        ax4 = plt.subplot(2, 2, 4)
        plt.hist(episode_steps, bins=min(30, len(set(episode_steps))), edgecolor='black', alpha=0.7, color='#F18F01')
        plt.xlabel('Episode Steps')
        plt.ylabel('Frequency')
        plt.title('Episode Steps Distribution')
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'ovmse_evaluation_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存到: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='OVMSE算法评估')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--env_name', type=str, default='hanabi_v5',
                       help='环境名称（hanabi_v5 或 Gym环境名称）')
    parser.add_argument('--state_dim', type=int, default=None,
                       help='状态维度（如果环境不支持自动获取）')
    parser.add_argument('--action_dim', type=int, default=None,
                       help='动作维度（如果环境不支持自动获取）')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度（需与训练时一致）')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='网络层数（需与训练时一致）')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='评估episode数量')
    parser.add_argument('--deterministic', action='store_true',
                       help='使用确定性策略（不采样）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='计算设备')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='结果输出目录')
    parser.add_argument('--plot', action='store_true',
                       help='生成评估图表')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建环境
    print(f"Creating environment: {args.env_name}")
    is_hanabi = 'hanabi' in args.env_name.lower()
    
    if is_hanabi:
        try:
            from pettingzoo.classic import hanabi_v5
            # 使用与训练时相同的配置
            players = 4
            max_life_tokens = 6
            env = hanabi_v5.env(players=players, max_life_tokens=max_life_tokens)
            env.reset(seed=0)
            
            # 获取实际维度
            obs, _, _, _, _ = env.last()
            state_dim = len(obs["observation"])
            action_dim = env.action_space(env.agents[0]).n
            
            print(f"Hanabi环境配置: {players}个智能体, max_life_tokens={max_life_tokens}")
            print(f"观察维度: {state_dim}, 动作维度: {action_dim}")
        except ImportError:
            print("错误: 无法导入Hanabi环境，请安装pettingzoo")
            return
    else:
        try:
            import gymnasium as gym
            env = gym.make(args.env_name)
            if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'shape'):
                state_dim = env.observation_space.shape[0] if len(env.observation_space.shape) > 0 else args.state_dim
            elif hasattr(env, 'state_dim'):
                state_dim = env.state_dim
            else:
                state_dim = args.state_dim or 128
            
            if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
                action_dim = env.action_space.n
            elif hasattr(env, 'action_dim'):
                action_dim = env.action_dim
            else:
                action_dim = args.action_dim or 4
        except Exception as e:
            print(f"错误: 无法创建环境: {e}")
            return
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # 创建模型
    print("Creating OVMSE model...")
    # 确定智能体数量
    num_agents = 4 if is_hanabi else 1
    
    model = OVMSE(
        obs_dim=state_dim,  # OVMSE使用obs_dim而不是state_dim
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    
    # 加载检查点
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 检查点文件不存在: {args.checkpoint_path}")
        return
    
    model.load_checkpoint(args.checkpoint_path)
    model.eval()
    print("✓ 模型加载成功")
    
    # 评估
    print(f"\n开始评估 ({args.num_episodes} episodes)...")
    all_episode_stats = []
    
    for episode in tqdm(range(1, args.num_episodes + 1)):
        if is_hanabi:
            episode_stats = evaluate_episode_hanabi(env, model, args.device, args.deterministic)
        else:
            episode_stats = evaluate_episode_gym(env, model, args.device, args.deterministic)
        
        all_episode_stats.append(episode_stats)
        
        # 每10个episode打印一次进度
        if episode % 10 == 0:
            recent_rewards = [s['episode_reward'] for s in all_episode_stats[-10:]]
            avg_reward = np.mean(recent_rewards)
            print(f"\nEpisode {episode}: 最近10个episode平均奖励 = {avg_reward:.2f}")
            if 'final_score' in episode_stats:
                recent_scores = [s['final_score'] for s in all_episode_stats[-10:]]
                avg_score = np.mean(recent_scores)
                print(f"  最近10个episode平均得分 = {avg_score:.2f}")
    
    env.close()
    
    # 计算统计指标
    print("\n" + "=" * 60)
    print("评估结果统计")
    print("=" * 60)
    
    episode_rewards = [s['episode_reward'] for s in all_episode_stats]
    episode_steps = [s['episode_steps'] for s in all_episode_stats]
    
    print(f"\nEpisode奖励统计:")
    print(f"  平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  最小奖励: {np.min(episode_rewards):.2f}")
    print(f"  最大奖励: {np.max(episode_rewards):.2f}")
    print(f"  中位数奖励: {np.median(episode_rewards):.2f}")
    
    print(f"\nEpisode步数统计:")
    print(f"  平均步数: {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")
    print(f"  最小步数: {np.min(episode_steps)}")
    print(f"  最大步数: {np.max(episode_steps)}")
    print(f"  中位数步数: {np.median(episode_steps):.2f}")
    
    if 'final_score' in all_episode_stats[0]:
        final_scores = [s['final_score'] for s in all_episode_stats]
        perfect_games = sum(1 for s in all_episode_stats if s.get('final_score', 0) == 25)
        
        print(f"\n最终得分统计 (Hanabi):")
        print(f"  平均得分: {np.mean(final_scores):.2f} ± {np.std(final_scores):.2f}")
        print(f"  最小得分: {np.min(final_scores)}")
        print(f"  最大得分: {np.max(final_scores)}")
        print(f"  中位数得分: {np.median(final_scores):.2f}")
        print(f"  完美游戏数: {perfect_games}/{args.num_episodes} ({100*perfect_games/args.num_episodes:.1f}%)")
    
    # 信息效率统计
    if 'information_efficiency' in all_episode_stats[0]:
        information_efficiencies = [s.get('information_efficiency', 0.0) for s in all_episode_stats]
        information_tokens_used = [s.get('information_tokens_used', 0) for s in all_episode_stats]
        total_hints = [s.get('total_hints_given', 0) for s in all_episode_stats]
        
        print(f"\n信息效率统计 (Information Efficiency):")
        print(f"  平均信息效率: {np.mean(information_efficiencies):.3f} ± {np.std(information_efficiencies):.3f}")
        print(f"  最小信息效率: {np.min(information_efficiencies):.3f}")
        print(f"  最大信息效率: {np.max(information_efficiencies):.3f}")
        print(f"  中位数信息效率: {np.median(information_efficiencies):.3f}")
        print(f"  平均信息令牌使用数: {np.mean(information_tokens_used):.2f} ± {np.std(information_tokens_used):.2f}")
        print(f"  平均提示动作数: {np.mean(total_hints):.2f} ± {np.std(total_hints):.2f}")
    
    # 保存结果
    results = {
        'checkpoint_path': args.checkpoint_path,
        'env_name': args.env_name,
        'num_episodes': args.num_episodes,
        'deterministic': args.deterministic,
        'statistics': {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'median_reward': float(np.median(episode_rewards)),
            'mean_steps': float(np.mean(episode_steps)),
            'std_steps': float(np.std(episode_steps)),
            'min_steps': int(np.min(episode_steps)),
            'max_steps': int(np.max(episode_steps)),
            'median_steps': float(np.median(episode_steps)),
        },
        'episode_stats': all_episode_stats,
    }
    
    if 'final_score' in all_episode_stats[0]:
        final_scores = [s['final_score'] for s in all_episode_stats]
        results['statistics']['mean_score'] = float(np.mean(final_scores))
        results['statistics']['std_score'] = float(np.std(final_scores))
        results['statistics']['min_score'] = int(np.min(final_scores))
        results['statistics']['max_score'] = int(np.max(final_scores))
        results['statistics']['median_score'] = float(np.median(final_scores))
        results['statistics']['perfect_games'] = sum(1 for s in all_episode_stats if s.get('final_score', 0) == 25)
        results['statistics']['perfect_game_rate'] = float(results['statistics']['perfect_games'] / args.num_episodes)
    
    # 添加信息效率统计
    if 'information_efficiency' in all_episode_stats[0]:
        information_efficiencies = [s.get('information_efficiency', 0.0) for s in all_episode_stats]
        information_tokens_used = [s.get('information_tokens_used', 0) for s in all_episode_stats]
        total_hints = [s.get('total_hints_given', 0) for s in all_episode_stats]
        
        results['statistics']['information_efficiency_mean'] = float(np.mean(information_efficiencies))
        results['statistics']['information_efficiency_std'] = float(np.std(information_efficiencies))
        results['statistics']['information_efficiency_min'] = float(np.min(information_efficiencies))
        results['statistics']['information_efficiency_max'] = float(np.max(information_efficiencies))
        results['statistics']['information_efficiency_median'] = float(np.median(information_efficiencies))
        results['statistics']['information_tokens_used_mean'] = float(np.mean(information_tokens_used))
        results['statistics']['information_tokens_used_std'] = float(np.std(information_tokens_used))
        results['statistics']['total_hints_mean'] = float(np.mean(total_hints))
        results['statistics']['total_hints_std'] = float(np.std(total_hints))
    
    json_path = os.path.join(args.output_dir, 'ovmse_evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 结果已保存到: {json_path}")
    
    # 生成图表
    if args.plot:
        plot_evaluation_results(all_episode_stats, args.output_dir)
    
    print("\n评估完成！")


if __name__ == '__main__':
    main()

