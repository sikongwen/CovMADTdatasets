"""
OVMSE算法训练脚本（新版本 - 基于论文实现）

使用示例：
    python train_ovmse_new.py --env_name hanabi_v5 --num_iterations 100000 --eval_episodes 200
"""
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
from pettingzoo.classic import hanabi_v5

from algorithms.ovmse import OVMSE


def create_hanabi_env(players=4, max_life_tokens=6, seed=None):
    """创建Hanabi环境"""
    env = hanabi_v5.env(players=players, max_life_tokens=max_life_tokens)
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    return env


def evaluate_episode_hanabi(env, model, device, deterministic=True):
    """评估Hanabi环境的一个episode"""
    env.reset()
    
    episode_stats = {
        'final_score': 0,
        'episode_reward': 0,
        'episode_steps': 0,
    }
    
    step = 0
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        done = terminated or truncated
        
        if done:
            env.step(None)
            # 尝试获取最终得分
            try:
                if hasattr(env, 'env') and hasattr(env.env, 'env'):
                    raw_env = env.env.env
                    if hasattr(raw_env, '_state'):
                        state = raw_env._state
                        fireworks = getattr(state, 'fireworks', None)
                        if fireworks is not None:
                            if isinstance(fireworks, dict):
                                score = sum(fireworks.values())
                            elif isinstance(fireworks, (list, np.ndarray)):
                                score = sum(fireworks)
                            else:
                                score = 0
                            if score > 0:
                                episode_stats['final_score'] = score
            except Exception:
                pass
            
            if episode_stats['final_score'] == 0:
                episode_stats['final_score'] = int(episode_stats['episode_reward'])
            break
        
        # 获取观察和动作掩码
        state = obs["observation"]
        action_mask = obs["action_mask"]
        
        # 获取当前智能体ID
        agent_id = env.agents.index(env.agent_selection)
        
        # 使用OVMSE选择动作（评估时使用确定性策略）
        action = model.get_action(state, action_mask, step=step, agent_id=agent_id, deterministic=deterministic)
        
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
        
        # 执行动作
        env.step(action)
        episode_stats['episode_reward'] += reward
        episode_stats['episode_steps'] += 1
        step += 1
    
    return episode_stats


def evaluate_model(model, env_name, players=4, max_life_tokens=6, num_episodes=200, device='cuda', seed=None):
    """评估模型，返回平均奖励"""
    episode_rewards = []
    episode_scores = []
    
    for episode in range(num_episodes):
        # 创建环境
        env = create_hanabi_env(players=players, max_life_tokens=max_life_tokens, seed=seed + episode if seed is not None else None)
        
        # 评估episode
        episode_stats = evaluate_episode_hanabi(env, model, device, deterministic=True)
        
        episode_rewards.append(episode_stats['episode_reward'])
        if 'final_score' in episode_stats:
            episode_scores.append(episode_stats['final_score'])
        
        env.close()
    
    result = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
    }
    
    if episode_scores:
        result['mean_score'] = np.mean(episode_scores)
        result['std_score'] = np.std(episode_scores)
    
    return result


def setup_chinese_font():
    """设置中文字体，如果找不到则返回False"""
    import matplotlib.font_manager as fm
    
    # 常见的中文字体名称
    chinese_fonts = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'WenQuanYi Micro Hei',  # 文泉驿微米黑
        'WenQuanYi Zen Hei',    # 文泉驿正黑
        'Noto Sans CJK SC',    # Noto Sans 中文字体
        'Source Han Sans CN',   # 思源黑体
        'STHeiti',             # 华文黑体
        'STSong',              # 华文宋体
    ]
    
    # 获取系统所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找可用的中文字体
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return True
    
    return False


def plot_training_curve(eval_results, save_path):
    """绘制训练曲线"""
    iterations = [r['iteration'] for r in eval_results]
    mean_rewards = [r['mean_reward'] for r in eval_results]
    std_rewards = [r['std_reward'] for r in eval_results]
    
    # 尝试设置中文字体
    has_chinese_font = setup_chinese_font()
    
    # 根据是否有中文字体选择标签
    if has_chinese_font:
        mean_label = '平均奖励'
        std_label = '标准差'
        ylabel = '平均奖励 (200 episodes)'
        title = 'OVMSE训练曲线'
    else:
        mean_label = 'Mean Reward'
        std_label = 'Std Dev'
        ylabel = 'Mean Reward (200 episodes)'
        title = 'OVMSE Training Curve'
    
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, mean_rewards, 'b-', label=mean_label, linewidth=2)
    plt.fill_between(iterations, 
                     np.array(mean_rewards) - np.array(std_rewards),
                     np.array(mean_rewards) + np.array(std_rewards),
                     alpha=0.3, color='blue', label=std_label)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='OVMSE算法训练（新版本）')
    
    # 环境参数
    parser.add_argument('--env_name', type=str, default='hanabi_v5',
                       help='环境名称')
    parser.add_argument('--players', type=int, default=4,
                       help='Hanabi环境玩家数量')
    parser.add_argument('--max_life_tokens', type=int, default=6,
                       help='Hanabi环境最大生命令牌数')
    
    # 训练参数
    parser.add_argument('--num_iterations', type=int, default=100000,
                       help='训练迭代次数')
    parser.add_argument('--train_freq', type=int, default=4,
                       help='每N步训练一次')
    parser.add_argument('--eval_freq', type=int, default=1000,
                       help='每N个iteration评估一次')
    parser.add_argument('--eval_episodes', type=int, default=200,
                       help='每次评估的episode数')
    
    # 模型参数（如果未指定，将从环境自动获取）
    parser.add_argument('--obs_dim', type=int, default=None,
                       help='观察维度（如果未指定，将从环境自动获取）')
    parser.add_argument('--action_dim', type=int, default=None,
                       help='动作维度（如果未指定，将从环境自动获取）')
    parser.add_argument('--num_agents', type=int, default=4,
                       help='智能体数量')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='隐藏层维度')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--buffer_size', type=int, default=100000,
                       help='经验回放缓冲区大小')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批次大小')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    parser.add_argument('--tau', type=float, default=0.005,
                       help='目标网络软更新参数')
    
    # OVMSE特定参数
    parser.add_argument('--lambda_memory_start', type=float, default=1.0,
                       help='λ_memory初始值')
    parser.add_argument('--lambda_memory_end', type=float, default=0.1,
                       help='λ_memory最终值')
    parser.add_argument('--lambda_annealing_steps', type=int, default=50000,
                       help='λ_memory退火步数')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                       help='ε初始值')
    parser.add_argument('--epsilon_end', type=float, default=0.05,
                       help='ε最终值')
    parser.add_argument('--epsilon_annealing_steps', type=int, default=50000,
                       help='ε退火步数')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='CQL正则化系数')
    parser.add_argument('--mixing_ratio', type=float, default=0.1,
                       help='离线数据混合比例')
    parser.add_argument('--use_sequential_exploration', action='store_true',
                       help='使用顺序探索')
    
    # 数据参数
    parser.add_argument('--offline_data_path', type=str, default=None,
                       help='离线数据文件路径（H5格式）')
    parser.add_argument('--offline_model_path', type=str, default=None,
                       help='离线预训练模型路径')
    parser.add_argument('--data_usage_ratio', type=float, default=None,
                       help='数据集使用比例（0.0-1.0，例如0.5表示只使用50%%的数据，None表示使用全部数据）')
    parser.add_argument('--random_sample', action='store_true',
                       help='是否随机采样数据（默认False，使用前N%%的数据）')
    
    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志保存目录')
    parser.add_argument('--experiment_name', type=str, default='ovmse_training',
                       help='实验名称')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("OVMSE算法训练（新版本）")
    print("=" * 60)
    print(f"环境: {args.env_name}")
    print(f"设备: {device}")
    print(f"训练迭代次数: {args.num_iterations}")
    print(f"评估频率: 每 {args.eval_freq} 个iteration评估一次")
    print(f"每次评估: {args.eval_episodes} 个episode")
    print("=" * 60)
    
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 创建环境以获取实际维度
    print("\n初始化环境以获取维度信息...")
    test_env = create_hanabi_env(players=args.players, max_life_tokens=args.max_life_tokens, seed=args.seed)
    test_obs, _, _, _, _ = test_env.last()
    if isinstance(test_obs, dict):
        actual_obs_dim = len(test_obs["observation"])
        actual_action_dim = test_env.action_space(test_env.agents[0]).n
    else:
        actual_obs_dim = len(test_obs)
        actual_action_dim = test_env.action_space.n
    test_env.close()
    
    print(f"实际观察维度: {actual_obs_dim}")
    print(f"实际动作维度: {actual_action_dim}")
    print(f"智能体数量: {args.num_agents}")
    
    # 使用实际维度创建模型
    print("\n创建OVMSE模型...")
    model = OVMSE(
        obs_dim=actual_obs_dim,  # 使用实际观察维度
        action_dim=actual_action_dim,  # 使用实际动作维度
        num_agents=args.num_agents,
        device=device,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        lambda_memory_start=args.lambda_memory_start,
        lambda_memory_end=args.lambda_memory_end,
        lambda_annealing_steps=args.lambda_annealing_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_annealing_steps=args.epsilon_annealing_steps,
        alpha=args.alpha,
        mixing_ratio=args.mixing_ratio,
        use_sequential_exploration=args.use_sequential_exploration,
    )
    
    print(f"模型参数量: {model.get_num_params():,}")
    
    # 加载离线数据（如果提供）
    if args.offline_data_path is not None:
        print(f"\n加载离线数据: {args.offline_data_path}")
        if args.data_usage_ratio is not None:
            sample_type = "随机采样" if args.random_sample else "顺序采样（前N%）"
            print(f"⚠️  使用数据集比例: {args.data_usage_ratio*100:.1f}% ({sample_type})")
        model.load_offline_data(
            args.offline_data_path,
            data_usage_ratio=args.data_usage_ratio,
            random_sample=args.random_sample
        )
    
    # 加载离线预训练模型（如果提供）
    if args.offline_model_path is not None:
        print(f"\n加载离线预训练模型: {args.offline_model_path}")
        model.load_offline_model(args.offline_model_path)
        # 将离线模型参数复制给在线模型（作为初始化）
        model.online_network.load_state_dict(model.offline_network.state_dict())
        model.target_network.load_state_dict(model.offline_network.state_dict())
    
    # 评估结果记录
    eval_results = []
    
    # 创建环境
    print("\n创建训练环境...")
    env = create_hanabi_env(players=args.players, max_life_tokens=args.max_life_tokens, seed=args.seed)
    
    # 训练循环
    print("\n开始训练...")
    print("=" * 60)
    
    step = 0
    best_mean_reward = -float('inf')
    
    # 进度条
    pbar = tqdm(total=args.num_iterations, desc="训练进度")
    
    while step < args.num_iterations:
        # 重置环境
        env.reset()
        episode_reward = 0
        done = False
        episode_step = 0
        
        while not done and episode_step < 100:
            # 获取当前观察
            obs, _, _, _, info = env.last()
            if isinstance(obs, dict):
                current_obs = obs["observation"]
                action_mask = obs.get('action_mask', None)
            else:
                current_obs = obs
                action_mask = None
            
            # 获取当前智能体ID
            agent_id = env.agents.index(env.agent_selection)
            
            # 使用OVMSE选择动作
            action = model.get_action(current_obs, action_mask, step, agent_id)
            
            # 执行动作
            env.step(action)
            
            # 获取下一个观察和奖励（Hanabi环境使用last()方法）
            next_obs, reward, termination, truncation, info = env.last()
            done = termination or truncation
            
            if isinstance(next_obs, dict):
                next_obs_processed = next_obs["observation"]
                next_action_mask = next_obs.get('action_mask', None)
            else:
                next_obs_processed = next_obs
                next_action_mask = None
            
            # 存储经验
            transition = {
                'obs': current_obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs_processed,
                'done': done,
                'action_mask': action_mask,
                'next_action_mask': next_action_mask
            }
            model.add_to_online_buffer(transition)
            
            episode_reward += reward
            episode_step += 1
            step += 1
            
            # 定期训练
            if step % args.train_freq == 0 and len(model.online_buffer) >= model.batch_size:
                stats = model.train_step(step)
                # 每1000步打印一次训练统计（帮助诊断问题）
                if step % 1000 == 0 and stats is not None:
                    print(f"  [训练统计] Loss: {stats['total_loss']:.4f}, "
                          f"TD: {stats['td_loss']:.4f}, "
                          f"OVM: {stats['ovm_loss']:.4f}, "
                          f"CQL: {stats['cql_loss']:.4f}, "
                          f"λ_mem: {stats['lambda_memory']:.3f}, "
                          f"ε: {stats['epsilon']:.3f}")
            
            # 定期评估
            if step % args.eval_freq == 0:
                print(f"\nIteration {step}: 评估模型...")
                model.eval()
                eval_stats = evaluate_model(
                    model, args.env_name, args.players, args.max_life_tokens,
                    num_episodes=args.eval_episodes, device=device, 
                    seed=args.seed + step * 10000
                )
                
                eval_results.append({
                    'iteration': step,
                    'mean_reward': eval_stats['mean_reward'],
                    'std_reward': eval_stats['std_reward'],
                    'min_reward': eval_stats['min_reward'],
                    'max_reward': eval_stats['max_reward'],
                })
                
                if 'mean_score' in eval_stats:
                    eval_results[-1]['mean_score'] = eval_stats['mean_score']
                    eval_results[-1]['std_score'] = eval_stats['std_score']
                    print(f"  平均奖励: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
                    print(f"  平均得分: {eval_stats['mean_score']:.4f} ± {eval_stats['std_score']:.4f}")
                else:
                    print(f"  平均奖励: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
                
                # 保存评估结果
                eval_file = os.path.join(log_dir, 'eval_results.json')
                with open(eval_file, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                
                # 保存最佳模型
                if eval_stats['mean_reward'] > best_mean_reward:
                    best_mean_reward = eval_stats['mean_reward']
                    best_path = os.path.join(args.checkpoint_dir, "best_ovmse_model.pt")
                    model.save_model(best_path)
                    print(f"  最佳模型已保存，平均奖励: {best_mean_reward:.4f}")
                
                model.train()
            
            # 更新进度条
            pbar.update(1)
            
            if step >= args.num_iterations:
                break
    
    env.close()
    pbar.close()
    
    # 保存最终模型
    final_path = os.path.join(args.checkpoint_dir, "final_ovmse_model.pt")
    model.save_model(final_path)
    print(f"\n最终模型已保存到: {final_path}")
    
    # 绘制训练曲线
    if len(eval_results) > 0:
        plot_path = os.path.join(log_dir, 'training_curve.png')
        plot_training_curve(eval_results, plot_path)
        
        # 保存训练统计
        stats = {
            'final_mean_reward': eval_results[-1]['mean_reward'],
            'best_mean_reward': max([r['mean_reward'] for r in eval_results]),
            'total_iterations': args.num_iterations,
            'eval_episodes': args.eval_episodes,
        }
        if 'mean_score' in eval_results[-1]:
            stats['final_mean_score'] = eval_results[-1]['mean_score']
            stats['best_mean_score'] = max([r.get('mean_score', 0) for r in eval_results])
        
        stats_file = os.path.join(log_dir, 'training_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n训练完成！")
        print(f"最终平均奖励: {eval_results[-1]['mean_reward']:.4f}")
        print(f"最佳平均奖励: {max([r['mean_reward'] for r in eval_results]):.4f}")
        if 'mean_score' in eval_results[-1]:
            print(f"最终平均得分: {eval_results[-1]['mean_score']:.4f}")
            print(f"最佳平均得分: {max([r.get('mean_score', 0) for r in eval_results]):.4f}")
        print(f"训练曲线已保存到: {plot_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"模型保存在: {args.checkpoint_dir}")
    print(f"日志保存在: {log_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

