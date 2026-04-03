"""
BAD算法训练脚本

使用示例：
    python train_bad.py --env_name hanabi_v5 --num_iterations 100000 --eval_episodes 200 --eval_freq 1
"""
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from datetime import datetime
from tqdm import tqdm
from pettingzoo.classic import hanabi_v5

from algorithms.bad import BADAgent


def create_hanabi_env(players=4, max_life_tokens=6, seed=None):
    """创建Hanabi环境"""
    env = hanabi_v5.env(players=players, max_life_tokens=max_life_tokens)
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    return env


def evaluate_episode_hanabi(env, agent, device, deterministic=True):
    """评估Hanabi环境的一个episode"""
    env.reset()
    
    episode_stats = {
        'final_score': 0,
        'episode_reward': 0,
        'episode_steps': 0,
    }
    
    step = 0
    for agent_name in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        done = terminated or truncated
        
        if done:
            env.step(None)
            # 尝试获取最终得分
            try:
                if hasattr(env, 'env') and hasattr(env.env, 'env'):
                    raw_env = env.env.env
                    if hasattr(raw_env, 'state'):
                        state = raw_env.state
                        if hasattr(state, 'score'):
                            episode_stats['final_score'] = state.score
            except:
                pass
            break
        
        # 获取玩家索引
        try:
            player_idx = int(agent_name.split('_')[-1]) - 1
        except:
            player_idx = 0
        
        # 获取动作掩码
        action_mask = None
        if isinstance(obs, dict) and 'action_mask' in obs:
            action_mask = obs['action_mask']
        elif isinstance(obs, dict) and 'observation' in obs:
            # 尝试从观察中提取动作掩码
            obs_vec = obs['observation']
            if isinstance(obs_vec, np.ndarray) and len(obs_vec) > 20:
                # Hanabi动作掩码通常在观察向量的末尾
                action_mask = obs_vec[-20:] if len(obs_vec) >= 20 else None
        
        # 使用BAD智能体选择动作（评估时使用确定性策略）
        action = agent.get_action(env, player_idx, obs, action_mask, deterministic=deterministic)
        
        # 执行动作
        env.step(action)
        
        # 获取奖励
        next_obs, next_reward, next_terminated, next_truncated, next_info = env.last()
        episode_stats['episode_reward'] += next_reward if next_reward is not None else 0
        step += 1
    
    episode_stats['episode_steps'] = step
    return episode_stats


def evaluate_model(agent, device, num_episodes=200, players=4, max_life_tokens=6, seed=None):
    """评估模型"""
    episode_rewards = []
    episode_scores = []
    
    for episode in range(num_episodes):
        # 创建环境
        env = create_hanabi_env(players=players, max_life_tokens=max_life_tokens, seed=seed + episode if seed is not None else None)
        
        # 评估episode（使用确定性策略）
        episode_stats = evaluate_episode_hanabi(env, agent, device, deterministic=True)
        
        episode_rewards.append(episode_stats['episode_reward'])
        if 'final_score' in episode_stats and episode_stats['final_score'] > 0:
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
        title = 'BAD训练曲线'
    else:
        mean_label = 'Mean Reward'
        std_label = 'Std Dev'
        ylabel = 'Mean Reward (200 episodes)'
        title = 'BAD Training Curve'
    
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
    parser = argparse.ArgumentParser(description='BAD算法训练')
    
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
    parser.add_argument('--episodes_per_iteration', type=int, default=10,
                       help='每次迭代收集的episode数量')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=384,
                       help='隐藏层维度')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='训练批次大小')
    parser.add_argument('--train_freq', type=int, default=10,
                       help='训练频率（每N个episode训练一次）')
    
    # 评估参数
    parser.add_argument('--eval_episodes', type=int, default=200,
                       help='评估时使用的episode数量')
    parser.add_argument('--eval_freq', type=int, default=1,
                       help='评估频率（每N次迭代评估一次）')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志保存目录')
    parser.add_argument('--experiment_name', type=str, default='bad_hanabi',
                       help='实验名称')
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.log_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建环境
    env = create_hanabi_env(players=args.players, max_life_tokens=args.max_life_tokens, seed=args.seed)
    
    # 获取环境的动作空间大小
    first_agent = env.agents[0] if env.agents else None
    if first_agent is not None:
        action_dim = env.action_space(first_agent).n
    else:
        action_dim = 20  # 默认值
    
    print(f"检测到动作空间大小: {action_dim}")
    
    # 创建BAD智能体
    agent = BADAgent(
        num_players=args.players,
        device=args.device,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        action_dim=action_dim  # 传入实际的动作空间大小
    )
    
    # 评估结果
    eval_results = []
    
    print(f"开始训练BAD算法...")
    print(f"环境: {args.env_name}, 玩家数: {args.players}")
    print(f"训练迭代: {args.num_iterations}, 每次迭代episode数: {args.episodes_per_iteration}")
    print(f"评估频率: 每{args.eval_freq}次迭代, 评估episode数: {args.eval_episodes}")
    print(f"实验目录: {experiment_dir}")
    
    # 训练循环
    total_episodes = 0
    episode_rewards_buffer = []
    
    for iteration in tqdm(range(args.num_iterations), desc="训练进度"):
        # 重置公共随机种子（用于确定性策略采样）
        agent.public_random_seed = iteration
        
        # 收集episode数据
        for episode_idx in range(args.episodes_per_iteration):
            env.reset()
            episode_reward = 0
            episode_memory = []
            
            # 游戏循环
            for agent_name in env.agent_iter():
                obs, reward, terminated, truncated, info = env.last()
                done = terminated or truncated
                
                if done:
                    env.step(None)
                    # 尝试获取最终得分
                    try:
                        if hasattr(env, 'env') and hasattr(env.env, 'env'):
                            raw_env = env.env.env
                            if hasattr(raw_env, 'state'):
                                state = raw_env.state
                                if hasattr(state, 'score'):
                                    episode_reward = state.score
                    except:
                        pass
                    break
                
                # 获取玩家索引
                try:
                    player_idx = int(agent_name.split('_')[-1]) - 1
                except:
                    player_idx = 0
                
                # 获取动作掩码
                action_mask = None
                if isinstance(obs, dict) and 'action_mask' in obs:
                    action_mask = obs['action_mask']
                elif isinstance(obs, dict) and 'observation' in obs:
                    obs_vec = obs['observation']
                    if isinstance(obs_vec, np.ndarray) and len(obs_vec) > 20:
                        action_mask = obs_vec[-20:] if len(obs_vec) >= 20 else None
                
                # 使用BAD智能体选择动作（训练时使用随机探索）
                action = agent.get_action(env, player_idx, obs, action_mask, deterministic=False)
                
                # 执行动作
                env.step(action)
                
                # 获取奖励
                next_obs, next_reward, next_terminated, next_truncated, next_info = env.last()
                if next_reward is not None:
                    episode_reward += next_reward
                    episode_memory.append(next_reward)
            
            # 更新奖励
            if episode_memory:
                agent.update_rewards(episode_memory)
            
            episode_rewards_buffer.append(episode_reward)
            total_episodes += 1
            
            # 定期训练
            if total_episodes % args.train_freq == 0 and len(agent.memory) >= args.batch_size:
                agent.train_with_importance_sampling(batch_size=args.batch_size)
        
        # 评估模型
        if (iteration + 1) % args.eval_freq == 0:
            print(f"\n迭代 {iteration + 1}/{args.num_iterations}: 评估模型...")
            eval_result = evaluate_model(
                agent, args.device, 
                num_episodes=args.eval_episodes,
                players=args.players,
                max_life_tokens=args.max_life_tokens,
                seed=args.seed
            )
            eval_result['iteration'] = iteration + 1
            eval_results.append(eval_result)
            
            print(f"  平均奖励: {eval_result['mean_reward']:.4f} ± {eval_result['std_reward']:.4f}")
            if 'mean_score' in eval_result:
                print(f"  平均得分: {eval_result['mean_score']:.4f} ± {eval_result['std_score']:.4f}")
            
            # 保存评估结果
            eval_results_path = os.path.join(experiment_dir, 'eval_results.json')
            with open(eval_results_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            # 绘制训练曲线
            if len(eval_results) > 0:
                plot_path = os.path.join(experiment_dir, 'training_curve.png')
                plot_training_curve(eval_results, plot_path)
        
        # 保存模型
        if (iteration + 1) % 100 == 0:
            model_path = os.path.join(args.save_dir, f"bad_{args.experiment_name}_iter_{iteration+1}.pth")
            agent.save(model_path)
    
    # 最终评估
    print("\n最终评估...")
    final_eval = evaluate_model(
        agent, args.device,
        num_episodes=args.eval_episodes,
        players=args.players,
        max_life_tokens=args.max_life_tokens,
        seed=args.seed
    )
    print(f"最终平均奖励: {final_eval['mean_reward']:.4f} ± {final_eval['std_reward']:.4f}")
    if 'mean_score' in final_eval:
        print(f"最终平均得分: {final_eval['mean_score']:.4f} ± {final_eval['std_score']:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, f"bad_{args.experiment_name}_final.pth")
    agent.save(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 保存最终评估结果
    eval_results.append({
        'iteration': args.num_iterations,
        **final_eval
    })
    eval_results_path = os.path.join(experiment_dir, 'eval_results.json')
    with open(eval_results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # 绘制最终训练曲线
    plot_path = os.path.join(experiment_dir, 'training_curve.png')
    plot_training_curve(eval_results, plot_path)
    
    print(f"\n训练完成！")
    print(f"评估结果保存在: {eval_results_path}")
    print(f"训练曲线保存在: {plot_path}")


if __name__ == "__main__":
    main()

