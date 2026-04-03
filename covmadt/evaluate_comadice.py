"""
ComaDICE算法评估脚本

评估训练好的ComaDICE模型性能
"""
import argparse
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict

from algorithms.comadice import ComaDICE

# 设置绘图库
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
except ImportError:
    HAS_PLOTTING = False
    print("警告: matplotlib/seaborn未安装，将跳过绘图功能")


def evaluate_episode_hanabi(env, model, device, deterministic=True):
    """评估Hanabi环境的episode"""
    env.reset()
    
    episode_stats = {
        'final_score': 0,
        'episode_reward': 0,
        'episode_steps': 0,
        'actions_taken': [],
    }
    
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
        
        # 执行动作
        env.step(action)
        episode_stats['episode_reward'] += reward
        episode_stats['episode_steps'] += 1
        episode_stats['actions_taken'].append(action)
    
    return episode_stats


def main():
    parser = argparse.ArgumentParser(description='ComaDICE算法评估')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--env_name', type=str, default='hanabi_v5',
                       help='环境名称（hanabi_v5 或 Gym环境名称）')
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
            players = 4
            max_life_tokens = 6
            env = hanabi_v5.env(players=players, max_life_tokens=max_life_tokens)
            env.reset(seed=0)
            
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
                state_dim = env.observation_space.shape[0] if len(env.observation_space.shape) > 0 else 128
            else:
                state_dim = 128
            
            if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
                action_dim = env.action_space.n
            else:
                action_dim = 4
        except Exception as e:
            print(f"错误: 无法创建环境: {e}")
            return
    
    # 创建模型
    print("Creating ComaDICE model...")
    model = ComaDICE(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
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
            # Gym环境评估（类似OVMSE）
            obs, info = env.reset()
            episode_stats = {
                'episode_reward': 0,
                'episode_steps': 0,
            }
            
            done = False
            truncated = False
            
            while not (done or truncated):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(args.device)
                    actions, _, _ = model.predict_action(state_tensor, deterministic=args.deterministic)
                    action = actions.cpu().numpy().item()
                
                next_obs, reward, done, truncated, info = env.step(action)
                episode_stats['episode_reward'] += reward
                episode_stats['episode_steps'] += 1
                obs = next_obs
        
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
    
    json_path = os.path.join(args.output_dir, 'comadice_evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 结果已保存到: {json_path}")
    
    print("\n评估完成！")


if __name__ == '__main__':
    main()


















