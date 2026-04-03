"""
Population MADDPG模型评估脚本

使用示例：
    python evaluate_population_maddpg.py --checkpoint_path ./checkpoints/population_maddpg/best_model.pt --num_episodes 200
"""
import argparse
import numpy as np
import torch
from pettingzoo.classic import hanabi_v5
from train_population_maddpg import PopulationMADDPG, evaluate_episode


def evaluate_episode_deterministic(env, model, population_idx=None):
    """
    确定性评估episode（使用argmax，完全无随机性）
    
    注意：这个函数确保评估时使用完全确定性的策略（argmax），
    而不是概率采样，以保证评估结果的可重复性。
    """
    env.reset()
    episode_reward = 0.0
    episode_steps = 0
    done = False
    
    # 获取初始观察
    obs, _, _, _, _ = env.last()
    state = obs["observation"]
    
    # 选择要使用的population
    if population_idx is None:
        population_idx = model.best_population_idx
    
    while not done:
        # 获取动作掩码
        action_mask = obs.get("action_mask", None) if isinstance(obs, dict) else None
        
        # 使用确定性策略选择动作（argmax，而不是随机采样）
        # 获取当前智能体的状态
        agent_idx = 0  # Hanabi是轮流行动，当前总是第一个智能体
        actor = model.populations[population_idx].actors[agent_idx]
        
        actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(model.device)
            probs = actor.forward(state_tensor)[0]  # [action_dim]
            
            # 应用动作掩码（如果提供）
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(model.device)
                probs = probs * mask_tensor
                # 重新归一化
                probs = probs / (probs.sum() + 1e-8)
            
            # 确定性选择：使用argmax而不是multinomial
            action = torch.argmax(probs).item()
            
            # 验证动作是否有效
            if action_mask is not None and action_mask[action] == 0:
                # 如果argmax的动作无效，从有效动作中选择概率最高的
                valid_actions = np.where(action_mask > 0)[0]
                if len(valid_actions) > 0:
                    valid_probs = probs[valid_actions].cpu().numpy()
                    best_valid_idx = np.argmax(valid_probs)
                    action = valid_actions[best_valid_idx]
                else:
                    action = 0
        
        env.step(action)
        
        # 获取奖励和下一个观察
        obs, reward, terminated, truncated, info = env.last()
        done = terminated or truncated
        episode_reward += reward
        episode_steps += 1
        
        if not done:
            next_state = obs["observation"]
            state = next_state
    
    # 获取最终得分
    final_score = info.get('score', episode_reward) if isinstance(info, dict) else episode_reward
    
    return {
        'episode_reward': episode_reward,
        'final_score': final_score,
        'episode_steps': episode_steps,
    }


def main():
    parser = argparse.ArgumentParser(description='评估Population MADDPG模型')
    
    parser.add_argument('--checkpoint_path', type=str, 
                       default='/root/autodl-tmp/covmadt/ma/ho/h/checkpoints/population_maddpg/best_model.pt',
                       help='模型检查点路径')
    parser.add_argument('--num_episodes', type=int, default=200,
                       help='评估episode数量')
    parser.add_argument('--population_size', type=int, default=3,
                       help='Population大小（需与训练时一致）')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='隐藏层维度（需与训练时一致）')
    parser.add_argument('--players', type=int, default=4,
                       help='Hanabi环境玩家数量')
    parser.add_argument('--max_life_tokens', type=int, default=6,
                       help='Hanabi环境最大生命令牌数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--population_idx', type=int, default=None,
                       help='指定评估的population索引（None表示使用最佳population）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("Population MADDPG 模型评估")
    print("=" * 60)
    print(f"模型路径: {args.checkpoint_path}")
    print(f"评估episode数: {args.num_episodes}")
    print(f"Population大小: {args.population_size}")
    print(f"隐藏层维度: {args.hidden_dim}")
    print(f"设备: {device}")
    if args.population_idx is not None:
        print(f"指定Population索引: {args.population_idx}")
    else:
        print(f"使用最佳Population")
    print("=" * 60)
    
    # 初始化环境以获取维度信息
    print("\n初始化环境...")
    env = hanabi_v5.env(players=args.players, max_life_tokens=args.max_life_tokens)
    env.reset(seed=args.seed)
    
    obs, _, _, _, _ = env.last()
    obs_dim = len(obs["observation"])
    act_dim = env.action_space(env.agents[0]).n
    num_agents = len(env.agents)
    
    print(f"观察维度: {obs_dim}")
    print(f"动作维度: {act_dim}")
    print(f"智能体数量: {num_agents}")
    
    # 创建模型
    print("\n创建模型...")
    model = PopulationMADDPG(
        state_dim=obs_dim,
        action_dim=act_dim,
        n_agents=num_agents,
        population_size=args.population_size,
        hidden_dim=args.hidden_dim,
        device=device,
    )
    
    # 加载检查点
    print(f"\n加载模型: {args.checkpoint_path}")
    try:
        model.load_checkpoint(args.checkpoint_path)
        print(f"✓ 模型加载成功")
        print(f"  最佳Population索引: {model.best_population_idx}")
        print(f"  Population分数: {model.population_scores}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 评估
    print(f"\n开始评估（{args.num_episodes} 个episode）...")
    print("=" * 60)
    
    episode_rewards = []
    episode_scores = []
    episode_steps = []
    
    for episode in range(args.num_episodes):
        # 创建新环境（每个episode使用不同的种子）
        env = hanabi_v5.env(players=args.players, max_life_tokens=args.max_life_tokens)
        env.reset(seed=args.seed + episode)
        
        # 评估episode（使用确定性策略：argmax，而不是随机采样）
        # 注意：evaluate_episode 使用 epsilon=0.0，但 Actor.select_action 仍可能使用随机采样
        # 为了确保完全确定性，我们需要修改评估逻辑
        stats = evaluate_episode_deterministic(
            env, 
            model, 
            population_idx=args.population_idx  # 如果指定，使用该population；否则使用最佳population
        )
        
        episode_rewards.append(stats['episode_reward'])
        episode_scores.append(stats['final_score'])
        episode_steps.append(stats['episode_steps'])
        
        env.close()
        
        # 每10个episode显示进度
        if (episode + 1) % 10 == 0:
            current_mean = np.mean(episode_scores)
            current_std = np.std(episode_scores)
            print(f"Episode {episode + 1}/{args.num_episodes}: "
                  f"平均得分 = {current_mean:.2f} ± {current_std:.2f}")
    
    # 统计结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总episode数: {args.num_episodes}")
    print(f"\n奖励统计:")
    print(f"  平均奖励: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"  最小奖励: {np.min(episode_rewards):.4f}")
    print(f"  最大奖励: {np.max(episode_rewards):.4f}")
    print(f"\n得分统计:")
    print(f"  平均得分: {np.mean(episode_scores):.4f} ± {np.std(episode_scores):.4f}")
    print(f"  最小得分: {np.min(episode_scores):.4f}")
    print(f"  最大得分: {np.max(episode_scores):.4f}")
    print(f"  中位数得分: {np.median(episode_scores):.4f}")
    print(f"\n步数统计:")
    print(f"  平均步数: {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")
    print(f"  最小步数: {np.min(episode_steps)}")
    print(f"  最大步数: {np.max(episode_steps)}")
    print("=" * 60)
    
    # 保存结果
    import json
    import os
    from datetime import datetime
    
    results = {
        'checkpoint_path': args.checkpoint_path,
        'num_episodes': args.num_episodes,
        'population_size': args.population_size,
        'hidden_dim': args.hidden_dim,
        'best_population_idx': model.best_population_idx,
        'evaluated_population_idx': args.population_idx,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'statistics': {
            'rewards': {
                'mean': float(np.mean(episode_rewards)),
                'std': float(np.std(episode_rewards)),
                'min': float(np.min(episode_rewards)),
                'max': float(np.max(episode_rewards)),
            },
            'scores': {
                'mean': float(np.mean(episode_scores)),
                'std': float(np.std(episode_scores)),
                'min': float(np.min(episode_scores)),
                'max': float(np.max(episode_scores)),
                'median': float(np.median(episode_scores)),
            },
            'steps': {
                'mean': float(np.mean(episode_steps)),
                'std': float(np.std(episode_steps)),
                'min': int(np.min(episode_steps)),
                'max': int(np.max(episode_steps)),
            },
        },
        'all_scores': episode_scores,
        'all_rewards': episode_rewards,
        'all_steps': episode_steps,
    }
    
    # 保存到文件
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    results_path = os.path.join(checkpoint_dir, f'evaluation_results_{results["timestamp"]}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n评估结果已保存到: {results_path}")


if __name__ == '__main__':
    main()

