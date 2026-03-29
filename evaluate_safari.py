"""
SAFARI算法评估脚本

评估训练好的SAFARI模型
"""
import argparse
import os
import torch
import numpy as np
from pettingzoo.classic import hanabi_v5

from algorithms.safari import SAFARI


def evaluate_episode(env, model, device, deterministic=True):
    """评估一个episode"""
    env.reset(seed=42)
    
    episode_reward = 0.0
    episode_length = 0
    done = False
    
    while not done:
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation
            
            if done:
                env.step(None)
                break
            
            # 处理观察
            if isinstance(obs, dict):
                state = np.array(obs["observation"], dtype=np.float32)
            else:
                state = np.array(obs, dtype=np.float32)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, S]
            
            # 预测动作
            with torch.no_grad():
                action, _, _ = model.predict_action(
                    state_tensor,
                    deterministic=deterministic,
                )
                action = action.item()
            
            # 执行动作
            env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
    
    return {
        'reward': episode_reward,
        'length': episode_length,
    }


def main():
    parser = argparse.ArgumentParser(description='评估SAFARI模型')
    
    # 模型参数
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--env_name', type=str, default='hanabi_v5',
                       help='环境名称')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='评估episode数')
    parser.add_argument('--deterministic', action='store_true',
                       help='使用确定性策略')
    
    # 环境参数
    parser.add_argument('--players', type=int, default=4,
                       help='玩家数量（Hanabi）')
    parser.add_argument('--max_life_tokens', type=int, default=6,
                       help='最大生命令牌（Hanabi）')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("SAFARI 模型评估")
    print("=" * 60)
    print(f"检查点路径: {args.checkpoint_path}")
    print(f"环境: {args.env_name}")
    print(f"评估episode数: {args.num_episodes}")
    print(f"确定性策略: {args.deterministic}")
    print("=" * 60)
    
    # 加载检查点
    print("\n加载模型检查点...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    print(f"观察维度: {config['state_dim']}")
    print(f"动作维度: {config['action_dim']}")
    print(f"智能体数量: {config['n_agents']}")
    
    # 创建模型
    model = SAFARI(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        n_agents=config['n_agents'],
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        gamma=config['gamma'],
        beta=config['beta'],
        alpha=config['alpha'],
        device=device,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ 模型加载成功")
    
    # 创建环境
    print(f"\n创建环境: {args.env_name}")
    if args.env_name == 'hanabi_v5':
        env = hanabi_v5.env(
            players=args.players,
            max_life_tokens=args.max_life_tokens,
        )
    else:
        raise ValueError(f"不支持的环境: {args.env_name}")
    
    # 评估
    print(f"\n开始评估（{args.num_episodes} episodes）...")
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(args.num_episodes):
        stats = evaluate_episode(env, model, device, args.deterministic)
        episode_rewards.append(stats['reward'])
        episode_lengths.append(stats['length'])
        
        print(f"Episode {episode + 1}: 奖励 = {stats['reward']:.2f}, 长度 = {stats['length']}")
    
    env.close()
    
    # 统计结果
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"平均长度: {avg_length:.2f}")
    print(f"最佳奖励: {np.max(episode_rewards):.2f}")
    print(f"最差奖励: {np.min(episode_rewards):.2f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
















