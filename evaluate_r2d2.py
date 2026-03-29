"""
评估R2D2模型性能（记录Hanabi游戏指标）
使用模块化评估器
支持评估+微调模式
"""
import argparse
import os
import torch
import numpy as np
from pettingzoo.classic import hanabi_v5
from tqdm import tqdm

from main import DEVICE, R2D2Net, select_action, SequenceReplayBuffer
from evaluators.r2d2_evaluator import R2D2Evaluator, load_r2d2_model



def main():
    parser = argparse.ArgumentParser(description='评估R2D2模型性能（记录Hanabi指标）')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='R2D2模型检查点路径')
    
    # 评估参数
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='评估episode数量')
    parser.add_argument('--epsilon', type=float, default=0.0,
                       help='探索率（0表示完全贪婪，0.05表示5%%探索）')
    parser.add_argument('--env_name', type=str, default='hanabi_v5',
                       help='环境名称')
    
    # 微调参数
    parser.add_argument('--enable_finetune', action='store_true',
                       help='在评估过程中进行微调')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='微调学习率（仅在启用微调时使用）')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='训练批量大小（episode数量，仅在启用微调时使用）')
    parser.add_argument('--replay_buffer_size', type=int, default=5000,
                       help='经验回放缓冲区大小（仅在启用微调时使用）')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子（仅在启用微调时使用）')
    parser.add_argument('--target_update_freq', type=int, default=200,
                       help='目标网络更新频率（episode数，仅在启用微调时使用）')
    parser.add_argument('--save_checkpoint_freq', type=int, default=50,
                       help='（已弃用）不再保存定期检查点，只保留最佳和最终模型')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./evaluation_results/r2d2',
                       help='结果保存目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='检查点保存目录（仅在启用微调时使用）')
    parser.add_argument('--save_detailed', action='store_true',
                       help='保存每个episode的详细指标')
    parser.add_argument('--device', type=str, default=None,
                       help='设备（cuda/cpu，默认: 自动检测）')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = DEVICE
    
    print("=" * 60)
    print("R2D2 模型评估")
    print("=" * 60)
    print(f"模型检查点: {args.checkpoint}")
    print(f"环境: {args.env_name}")
    print(f"设备: {device}")
    print(f"Episode数: {args.num_episodes}")
    print(f"探索率: {args.epsilon}")
    if args.enable_finetune:
        print(f"微调模式: 启用")
        print(f"  学习率: {args.learning_rate}")
        print(f"  批量大小: {args.batch_size}")
        print(f"  折扣因子: {args.gamma}")
    else:
        print(f"微调模式: 禁用（纯评估）")
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
    print(f"\n加载R2D2模型: {args.checkpoint}")
    net = load_r2d2_model(args.checkpoint, obs_dim, act_dim, device)
    
    # 如果启用微调，创建目标网络和优化器
    if args.enable_finetune:
        net.train()  # 训练模式
        target_net = R2D2Net(obs_dim, act_dim).to(device)
        target_net.load_state_dict(net.state_dict())
        target_net.eval()
        
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
        replay_buffer = SequenceReplayBuffer(capacity=args.replay_buffer_size)
        
        print(f"✓ 模型加载成功（评估+微调模式）")
        print(f"  学习率: {args.learning_rate}")
        print(f"  批量大小: {args.batch_size}")
        print(f"  探索率: {args.epsilon}")
    else:
        net.eval()  # 评估模式
        print("✓ 模型加载成功（评估模式）")
        target_net = None
        optimizer = None
        replay_buffer = None
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    if args.enable_finetune:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 开始评估（可能包含微调）
    mode_str = "评估+微调" if args.enable_finetune else "评估"
    print(f"\n开始{mode_str} {args.num_episodes} 个episodes...")
    print("=" * 60)
    
    # 创建评估器（用于评估指标）
    evaluator = R2D2Evaluator(
        model=net,
        env=env,
        device=device,
        epsilon=args.epsilon,
    )
    
    all_episode_stats = []
    best_score = float('-inf')
    
    # 手动评估循环（支持微调）
    for episode in tqdm(range(1, args.num_episodes + 1), desc=f"{mode_str}进度"):
        if args.enable_finetune:
            # 评估+微调模式：收集episode数据并训练
            episode_stats, episode_data = evaluate_episode_with_data(
                env, net, device, args.epsilon
            )
            
            # 存储到回放缓冲区
            replay_buffer.add(episode_data)
            
            # 训练（如果缓冲区有足够数据）
            train_metrics = {}
            if len(replay_buffer) >= args.batch_size:
                batch = replay_buffer.sample(args.batch_size)
                train_loss = train_step_r2d2(
                    net, target_net, optimizer, batch, device, args.gamma
                )
                train_metrics = {"train_loss": train_loss}
                episode_stats.update(train_metrics)
            
            # 更新目标网络
            if episode % args.target_update_freq == 0:
                target_net.load_state_dict(net.state_dict())
        else:
            # 纯评估模式
            episode_stats = evaluator.evaluate_episode()
        
        all_episode_stats.append(episode_stats)
        
        # 每10个episode打印一次进度
        if episode % 10 == 0:
            recent_scores = [s['final_score'] for s in all_episode_stats[-10:]]
            avg_score = np.mean(recent_scores)
            print(f"\nEpisode {episode}: 最近10个episode平均得分 = {avg_score:.2f}")
            if args.enable_finetune and train_metrics:
                print(f"  训练损失: {train_metrics.get('train_loss', 0):.4f}")
        
        # 保存最佳模型（如果启用微调）
        if args.enable_finetune:
            if episode_stats['final_score'] > best_score:
                best_score = episode_stats['final_score']
                best_path = os.path.join(args.checkpoint_dir, "best_r2d2_eval_model_4.pt")
                save_r2d2_checkpoint(net, target_net, optimizer, episode, args.epsilon, 
                                    obs_dim, act_dim, best_path)
    
    env.close()
    
    # 保存最终模型（如果启用微调）
    if args.enable_finetune:
        final_path = os.path.join(args.checkpoint_dir, "final_r2d2_eval_model_4.pt")
        save_r2d2_checkpoint(net, target_net, optimizer, args.num_episodes, args.epsilon,
                            obs_dim, act_dim, final_path)
        print(f"\n✓ 最终模型已保存到: {final_path}")
    
    # 计算统计摘要
    stats_summary = evaluator.compute_statistics(all_episode_stats)
    stats_summary['checkpoint'] = args.checkpoint
    stats_summary['epsilon'] = args.epsilon
    
    # 打印统计摘要
    evaluator.print_summary(stats_summary)
    
    # 保存结果
    saved_files = evaluator.save_results(
        all_episode_stats,
        stats_summary,
        args.output_dir,
        save_detailed=args.save_detailed,
    )
    
    # 生成图表
    evaluator.plot_results(all_episode_stats, args.output_dir)
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print(f"结果保存在: {args.output_dir}")
    print("=" * 60)
    
    return stats_summary


def evaluate_episode_with_data(env, net, device, epsilon=0.0):
    """评估一个episode并返回指标和数据（用于微调）"""
    env.reset()
    
    episode_stats = {
        'final_score': 0,
        'life_tokens_lost': 0,
        'information_tokens_used': 0,
        'total_hints_given': 0,
        'total_discards': 0,
        'total_plays': 0,
        'is_perfect_game': False,
        'episode_reward': 0,
        'episode_steps': 0,
        'actions_taken': [],
    }
    
    # 用于训练的数据
    episode_data = []
    
    hidden_states = {}
    
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, _ = env.last()
        done = terminated or truncated
        
        if done:
            env.step(None)
            # 获取最终统计信息
            final_stats = get_game_statistics(env, cumulative_reward=episode_stats['episode_reward'])
            episode_stats.update(final_stats)
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
        
        # 记录动作
        episode_stats['actions_taken'].append(action)
        
        # 存储经验（用于训练）
        episode_data.append((
            obs["observation"],
            action,
            reward
        ))
        
        # 执行动作
        env.step(action)
        episode_stats['episode_reward'] += reward
        episode_stats['episode_steps'] += 1
    
    # 如果episode结束时没有获取到统计信息，再次尝试
    if episode_stats['final_score'] == 0:
        final_stats = get_game_statistics(env, cumulative_reward=episode_stats['episode_reward'])
        episode_stats.update(final_stats)
    
    # 如果仍然无法获取，使用累计奖励作为得分
    if episode_stats['final_score'] == 0 and episode_stats['episode_reward'] > 0:
        episode_stats['final_score'] = int(episode_stats['episode_reward'])
        episode_stats['is_perfect_game'] = (episode_stats['final_score'] == 25)
    
    # 计算信息效率
    if episode_stats['information_tokens_used'] > 0:
        episode_stats['information_efficiency'] = (
            episode_stats['final_score'] / episode_stats['information_tokens_used']
        )
    else:
        episode_stats['information_efficiency'] = episode_stats['final_score']
    
    # 计算风险控制能力
    max_life = 6  # 4个智能体环境使用max_life=6
    episode_stats['life_loss_rate'] = (
        episode_stats['life_tokens_lost'] / max_life if max_life > 0 else 0
    )
    episode_stats['risk_control_score'] = 1.0 - episode_stats['life_loss_rate']
    
    return episode_stats, episode_data


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
    
    try:
        # 尝试从环境获取状态
        if hasattr(env, 'aec_env') and hasattr(env.aec_env, 'env'):
            raw_env = env.aec_env.env
            if hasattr(raw_env, 'state'):
                state = raw_env.state
                
                # 获取最终得分
                if hasattr(state, 'score'):
                    stats['final_score'] = state.score
                    stats['is_perfect_game'] = (state.score == 25)
                
                # 获取生命令牌
                if hasattr(state, 'life_tokens'):
                    life_tokens = state.life_tokens
                    max_life = getattr(raw_env, 'max_life_tokens', 6)
                    stats['life_tokens_lost'] = max(0, max_life - life_tokens)
                
                # 获取信息令牌
                if hasattr(state, 'information_tokens'):
                    information_tokens = state.information_tokens
                    max_information = getattr(raw_env, 'max_information_tokens', 8)
                    stats['information_tokens_used'] = max(0, max_information - information_tokens)
                
                # 尝试获取动作历史
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
    except Exception:
        pass
    
    # 如果无法获取得分，使用累计奖励
    if stats['final_score'] == 0 and cumulative_reward > 0:
        stats['final_score'] = int(cumulative_reward)
        stats['is_perfect_game'] = (stats['final_score'] == 25)
    
    return stats


def train_step_r2d2(net, target_net, optimizer, batch, device, gamma=0.99):
    """执行一个R2D2训练步骤"""
    import torch.nn.functional as F
    
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
        loss = F.mse_loss(q_taken, targets)
        total_loss += loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    optimizer.step()
    
    return total_loss.item() / len(batch)


def save_r2d2_checkpoint(net, target_net, optimizer, episode, eps, obs_dim, act_dim, path):
    """保存R2D2检查点"""
    checkpoint = {
        'episode': episode,
        'net_state_dict': net.state_dict(),
        'target_net_state_dict': target_net.state_dict() if target_net is not None else None,
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'eps': eps,
        'obs_dim': obs_dim,
        'act_dim': act_dim,
    }
    torch.save(checkpoint, path)


if __name__ == "__main__":
    main()

