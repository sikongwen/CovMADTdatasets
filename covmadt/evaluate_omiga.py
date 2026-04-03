"""
OMIGA算法评估脚本

评估训练好的OMIGA模型性能
"""
import argparse
import os
import torch
import numpy as np
import json
import csv
from tqdm import tqdm

from algorithms.omiga import OMIGA

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
        
        # 验证动作是否合法
        if action_mask is not None and action_mask[action] == 0:
            # 如果动作不合法，从合法动作中随机选择
            valid_actions = np.where(action_mask > 0)[0]
            if len(valid_actions) > 0:
                action = int(np.random.choice(valid_actions))
            else:
                # 如果没有合法动作，使用动作0
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


def save_episode_rewards(all_episode_stats, output_dir, current_episode, append_mode=False):
    """
    保存episode奖励到JSON、NPY和图像文件
    
    参数:
        all_episode_stats: 所有episode的统计信息列表（可能只包含最近的数据）
        output_dir: 输出目录
        current_episode: 当前episode编号
        append_mode: 是否为追加模式（True=追加到CSV，False=覆盖所有文件）
        
    返回:
        (json_path, csv_path, npy_path, image_path, total_episodes): 保存的文件路径和总episode数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果之前有保存的文件，读取并合并所有数据
    npy_path = os.path.join(output_dir, 'episode_rewards.npy')
    all_episode_rewards = []
    all_episode_scores = []
    
    # 读取之前保存的数据（如果存在）
    if append_mode and os.path.exists(npy_path):
        try:
            existing_data = np.load(npy_path)
            # 获取已有数据的最大episode编号
            max_existing_episode = int(max(existing_data['episode'])) if len(existing_data) > 0 else 0
            
            # 计算当前内存中的数据对应的episode范围
            start_episode_in_memory = current_episode - len(all_episode_stats) + 1
            
            # 只添加新数据（episode编号大于已有数据的）
            new_rewards = []
            new_scores = []
            for i, stats in enumerate(all_episode_stats):
                episode_num = start_episode_in_memory + i
                if episode_num > max_existing_episode:
                    new_rewards.append(stats['episode_reward'])
                    new_scores.append(stats.get('final_score', 0))
            
            # 合并旧数据和新数据
            if len(existing_data) > 0:
                old_rewards = existing_data['reward'].tolist()
                old_scores = existing_data['score'].tolist()
                all_episode_rewards = old_rewards + new_rewards
                all_episode_scores = old_scores + new_scores
            else:
                all_episode_rewards = new_rewards
                all_episode_scores = new_scores
                
            if len(new_rewards) > 0:
                print(f"  🔄 合并数据: 已有{len(existing_data)}个episode，新增{len(new_rewards)}个episode (episode {max_existing_episode+1} 到 {current_episode})")
        except Exception as e:
            print(f"⚠️  警告: 读取之前的数据失败: {e}，将只保存当前数据")
            all_episode_rewards = [s['episode_reward'] for s in all_episode_stats]
            all_episode_scores = [s.get('final_score', 0) for s in all_episode_stats]
    else:
        # 第一次保存，使用所有当前数据
        all_episode_rewards = [s['episode_reward'] for s in all_episode_stats]
        all_episode_scores = [s.get('final_score', 0) for s in all_episode_stats]
    
    # 转换为numpy数组
    episode_rewards = np.array(all_episode_rewards, dtype=np.float32)
    episode_scores = np.array(all_episode_scores, dtype=np.float32)
    episodes = np.arange(1, len(episode_rewards) + 1, dtype=np.int32)
    
    # 创建奖励数据
    rewards_data = {
        'current_episode': current_episode,
        'total_episodes': len(episode_rewards),
        'episode_rewards': episode_rewards.tolist(),
        'episode_scores': episode_scores.tolist(),
        'statistics': {
            'mean_reward': float(np.mean(episode_rewards)) if len(episode_rewards) > 0 else 0.0,
            'std_reward': float(np.std(episode_rewards)) if len(episode_rewards) > 0 else 0.0,
            'min_reward': float(np.min(episode_rewards)) if len(episode_rewards) > 0 else 0.0,
            'max_reward': float(np.max(episode_rewards)) if len(episode_rewards) > 0 else 0.0,
            'mean_score': float(np.mean(episode_scores)) if len(episode_scores) > 0 else 0.0,
            'std_score': float(np.std(episode_scores)) if len(episode_scores) > 0 else 0.0,
            'min_score': float(np.min(episode_scores)) if len(episode_scores) > 0 else 0.0,
            'max_score': float(np.max(episode_scores)) if len(episode_scores) > 0 else 0.0,
        }
    }
    
    # 1. 保存到JSON文件
    json_path = os.path.join(output_dir, 'episode_rewards.json')
    with open(json_path, 'w') as f:
        json.dump(rewards_data, f, indent=2)
    
    # 2. 保存到NPY文件
    npy_data = np.array(
        list(zip(episodes, episode_rewards, episode_scores)),
        dtype=[('episode', np.int32), ('reward', np.float32), ('score', np.float32)]
    )
    np.save(npy_path, npy_data)
    
    # 3. 保存CSV格式
    csv_path = os.path.join(output_dir, 'episode_rewards.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'score'])
        for i, (reward, score) in enumerate(zip(episode_rewards, episode_scores)):
            writer.writerow([i + 1, float(reward), float(score)])
    
    # 4. 保存图像
    image_path = None
    if HAS_PLOTTING and len(episode_rewards) > 0:
        image_path = os.path.join(output_dir, 'episode_rewards.png')
        try:
            plt.clf()
            plt.close('all')
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            # 绘制移动平均线
            if len(episode_rewards) >= 100:
                window_size = 100
                moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
                moving_episodes = episodes[window_size-1:]
                ax.plot(moving_episodes, moving_avg, color='red', 
                        label=f'Moving Average ({window_size})', linewidth=2)
            
            # 添加均值线
            mean_reward = np.mean(episode_rewards)
            ax.axhline(y=mean_reward, color='green', linestyle='--', 
                      label=f'Mean: {mean_reward:.2f}', linewidth=1.5)
            
            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Reward', fontsize=12)
            ax.set_title(f'Episode Rewards (Total: {len(episode_rewards)} episodes)', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        except Exception as e:
            print(f"⚠️  警告: 无法保存图像: {e}")
            image_path = None
    
    return json_path, csv_path, npy_path, image_path, len(episode_rewards)


def main():
    parser = argparse.ArgumentParser(description='OMIGA算法评估')
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
    parser.add_argument('--save_rewards_freq', type=int, default=100,
                       help='每N个episode保存一次奖励数据（默认：100）')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    args.output_dir = os.path.abspath(args.output_dir)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 输出目录: {args.output_dir}")
    
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
            num_agents = len(env.agents)
            
            print(f"Hanabi环境配置: {num_agents}个智能体, max_life_tokens={max_life_tokens}")
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
            
            num_agents = 1  # Gym环境通常是单智能体
        except Exception as e:
            print(f"错误: 无法创建环境: {e}")
            return
    
    # 先加载检查点以获取配置
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 检查点文件不存在: {args.checkpoint_path}")
        return
    
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    
    # 从检查点读取配置（如果存在），否则使用命令行参数
    if 'config' in checkpoint:
        config = checkpoint['config']
        hidden_dim = config.get('hidden_dim', args.hidden_dim)
        num_layers = config.get('num_layers', args.num_layers)
        checkpoint_n_agents = config.get('n_agents', num_agents)
        
        # 如果检查点中有n_agents，使用检查点中的值
        if checkpoint_n_agents != num_agents:
            print(f"警告: 检查点中的n_agents ({checkpoint_n_agents}) 与环境不匹配 ({num_agents})")
            print(f"使用检查点中的n_agents: {checkpoint_n_agents}")
            num_agents = checkpoint_n_agents
        
        # 如果检查点中没有num_layers，尝试从模型结构推断
        if num_layers is None:
            # 尝试从state_dict推断层数
            policy_keys = [k for k in checkpoint.get('model_state_dict', {}).keys() if 'policy_net.policy_net' in k]
            if policy_keys:
                # 计算层数：每层有3个组件（Linear, LayerNorm, ReLU），最后一层只有Linear
                # 格式：policy_net.policy_net.{i}.weight
                max_idx = max([int(k.split('.')[2]) for k in policy_keys if k.split('.')[2].isdigit()])
                # 估算：如果有索引0,3,6，说明有3层（0是第一层，3是第二层，6是第三层）
                num_layers = (max_idx // 3) + 1
                print(f"从模型结构推断 num_layers={num_layers}")
            else:
                num_layers = args.num_layers
                print(f"无法推断num_layers，使用默认值: {num_layers}")
        
        print(f"从检查点读取配置: hidden_dim={hidden_dim}, num_layers={num_layers}, n_agents={num_agents}")
        
        # 检查state_dim和action_dim是否匹配
        checkpoint_state_dim = config.get('state_dim', state_dim)
        checkpoint_action_dim = config.get('action_dim', action_dim)
        
        if checkpoint_state_dim != state_dim:
            print(f"警告: 检查点中的state_dim ({checkpoint_state_dim}) 与环境不匹配 ({state_dim})")
            print(f"使用检查点中的state_dim: {checkpoint_state_dim}")
            state_dim = checkpoint_state_dim
        
        if checkpoint_action_dim != action_dim:
            print(f"警告: 检查点中的action_dim ({checkpoint_action_dim}) 与环境不匹配 ({action_dim})")
            print(f"使用检查点中的action_dim: {checkpoint_action_dim}")
            action_dim = checkpoint_action_dim
    else:
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        print(f"检查点中没有配置信息，使用命令行参数: hidden_dim={hidden_dim}, num_layers={num_layers}, n_agents={num_agents}")
    
    # 创建模型（使用从检查点读取的配置）
    print("Creating OMIGA model...")
    model = OMIGA(
        state_dim=state_dim,
        action_dim=action_dim,
        n_agents=num_agents,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=args.device,
    )
    
    # 加载检查点权重
    model.load_checkpoint(args.checkpoint_path)
    model.eval()
    print("✓ 模型加载成功")
    
    # 评估
    print(f"\n开始评估 ({args.num_episodes} episodes)...")
    print(f"📊 每 {args.save_rewards_freq} 个episode保存一次数据")
    all_episode_stats = []
    max_stats_history = 2000  # 内存中保留的最大episode数
    
    for episode in tqdm(range(1, args.num_episodes + 1)):
        if is_hanabi:
            episode_stats = evaluate_episode_hanabi(env, model, args.device, args.deterministic)
        else:
            # Gym环境评估
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
        
        # 限制内存中的数据量
        if len(all_episode_stats) > max_stats_history:
            all_episode_stats = all_episode_stats[-max_stats_history:]
        
        # 每10个episode打印一次进度
        if episode % 10 == 0:
            recent_rewards = [s['episode_reward'] for s in all_episode_stats[-10:]]
            avg_reward = np.mean(recent_rewards)
            print(f"\nEpisode {episode}: 最近10个episode平均奖励 = {avg_reward:.2f}")
            if 'final_score' in episode_stats:
                recent_scores = [s.get('final_score', 0) for s in all_episode_stats[-10:]]
                avg_score = np.mean(recent_scores)
                print(f"  最近10个episode平均得分 = {avg_score:.2f}")
            if 'information_efficiency' in episode_stats:
                recent_efficiencies = [s.get('information_efficiency', 0.0) for s in all_episode_stats[-10:] 
                                     if s.get('information_tokens_used', 0) > 0]
                if len(recent_efficiencies) > 0:
                    avg_efficiency = np.mean(recent_efficiencies)
                    print(f"  最近10个episode平均信息效率 = {avg_efficiency:.3f}")
        
        # 定期保存数据
        if episode % args.save_rewards_freq == 0:
            append_mode = episode > args.save_rewards_freq
            json_path, csv_path, npy_path, image_path, total_episodes = save_episode_rewards(
                all_episode_stats, args.output_dir, episode, append_mode=append_mode
            )
            print(f"\n💾 Episode {episode}: 已保存数据到 {args.output_dir}")
            print(f"   - JSON: {json_path}")
            print(f"   - CSV: {csv_path}")
            print(f"   - NPY: {npy_path}")
            if image_path:
                print(f"   - 图像: {image_path}")
            print(f"   - 总episode数: {total_episodes}")
    
    env.close()
    
    # 最终保存所有数据
    print("\n💾 保存最终评估结果...")
    append_mode = args.num_episodes > args.save_rewards_freq
    json_path, csv_path, npy_path, image_path, total_episodes = save_episode_rewards(
        all_episode_stats, args.output_dir, args.num_episodes, append_mode=append_mode
    )
    
    # 计算统计指标
    print("\n" + "=" * 60)
    print("评估结果统计")
    print("=" * 60)
    
    # 从保存的文件中读取完整数据以计算统计
    try:
        if os.path.exists(npy_path):
            saved_data = np.load(npy_path)
            episode_rewards = saved_data['reward']
            episode_scores = saved_data['score']
        else:
            episode_rewards = [s['episode_reward'] for s in all_episode_stats]
            episode_scores = [s.get('final_score', 0) for s in all_episode_stats]
    except:
        episode_rewards = [s['episode_reward'] for s in all_episode_stats]
        episode_scores = [s.get('final_score', 0) for s in all_episode_stats]
    
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
    
    if len(episode_scores) > 0 and any(s > 0 for s in episode_scores):
        perfect_games = sum(1 for s in episode_scores if s == 25)
        
        print(f"\n最终得分统计 (Hanabi):")
        print(f"  平均得分: {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}")
        print(f"  最小得分: {np.min(episode_scores)}")
        print(f"  最大得分: {np.max(episode_scores)}")
        print(f"  中位数得分: {np.median(episode_scores):.2f}")
        print(f"  完美游戏数: {perfect_games}/{len(episode_scores)} ({100*perfect_games/len(episode_scores):.1f}%)")
    
    # 信息效率统计（Hanabi环境）
    if 'information_efficiency' in all_episode_stats[0] if len(all_episode_stats) > 0 else False:
        information_efficiencies = [s.get('information_efficiency', 0.0) for s in all_episode_stats]
        information_tokens_used = [s.get('information_tokens_used', 0) for s in all_episode_stats]
        total_hints = [s.get('total_hints_given', 0) for s in all_episode_stats]
        
        # 过滤掉无效值（信息效率为0且信息令牌使用数为0的情况）
        valid_efficiencies = [eff for eff, tokens in zip(information_efficiencies, information_tokens_used) 
                             if tokens > 0 or eff > 0]
        
        if len(valid_efficiencies) > 0:
            print(f"\n信息效率统计 (Information Efficiency):")
            print(f"  平均信息效率: {np.mean(valid_efficiencies):.3f} ± {np.std(valid_efficiencies):.3f}")
            print(f"  最小信息效率: {np.min(valid_efficiencies):.3f}")
            print(f"  最大信息效率: {np.max(valid_efficiencies):.3f}")
            print(f"  中位数信息效率: {np.median(valid_efficiencies):.3f}")
            print(f"  平均信息令牌使用数: {np.mean(information_tokens_used):.2f} ± {np.std(information_tokens_used):.2f}")
            print(f"  平均提示动作数: {np.mean(total_hints):.2f} ± {np.std(total_hints):.2f}")
    
    # 保存详细结果JSON
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
    }
    
    if len(episode_scores) > 0 and any(s > 0 for s in episode_scores):
        results['statistics']['mean_score'] = float(np.mean(episode_scores))
        results['statistics']['std_score'] = float(np.std(episode_scores))
        results['statistics']['min_score'] = int(np.min(episode_scores))
        results['statistics']['max_score'] = int(np.max(episode_scores))
        results['statistics']['median_score'] = float(np.median(episode_scores))
        results['statistics']['perfect_games'] = perfect_games
        results['statistics']['perfect_game_rate'] = float(perfect_games / len(episode_scores))
    
    # 添加信息效率统计
    if 'information_efficiency' in all_episode_stats[0] if len(all_episode_stats) > 0 else False:
        information_efficiencies = [s.get('information_efficiency', 0.0) for s in all_episode_stats]
        information_tokens_used = [s.get('information_tokens_used', 0) for s in all_episode_stats]
        total_hints = [s.get('total_hints_given', 0) for s in all_episode_stats]
        
        # 过滤掉无效值
        valid_efficiencies = [eff for eff, tokens in zip(information_efficiencies, information_tokens_used) 
                             if tokens > 0 or eff > 0]
        
        if len(valid_efficiencies) > 0:
            results['statistics']['information_efficiency_mean'] = float(np.mean(valid_efficiencies))
            results['statistics']['information_efficiency_std'] = float(np.std(valid_efficiencies))
            results['statistics']['information_efficiency_min'] = float(np.min(valid_efficiencies))
            results['statistics']['information_efficiency_max'] = float(np.max(valid_efficiencies))
            results['statistics']['information_efficiency_median'] = float(np.median(valid_efficiencies))
            results['statistics']['information_tokens_used_mean'] = float(np.mean(information_tokens_used))
            results['statistics']['information_tokens_used_std'] = float(np.std(information_tokens_used))
            results['statistics']['total_hints_mean'] = float(np.mean(total_hints))
            results['statistics']['total_hints_std'] = float(np.std(total_hints))
    
    detailed_json_path = os.path.join(args.output_dir, 'omiga_evaluation_results.json')
    with open(detailed_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 详细结果已保存到: {detailed_json_path}")
    print(f"✓ 奖励数据已保存到: {json_path}")
    print(f"✓ CSV数据已保存到: {csv_path}")
    print(f"✓ NPY数据已保存到: {npy_path}")
    if image_path:
        print(f"✓ 图像已保存到: {image_path}")
    
    print("\n评估完成！")


if __name__ == '__main__':
    main()

