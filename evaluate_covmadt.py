"""
评估CovMADT模型（记录Hanabi游戏指标）
"""
import argparse
import os
import torch
import numpy as np
import json
from pettingzoo.classic import hanabi_v5
from tqdm import tqdm
from collections import defaultdict

from algorithms.covmadt import CovMADT
from data.replay_buffer import ReplayBuffer
try:
    from model_change_detector import ModelChangeDetector
    HAS_CHANGE_DETECTOR = True
except ImportError:
    HAS_CHANGE_DETECTOR = False
    print("⚠️  警告: 无法导入 ModelChangeDetector，将跳过模型变化检测")
from model_change_detector import ModelChangeDetector


def save_episode_rewards(all_episode_stats, output_dir, current_episode, append_mode=False):
    """
    保存episode奖励到JSON、NPY和图像文件
    
    参数:
        all_episode_stats: 所有episode的统计信息列表（可能只包含最近的数据）
        output_dir: 输出目录
        current_episode: 当前episode编号
        append_mode: 是否为追加模式（True=追加到CSV，False=覆盖所有文件）
        
    返回:
        (json_path, csv_path, npy_path, image_path): 保存的文件路径
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
            # all_episode_stats 只包含最近的数据，需要根据 current_episode 推断
            # 假设 all_episode_stats 包含从 (current_episode - len(all_episode_stats) + 1) 到 current_episode 的数据
            start_episode_in_memory = current_episode - len(all_episode_stats) + 1
            
            # 只添加新数据（episode编号大于已有数据的）
            new_rewards = []
            new_scores = []
            for i, stats in enumerate(all_episode_stats):
                episode_num = start_episode_in_memory + i
                if episode_num > max_existing_episode:
                    new_rewards.append(stats['episode_reward'])
                    new_scores.append(stats['final_score'])
            
            # 合并旧数据和新数据
            if len(existing_data) > 0:
                old_rewards = existing_data['reward'].tolist()
                old_scores = existing_data['score'].tolist()
                all_episode_rewards = old_rewards + new_rewards
                all_episode_scores = old_scores + new_scores
            else:
                # 如果旧数据为空，只使用新数据
                all_episode_rewards = new_rewards
                all_episode_scores = new_scores
                
            # 调试信息
            if len(new_rewards) > 0:
                print(f"  🔄 合并数据: 已有{len(existing_data)}个episode，新增{len(new_rewards)}个episode (episode {max_existing_episode+1} 到 {current_episode})")
        except Exception as e:
            print(f"⚠️  警告: 读取之前的数据失败: {e}，将只保存当前数据")
            import traceback
            traceback.print_exc()
            # 如果读取失败，只使用当前数据
            all_episode_rewards = [s['episode_reward'] for s in all_episode_stats]
            all_episode_scores = [s['final_score'] for s in all_episode_stats]
    else:
        # 第一次保存，使用所有当前数据
        all_episode_rewards = [s['episode_reward'] for s in all_episode_stats]
        all_episode_scores = [s['final_score'] for s in all_episode_stats]
    
    # 转换为numpy数组
    episode_rewards = np.array(all_episode_rewards, dtype=np.float32)
    episode_scores = np.array(all_episode_scores, dtype=np.float32)
    episodes = np.arange(1, len(episode_rewards) + 1, dtype=np.int32)
    
    # 创建奖励数据（使用合并后的总数据）
    rewards_data = {
        'current_episode': current_episode,
        'total_episodes': len(episode_rewards),  # 使用合并后的总数据长度
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
    
    # 1. 保存到JSON文件（总是覆盖，包含最新统计）
    json_path = os.path.join(output_dir, 'episode_rewards.json')
    with open(json_path, 'w') as f:
        json.dump(rewards_data, f, indent=2)
    
    # 2. 保存到NPY文件（总是覆盖，包含所有数据）
    npy_path = os.path.join(output_dir, 'episode_rewards.npy')
    # 创建结构化数组
    npy_data = np.array(
        list(zip(episodes, episode_rewards, episode_scores)),
        dtype=[('episode', np.int32), ('reward', np.float32), ('score', np.float32)]
    )
    np.save(npy_path, npy_data)
    
    # 3. 保存CSV格式（总是覆盖，包含所有合并后的数据）
    csv_path = os.path.join(output_dir, 'episode_rewards.csv')
    import csv
    
    # 总是覆盖模式，写入所有合并后的数据（因为我们已经从NPY文件合并了所有数据）
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'score'])
        for i, (reward, score) in enumerate(zip(episode_rewards, episode_scores)):
            writer.writerow([i + 1, float(reward), float(score)])
    
    # 4. 保存图像（如果matplotlib可用）
    image_path = None
    if HAS_PLOTTING and len(episode_rewards) > 0:
        image_path = os.path.join(output_dir, 'episode_rewards.png')
        try:
            # 清除之前的图形状态，确保每次都是新图
            plt.clf()
            plt.close('all')
            
            # 只绘制一个子图（移动平均和均值线）
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            # 只绘制移动平均线（不绘制原始reward曲线）
            if len(episode_rewards) >= 100:
                # 计算移动平均（窗口大小固定为100）
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
            # 强制保存并刷新
            plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # 验证文件是否真的被更新了
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                if file_size == 0:
                    print(f"⚠️  警告: 图像文件大小为0，可能保存失败")
                # 不打印文件大小，避免输出过多
        except Exception as e:
            print(f"⚠️  警告: 无法保存图像: {e}")
            import traceback
            traceback.print_exc()
            image_path = None
    
    return json_path, csv_path, npy_path, image_path, len(episode_rewards)


# 设置绘图库
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    # 设置中文字体（如果需要）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
except ImportError:
    HAS_PLOTTING = False
    print("警告: matplotlib/seaborn未安装，将跳过绘图功能")
from data.replay_buffer import ReplayBuffer


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


def evaluate_episode_hanabi(env, model, device, deterministic=True, collect_data=False, 
                             print_interaction=False, episode_num=0, print_top_k=5, epsilon=0.1):
    """
    评估一个episode并返回指标，可选择收集数据用于训练
    
    参数:
        env: 环境
        model: 模型
        device: 设备
        deterministic: 是否使用确定性策略
        collect_data: 是否收集数据用于训练
        print_interaction: 是否打印详细的交互信息
        episode_num: episode编号（用于打印）
        print_top_k: 打印Top-K动作的数量
        epsilon: epsilon-greedy探索率（仅在collect_data=True且deterministic=False时使用）
    """
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
    
    # 如果收集数据，存储经验（包含PPO需要的old_log_probs）
    episode_data = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": [],
        "old_log_probs": [],  # PPO需要旧的对数概率
    } if collect_data else None
    
    # 跟踪动作历史
    action_history = []
    life_tokens_start = 3  # Hanabi默认3个生命令牌
    information_tokens_start = 8  # Hanabi默认8个信息令牌
    
    # 用于打印的统计信息
    step_count = 0
    cumulative_reward = 0.0
    
    # 动作类型统计（用于监控和强制通信）
    action_type_stats = {
        'play': 0,      # 打牌动作（假设0-9）
        'hint': 0,      # 提示动作（假设10-29）
        'discard': 0,   # 弃牌动作（假设30-39）
        'other': 0,     # 其他动作
    }
    last_hint_step = -10  # 记录最后一次提示的步数
    hint_interval = 5     # 每5步至少一次提示
    
    # 打印episode开始信息
    # 只在启用打印交互信息时显示episode开始信息
    # if print_interaction:
    #     print(f"\n{'='*80}")
    #     print(f"🎮 Episode {episode_num} 开始")
    #     print(f"{'='*80}")
    
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        done = terminated or truncated
        
        if done:
            env.step(None)
            # 获取最终统计信息（传入累计奖励）
            final_stats = get_game_statistics(env, cumulative_reward=episode_stats['episode_reward'])
            episode_stats.update(final_stats)
            break
        
        # 获取观察和动作掩码
        state = obs["observation"]
        action_mask = obs["action_mask"]
        
        # 预测动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(device)
            
            # 使用模型预测动作
            actions, log_probs, info = model.predict_action(
                states=state_tensor,
                mask=mask_tensor,
                deterministic=deterministic,
            )
            
            # 提取单个元素并转换为int（避免NumPy deprecation warning）
            action = int(actions.cpu().numpy().item())
            # 提取对数概率（用于PPO）
            old_log_prob = float(log_probs.cpu().numpy().item()) if log_probs.numel() > 0 else 0.0
            
            # 获取动作概率分布（用于智能探索）
            action_dist = info.get("action_dist", None)
            action_probs_np = None
            if action_dist is not None:
                action_probs_np = action_dist.squeeze().cpu().numpy()
            
            # Epsilon-greedy探索（仅在非确定性且收集数据时）
            # 如果collect_data=True且deterministic=False，使用epsilon-greedy
            # 改进：探索时优先选择概率高的动作，而不是完全随机
            # 当epsilon很小时，更倾向于选择概率最高的动作（贪婪策略）
            if collect_data and not deterministic and epsilon > 0:
                if np.random.random() < epsilon:
                    valid_actions = np.where(action_mask > 0)[0]
                    if len(valid_actions) > 0:
                        if action_probs_np is not None:
                            # 获取有效动作的概率
                            valid_probs = action_probs_np[valid_actions]
                            # 归一化有效动作的概率
                            valid_probs = valid_probs / (valid_probs.sum() + 1e-8)
                            
                            # 根据epsilon大小决定探索策略
                            # epsilon < 0.1: 直接选择概率最高的动作（贪婪）
                            # epsilon >= 0.1: 根据概率分布采样（概率高的更容易被选中）
                            if epsilon < 0.1:
                                # 小epsilon：直接选择概率最高的动作
                                best_action_idx = np.argmax(valid_probs)
                                action = int(valid_actions[best_action_idx])
                                old_log_prob = np.log(valid_probs[best_action_idx] + 1e-8)
                            else:
                                # 大epsilon：根据概率分布采样（概率高的更容易被选中）
                                action = int(np.random.choice(valid_actions, p=valid_probs))
                                # 找到选中动作在valid_actions中的索引
                                action_idx = np.where(valid_actions == action)[0]
                                if len(action_idx) > 0:
                                    old_log_prob = np.log(valid_probs[action_idx[0]] + 1e-8)
                                else:
                                    old_log_prob = np.log(1.0 / len(valid_actions))
                        else:
                            # 如果没有概率信息，回退到均匀随机
                            action = int(np.random.choice(valid_actions))
                            old_log_prob = np.log(1.0 / len(valid_actions))
            
            # 安全检查：确保选择的动作始终在有效动作列表中
            # 防止模型预测或epsilon-greedy选择非法动作
            valid_actions = np.where(action_mask > 0)[0]
            if action not in valid_actions:
                # 如果动作无效，从有效动作中随机选择一个
                if len(valid_actions) > 0:
                    action = int(np.random.choice(valid_actions))
                    old_log_prob = np.log(1.0 / len(valid_actions))
                else:
                    # 如果没有有效动作（不应该发生），使用动作0作为fallback
                    action = 0
                    old_log_prob = 0.0
            
            # 通信强制机制：每N步至少一次提示动作
            # 假设动作10-29是提示动作（需要根据实际Hanabi动作空间调整）
            if collect_data:
                hint_action_range = (10, 29)  # 假设的提示动作范围
                steps_since_last_hint = step_count - last_hint_step
                
                # 如果超过hint_interval步没有提示，且当前动作不是提示
                if (steps_since_last_hint >= hint_interval and 
                    not (hint_action_range[0] <= action <= hint_action_range[1])):
                    # 查找有效的提示动作
                    valid_hint_actions = [a for a in valid_actions 
                                        if hint_action_range[0] <= a <= hint_action_range[1]]
                    if len(valid_hint_actions) > 0:
                        # 强制选择提示动作
                        action = int(np.random.choice(valid_hint_actions))
                        old_log_prob = np.log(1.0 / len(valid_hint_actions))
                        last_hint_step = step_count
                        if print_interaction:
                            print(f"  🔔 强制通信：已超过{hint_interval}步未提示，强制选择提示动作 {action}")
            
            # 更新动作类型统计
            if action < 10:
                action_type_stats['play'] += 1
            elif 10 <= action <= 29:
                action_type_stats['hint'] += 1
                last_hint_step = step_count
            elif 30 <= action <= 39:
                action_type_stats['discard'] += 1
            else:
                action_type_stats['other'] += 1
            
            # 提取详细信息用于打印
            logits = info.get("logits", None)
            values = info.get("values", None)
            action_dist = info.get("action_dist", None)
            
            # 转换为numpy用于打印（初始化默认值）
            logits_np = None
            action_probs_np = None
            value_np = 0.0
            entropy = 0.0
            
            if logits is not None:
                logits_np = logits.squeeze().cpu().numpy()
            if action_dist is not None:
                action_probs_np = action_dist.squeeze().cpu().numpy()
                # 计算策略熵（探索性指标）
                probs = action_probs_np
                # 只考虑有效动作（概率>0）
                valid_probs = probs[probs > 1e-8]
                if len(valid_probs) > 0:
                    entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-8))
            if values is not None:
                value_np = float(values.squeeze().cpu().numpy().item()) if values.numel() > 0 else 0.0
        
        # 记录动作
        action_history.append(action)
        episode_stats['actions_taken'].append(action)
        step_count += 1
        cumulative_reward += reward
        
        # 打印交互信息
        if print_interaction:
            print(f"\n📊 Step {step_count} (Agent: {agent})")
            if action_probs_np is not None:
                action_prob = action_probs_np[action] if action < len(action_probs_np) else 0.0
                print(f"  🎯 选择的动作: {action} (概率: {action_prob:.4f}, log_prob: {old_log_prob:.4f})")
            else:
                print(f"  🎯 选择的动作: {action} (log_prob: {old_log_prob:.4f})")
            print(f"  💰 价值估计: {value_np:.4f}")
            print(f"  🎁 即时奖励: {reward:.4f} | 累计奖励: {cumulative_reward:.4f}")
            print(f"  🔀 策略熵: {entropy:.4f} (探索性指标，越高越探索)")
            
            # 打印Top-K动作
            if action_probs_np is not None and logits_np is not None:
                top_k_indices = np.argsort(action_probs_np)[-print_top_k:][::-1]
                print(f"  📈 Top-{print_top_k} 动作:")
                for rank, idx in enumerate(top_k_indices, 1):
                    marker = "✓" if idx == action else " "
                    prob_str = f"{action_probs_np[idx]:.4f}" if idx < len(action_probs_np) else "N/A"
                    logit_str = f"{logits_np[idx]:.4f}" if idx < len(logits_np) else "N/A"
                    print(f"    {marker} {rank}. 动作 {idx}: 概率={prob_str}, logit={logit_str}")
            
            # 打印动作掩码信息
            valid_actions = np.where(action_mask > 0)[0]
            print(f"  ✅ 有效动作数: {len(valid_actions)}/{len(action_mask)}")
            if len(valid_actions) <= 10:
                print(f"  ✅ 有效动作列表: {valid_actions.tolist()}")
        
        # 如果收集数据，存储当前状态和旧的对数概率（PPO需要）
        if collect_data:
            episode_data["states"].append(state)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["dones"].append(done)
            episode_data["old_log_probs"] = episode_data.get("old_log_probs", [])
            episode_data["old_log_probs"].append(old_log_prob)
        
        # 奖励塑形：鼓励使用提示动作和动作多样性
        reward_shaping = 0.0
        
        # 1. 提示动作奖励（假设动作10-29是提示动作）
        if collect_data and 10 <= action <= 29:
            reward_shaping += 0.1  # 鼓励使用提示动作
            reward += reward_shaping
        
        # 2. 动作多样性奖励（鼓励探索不同动作类型）
        if collect_data and step_count > 0:
            total_actions = sum(action_type_stats.values())
            if total_actions > 0:
                # 计算动作类型多样性（熵）
                type_probs = [count / total_actions for count in action_type_stats.values() if count > 0]
                diversity_entropy = -sum(p * np.log(p + 1e-8) for p in type_probs)
                # 多样性奖励（熵越高，奖励越多）
                diversity_bonus = 0.02 * diversity_entropy / np.log(len(action_type_stats))  # 归一化到[0, 0.02]
                reward += diversity_bonus
                reward_shaping += diversity_bonus
        
        # 执行动作
        env.step(action)
        episode_stats['episode_reward'] += reward
        episode_stats['episode_steps'] += 1
        
        # 获取下一个观察（下一个智能体的观察）
        if collect_data:
            next_obs = None
            if env.agent_selection:  # 如果还有下一个智能体
                next_obs, _, _, _, _ = env.last()
                next_state = next_obs["observation"]
            else:
                # Episode结束，使用当前状态
                next_state = state
            episode_data["next_states"].append(next_state)
    
    # 如果episode结束时没有获取到统计信息，再次尝试
    if episode_stats['final_score'] == 0:
        final_stats = get_game_statistics(env, cumulative_reward=episode_stats['episode_reward'])
        episode_stats.update(final_stats)
    
    # 如果仍然无法获取，使用累计奖励作为得分
    if episode_stats['final_score'] == 0 and episode_stats['episode_reward'] > 0:
        episode_stats['final_score'] = int(episode_stats['episode_reward'])
        episode_stats['is_perfect_game'] = (episode_stats['final_score'] == 25)
    
    # 更新信息令牌使用数：使用提示动作数量（更准确）
    # 因为每次提示动作都会消耗1个信息令牌，而弃牌会恢复信息令牌
    # 所以使用提示动作数量比使用令牌差值更准确
    if episode_stats['total_hints_given'] > 0:
        episode_stats['information_tokens_used'] = episode_stats['total_hints_given']
    # 如果无法从动作历史获取提示数量，使用动作类型统计中的hint数量
    elif action_type_stats['hint'] > 0:
        episode_stats['information_tokens_used'] = action_type_stats['hint']
        episode_stats['total_hints_given'] = action_type_stats['hint']
    # 如果两种方法都失败，保持从get_game_statistics获取的值（可能是0）
    
    # 计算信息效率
    # 信息效率 = 最终得分 / (信息令牌使用数 + 1) 或使用其他定义
    if episode_stats['information_tokens_used'] > 0:
        episode_stats['information_efficiency'] = episode_stats['final_score'] / episode_stats['information_tokens_used']
    else:
        episode_stats['information_efficiency'] = episode_stats['final_score']
    
    # 计算风险控制能力（生命损失率）
    # 风险控制 = 1 - (生命损失数 / 最大生命数)
    max_life = 6  # 4个智能体环境使用max_life=6
    episode_stats['life_loss_rate'] = episode_stats['life_tokens_lost'] / max_life if max_life > 0 else 0
    episode_stats['risk_control_score'] = 1.0 - episode_stats['life_loss_rate']
    
    # 计算动作类型统计和多样性（在episode结束前）
    total_actions = sum(action_type_stats.values())
    if total_actions > 0:
        type_probs = [count / total_actions for count in action_type_stats.values() if count > 0]
        diversity_entropy = -sum(p * np.log(p + 1e-8) for p in type_probs)
        max_entropy = np.log(len([c for c in action_type_stats.values() if c > 0]))
        diversity_score = diversity_entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        diversity_score = 0.0
    
    # 打印episode结束信息
    if print_interaction:
        print(f"\n{'='*80}")
        print(f"🏁 Episode {episode_num} 结束")
        print(f"  📊 总步数: {episode_stats['episode_steps']}")
        print(f"  🎯 最终得分: {episode_stats['final_score']}")
        print(f"  🎁 累计奖励: {episode_stats['episode_reward']:.4f}")
        print(f"  💎 完美游戏: {'是' if episode_stats['is_perfect_game'] else '否'}")
        print(f"  ❤️  生命损失: {episode_stats['life_tokens_lost']}")
        print(f"  💡 信息令牌使用: {episode_stats['information_tokens_used']}")
        print(f"  📈 信息效率: {episode_stats.get('information_efficiency', 0):.4f}")
        print(f"  🎲 动作序列长度: {len(action_history)}")
        
        # 打印动作类型分布
        if total_actions > 0:
            print(f"  📊 动作类型分布:")
            print(f"     🎴 打牌动作 (0-9): {action_type_stats['play']} ({100*action_type_stats['play']/total_actions:.1f}%)")
            print(f"     💬 提示动作 (10-29): {action_type_stats['hint']} ({100*action_type_stats['hint']/total_actions:.1f}%)")
            print(f"     🗑️  弃牌动作 (30-39): {action_type_stats['discard']} ({100*action_type_stats['discard']/total_actions:.1f}%)")
            print(f"     ❓ 其他动作: {action_type_stats['other']} ({100*action_type_stats['other']/total_actions:.1f}%)")
            print(f"     🔀 动作多样性: {diversity_score:.3f} (1.0=完全均匀, 0.0=完全集中)")
            
            # 警告：如果提示动作比例过低
            hint_ratio = action_type_stats['hint'] / total_actions
            if hint_ratio < 0.1:
                print(f"     ⚠️  警告: 提示动作比例过低 ({hint_ratio:.1%})，建议至少20-30%")
        
        if len(action_history) <= 20:
            print(f"  🎲 动作序列: {action_history}")
        print(f"{'='*80}\n")
    
    # 将动作类型统计添加到episode_stats
    episode_stats['action_type_stats'] = action_type_stats.copy()
    episode_stats['action_diversity'] = diversity_score
    if total_actions > 0:
        episode_stats['hint_ratio'] = action_type_stats['hint'] / total_actions
        episode_stats['play_ratio'] = action_type_stats['play'] / total_actions
        episode_stats['discard_ratio'] = action_type_stats['discard'] / total_actions
    else:
        episode_stats['hint_ratio'] = 0.0
        episode_stats['play_ratio'] = 0.0
        episode_stats['discard_ratio'] = 0.0
    
    # 返回数据和统计信息
    if collect_data:
        return episode_stats, episode_data
    else:
        return episode_stats


def train_step(
    model, 
    batch, 
    device, 
    optimizer, 
    update_target=True, 
    lambda_convex=0.1,
    lambda_rkhs=0.1,
    lambda_value=1.0,  # 价值损失权重（新增）
    use_ppo=False,
    clip_ratio=0.2,
    entropy_coef=0.01,
    normalize_advantages=False,
    use_gae=False,  # 新增：是否使用GAE
    gae_lambda=0.95,  # 新增：GAE lambda参数
    value_clip=10.0,
    grad_clip=0.3,
    grad_skip_threshold=3.0,
    grad_accumulation_steps=1,
    accumulation_step=0,
    change_detector=None,  # 添加检测器参数
):
    """
    执行一个训练步骤
    
    参数:
        model: CovMADT模型
        batch: 训练批次
        device: 设备
        optimizer: 优化器
        update_target: 是否更新目标网络
        lambda_convex: 凸正则化损失权重（微调时降低）
        lambda_rkhs: RKHS损失权重（设为0可禁用）
        use_ppo: 是否使用PPO策略损失
        clip_ratio: PPO裁剪比例
        entropy_coef: 熵正则化系数
    """
    model.train()
    
    # 准备批次数据
    states = torch.FloatTensor(batch["states"]).to(device)
    actions = torch.LongTensor(batch["actions"]).to(device)
    rewards = torch.FloatTensor(batch["rewards"]).to(device)
    next_states = torch.FloatTensor(batch["next_states"]).to(device)
    
    # 检查数据维度
    if states.dim() == 2:
        # 单步数据 [B, S]，需要添加序列维度
        states = states.unsqueeze(1)  # [B, 1, obs_dim]
        next_states = next_states.unsqueeze(1)  # [B, 1, obs_dim]
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)  # [B, 1]
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)  # [B, 1]
    # 如果已经是序列数据 [B, T, S]，直接使用
    
    # 将actions转换为one-hot编码
    action_dim = model.action_dim
    batch_size, seq_len = states.shape[0], states.shape[1]
    
    # 处理actions：可能是 [B, T] 或 [B]
    if actions.dim() == 1:
        actions = actions.unsqueeze(1)  # [B] -> [B, 1]
    if actions.dim() == 2 and actions.shape[1] != seq_len:
        # 如果actions是 [B]，需要扩展到 [B, T]
        actions = actions.unsqueeze(1).expand(-1, seq_len)  # [B] -> [B, 1] -> [B, T]
    
    # 创建one-hot编码 [B, T, A]
    actions_onehot = torch.zeros(batch_size, seq_len, action_dim).to(device)
    actions_flat = actions.view(-1)  # [B*T]
    actions_onehot_flat = actions_onehot.view(-1, action_dim)  # [B*T, A]
    actions_onehot_flat.scatter_(1, actions_flat.unsqueeze(1), 1.0)
    actions_onehot = actions_onehot_flat.view(batch_size, seq_len, action_dim)  # [B, T, A]
    
    # 处理rewards：确保是 [B, T, 1]
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1)  # [B] -> [B, 1]
    if rewards.dim() == 2 and rewards.shape[1] != seq_len:
        rewards = rewards.unsqueeze(1).expand(-1, seq_len)  # [B] -> [B, 1] -> [B, T]
    if rewards.dim() == 2:
        rewards = rewards.unsqueeze(-1)  # [B, T] -> [B, T, 1]
    
    # 前向传播
    # 诊断：检查 batch 形状（只在第一次打印）
    if accumulation_step == 0:
        if not hasattr(train_step, '_shape_printed'):
            print(f"🔍 诊断: batch 形状 - states: {states.shape}, actions: {actions.shape}, rewards: {rewards.shape}")
            print(f"🔍 诊断: 序列长度 T = {states.shape[1] if states.dim() >= 2 else 1}")
            train_step._shape_printed = True
    
    outputs = model(
        states=states,
        actions=actions_onehot,
        rewards=rewards,
        next_states=next_states,
        compute_loss=True,
    )
    
    # 计算损失
    loss_dict = outputs.get("loss_dict", {})
    
    # 诊断：检查 logits 形状（只在第一次打印）
    if accumulation_step == 0:
        if not hasattr(train_step, '_logits_printed'):
            policy_logits = outputs.get("logits")
            if policy_logits is not None:
                print(f"🔍 诊断: policy_logits.shape = {policy_logits.shape}")
            train_step._logits_printed = True
    
    # 获取序列长度
    batch_size, seq_len = states.shape[0], states.shape[1]
    
    # 先计算价值相关的内容（PPO需要这些来计算优势）
    values = outputs["values"]  # [B, T, 1] 或 [B, T] 或 [B, 1, 1] 或 [B, 1]
    
    with torch.no_grad():
        next_values = model.compute_value(next_states, use_target=True)
        # 处理 next_values 的维度
        if next_values.dim() == 3:
            if next_values.shape[2] == 1:
                next_values = next_values.squeeze(-1)  # [B, T, 1] -> [B, T]
            # 如果 next_values 是 [B, 1, 1]，需要扩展到 [B, T]
            elif next_values.shape[1] == 1:
                next_values = next_values.squeeze(1).unsqueeze(1).expand(-1, seq_len)  # [B, 1, 1] -> [B, T]
        elif next_values.dim() == 2 and next_values.shape[1] == 1:
            next_values = next_values.expand(-1, seq_len)  # [B, 1] -> [B, T]
        
        # 计算 TD 目标：r + γ * V(s')
        # rewards 的形状应该是 [B, T, 1] 或 [B, T]
        if rewards.dim() == 3:
            rewards_flat = rewards.squeeze(-1)  # [B, T, 1] -> [B, T]
        else:
            rewards_flat = rewards  # [B, T]
        
        target_values = rewards_flat + model.gamma * next_values  # [B, T]
        
        # 价值裁剪（防止价值爆炸）
        target_values = torch.clamp(target_values, -value_clip, value_clip)
    
    # 处理 values 的维度
    if values.dim() == 3:
        if values.shape[2] == 1:
            values = values.squeeze(-1)  # [B, T, 1] -> [B, T]
        # 如果 values 是 [B, 1, 1]，需要扩展到 [B, T]
        elif values.shape[1] == 1:
            values = values.squeeze(1).unsqueeze(1).expand(-1, seq_len)  # [B, 1, 1] -> [B, T]
    elif values.dim() == 2 and values.shape[1] == 1:
        values = values.expand(-1, seq_len)  # [B, 1] -> [B, T]
    
    # 价值裁剪（防止价值爆炸）
    values_clipped = torch.clamp(values, -value_clip, value_clip)
    target_values_clipped = torch.clamp(target_values, -value_clip, value_clip)
    
    # 使用Huber损失代替MSE（对异常值更鲁棒）
    value_loss = torch.nn.functional.smooth_l1_loss(values_clipped, target_values_clipped)
    
    # 策略损失（PPO或标准交叉熵）
    policy_logits = outputs["logits"]  # [B, 1, action_dim]
    
    if use_ppo and "old_log_probs" in batch:
        # PPO策略损失
        old_log_probs = torch.FloatTensor(batch["old_log_probs"]).to(device)  # [B, T] 或 [B]
        
        # 计算当前策略的对数概率
        log_probs = torch.nn.functional.log_softmax(policy_logits.view(-1, action_dim), dim=-1)  # [B*T, A]
        selected_log_probs = torch.gather(log_probs, 1, actions.view(-1, 1)).squeeze(-1)  # [B*T]
        
        # 处理 old_log_probs 的维度
        # batch 中的 old_log_probs 可能是 [B, T] 或 [B]
        if old_log_probs.dim() == 1:
            # 如果是单步数据 [B]，需要扩展到序列 [B, T]
            # 假设每个样本的 old_log_prob 适用于整个序列
            if old_log_probs.shape[0] == batch_size:
                # [B] -> [B, T]
                old_log_probs = old_log_probs.unsqueeze(1).expand(-1, seq_len).contiguous()
            else:
                # 如果形状不匹配，可能是 [B*T]，需要reshape
                if old_log_probs.shape[0] == batch_size * seq_len:
                    old_log_probs = old_log_probs.view(batch_size, seq_len)
                else:
                    # 其他情况，尝试reshape或扩展
                    old_log_probs = old_log_probs[:batch_size].unsqueeze(1).expand(-1, seq_len).contiguous()
        elif old_log_probs.dim() == 2:
            # 如果已经是 [B, T]，检查维度是否匹配
            if old_log_probs.shape[0] != batch_size or old_log_probs.shape[1] != seq_len:
                # 如果维度不匹配，尝试调整
                if old_log_probs.shape[0] == batch_size:
                    # 如果只有 batch_size 匹配，扩展或截断序列维度
                    if old_log_probs.shape[1] == 1:
                        old_log_probs = old_log_probs.expand(-1, seq_len)
                    elif old_log_probs.shape[1] > seq_len:
                        old_log_probs = old_log_probs[:, :seq_len]
                    else:
                        # 填充到 seq_len（使用最后一个值）
                        last_val = old_log_probs[:, -1:]
                        padding = last_val.expand(-1, seq_len - old_log_probs.shape[1])
                        old_log_probs = torch.cat([old_log_probs, padding], dim=1)
                else:
                    # 如果 batch_size 不匹配，截断或填充
                    if old_log_probs.shape[0] > batch_size:
                        old_log_probs = old_log_probs[:batch_size]
                    else:
                        padding = old_log_probs[-1:].expand(batch_size - old_log_probs.shape[0], -1)
                        old_log_probs = torch.cat([old_log_probs, padding], dim=0)
                    # 然后处理序列维度
                    if old_log_probs.shape[1] != seq_len:
                        if old_log_probs.shape[1] == 1:
                            old_log_probs = old_log_probs.expand(-1, seq_len)
                        elif old_log_probs.shape[1] > seq_len:
                            old_log_probs = old_log_probs[:, :seq_len]
                        else:
                            last_val = old_log_probs[:, -1:]
                            padding = last_val.expand(-1, seq_len - old_log_probs.shape[1])
                            old_log_probs = torch.cat([old_log_probs, padding], dim=1)
        
        old_log_probs = old_log_probs.view(-1)  # [B, T] -> [B*T]
        
        # 计算优势（使用TD误差或GAE）
        # target_values 和 values 的形状应该是 [B, T, 1] 或 [B, T]
        if target_values.dim() == 3:
            target_values = target_values.squeeze(-1)  # [B, T, 1] -> [B, T]
        if values.dim() == 3:
            values = values.squeeze(-1)  # [B, T, 1] -> [B, T]
        
        if use_gae:
            # 使用GAE计算优势
            # 需要从batch中获取dones信息
            dones = batch.get("dones", None)
            if dones is not None:
                dones_tensor = torch.FloatTensor(dones).to(device)  # [B, T] 或 [B]
                
                # 处理dones的维度
                if dones_tensor.dim() == 1:
                    # [B] -> [B, T]
                    dones_tensor = dones_tensor.unsqueeze(1).expand(-1, seq_len)
                elif dones_tensor.dim() == 2:
                    # [B, T] 确保维度匹配
                    if dones_tensor.shape[0] != batch_size or dones_tensor.shape[1] != seq_len:
                        if dones_tensor.shape[0] == batch_size:
                            if dones_tensor.shape[1] == 1:
                                dones_tensor = dones_tensor.expand(-1, seq_len)
                            elif dones_tensor.shape[1] > seq_len:
                                dones_tensor = dones_tensor[:, :seq_len]
                            else:
                                last_val = dones_tensor[:, -1:]
                                padding = last_val.expand(-1, seq_len - dones_tensor.shape[1])
                                dones_tensor = torch.cat([dones_tensor, padding], dim=1)
                        else:
                            # 截断或填充batch维度
                            if dones_tensor.shape[0] > batch_size:
                                dones_tensor = dones_tensor[:batch_size]
                            else:
                                padding = dones_tensor[-1:].expand(batch_size - dones_tensor.shape[0], -1)
                                dones_tensor = torch.cat([dones_tensor, padding], dim=0)
                            # 处理序列维度
                            if dones_tensor.shape[1] != seq_len:
                                if dones_tensor.shape[1] == 1:
                                    dones_tensor = dones_tensor.expand(-1, seq_len)
                                elif dones_tensor.shape[1] > seq_len:
                                    dones_tensor = dones_tensor[:, :seq_len]
                                else:
                                    last_val = dones_tensor[:, -1:]
                                    padding = last_val.expand(-1, seq_len - dones_tensor.shape[1])
                                    dones_tensor = torch.cat([dones_tensor, padding], dim=1)
                else:
                    # 如果dones维度不对，回退到TD误差
                    use_gae = False
                
                if use_gae:
                    # 准备GAE输入：需要 [T, B] 格式
                    # 转换维度：[B, T] -> [T, B]
                    rewards_gae = rewards_flat.permute(1, 0)  # [B, T] -> [T, B]
                    values_gae = values.permute(1, 0)  # [B, T] -> [T, B]
                    dones_gae = dones_tensor.permute(1, 0)  # [B, T] -> [T, B]
                    
                    # 计算下一个状态的价值（用于GAE）
                    # 对于序列的最后一步，使用目标网络计算next_value
                    with torch.no_grad():
                        # 获取序列的最后一个next_state
                        last_next_states = next_states[:, -1:, :]  # [B, 1, obs_dim]
                        last_next_values = model.compute_value(last_next_states, use_target=True)  # [B, 1, 1] 或 [B, 1]
                        if last_next_values.dim() == 3:
                            last_next_values = last_next_values.squeeze(-1)  # [B, 1, 1] -> [B, 1]
                        elif last_next_values.dim() == 2 and last_next_values.shape[1] == 1:
                            pass  # 已经是 [B, 1]
                        else:
                            last_next_values = last_next_values.unsqueeze(1)  # [B] -> [B, 1]
                    
                    # 扩展values以包含最后一个next_value
                    # GAE需要知道序列结束后的价值
                    values_with_next = torch.cat([values_gae, last_next_values.permute(1, 0)], dim=0)  # [T+1, B]
                    
                    # 计算GAE
                    from utils.math_utils import compute_gae
                    advantages_gae, returns_gae = compute_gae(
                        rewards=rewards_gae,  # [T, B]
                        values=values_with_next[:-1],  # [T, B] (不包括最后一个)
                        dones=dones_gae,  # [T, B]
                        gamma=model.gamma,
                        lambda_=gae_lambda,
                    )
                    
                    # 转换回 [B, T] 格式
                    advantages = advantages_gae.permute(1, 0)  # [T, B] -> [B, T]
                    advantages = advantages.detach()  # 分离梯度
                    advantages = advantages.view(-1)  # [B, T] -> [B*T]
                    
                    # 使用GAE计算的returns更新target_values（用于价值损失）
                    target_values = returns_gae.permute(1, 0)  # [T, B] -> [B, T]
            else:
                # 如果没有dones信息，回退到TD误差
                use_gae = False
        
        if not use_gae:
            # 使用TD误差计算优势（原始方法）
            advantages = (target_values - values).detach()  # [B, T]
            advantages = advantages.view(-1)  # [B, T] -> [B*T]
        
        # 优势归一化（减少方差）
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 重要性采样比率（防止数值不稳定）
        log_ratio = selected_log_probs - old_log_probs  # [B*T]
        # 裁剪log_ratio防止exp溢出
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
        ratio = torch.exp(log_ratio)  # [B]
        
        # PPO裁剪
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # 熵正则化（鼓励探索）
        probs = torch.nn.functional.softmax(policy_logits.view(-1, action_dim), dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # 熵下限机制：防止策略过度收敛（过度随机或过度确定）
        # 目标熵范围：1.5-2.5（平衡探索和利用）
        target_entropy_min = 1.5
        target_entropy_max = 2.5
        
        # 如果熵过低（<1.5），增加熵正则化权重
        if entropy.item() < target_entropy_min:
            # 动态调整熵系数：熵越低，系数越大
            entropy_coef_adjusted = entropy_coef * (1.0 + (target_entropy_min - entropy.item()) / target_entropy_min)
        # 如果熵过高（>2.5），减少熵正则化权重
        elif entropy.item() > target_entropy_max:
            # 动态调整熵系数：熵越高，系数越小
            entropy_coef_adjusted = entropy_coef * max(0.1, 1.0 - (entropy.item() - target_entropy_max) / target_entropy_max)
        else:
            entropy_coef_adjusted = entropy_coef
        
        entropy_bonus = entropy_coef_adjusted * entropy
    else:
        # 标准交叉熵损失
        policy_loss = torch.nn.functional.cross_entropy(
            policy_logits.view(-1, action_dim),
            actions.view(-1),
        )
        entropy_bonus = torch.tensor(0.0, device=device)
    
    # RKHS损失
    rkhs_loss = torch.tensor(0.0, device=device)
    if "next_state_pred" in outputs and outputs["next_state_pred"] is not None:
        next_state_pred = outputs["next_state_pred"]
        if next_state_pred.shape == next_states.shape:
            rkhs_loss = torch.nn.functional.mse_loss(next_state_pred, next_states)
    
    # 凸正则化损失（微调时降低权重）
    convex_loss = loss_dict.get("loss", torch.tensor(0.0, device=device))
    
    # 总损失（调整权重以适应在线微调）
    # 梯度累积：除以累积步数
    # 价值损失权重：默认1.0，如果value_loss >> policy_loss，建议增加到1.5-2.0
    total_loss = (
        1.0 * policy_loss +
        lambda_value * value_loss +  # 可配置的价值损失权重（新增）
        lambda_rkhs * rkhs_loss +  # 可配置的RKHS损失权重
        lambda_convex * convex_loss -  # 可配置的凸正则化权重
        entropy_bonus  # 熵正则化（PPO时使用）
    ) / grad_accumulation_steps
    
    # 反向传播（累积梯度）
    # 注意：第一次调用时需要zero_grad，后续累积时不需要
    if accumulation_step == 0:
        optimizer.zero_grad()
        # 诊断：保存更新前的参数（用于验证参数是否真的更新）
        # 优先选择有梯度的参数
        old_param_value = None
        old_param_name = None
        # 先尝试找transformer的参数
        for name, param in model.named_parameters():
            if param.requires_grad and 'transformer.transformer_blocks.0.attention.q_proj.weight' in name:
                old_param_value = param.data[0, 0].item()
                old_param_name = name
                break
        # 如果没找到，找第一个有梯度的参数
        if old_param_value is None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    old_param_value = param.data.flatten()[0].item()
                    old_param_name = name
                    break
    else:
        old_param_value = None
        old_param_name = None
    
    total_loss.backward()
    
    # 只在累积步数达到时才更新
    if (accumulation_step + 1) % grad_accumulation_steps == 0:
        # 诊断：计算梯度裁剪前的梯度范数
        grad_norm_before = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_before += param.grad.data.norm(2).item() ** 2
        grad_norm_before = grad_norm_before ** (1. / 2)
        
        # 分别裁剪策略梯度和价值梯度（防止单次更新破坏现有知识）
        # 策略梯度裁剪到0.5，价值梯度裁剪到1.0
        policy_grad_norm = 0.0
        value_grad_norm = 0.0
        
        # 分别处理策略网络和价值网络的梯度
        policy_params = []
        value_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if 'transformer' in name and 'value_head' not in name:
                    policy_params.append(param)
                elif 'value_head' in name or 'critic' in name:
                    value_params.append(param)
                else:
                    # 其他参数（如RKHS）归入策略网络
                    policy_params.append(param)
        
        # 裁剪策略梯度（更严格：0.5）
        if len(policy_params) > 0:
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(policy_params, 0.5)
        
        # 裁剪价值梯度（1.0）
        if len(value_params) > 0:
            value_grad_norm = torch.nn.utils.clip_grad_norm_(value_params, 1.0)
        
        # 总梯度范数（用于诊断）
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # 诊断：检查梯度是否存在
        has_grad = False
        param_count = 0
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count += 1
                if param.grad is not None:
                    has_grad = True
                    grad_count += 1
        
        # 检查梯度是否异常
        if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > grad_skip_threshold:
            # 静默跳过，只在严重异常时打印（避免输出过多）
            if grad_norm > grad_skip_threshold * 2:
                print(f"⚠️  警告: 梯度异常 (grad_norm={grad_norm:.2f})，跳过此更新")
            optimizer.zero_grad()  # 清零梯度，不更新
            skip_update = True
        else:
            # 在 zero_grad 之前保存梯度信息（用于检测器）
            if change_detector is not None:
                # 保存梯度信息（在 zero_grad 之前）
                grad_info = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_info[name] = param.grad.data.clone().cpu()
                    else:
                        grad_info[name] = None
                # 临时保存到检测器
                if not hasattr(change_detector, '_last_grad_info'):
                    change_detector._last_grad_info = {}
                change_detector._last_grad_info = grad_info
            
            # 诊断：验证参数是否真的更新
            optimizer.step()
            skip_update = False
            
            # 诊断：检查参数是否真的变化了
            if old_param_value is not None and old_param_name is not None:
                new_param_value = None
                for name, param in model.named_parameters():
                    if name == old_param_name:
                        if 'weight' in name and param.data.dim() >= 2:
                            new_param_value = param.data[0, 0].item()
                        else:
                            new_param_value = param.data.flatten()[0].item()
                        break
                if new_param_value is not None:
                    param_change = abs(new_param_value - old_param_value)
                    # 每次更新都检查，但只在有问题时打印
                    if param_change < 1e-8:
                        # 获取当前学习率
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"⚠️  警告: 参数几乎没有变化 ({param_change:.2e})")
                        print(f"   检查的参数: {old_param_name}")
                        print(f"   梯度裁剪前范数: {grad_norm_before:.6f}")
                        print(f"   梯度裁剪后范数: {grad_norm:.6f}")
                        print(f"   裁剪比例: {grad_norm / grad_norm_before if grad_norm_before > 0 else 0:.2%}")
                        print(f"   学习率: {current_lr:.2e}")
                        print(f"   预期更新幅度: {current_lr * grad_norm:.2e}")
                        print(f"   有梯度参数: {grad_count}/{param_count}")
                        
                        # 诊断：列出没有梯度的参数类型
                        no_grad_params = []
                        for name, param in model.named_parameters():
                            if param.requires_grad and param.grad is None:
                                no_grad_params.append(name)
                        if no_grad_params:
                            print(f"   ⚠️  以下参数没有梯度（前5个）:")
                            for name in no_grad_params[:5]:
                                print(f"      - {name}")
                        
                        # 诊断：列出有梯度的参数类型
                        has_grad_params = []
                        for name, param in model.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                grad_norm_param = param.grad.data.norm(2).item()
                                has_grad_params.append((name, grad_norm_param))
                        if has_grad_params:
                            print(f"   ✓ 有梯度的参数（前5个）:")
                            for name, grad_norm_param in has_grad_params[:5]:
                                print(f"      - {name}: 梯度范数={grad_norm_param:.6f}")
                        
                        if grad_norm_before > grad_clip:
                            print(f"   ⚠️  梯度被裁剪！原始梯度 ({grad_norm_before:.6f}) > 裁剪阈值 ({grad_clip})")
                        if grad_norm < 1e-6:
                            print(f"   ⚠️  梯度范数太小！可能是梯度消失问题")
                        if grad_count < param_count * 0.5:
                            print(f"   ⚠️  只有 {grad_count}/{param_count} ({grad_count/param_count:.1%}) 的参数有梯度！")
                    # 每100次更新打印一次正常诊断信息
                    elif (accumulation_step + 1) % (grad_accumulation_steps * 100) == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"🔍 诊断: 参数变化 = {param_change:.8f}, 梯度范数 = {grad_norm:.4f}, 学习率 = {current_lr:.2e}, 有梯度参数 = {grad_count}/{param_count}")
        
        # 清零梯度（为下一次累积做准备）
        optimizer.zero_grad()
    else:
        # 还在累积中，不更新
        grad_norm = torch.tensor(0.0, device=device)
        skip_update = False  # 不算跳过，只是还没到更新时机
    
    # 更新目标网络（可控制频率）
    if update_target:
        model.update_target_networks()
    
    metrics = {
        "total_loss": total_loss.item() if not skip_update else 0.0,
        "policy_loss": policy_loss.item() if not skip_update else 0.0,
        "value_loss": value_loss.item() if not skip_update else 0.0,
        "rkhs_loss": rkhs_loss.item() if isinstance(rkhs_loss, torch.Tensor) and not skip_update else 0.0,
        "convex_loss": convex_loss.item() if isinstance(convex_loss, torch.Tensor) and not skip_update else 0.0,
        "grad_norm": grad_norm.item() if 'grad_norm' in locals() else 0.0,
        "skip_update": skip_update,
    }
    
    if use_ppo:
        metrics["entropy"] = entropy_bonus.item() / entropy_coef if entropy_coef > 0 else 0.0
        if "old_log_probs" in batch and 'ratio' in locals():
            metrics["ppo_ratio_mean"] = ratio.mean().item()
            metrics["ppo_ratio_std"] = ratio.std().item()
            metrics["advantages_mean"] = advantages.mean().item() if 'advantages' in locals() else 0.0
            metrics["advantages_std"] = advantages.std().item() if 'advantages' in locals() else 0.0
            
            # 诊断：检查PPO ratio是否在合理范围内
            ratio_in_clip = ((ratio >= 1 - clip_ratio) & (ratio <= 1 + clip_ratio)).float().mean().item()
            metrics["ppo_ratio_in_clip"] = ratio_in_clip
            # 每100次更新打印一次诊断信息
            if (accumulation_step + 1) % (grad_accumulation_steps * 100) == 0:
                print(f"🔍 诊断: PPO ratio 在裁剪范围内比例 = {ratio_in_clip:.2%}, 优势均值 = {advantages.mean().item():.4f}")
                if ratio_in_clip > 0.95:
                    print(f"⚠️  警告: 几乎所有ratio都在裁剪范围内，裁剪可能太严格！")
    
    return metrics


def plot_evaluation_results(all_episode_stats, output_dir, enable_finetune=False):
    """绘制评估结果图表"""
    if not HAS_PLOTTING:
        print("跳过绘图（matplotlib/seaborn未安装）")
        return
    
    print("\n生成评估图表...")
    
    # 提取数据
    episodes = list(range(1, len(all_episode_stats) + 1))
    final_scores = [s['final_score'] for s in all_episode_stats]
    information_efficiencies = [s['information_efficiency'] for s in all_episode_stats]
    risk_control_scores = [s['risk_control_score'] for s in all_episode_stats]
    life_loss_rates = [s['life_loss_rate'] for s in all_episode_stats]
    episode_rewards = [s['episode_reward'] for s in all_episode_stats]
    
    # 如果有训练损失，也提取
    train_losses = []
    if enable_finetune:
        train_losses = [s.get('total_loss', 0) for s in all_episode_stats if 'total_loss' in s]
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Final Score趋势图
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(episodes, final_scores, alpha=0.6, linewidth=1.5, color='#2E86AB')
    # 添加移动平均线
    window = min(20, len(final_scores) // 5)
    if window > 1:
        moving_avg = np.convolve(final_scores, np.ones(window)/window, mode='valid')
        moving_episodes = episodes[window-1:]
        plt.plot(moving_episodes, moving_avg, linewidth=2, color='#A23B72', label=f'Moving Avg ({window})')
        plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Final Score')
    plt.title('Final Score Over Episodes')
    plt.grid(True, alpha=0.3)
    
    # 2. Final Score分布直方图
    ax2 = plt.subplot(3, 3, 2)
    plt.hist(final_scores, bins=min(25, len(set(final_scores))), edgecolor='black', alpha=0.7, color='#2E86AB')
    plt.axvline(np.mean(final_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_scores):.2f}')
    plt.axvline(np.median(final_scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(final_scores):.2f}')
    plt.xlabel('Final Score')
    plt.ylabel('Frequency')
    plt.title('Final Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Perfect Game Rate（累计）
    ax3 = plt.subplot(3, 3, 3)
    perfect_games = [s['is_perfect_game'] for s in all_episode_stats]
    cumulative_perfect = np.cumsum(perfect_games)
    cumulative_rate = cumulative_perfect / np.arange(1, len(perfect_games) + 1)
    plt.plot(episodes, cumulative_rate * 100, linewidth=2, color='#F18F01')
    plt.xlabel('Episode')
    plt.ylabel('Perfect Game Rate (%)')
    plt.title('Cumulative Perfect Game Rate')
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3)
    
    # 4. Information Efficiency趋势
    ax4 = plt.subplot(3, 3, 4)
    plt.plot(episodes, information_efficiencies, alpha=0.6, linewidth=1.5, color='#C73E1D')
    if window > 1:
        moving_avg = np.convolve(information_efficiencies, np.ones(window)/window, mode='valid')
        plt.plot(moving_episodes, moving_avg, linewidth=2, color='#A23B72', label=f'Moving Avg ({window})')
        plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Information Efficiency')
    plt.title('Information Efficiency Over Episodes')
    plt.grid(True, alpha=0.3)
    
    # 5. Risk Control Score趋势
    ax5 = plt.subplot(3, 3, 5)
    plt.plot(episodes, risk_control_scores, alpha=0.6, linewidth=1.5, color='#6A994E')
    if window > 1:
        moving_avg = np.convolve(risk_control_scores, np.ones(window)/window, mode='valid')
        plt.plot(moving_episodes, moving_avg, linewidth=2, color='#A23B72', label=f'Moving Avg ({window})')
        plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Risk Control Score')
    plt.title('Risk Control Score Over Episodes')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    
    # 6. Life Loss Rate趋势
    ax6 = plt.subplot(3, 3, 6)
    plt.plot(episodes, life_loss_rates, alpha=0.6, linewidth=1.5, color='#BC4749')
    if window > 1:
        moving_avg = np.convolve(life_loss_rates, np.ones(window)/window, mode='valid')
        plt.plot(moving_episodes, moving_avg, linewidth=2, color='#A23B72', label=f'Moving Avg ({window})')
        plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Life Loss Rate')
    plt.title('Life Loss Rate Over Episodes')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    
    # 7. 指标对比箱线图
    ax7 = plt.subplot(3, 3, 7)
    data_to_plot = [
        final_scores,
        [e * 10 for e in information_efficiencies],  # 缩放以便对比
        [s * 25 for s in risk_control_scores],  # 缩放以便对比
    ]
    labels = ['Final Score', 'Info Eff (×10)', 'Risk Ctrl (×25)']
    bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['#2E86AB', '#C73E1D', '#6A994E']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.ylabel('Value (Scaled)')
    plt.title('Metrics Comparison (Box Plot)')
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 8. 训练损失（如果启用微调）
    if enable_finetune and train_losses:
        ax8 = plt.subplot(3, 3, 8)
        train_episodes = [i+1 for i, s in enumerate(all_episode_stats) if 'total_loss' in s]
        plt.plot(train_episodes, train_losses, alpha=0.6, linewidth=1.5, color='#8B5A3C')
        if len(train_losses) > window:
            moving_avg = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            moving_train_episodes = train_episodes[window-1:]
            plt.plot(moving_train_episodes, moving_avg, linewidth=2, color='#A23B72', label=f'Moving Avg ({window})')
            plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Over Episodes')
        plt.grid(True, alpha=0.3)
    else:
        ax8 = plt.subplot(3, 3, 8)
        plt.text(0.5, 0.5, 'Training Loss\n(Not Available)', 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    # 9. 指标相关性热力图
    ax9 = plt.subplot(3, 3, 9)
    try:
        import pandas as pd
        metrics_df = {
            'Final Score': final_scores,
            'Info Efficiency': information_efficiencies,
            'Risk Control': risk_control_scores,
            'Life Loss Rate': life_loss_rates,
        }
        df = pd.DataFrame(metrics_df)
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax9)
        plt.title('Metrics Correlation')
    except ImportError:
        # 如果没有pandas，使用numpy计算相关性
        metrics_array = np.array([
            final_scores,
            information_efficiencies,
            risk_control_scores,
            life_loss_rates,
        ]).T
        corr = np.corrcoef(metrics_array.T)
        labels = ['Final Score', 'Info Eff', 'Risk Ctrl', 'Life Loss']
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, 
                    xticklabels=labels, yticklabels=labels, ax=ax9)
        plt.title('Metrics Correlation')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'evaluation_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 评估图表已保存到: {plot_path}")
    plt.close()
    
    # 创建单独的详细图表
    create_detailed_plots(all_episode_stats, output_dir, enable_finetune)


def create_detailed_plots(all_episode_stats, output_dir, enable_finetune=False):
    """创建更详细的单独图表"""
    if not HAS_PLOTTING:
        return
    
    episodes = list(range(1, len(all_episode_stats) + 1))
    final_scores = [s['final_score'] for s in all_episode_stats]
    
    # 1. Final Score详细趋势图（带置信区间）
    fig, ax = plt.subplots(figsize=(12, 6))
    window = min(20, len(final_scores) // 5)
    if window > 1:
        moving_avg = np.convolve(final_scores, np.ones(window)/window, mode='valid')
        moving_std = []
        for i in range(window-1, len(final_scores)):
            window_scores = final_scores[i-window+1:i+1]
            moving_std.append(np.std(window_scores))
        moving_episodes = episodes[window-1:]
        
        ax.plot(episodes, final_scores, alpha=0.3, linewidth=1, color='#2E86AB', label='Raw Scores')
        ax.plot(moving_episodes, moving_avg, linewidth=2, color='#A23B72', label=f'Moving Average ({window})')
        ax.fill_between(moving_episodes, 
                        np.array(moving_avg) - np.array(moving_std),
                        np.array(moving_avg) + np.array(moving_std),
                        alpha=0.2, color='#A23B72', label='±1 Std')
    else:
        ax.plot(episodes, final_scores, linewidth=1.5, color='#2E86AB')
    
    ax.axhline(np.mean(final_scores), color='red', linestyle='--', linewidth=2, 
               label=f'Overall Mean: {np.mean(final_scores):.2f}')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Final Score', fontsize=12)
    ax.set_title('Final Score Trend with Confidence Interval', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    detailed_path = os.path.join(output_dir, 'final_score_trend.png')
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 所有指标的综合趋势图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Final Score
    axes[0, 0].plot(episodes, final_scores, alpha=0.6, linewidth=1.5, color='#2E86AB')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Final Score')
    axes[0, 0].set_title('Final Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Information Efficiency
    info_eff = [s['information_efficiency'] for s in all_episode_stats]
    axes[0, 1].plot(episodes, info_eff, alpha=0.6, linewidth=1.5, color='#C73E1D')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Information Efficiency')
    axes[0, 1].set_title('Information Efficiency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Risk Control Score
    risk_ctrl = [s['risk_control_score'] for s in all_episode_stats]
    axes[1, 0].plot(episodes, risk_ctrl, alpha=0.6, linewidth=1.5, color='#6A994E')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Risk Control Score')
    axes[1, 0].set_title('Risk Control Score')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Life Loss Rate
    life_loss = [s['life_loss_rate'] for s in all_episode_stats]
    axes[1, 1].plot(episodes, life_loss, alpha=0.6, linewidth=1.5, color='#BC4749')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Life Loss Rate')
    axes[1, 1].set_title('Life Loss Rate')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comprehensive_path = os.path.join(output_dir, 'all_metrics_trend.png')
    plt.savefig(comprehensive_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 详细图表已保存到: {detailed_path}, {comprehensive_path}")


def main():
    parser = argparse.ArgumentParser(description='评估CovMADT模型（记录Hanabi指标）')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    
    # 评估参数
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='评估episode数量')
    parser.add_argument('--deterministic', action='store_true',
                       help='使用确定性策略（不采样）')
    parser.add_argument('--env_name', type=str, default=None,
                       help='环境名称（如果为None，将从检查点自动推断；仅支持 hanabi_v5）')
    
    # 微调参数
    parser.add_argument('--enable_finetune', action='store_true',
                       help='在评估过程中进行微调')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='微调学习率（仅在启用微调时使用，默认5e-5，比训练时小）')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'],
                       help='优化器类型（默认adam，可选adamw，adamw通常更稳定）')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='权重衰减系数（默认0.0，AdamW推荐0.01-0.1）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='训练批量大小（仅在启用微调时使用）')
    parser.add_argument('--replay_buffer_size', type=int, default=20000,
                       help='经验回放缓冲区大小（仅在启用微调时使用，默认20000）')
    parser.add_argument('--epsilon', type=float, default=0.01,
                       help='探索率（仅在启用微调时使用，默认0.01，保守探索）')
    parser.add_argument('--epsilon_start', type=float, default=None,
                       help='探索率起始值（如果指定，将使用epsilon衰减，默认None使用固定epsilon）')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                       help='探索率结束值（epsilon衰减的最小值，默认0.01）')
    parser.add_argument('--epsilon_decay', type=float, default=0.9995,
                       help='探索率衰减系数（每episode衰减，默认0.9995，即每1000步约衰减到60%%）')
    parser.add_argument('--deterministic_after_episode', type=int, default=None,
                       help='在指定episode后切换到确定性策略（默认None，不切换。例如：--deterministic_after_episode 1000 表示前1000个episode使用探索策略，之后使用确定性策略）')
    parser.add_argument('--train_freq', type=int, default=3,
                       help='每N个episode训练一次（默认3，避免更新太频繁）')
    parser.add_argument('--target_update_freq', type=int, default=5,
                       help='每N个batch更新一次目标网络（默认5，稳定训练）')
    parser.add_argument('--min_buffer_size', type=int, default=1000,
                       help='开始训练的最小缓冲区大小（默认1000）')
    parser.add_argument('--lambda_convex_finetune', type=float, default=0.1,
                       help='微调时凸正则化损失权重（默认0.1，降低对离线数据的依赖）')
    parser.add_argument('--lambda_rkhs_finetune', type=float, default=0.1,
                       help='微调时RKHS损失权重（默认0.1，设为0可禁用RKHS损失）')
    parser.add_argument('--lambda_value_finetune', type=float, default=1.0,
                       help='微调时价值损失权重（默认1.0，建议1.0-2.0以加速价值网络收敛）')
    parser.add_argument('--print_interaction', action='store_true',
                       help='打印详细的交互信息和行为选择（用于调试和调整模型）')
    parser.add_argument('--print_interaction_freq', type=int, default=1,
                       help='打印交互信息的频率（每N个episode打印一次，默认1）')
    parser.add_argument('--print_top_k_actions', type=int, default=5,
                       help='打印Top-K动作及其概率（默认5）')
    parser.add_argument('--use_lr_scheduler', action='store_true',
                       help='使用学习率调度器（余弦退火）')
    parser.add_argument('--use_ppo', action='store_true',
                       help='使用PPO策略损失（默认使用交叉熵）')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                       help='PPO裁剪比例（默认0.2）')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                       help='熵正则化系数（默认0.01）')
    parser.add_argument('--normalize_advantages', action='store_true',
                       help='归一化优势（减少方差）')
    parser.add_argument('--use_gae', action='store_true',
                       help='使用GAE（Generalized Advantage Estimation）计算优势函数（仅在启用PPO时有效）')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda参数（默认0.95，范围0-1，越大越偏向长期优势）')
    parser.add_argument('--value_clip', type=float, default=10.0,
                       help='价值裁剪范围（默认10.0，防止价值爆炸）')
    parser.add_argument('--grad_clip', type=float, default=0.3,
                       help='梯度裁剪阈值（默认0.3，更保守）')
    parser.add_argument('--grad_skip_threshold', type=float, default=3.0,
                       help='梯度异常阈值（超过此值跳过更新，默认3.0）')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1,
                       help='梯度累积步数（默认1，相当于增大批次大小）')
    parser.add_argument('--warmup_episodes', type=int, default=50,
                       help='预热episode数（前N个episode不训练，只收集数据）')
    parser.add_argument('--save_checkpoint_freq', type=int, default=50,
                       help='（已弃用）不再保存定期检查点，只保留最佳和最终模型')
    parser.add_argument('--use_mfvi', action='store_true',
                       help='使用MFVI Critic（默认使用标准Critic）')
    parser.add_argument('--use_transformer_critic', action='store_true',
                       help='使用Transformer Critic（默认使用标准Critic）')
    parser.add_argument('--critic_use_action', type=str, default='True',
                       choices=['True', 'False', 'true', 'false'],
                       help='Transformer Critic是否使用动作作为输入（默认True）')
    parser.add_argument('--recalibrate_value_head', action='store_true',
                       help='重新初始化价值网络最后一层（用于微调时重新校准价值估计）')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='结果保存目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='检查点保存目录（仅在启用微调时使用）')
    # save_detailed 默认启用
    parser.add_argument('--save_detailed', type=lambda x: (str(x).lower() not in ['false', '0', 'no']), 
                       default=True, nargs='?', const=True,
                       help='保存每个episode的详细指标（默认启用，可用 --save_detailed False 禁用）')
    parser.add_argument('--save_rewards_freq', type=int, default=100,
                       help='每N个episode保存一次奖励（默认100=每100个episode保存一次，0表示不实时保存，只在最后保存）')
    parser.add_argument('--print_freq', type=int, default=10,
                       help='每N个episode打印一次进度统计（默认10=每10个episode打印一次，0表示不打印进度）')
    parser.add_argument('--print_stats_window', type=int, default=10,
                       help='打印统计时使用的窗口大小（默认10=显示最近10个episode的平均值）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("CovMADT 模型评估")
    print("=" * 60)
    print(f"模型检查点: {args.checkpoint}")
    print(f"环境: {args.env_name}")
    print(f"设备: {device}")
    print(f"Episode数: {args.num_episodes}")
    print(f"策略模式: {'确定性' if args.deterministic else '随机性'}")
    if args.deterministic_after_episode is not None:
        print(f"策略切换: 前 {args.deterministic_after_episode} 个episode使用探索策略，之后切换到确定性策略")
    print(f"进度打印频率: 每 {args.print_freq} 个episode打印一次" if args.print_freq > 0 else "进度打印: 禁用")
    print(f"统计窗口大小: 最近 {args.print_stats_window} 个episode")
    print(f"\n📁 存储目录:")
    print(f"  结果输出目录: {os.path.abspath(args.output_dir)}")
    if args.enable_finetune:
        print(f"  检查点保存目录: {os.path.abspath(args.checkpoint_dir)}")
        print(f"\n🔧 微调模式: 启用")
        print(f"  学习率: {args.learning_rate}")
        print(f"  批量大小: {args.batch_size}")
        print(f"  探索率: {args.epsilon}")
        if args.use_transformer_critic:
            print(f"  Critic类型: Transformer Critic")
            print(f"    使用动作: {args.critic_use_action}")
        elif args.use_mfvi:
            print(f"  Critic类型: MFVI Critic")
        else:
            print(f"  Critic类型: 标准Critic（默认）")
    else:
        print(f"\n🔧 微调模式: 禁用（纯评估）")
    print(f"\n💾 数据保存:")
    print(f"  每 {args.save_rewards_freq} 个episode自动保存奖励数据")
    print(f"  保存文件: episode_rewards.json, .csv, .npy, .png")
    print("=" * 60)
    
    # 先加载检查点以确定环境配置
    print(f"\n加载模型: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件不存在: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # 从检查点读取环境信息
    checkpoint_obs_dim = None
    checkpoint_act_dim = None
    checkpoint_num_agents = None
    checkpoint_env_name = None
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        checkpoint_obs_dim = config.get('state_dim')
        checkpoint_act_dim = config.get('action_dim')
        checkpoint_num_agents = config.get('n_agents')
        checkpoint_env_name = config.get('env_name')
    
    # 初始化环境（根据检查点信息或命令行参数）
    print("\n初始化环境...")
    
    # 判断环境类型（仅支持 hanabi_v5）
    if args.env_name:
        env_name = args.env_name
    elif checkpoint_env_name:
        env_name = checkpoint_env_name
    else:
        env_name = 'hanabi_v5'
        print(f"⚠️  未指定环境，默认使用: {env_name}")
    
    if env_name != 'hanabi_v5':
        raise ValueError(
            f"不支持的环境: {env_name}。本脚本仅支持 hanabi_v5。"
        )
    
    # 创建环境
    env = hanabi_v5.env(players=4, max_life_tokens=6)  # 4个智能体，max_life=6
    env.reset()

    obs, _, _, _, _ = env.last()
    env_obs_dim = len(obs["observation"])
    env_act_dim = env.action_space(env.agents[0]).n
    env_num_agents = len(env.agents)
    
    # 使用检查点的维度信息（如果存在），否则使用环境的维度
    if checkpoint_obs_dim:
        obs_dim = checkpoint_obs_dim
        if obs_dim != env_obs_dim:
            print(f"⚠️  警告: 环境观察维度 ({env_obs_dim}) 与检查点中的观察维度 ({obs_dim}) 不匹配")
            print(f"   使用检查点中的观察维度: {obs_dim}")
    else:
        obs_dim = env_obs_dim
    
    if checkpoint_act_dim:
        act_dim = checkpoint_act_dim
        if act_dim != env_act_dim:
            print(f"⚠️  警告: 环境动作维度 ({env_act_dim}) 与检查点中的动作维度 ({act_dim}) 不匹配")
            print(f"   使用检查点中的动作维度: {act_dim}")
    else:
        act_dim = env_act_dim
    
    if checkpoint_num_agents:
        num_agents = checkpoint_num_agents
        if num_agents != env_num_agents:
            print(f"⚠️  警告: 环境智能体数量 ({env_num_agents}) 与检查点中的智能体数量 ({num_agents}) 不匹配")
            print(f"   使用检查点中的智能体数量: {num_agents}")
    else:
        num_agents = env_num_agents
    
    print(f"环境类型: {env_name}")
    print(f"观察维度: {obs_dim}")
    print(f"动作维度: {act_dim}")
    print(f"智能体数量: {num_agents}")
    
    # 确定检查点使用的Critic类型
    # 优先检查state_dict（更准确），然后检查config
    checkpoint_use_mfvi = None
    checkpoint_use_transformer_critic = None
    checkpoint_critic_use_action = True
    
    # 方法1: 通过检查state_dict判断Critic类型（最准确）
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # 检查是否有MFVI Critic特有的键
        has_mfvi_keys = any('critic.rkhs_embedding' in k for k in state_dict.keys())
        # 检查是否有Transformer Critic特有的键
        has_transformer_critic_keys = any('critic.transformer_blocks' in k or 'critic.pos_encoding' in k for k in state_dict.keys())
        checkpoint_use_mfvi = has_mfvi_keys
        checkpoint_use_transformer_critic = has_transformer_critic_keys
        print(f"从检查点state_dict推断: use_mfvi = {checkpoint_use_mfvi}, use_transformer_critic = {checkpoint_use_transformer_critic}")
    elif 'critic_state_dict' in checkpoint:
        # 检查critic_state_dict
        critic_keys = list(checkpoint['critic_state_dict'].keys())
        has_mfvi_keys = any('rkhs_embedding' in k for k in critic_keys)
        has_transformer_critic_keys = any('transformer_blocks' in k or 'pos_encoding' in k for k in critic_keys)
        checkpoint_use_mfvi = has_mfvi_keys
        checkpoint_use_transformer_critic = has_transformer_critic_keys
        print(f"从检查点critic_state_dict推断: use_mfvi = {checkpoint_use_mfvi}, use_transformer_critic = {checkpoint_use_transformer_critic}")
    elif 'config' in checkpoint:
        # 方法2: 如果state_dict检查不到，再检查config
        checkpoint_use_mfvi = checkpoint['config'].get('use_mfvi', False)
        checkpoint_use_transformer_critic = checkpoint['config'].get('use_transformer_critic', False)
        checkpoint_critic_use_action = checkpoint['config'].get('critic_use_action', True)
        print(f"从检查点配置读取: use_mfvi = {checkpoint_use_mfvi}, use_transformer_critic = {checkpoint_use_transformer_critic}")
    
    # 如果命令行参数与检查点不匹配，使用检查点的配置并给出警告
    if checkpoint_use_transformer_critic is not None and checkpoint_use_transformer_critic != args.use_transformer_critic:
        print(f"⚠️  警告: 命令行参数 --use_transformer_critic={args.use_transformer_critic} 与检查点不匹配")
        print(f"   检查点使用: {'Transformer Critic' if checkpoint_use_transformer_critic else '其他Critic'}")
        print(f"   将使用检查点的配置: {'Transformer Critic' if checkpoint_use_transformer_critic else '其他Critic'}")
        actual_use_transformer_critic = checkpoint_use_transformer_critic
    else:
        actual_use_transformer_critic = args.use_transformer_critic
    
    if checkpoint_use_mfvi is not None and checkpoint_use_mfvi != args.use_mfvi:
        print(f"⚠️  警告: 命令行参数 --use_mfvi={args.use_mfvi} 与检查点不匹配")
        print(f"   检查点使用: {'MFVI Critic' if checkpoint_use_mfvi else '标准Critic'}")
        print(f"   将使用检查点的配置: {'MFVI Critic' if checkpoint_use_mfvi else '标准Critic'}")
        actual_use_mfvi = checkpoint_use_mfvi
    else:
        actual_use_mfvi = args.use_mfvi
    
    # 如果检查点使用了Transformer Critic，从配置中读取critic_use_action
    if checkpoint_use_transformer_critic and 'config' in checkpoint:
        checkpoint_critic_use_action = checkpoint['config'].get('critic_use_action', True)
        if args.critic_use_action.lower() != str(checkpoint_critic_use_action).lower():
            print(f"⚠️  警告: 命令行参数 --critic_use_action={args.critic_use_action} 与检查点不匹配")
            print(f"   将使用检查点的配置: critic_use_action = {checkpoint_critic_use_action}")
        actual_critic_use_action = checkpoint_critic_use_action
    else:
        actual_critic_use_action = args.critic_use_action.lower() == 'true'
    
    # 从检查点配置中读取其他参数（如果存在）
    if 'config' in checkpoint:
        config = checkpoint['config']
        hidden_dim = config.get('hidden_dim', 128)
        transformer_layers = config.get('transformer_layers', 2)
        transformer_heads = config.get('transformer_heads', 4)
        rkhs_embedding_dim = config.get('rkhs_embedding_dim', 128)
        kernel_type = config.get('kernel_type', 'rbf')
        tau = config.get('tau', 0.1)
        gamma = config.get('gamma', 0.99)
        max_seq_len = config.get('max_seq_len', 100)
        print(f"从检查点配置读取模型参数:")
        print(f"  hidden_dim={hidden_dim}, transformer_layers={transformer_layers}")
        print(f"  transformer_heads={transformer_heads}, rkhs_embedding_dim={rkhs_embedding_dim}")
        print(f"  max_seq_len={max_seq_len}")
    else:
        # 使用默认值
        hidden_dim = 128
        transformer_layers = 2
        transformer_heads = 4
        rkhs_embedding_dim = 128
        kernel_type = "rbf"
        tau = 0.1
        gamma = 0.99
        max_seq_len = 100
    
    # 如果 config 中没有这些参数，尝试从 state_dict 推断
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # 推断 transformer_layers（通过计算 transformer_blocks 的数量）
        if transformer_layers == 2:  # 如果还是默认值，尝试推断
            max_layer_idx = -1
            for key in state_dict.keys():
                if 'transformer.transformer_blocks.' in key:
                    # 提取层索引，例如 "transformer.transformer_blocks.2.attention.q_proj.weight" -> 2
                    parts = key.split('transformer.transformer_blocks.')
                    if len(parts) > 1:
                        layer_idx = int(parts[1].split('.')[0])
                        max_layer_idx = max(max_layer_idx, layer_idx)
            if max_layer_idx >= 0:
                transformer_layers = max_layer_idx + 1
                print(f"从 state_dict 推断: transformer_layers={transformer_layers}")
        
        # 推断 max_seq_len（从位置编码的形状）
        if max_seq_len == 100:  # 如果还是默认值，尝试推断
            pos_encoding_keys = [k for k in state_dict.keys() if 'pos_encoding.pe' in k]
            if pos_encoding_keys:
                pe_shape = state_dict[pos_encoding_keys[0]].shape
                if len(pe_shape) >= 2:
                    inferred_max_seq_len = pe_shape[1]
                    if inferred_max_seq_len != 100:  # 如果不是默认值
                        max_seq_len = inferred_max_seq_len
                        print(f"从 state_dict 推断: max_seq_len={max_seq_len}")
        
        # 推断 transformer_heads（从 attention 权重形状）
        if transformer_heads == 4:  # 如果还是默认值，尝试推断
            # 查找 q_proj 权重，形状通常是 [hidden_dim, hidden_dim] 或 [num_heads * head_dim, hidden_dim]
            q_proj_keys = [k for k in state_dict.keys() if 'transformer.transformer_blocks.0.attention.q_proj.weight' in k]
            if q_proj_keys:
                q_proj_shape = state_dict[q_proj_keys[0]].shape
                if len(q_proj_shape) >= 1:
                    # q_proj 的输出维度通常是 num_heads * head_dim
                    # 假设 head_dim = hidden_dim / num_heads，所以 num_heads = hidden_dim / head_dim
                    # 但更简单的方法是：如果 q_proj 输出维度是 hidden_dim，通常 num_heads = 4 或 8
                    # 这里我们使用一个启发式方法
                    q_out_dim = q_proj_shape[0]
                    # 常见的 head_dim 是 32, 64，所以 num_heads 可能是 4, 8, 16
                    if q_out_dim % 32 == 0:
                        inferred_heads = q_out_dim // 32
                        if inferred_heads in [4, 8, 16]:
                            transformer_heads = inferred_heads
                            print(f"从 state_dict 推断: transformer_heads={transformer_heads}")
    
    # 准备模型配置
    # 注意：主Transformer策略网络也需要max_seq_len，不仅仅是Transformer Critic
    model_config = {
        'use_occupancy_measure': False,
        'max_seq_len': max_seq_len,  # 主Transformer策略网络需要这个参数
    }
    if actual_use_transformer_critic:
        model_config['critic_use_action'] = actual_critic_use_action
    
    # 创建模型（使用检查点的配置）
    model = CovMADT(
        state_dim=obs_dim,
        action_dim=act_dim,
        n_agents=num_agents,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        rkhs_embedding_dim=rkhs_embedding_dim,
        kernel_type=kernel_type,
        tau=tau,
        gamma=gamma,
        use_mfvi=actual_use_mfvi,
        use_transformer_critic=actual_use_transformer_critic,
        device=device,
        config=model_config,
    ).to(device)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif 'net_state_dict' in checkpoint:
        # 如果是R2D2模型，跳过（需要不同的评估方式）
        print("❌ 错误: 这是R2D2模型检查点，CovMADT评估器不支持")
        return
    else:
        # 尝试加载各个组件
        if 'transformer_state_dict' in checkpoint:
            model.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        if 'critic_state_dict' in checkpoint:
            model.critic.load_state_dict(checkpoint['critic_state_dict'])
        if 'rkhs_model_state_dict' in checkpoint:
            model.rkhs_model.load_state_dict(checkpoint['rkhs_model_state_dict'])
        if 'reference_policy_state_dict' in checkpoint:
            model.reference_policy.load_state_dict(checkpoint['reference_policy_state_dict'])
        print("⚠️  使用非严格模式加载模型组件")
    
    if args.enable_finetune:
        model.train()  # 微调模式，设置为训练模式
        print("✓ 模型加载成功（微调模式）")
        
        # 创建优化器
        if args.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
            print(f"  优化器: AdamW (weight_decay={args.weight_decay})")
        else:  # adam
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
            print(f"  优化器: Adam (weight_decay={args.weight_decay})")
        
        # 创建学习率调度器（可选）
        scheduler = None
        if args.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.num_episodes, eta_min=args.learning_rate * 0.1
            )
            print(f"  使用学习率调度器（余弦退火）")
        
        # 创建经验回放缓冲区
        replay_buffer = ReplayBuffer(
            capacity=args.replay_buffer_size,
            state_dim=obs_dim,
            action_dim=act_dim,
        )
        
        print(f"  学习率: {args.learning_rate}")
        print(f"  批量大小: {args.batch_size}")
        print(f"  探索率: {args.epsilon}")
        print(f"  训练频率: 每 {args.train_freq} 个episode训练一次")
        print(f"  目标网络更新频率: 每 {args.target_update_freq} 个batch更新一次")
        print(f"  最小缓冲区大小: {args.min_buffer_size}")
        print(f"  预热episode数: {args.warmup_episodes}")
        print(f"  凸正则化权重: {args.lambda_convex_finetune}")
        if args.use_ppo:
            print(f"  优势归一化: {args.normalize_advantages}")
            if args.use_gae:
                print(f"  使用GAE计算优势: lambda={args.gae_lambda}")
            else:
                print(f"  使用TD误差计算优势")
        print(f"  价值裁剪: ±{args.value_clip}")
        print(f"  梯度裁剪: {args.grad_clip}")
        print(f"  梯度跳过阈值: {args.grad_skip_threshold}")
    else:
        model.eval()  # 纯评估模式
        print("✓ 模型加载成功（评估模式）")
        optimizer = None
        replay_buffer = None
        scheduler = None
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✓ 输出目录已创建/确认: {os.path.abspath(args.output_dir)}")
    if args.enable_finetune:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"✓ 检查点目录已创建/确认: {os.path.abspath(args.checkpoint_dir)}")
    
    # 开始评估（可能包含微调）
    mode_str = "评估+微调" if args.enable_finetune else "评估"
    print(f"\n开始{mode_str} {args.num_episodes} 个episodes...")
    print("=" * 60)
    
    all_episode_stats = []
    best_score = float('-inf')
    train_count = 0  # 训练计数器（用于控制目标网络更新频率）
    accumulation_step = 0  # 梯度累积步数计数器
    # 由于每100个episode会自动保存到文件，内存中只保留最近的数据即可
    # 保留最近2000个episode的统计数据（足够用于统计和绘图，同时节省内存）
    # 旧数据已经保存到CSV/NPY文件中，不会丢失
    max_stats_history = min(2000, args.num_episodes)  # 最多保留2000个episode统计
    
    # 价值网络重新校准（如果启用微调且指定了该选项）
    if args.enable_finetune and args.recalibrate_value_head:
        print("\n🔄 价值网络重新校准...")
        # 重新初始化价值网络的最后一层，根据最近回报设置合理的偏置
        # 这有助于价值网络快速校准到正确的量级
        import torch.nn as nn
        recalibrated = False
        
        # 处理Transformer策略网络的价值头
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'value_head'):
            value_head = model.transformer.value_head
            if isinstance(value_head, nn.Sequential):
                # 找到最后一个Linear层
                for i in range(len(value_head) - 1, -1, -1):
                    if isinstance(value_head[i], nn.Linear):
                        # 重新初始化最后一层
                        nn.init.xavier_uniform_(value_head[i].weight, gain=0.1)  # 较小的gain
                        if value_head[i].bias is not None:
                            # 设置偏置为0（可以根据最近回报调整，这里先设为0）
                            nn.init.constant_(value_head[i].bias, 0.0)
                        print(f"✓ Transformer策略网络价值头最后一层已重新初始化")
                        recalibrated = True
                        break
        
        # 处理Critic网络（如果使用Transformer Critic）
        if hasattr(model, 'critic') and hasattr(model.critic, 'value_head'):
            value_head = model.critic.value_head
            if isinstance(value_head, nn.Sequential):
                # 找到最后一个Linear层
                for i in range(len(value_head) - 1, -1, -1):
                    if isinstance(value_head[i], nn.Linear):
                        # 重新初始化最后一层
                        nn.init.xavier_uniform_(value_head[i].weight, gain=0.1)
                        if value_head[i].bias is not None:
                            nn.init.constant_(value_head[i].bias, 0.0)
                        print(f"✓ Transformer Critic价值头最后一层已重新初始化")
                        recalibrated = True
                        break
        
        if not recalibrated:
            print("⚠️  警告: 未找到价值网络头部，跳过重新初始化")
    
    # 初始化模型变化检测器（如果启用微调）
    change_detector = None
    if args.enable_finetune and HAS_CHANGE_DETECTOR:
        detector_dir = os.path.join(args.output_dir, "model_change_logs")
        change_detector = ModelChangeDetector(model, device, save_dir=detector_dir)
        print(f"✓ 模型变化检测器已初始化，日志保存到: {detector_dir}")
    
    # 初始化epsilon衰减（如果启用）
    current_epsilon = args.epsilon
    if args.enable_finetune and args.epsilon_start is not None:
        current_epsilon = args.epsilon_start
        print(f"✓ Epsilon衰减已启用: {args.epsilon_start} -> {args.epsilon_end} (decay: {args.epsilon_decay})")
    
    # 检查是否启用了在指定episode后切换到确定性策略
    use_deterministic_after = (args.deterministic_after_episode is not None and 
                               args.deterministic_after_episode > 0)
    if use_deterministic_after:
        print(f"✓ 策略切换已启用: 前 {args.deterministic_after_episode} 个episode使用探索策略，之后切换到确定性策略")
    
    for episode in tqdm(range(1, args.num_episodes + 1), desc=f"{mode_str}进度"):
        # 评估episode（如果启用微调，同时收集数据）
        if args.enable_finetune:
            # 检查是否应该切换到确定性策略
            should_use_deterministic = False
            if use_deterministic_after and episode > args.deterministic_after_episode:
                should_use_deterministic = True
                # 在切换时打印提示（只打印一次）
                if episode == args.deterministic_after_episode + 1:
                    print(f"\n🎯 Episode {episode}: 已切换到确定性策略（epsilon=0）")
            
            # Epsilon衰减（如果启用了epsilon_start且未切换到确定性策略）
            if args.epsilon_start is not None and not should_use_deterministic:
                # 每episode衰减epsilon
                current_epsilon = max(args.epsilon_end, current_epsilon * args.epsilon_decay)
            
            # 周期性探索：每1000步增加探索（防止过早收敛）
            # 在探索阶段（每1000步的前100步），增加熵系数和探索率
            # 注意：如果已切换到确定性策略，则跳过周期性探索
            if should_use_deterministic:
                # 确定性策略：epsilon=0，熵系数正常
                adjusted_entropy_coef = args.entropy_coef
                adjusted_epsilon = 0.0
            else:
                exploration_cycle = 1000  # 探索周期
                exploration_phase_length = 100  # 探索阶段长度
                is_exploration_phase = (episode % exploration_cycle) < exploration_phase_length
                
                # 动态调整熵系数和探索率
                if is_exploration_phase:
                    # 探索阶段：增加探索（基于当前epsilon）
                    adjusted_entropy_coef = args.entropy_coef * 2.0  # 熵系数翻倍
                    base_epsilon = current_epsilon if args.epsilon_start is not None else args.epsilon
                    adjusted_epsilon = min(0.6, base_epsilon * 1.5)  # 探索率增加50%，最高0.6
                else:
                    # 利用阶段：使用当前衰减后的epsilon
                    adjusted_entropy_coef = args.entropy_coef
                    adjusted_epsilon = current_epsilon if args.epsilon_start is not None else args.epsilon
            
            # 决定是否打印交互信息
            should_print = (args.print_interaction and 
                          episode % args.print_interaction_freq == 0)
            
            # 定期打印epsilon值（每100个episode，且未切换到确定性策略）
            if args.epsilon_start is not None and episode % 100 == 0 and not should_use_deterministic:
                print(f"📊 Episode {episode}: 当前探索率 epsilon = {current_epsilon:.4f} (目标: {args.epsilon_end:.4f})")
            
            # 使用确定性策略的条件：args.deterministic 或 should_use_deterministic 或 adjusted_epsilon == 0
            episode_deterministic = args.deterministic or should_use_deterministic or adjusted_epsilon == 0
            episode_stats, episode_data = evaluate_episode_hanabi(
                env, model, device, 
                deterministic=episode_deterministic,
                collect_data=True,
                print_interaction=should_print,
                episode_num=episode,
                print_top_k=args.print_top_k_actions,
                epsilon=adjusted_epsilon,
            )
            
            # 存储到回放缓冲区（包含PPO需要的old_log_probs）
            for i in range(len(episode_data["states"])):
                old_log_prob = episode_data.get("old_log_probs", [None] * len(episode_data["states"]))[i]
                replay_buffer.add(
                    state=episode_data["states"][i],
                    action=episode_data["actions"][i],
                    reward=episode_data["rewards"][i],
                    next_state=episode_data["next_states"][i],
                    done=episode_data["dones"][i],
                    old_log_prob=old_log_prob if old_log_prob is not None else 0.0,
                )
            
            # 训练（增加频率控制和最小缓冲区要求，以及预热期）
            train_metrics = {}
            if (episode > args.warmup_episodes and  # 预热期
                len(replay_buffer) >= args.min_buffer_size and 
                episode % args.train_freq == 0):
                # 使用序列长度进行采样（默认使用模型的max_seq_len，但限制为合理值）
                seq_len = min(getattr(model.transformer, 'max_seq_len', 10), 10)  # 限制最大序列长度为10
                batch = replay_buffer.sample(args.batch_size, seq_len=seq_len)
                # 控制目标网络更新频率
                update_target = (train_count % args.target_update_freq == 0)
                train_metrics = train_step(
                    model, batch, device, optimizer,
                    update_target=update_target,
                    lambda_convex=args.lambda_convex_finetune,
                    lambda_rkhs=args.lambda_rkhs_finetune,
                    lambda_value=args.lambda_value_finetune,  # 传递价值损失权重
                    use_ppo=args.use_ppo,
                    clip_ratio=args.clip_ratio,
                    entropy_coef=adjusted_entropy_coef,  # 使用动态调整的熵系数
                    normalize_advantages=args.normalize_advantages,
                    use_gae=args.use_gae if args.use_ppo else False,  # 仅在启用PPO时使用GAE
                    gae_lambda=args.gae_lambda,
                    value_clip=args.value_clip,
                    grad_clip=args.grad_clip,
                    grad_skip_threshold=args.grad_skip_threshold,
                    grad_accumulation_steps=args.grad_accumulation_steps,
                    accumulation_step=accumulation_step,
                    change_detector=change_detector,  # 传递检测器
                )
                # 检查是否是更新步骤（累积完成）
                is_update_step = ((accumulation_step + 1) % args.grad_accumulation_steps == 0)
                # 只在真正更新时才增加train_count（累积完成且未跳过）
                if is_update_step and not train_metrics.get('skip_update', False):
                    train_count += 1
                    
                    # 检测模型变化（每次实际更新后）
                    if change_detector is not None:
                        loss_dict_for_detector = {
                            'total_loss': train_metrics.get('total_loss', 0.0),
                            'policy_loss': train_metrics.get('policy_loss', 0.0),
                            'value_loss': train_metrics.get('value_loss', 0.0),
                            'rkhs_loss': train_metrics.get('rkhs_loss', 0.0),
                            'convex_loss': train_metrics.get('convex_loss', 0.0),
                            'grad_norm': train_metrics.get('grad_norm', 0.0),
                        }
                        if args.use_ppo:
                            loss_dict_for_detector['entropy'] = train_metrics.get('entropy', 0.0)
                            loss_dict_for_detector['ppo_ratio_mean'] = train_metrics.get('ppo_ratio_mean', 0.0)
                            loss_dict_for_detector['advantages_mean'] = train_metrics.get('advantages_mean', 0.0)
                        
                        change_detector.check_update(optimizer, loss_dict_for_detector, step=episode)
                
                accumulation_step = (accumulation_step + 1) % args.grad_accumulation_steps
                episode_stats.update(train_metrics)
            
            # 更新学习率（如果使用调度器）
            if scheduler is not None:
                scheduler.step()
        else:
            # 评估模式：检查是否应该切换到确定性策略
            should_use_deterministic = False
            if use_deterministic_after and episode > args.deterministic_after_episode:
                should_use_deterministic = True
                # 在切换时打印提示（只打印一次）
                if episode == args.deterministic_after_episode + 1:
                    print(f"\n🎯 Episode {episode}: 已切换到确定性策略")
            
            # 评估模式也可以打印交互信息
            should_print = (args.print_interaction and 
                          episode % args.print_interaction_freq == 0)
            
            # 使用确定性策略的条件：args.deterministic 或 should_use_deterministic
            episode_deterministic = args.deterministic or should_use_deterministic
            episode_stats = evaluate_episode_hanabi(
                env, model, device, 
                deterministic=episode_deterministic,
                print_interaction=should_print,
                episode_num=episode,
                print_top_k=args.print_top_k_actions,
            )
        
        all_episode_stats.append(episode_stats)
        
        # 限制episode统计历史（只保留最近的数据，节省内存）
        if len(all_episode_stats) > max_stats_history:
            # 保留最近的数据，移除最旧的
            all_episode_stats = all_episode_stats[-max_stats_history:]
        
        # 定期清理内存（防止OOM）
        if episode % 100 == 0:
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # 实时保存奖励（每save_rewards_freq个episode保存一次）
        # 使用追加模式（append_mode=True）来追加CSV数据，避免重复写入
        should_save = (args.save_rewards_freq > 0 and episode % args.save_rewards_freq == 0)
        if should_save:
            # 确保使用正确的输出目录（使用绝对路径，避免相对路径问题）
            output_dir = os.path.abspath(args.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # 第一次保存时使用覆盖模式，后续使用追加模式
            # 判断是否为第一次保存：检查CSV文件是否存在且不为空
            csv_path_check = os.path.join(output_dir, 'episode_rewards.csv')
            append_mode = os.path.exists(csv_path_check) and os.path.getsize(csv_path_check) > 0
            
            # 如果启用详细保存，在保存奖励时也保存详细JSON（合并之前的数据）
            if args.save_detailed:
                detailed_path = os.path.join(output_dir, 'evaluation_detailed.json')
                # 读取之前保存的详细数据（如果有）
                saved_stats = []
                if os.path.exists(detailed_path):
                    try:
                        with open(detailed_path, 'r') as f:
                            saved_stats = json.load(f)
                    except:
                        pass
                
                # 计算当前内存中数据的episode范围
                if len(all_episode_stats) > 0:
                    start_episode_in_memory = episode - len(all_episode_stats) + 1
                    max_saved_episode = len(saved_stats)
                    
                    # 合并数据：确保不重复，只添加新数据
                    if start_episode_in_memory > max_saved_episode:
                        # 新数据在已保存数据之后，直接追加
                        complete_stats = saved_stats + all_episode_stats
                        print(f"  💾 保存详细数据: 已有{len(saved_stats)}个episode，新增{len(all_episode_stats)}个episode (episode {max_saved_episode+1} 到 {episode})，总计{len(complete_stats)}个episode")
                    elif start_episode_in_memory <= max_saved_episode:
                        # 有重叠，需要合并：保留旧数据的前半部分，用新数据覆盖重叠部分并追加新部分
                        overlap_start = max(0, start_episode_in_memory - 1)
                        # 只保留不重叠的旧数据
                        complete_stats = saved_stats[:overlap_start]
                        # 添加新数据（覆盖重叠部分并追加新部分）
                        complete_stats.extend(all_episode_stats)
                        print(f"  💾 更新详细数据: 已有{len(saved_stats)}个episode，更新/新增{len(all_episode_stats)}个episode (episode {start_episode_in_memory} 到 {episode})，总计{len(complete_stats)}个episode")
                    else:
                        # 如果已保存的数据更完整，使用已保存的数据
                        complete_stats = saved_stats
                        print(f"  💾 使用已保存的详细数据: {len(saved_stats)}个episode")
                    # 保存合并后的详细数据
                    with open(detailed_path, 'w') as f:
                        json.dump(complete_stats, f, indent=2)
                else:
                    # 如果内存中没有数据，使用已保存的数据
                    complete_stats = saved_stats
                    with open(detailed_path, 'w') as f:
                        json.dump(complete_stats, f, indent=2)
            
            json_path, csv_path, npy_path, image_path, total_saved_episodes = save_episode_rewards(
                all_episode_stats, output_dir, episode, append_mode=append_mode
            )
            # 显示保存信息（不依赖print_interaction，因为这是重要的进度信息）
            print(f"\n{'='*80}")
            print(f"💾 自动保存奖励数据 (Episode {episode}/{args.num_episodes})")
            print(f"{'='*80}")
            print(f"  📁 输出目录: {os.path.abspath(args.output_dir)}")
            print(f"  ✓ JSON: {os.path.abspath(json_path)}")
            print(f"  ✓ CSV:  {os.path.abspath(csv_path)} (覆盖模式，包含所有{total_saved_episodes}个episode)")
            print(f"  ✓ NPY:  {os.path.abspath(npy_path)} (包含所有{total_saved_episodes}个episode)")
            if image_path:
                print(f"  ✓ 图像: {os.path.abspath(image_path)} (显示所有{total_saved_episodes}个episode)")
            else:
                print(f"  ⚠️  图像: 未保存（matplotlib不可用或保存失败）")
            print(f"  📊 数据统计: 内存中{len(all_episode_stats)}个episode，文件中{total_saved_episodes}个episode")
            if total_saved_episodes > 0:
                # 从NPY文件读取所有数据进行统计
                try:
                    saved_data = np.load(npy_path)
                    recent_scores = saved_data['score'][-min(100, len(saved_data)):]
                    recent_rewards = saved_data['reward'][-min(100, len(saved_data)):]
                    print(f"  📈 最近100个episode平均得分: {np.mean(recent_scores):.2f}, 平均奖励: {np.mean(recent_rewards):.2f}")
                except:
                    if len(all_episode_stats) > 0:
                        recent_scores = [s['final_score'] for s in all_episode_stats[-min(100, len(all_episode_stats)):]]
                        recent_rewards = [s['episode_reward'] for s in all_episode_stats[-min(100, len(all_episode_stats)):]]
                        print(f"  📈 最近100个episode平均得分: {np.mean(recent_scores):.2f}, 平均奖励: {np.mean(recent_rewards):.2f}")
            print(f"{'='*80}\n")
        
        # 每N个episode打印一次进度（可配置）
        if args.print_freq > 0 and episode % args.print_freq == 0:
            window_size = min(args.print_stats_window, len(all_episode_stats))
            if window_size > 0:
                recent_scores = [s['final_score'] for s in all_episode_stats[-window_size:]]
                avg_score = np.mean(recent_scores)
                recent_rewards = [s['episode_reward'] for s in all_episode_stats[-window_size:]]
                avg_reward = np.mean(recent_rewards)
                print(f"\nEpisode {episode}: 最近{window_size}个episode平均得分 = {avg_score:.2f}, 平均奖励 = {avg_reward:.2f}")
            if args.enable_finetune and train_metrics:
                if not train_metrics.get('skip_update', False):
                    print(f"  训练损失: {train_metrics.get('total_loss', 0):.4f}")
                else:
                    print(f"  ⚠️  上次更新被跳过（梯度异常）")
                grad_norm = train_metrics.get('grad_norm', 0)
                if grad_norm > 0:
                    print(f"  梯度范数: {grad_norm:.3f}")
                if args.use_ppo and 'ppo_ratio_mean' in train_metrics:
                    ratio_mean = train_metrics.get('ppo_ratio_mean', 0)
                    ratio_std = train_metrics.get('ppo_ratio_std', 0)
                    print(f"  PPO比率: {ratio_mean:.3f} ± {ratio_std:.3f}")
        
        # 保存最佳模型（如果启用微调）
        if args.enable_finetune:
            if episode_stats['final_score'] > best_score:
                best_score = episode_stats['final_score']
                best_path = os.path.join(args.checkpoint_dir, "best_eval_model_4.pt")
                model.save_checkpoint(best_path)
    
    env.close()
    
    # 保存最终模型（如果启用微调）
    if args.enable_finetune:
        final_path = os.path.join(args.checkpoint_dir, "final_eval_model_4.pt")
        model.save_checkpoint(final_path)
        print(f"\n✓ 最终模型已保存到: {final_path}")
        
        # 保存最终变化报告
        if change_detector is not None:
            change_detector.save_final_report(step=args.num_episodes)
    
    # 计算统计指标
    print("\n" + "=" * 60)
    print("评估结果统计")
    print("=" * 60)
    
    # 提取所有指标（兼容不同环境）
    final_scores = [s.get('final_score', s.get('episode_reward', 0)) for s in all_episode_stats]
    perfect_games = [s.get('is_perfect_game', False) for s in all_episode_stats]
    information_efficiencies = [s.get('information_efficiency', 0.0) for s in all_episode_stats]
    life_loss_rates = [s.get('life_loss_rate', 0.0) for s in all_episode_stats]
    risk_control_scores = [s.get('risk_control_score', 0.0) for s in all_episode_stats]
    episode_rewards = [s.get('episode_reward', 0) for s in all_episode_stats]
    
    # 计算统计信息
    stats_summary = {
        'num_episodes': args.num_episodes,
        
        # Final Score (mean ± std)
        'final_score_mean': float(np.mean(final_scores)),
        'final_score_std': float(np.std(final_scores)),
        'final_score_min': float(np.min(final_scores)),
        'final_score_max': float(np.max(final_scores)),
        
        # Perfect Game Rate
        'perfect_game_count': int(sum(perfect_games)),
        'perfect_game_rate': float(sum(perfect_games) / len(perfect_games)),
        
        # Information Efficiency (mean ± std)
        'information_efficiency_mean': float(np.mean(information_efficiencies)),
        'information_efficiency_std': float(np.std(information_efficiencies)),
        
        # Life Loss Rate / Risk Control (mean ± std)
        'life_loss_rate_mean': float(np.mean(life_loss_rates)),
        'life_loss_rate_std': float(np.std(life_loss_rates)),
        'risk_control_score_mean': float(np.mean(risk_control_scores)),
        'risk_control_score_std': float(np.std(risk_control_scores)),
        
        # Episode Reward
        'episode_reward_mean': float(np.mean(episode_rewards)),
        'episode_reward_std': float(np.std(episode_rewards)),
    }
    
    # 打印结果
    print(f"\n1. Final Score（最终得分）:")
    print(f"   Mean ± Std: {stats_summary['final_score_mean']:.2f} ± {stats_summary['final_score_std']:.2f}")
    print(f"   Range: [{stats_summary['final_score_min']:.0f}, {stats_summary['final_score_max']:.0f}]")
    
    print(f"\n2. Perfect Game Rate（满分率）:")
    print(f"   Perfect Games: {stats_summary['perfect_game_count']}/{args.num_episodes}")
    print(f"   Rate: {stats_summary['perfect_game_rate']*100:.2f}%")
    
    print(f"\n3. Information Efficiency（信息效率）:")
    print(f"   Mean ± Std: {stats_summary['information_efficiency_mean']:.3f} ± {stats_summary['information_efficiency_std']:.3f}")
    
    print(f"\n4. Life Loss Rate / Risk Control（风险控制能力）:")
    print(f"   Life Loss Rate (Mean ± Std): {stats_summary['life_loss_rate_mean']:.3f} ± {stats_summary['life_loss_rate_std']:.3f}")
    print(f"   Risk Control Score (Mean ± Std): {stats_summary['risk_control_score_mean']:.3f} ± {stats_summary['risk_control_score_std']:.3f}")
    
    print(f"\n5. Episode Reward（Episode奖励）:")
    print(f"   Mean ± Std: {stats_summary['episode_reward_mean']:.2f} ± {stats_summary['episode_reward_std']:.2f}")
    
    # 保存结果
    summary_path = os.path.join(args.output_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print(f"\n✓ 统计摘要已保存到: {summary_path}")
    
    # ========== 读取并合并所有已保存的数据 ==========
    # 尝试读取之前保存的详细数据（如果有）
    detailed_path = os.path.join(args.output_dir, 'evaluation_detailed.json')
    complete_episode_stats = all_episode_stats.copy()  # 默认使用当前内存中的数据
    
    if os.path.exists(detailed_path):
        try:
            # 读取之前保存的详细数据
            with open(detailed_path, 'r') as f:
                saved_stats = json.load(f)
            
            # 计算当前内存中数据的episode范围
            # all_episode_stats 只包含最近的数据，需要根据 args.num_episodes 推断
            if len(all_episode_stats) > 0 and len(saved_stats) > 0:
                # 假设 all_episode_stats 包含从 (args.num_episodes - len(all_episode_stats) + 1) 到 args.num_episodes 的数据
                start_episode_in_memory = args.num_episodes - len(all_episode_stats) + 1
                
                # 合并数据：保留旧数据，用新数据覆盖或追加
                # 如果旧数据已经包含新数据的episode，则更新；否则追加
                max_saved_episode = len(saved_stats)
                
                # 如果内存中的数据是新的（episode编号大于已保存的），则合并
                if start_episode_in_memory > max_saved_episode:
                    # 新数据在已保存数据之后，直接追加
                    complete_episode_stats = saved_stats + all_episode_stats
                    print(f"  🔄 合并详细数据: 已有{len(saved_stats)}个episode，新增{len(all_episode_stats)}个episode (episode {max_saved_episode+1} 到 {args.num_episodes})")
                elif start_episode_in_memory <= max_saved_episode:
                    # 有重叠，需要合并：保留旧数据的前半部分，用新数据覆盖重叠部分并追加新部分
                    overlap_start = max(0, start_episode_in_memory - 1)  # 转换为0-based索引
                    
                    # 保留旧数据的前半部分（不重叠的部分）
                    complete_episode_stats = saved_stats[:overlap_start]
                    
                    # 添加新数据（覆盖重叠部分并追加新部分）
                    complete_episode_stats.extend(all_episode_stats)
                    
                    print(f"  🔄 合并详细数据: 已有{len(saved_stats)}个episode，更新/新增{len(all_episode_stats)}个episode (episode {start_episode_in_memory} 到 {args.num_episodes})")
                else:
                    # 使用已保存的数据（更完整）
                    complete_episode_stats = saved_stats
                    print(f"  🔄 使用已保存的详细数据: {len(saved_stats)}个episode")
            elif len(saved_stats) > 0:
                # 如果内存中没有数据，使用已保存的数据
                complete_episode_stats = saved_stats
                print(f"  🔄 使用已保存的详细数据: {len(saved_stats)}个episode")
        except Exception as e:
            print(f"⚠️  警告: 读取已保存的详细数据失败: {e}，将只使用当前内存中的数据")
            import traceback
            traceback.print_exc()
            complete_episode_stats = all_episode_stats
    
    # ========== 使用完整数据保存所有文件 ==========
    # 保存详细结果（如果启用）- 使用完整数据
    if args.save_detailed:
        detailed_path = os.path.join(args.output_dir, 'evaluation_detailed.json')
        with open(detailed_path, 'w') as f:
            json.dump(complete_episode_stats, f, indent=2)
        print(f"✓ 详细结果已保存到: {detailed_path} (包含 {len(complete_episode_stats)} 个episode)")
    
    # 保存CSV格式（便于分析）- 使用完整数据
    import csv
    csv_path = os.path.join(args.output_dir, 'evaluation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'episode', 'final_score', 'is_perfect_game', 'information_efficiency',
            'life_loss_rate', 'risk_control_score', 'episode_reward', 'episode_steps'
        ])
        writer.writeheader()
        for i, stats in enumerate(complete_episode_stats):
            writer.writerow({
                'episode': i + 1,
                'final_score': stats['final_score'],
                'is_perfect_game': 1 if stats['is_perfect_game'] else 0,
                'information_efficiency': stats['information_efficiency'],
                'life_loss_rate': stats['life_loss_rate'],
                'risk_control_score': stats['risk_control_score'],
                'episode_reward': stats['episode_reward'],
                'episode_steps': stats['episode_steps'],
            })
    print(f"✓ CSV结果已保存到: {csv_path} (包含 {len(complete_episode_stats)} 个episode)")
    
    # 保存episode奖励（最终版本）- JSON, NPY, CSV, 图像
    # 确保使用正确的输出目录（使用绝对路径）
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # 使用完整数据保存奖励
    json_path, csv_path, npy_path, image_path, total_saved_episodes = save_episode_rewards(complete_episode_stats, output_dir, args.num_episodes)
    print(f"✓ Episode奖励JSON已保存到: {json_path}")
    print(f"✓ Episode奖励CSV已保存到: {csv_path}")
    print(f"✓ Episode奖励NPY已保存到: {npy_path}")
    if image_path:
        print(f"✓ Episode奖励图像已保存到: {image_path}")
    
    # 保存NPY格式（包含所有指标）- 使用完整数据
    npy_path = os.path.join(args.output_dir, 'evaluation_results.npy')
    
    # 提取所有可能的指标字段（使用完整的数据）
    all_keys = set()
    for stats in complete_episode_stats:
        all_keys.update(stats.keys())
    
    # 定义数据类型（排除列表类型和字典类型字段）
    exclude_keys = {'actions_taken', 'action_type_stats'}  # 排除列表类型和字典类型字段
    
    # 进一步过滤：排除所有字典类型和列表类型的字段
    # 检查所有episode，确保字段在所有episode中都是数值类型
    numeric_keys = []
    for k in sorted(all_keys):
        if k in exclude_keys:
            continue
        # 检查所有episode中该字段的值，确保都不是字典或列表
        is_valid = True
        for stats in complete_episode_stats:
            value = stats.get(k, None)
            if value is not None and isinstance(value, (dict, list)):
                is_valid = False
                break
        if is_valid:
            numeric_keys.append(k)
    
    # 创建结构化数组
    # 再次过滤numeric_keys，移除任何可能包含字典或列表的字段
    final_numeric_keys = []
    dtype_list = []
    for key in numeric_keys:
        # 根据第一个episode的值推断类型（确保不是字典或列表）
        sample_value = complete_episode_stats[0].get(key, 0) if len(complete_episode_stats) > 0 else 0
        # 再次检查，确保不是字典或列表
        if isinstance(sample_value, (dict, list)):
            continue  # 跳过字典和列表类型
        # 添加到最终列表
        final_numeric_keys.append(key)
        # 推断数据类型
        if isinstance(sample_value, bool):
            dtype_list.append((key, np.bool_))
        elif isinstance(sample_value, (int, np.integer)):
            dtype_list.append((key, np.int64))
        elif isinstance(sample_value, (float, np.floating)):
            dtype_list.append((key, np.float64))
        else:
            dtype_list.append((key, np.float64))  # 默认使用float64
    
    # 更新numeric_keys为最终过滤后的列表
    numeric_keys = final_numeric_keys
    
    # 创建结构化数组（使用完整的数据）
    npy_array = np.zeros(len(complete_episode_stats), dtype=dtype_list)
    
    # 填充数据
    for i, stats in enumerate(complete_episode_stats):
        for key in numeric_keys:
            value = stats.get(key, 0)
            # 跳过字典和列表类型
            if isinstance(value, (dict, list)):
                continue
            # 处理布尔值
            if isinstance(value, bool):
                npy_array[i][key] = value
            else:
                try:
                    npy_array[i][key] = float(value) if value is not None else 0.0
                except (TypeError, ValueError):
                    # 如果无法转换为浮点数，使用默认值0.0
                    npy_array[i][key] = 0.0
    
    # 保存NPY文件
    np.save(npy_path, npy_array)
    print(f"✓ NPY结果已保存到: {npy_path}")
    print(f"  包含 {len(numeric_keys)} 个指标字段: {', '.join(numeric_keys)}")
    
    # 生成图表
    plot_evaluation_results(all_episode_stats, args.output_dir, enable_finetune=args.enable_finetune)
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print(f"结果保存在: {args.output_dir}")
    print("=" * 60)
    
    return stats_summary


if __name__ == "__main__":
    main()

