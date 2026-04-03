"""
MAPPO训练脚本 - 支持 Hanabi 与 Entombed Cooperative 环境
在每个iteration后评估200个episode并记录，最后绘图
"""
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import json
from datetime import datetime

# 环境导入
from pettingzoo.classic import hanabi_v5
from pettingzoo.atari import entombed_cooperative_v3
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1

from algorithms.mappo import MAPPO


def create_hanabi_env(seed=None):
    """创建Hanabi环境"""
    env = hanabi_v5.env(players=4, max_life_tokens=6)
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    return env


def create_entombed_env(preprocess=True, seed=None):
    """创建Entombed Cooperative环境（并可选预处理）"""
    env = entombed_cooperative_v3.parallel_env(render_mode=None)
    if preprocess:
        env = color_reduction_v0(env)
        env = resize_v1(env, 84, 84)
        env = frame_stack_v1(env, stack_size=4)
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    return env


def get_env_info(env, env_name):
    """获取环境信息"""
    if env_name == 'hanabi_v5':
        first_agent = env.agents[0]
        obs, _, _, _, _ = env.last()
        obs_dim = len(obs["observation"])
        action_dim = env.action_space(first_agent).n
        num_agents = len(env.agents)
    elif env_name == 'entombed_cooperative_v3':
        first_agent = env.agents[0]
        obs_space = env.observation_space(first_agent)
        action_space = env.action_space(first_agent)
        if hasattr(obs_space, 'shape'):
            obs_dim = int(np.prod(obs_space.shape))
        elif hasattr(obs_space, 'spaces'):
            obs_dim = int(np.sum([np.prod(space.shape) for space in obs_space.spaces.values()]))
        else:
            obs_dim = 128
        action_dim = action_space.n if hasattr(action_space, 'n') else 4
        num_agents = len(env.agents)
    else:
        raise ValueError(f"不支持的环境: {env_name}")
    
    return obs_dim, action_dim, num_agents


def process_observation(obs, env_name):
    """处理观察（转换为numpy数组）"""
    if env_name == 'hanabi_v5':
        if isinstance(obs, dict):
            return obs["observation"]
        return obs
    elif env_name == 'entombed_cooperative_v3':
        if isinstance(obs, dict) and 'observation' in obs:
            obs = obs['observation']
        if isinstance(obs, np.ndarray):
            return obs.flatten()
        return obs
    return obs


def collect_episode_data(env, model, env_name, device, epsilon=0.0, deterministic=False):
    """收集一个episode的数据
    
    Args:
        env: 环境
        model: MAPPO模型
        env_name: 环境名称
        device: 设备
        epsilon: 探索率（用于训练，评估时忽略）
        deterministic: 是否使用确定性策略（用于评估）
    """
    if env_name == 'hanabi_v5':
        env.reset()
        obs_dict = {}
        action_masks = {}
        # 初始化所有智能体的观察和动作掩码
        for agent_id in env.agents:
            obs, _, _, _, info = env.last()
            obs_dict[agent_id] = process_observation(obs, env_name)
            # 从观察中获取动作掩码（Hanabi的action_mask在obs中）
            if isinstance(obs, dict) and 'action_mask' in obs:
                action_masks[agent_id] = obs['action_mask']
            elif 'action_mask' in info:
                action_masks[agent_id] = info['action_mask']
            else:
                # 如果没有掩码，创建全1掩码（所有动作都允许）
                action_dim = env.action_space(agent_id).n
                action_masks[agent_id] = np.ones(action_dim, dtype=np.float32)
    elif env_name == 'entombed_cooperative_v3':
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _ = reset_out
        else:
            obs = reset_out
        all_agent_ids = list(env.possible_agents) if hasattr(env, 'possible_agents') else list(env.agents)
        sample_agent = all_agent_ids[0] if len(all_agent_ids) > 0 else None
        if sample_agent is not None and sample_agent in obs:
            sample_obs = process_observation(obs[sample_agent], env_name)
            zero_obs = np.zeros_like(sample_obs)
        else:
            zero_obs = None
        obs_dict = {}
        for agent_id in all_agent_ids:
            if agent_id in obs:
                obs_dict[agent_id] = process_observation(obs[agent_id], env_name)
            else:
                if zero_obs is None:
                    obs_dict[agent_id] = np.zeros(1, dtype=np.float32)
                else:
                    obs_dict[agent_id] = zero_obs.copy()
        action_masks = None
    else:
        raise ValueError(f"不支持的环境: {env_name}")
    
    episode_data = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'old_log_probs': [],
        'global_obs': [],
        'action_masks': [],
    }
    
    episode_reward = 0.0
    step_count = 0
    max_steps = 1000
    
    # 计算全局观察（所有智能体观察的拼接）
    def get_global_obs(obs_dict):
        agent_ids = sorted(obs_dict.keys())
        return np.concatenate([obs_dict[aid] for aid in agent_ids])
    
    # 存储初始观察
    initial_global_obs = get_global_obs(obs_dict)
    
    while step_count < max_steps:
        # 执行动作
        if env_name == 'hanabi_v5':
            # Hanabi是顺序环境，只有当前智能体需要选择动作
            agent_id = env.agent_selection
            
            # 获取当前智能体的最新观察和动作掩码
            obs, _, _, _, info = env.last()
            if isinstance(obs, dict):
                current_state = process_observation(obs, env_name)
                current_action_mask = obs.get('action_mask', None)
                if current_action_mask is not None:
                    current_action_mask = current_action_mask.astype(np.float32)
            else:
                current_state = process_observation(obs, env_name)
                current_action_mask = None
            
            # 如果没有动作掩码，尝试从info获取
            if current_action_mask is None and 'action_mask' in info:
                current_action_mask = np.array(info['action_mask'], dtype=np.float32)
            
            # 如果还是没有，创建全1掩码
            if current_action_mask is None:
                action_dim = env.action_space(agent_id).n
                current_action_mask = np.ones(action_dim, dtype=np.float32)
            
            # 更新当前智能体的观察和动作掩码
            obs_dict[agent_id] = current_state
            action_masks[agent_id] = current_action_mask
            
            # 存储当前观察（在执行动作之前）
            current_obs_dict = obs_dict.copy()
            current_global_obs = get_global_obs(obs_dict)
            
            # 只为当前智能体选择动作
            agent_idx = sorted(env.agents).index(agent_id)
            obs_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)
            mask_tensor = torch.FloatTensor(current_action_mask).unsqueeze(0).to(device)
            
            model.actors[agent_idx].eval()
            with torch.no_grad():
                action, log_prob = model.actors[agent_idx].get_action_and_log_prob(obs_tensor, mask_tensor, deterministic=deterministic)
                action = action.item()
            
            # 验证动作是否合法
            if action >= len(current_action_mask) or current_action_mask[action] == 0:
                # 如果动作不合法，从合法动作中随机选择
                valid_actions = np.where(current_action_mask > 0)[0]
                if len(valid_actions) > 0:
                    action = int(np.random.choice(valid_actions))
                else:
                    # 如果没有合法动作，跳过这一步
                    print(f"警告: 智能体 {agent_id} 没有合法动作，跳过这一步")
                    break
            
            # 创建动作字典（只为当前智能体）
            all_agent_ids = sorted(env.agents)
            actions = {aid: 0 for aid in all_agent_ids}  # 其他智能体动作设为0
            actions[agent_id] = action
            
            # 存储旧的对数概率（用于PPO）
            old_log_probs = {aid: 0.0 for aid in all_agent_ids}
            old_log_probs[agent_id] = log_prob.item()
            
            # 执行动作
            env.step(action)
            
            # 获取奖励和下一个观察
            next_obs, reward, termination, truncation, info = env.last()
            done = termination or truncation
            
            # 为所有智能体创建奖励字典（Hanabi是顺序环境，只有当前智能体有奖励）
            all_agent_ids = sorted(env.agents)
            rewards_dict = {aid: 0.0 for aid in all_agent_ids}
            rewards_dict[agent_id] = reward
            
            dones_dict = {aid: False for aid in all_agent_ids}
            dones_dict[agent_id] = done
            
            # 存储数据（使用执行动作前的观察）
            episode_data['obs'].append(current_obs_dict.copy())
            episode_data['actions'].append(actions.copy())
            episode_data['rewards'].append(rewards_dict)
            episode_data['dones'].append(dones_dict)
            episode_data['old_log_probs'].append(old_log_probs.copy())
            episode_data['global_obs'].append(current_global_obs)
            episode_data['action_masks'].append(action_masks.copy() if action_masks else None)
            
            episode_reward += reward
            
            # 更新观察和动作掩码
            if isinstance(next_obs, dict):
                obs_dict[agent_id] = process_observation(next_obs, env_name)
                # 从观察中获取动作掩码
                if 'action_mask' in next_obs:
                    action_masks[agent_id] = next_obs['action_mask'].astype(np.float32)
                elif 'action_mask' in info:
                    action_masks[agent_id] = np.array(info['action_mask'], dtype=np.float32)
            else:
                obs_dict[agent_id] = process_observation(next_obs, env_name)
                if 'action_mask' in info:
                    action_masks[agent_id] = np.array(info['action_mask'], dtype=np.float32)
            
            if done:
                break
                
        elif env_name == 'entombed_cooperative_v3':
            # 并行环境：为所有智能体选择动作
            current_obs_dict = obs_dict.copy()
            current_global_obs = get_global_obs(obs_dict)
            agent_ids = sorted(current_obs_dict.keys())

            actions = {}
            old_log_probs = {}
            for i, agent_id in enumerate(agent_ids):
                if agent_id in env.agents:
                    obs_tensor = torch.FloatTensor(current_obs_dict[agent_id]).unsqueeze(0).to(device)
                    model.actors[i].eval()
                    with torch.no_grad():
                        action, log_prob = model.actors[i].get_action_and_log_prob(obs_tensor, None, deterministic=deterministic)
                    actions[agent_id] = action.item()
                    old_log_probs[agent_id] = log_prob.item()
                else:
                    actions[agent_id] = 0
                    old_log_probs[agent_id] = 0.0

            step_actions = {aid: actions[aid] for aid in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(step_actions)
            
            rewards_dict = {aid: float(rewards.get(aid, 0.0)) for aid in agent_ids}
            dones_dict = {}
            for aid in agent_ids:
                if aid in terminations or aid in truncations:
                    dones_dict[aid] = bool(terminations.get(aid, False) or truncations.get(aid, False))
                else:
                    dones_dict[aid] = aid not in env.agents
            
            # 存储数据（使用执行动作前的观察）
            episode_data['obs'].append(current_obs_dict.copy())
            episode_data['actions'].append(actions.copy())
            episode_data['rewards'].append(rewards_dict)
            episode_data['dones'].append(dones_dict)
            episode_data['old_log_probs'].append(old_log_probs.copy())
            episode_data['global_obs'].append(current_global_obs)
            episode_data['action_masks'].append(None)
            
            # 更新观察
            for agent_id in agent_ids:
                if agent_id in obs:
                    obs_dict[agent_id] = process_observation(obs[agent_id], env_name)
                else:
                    if isinstance(next(iter(obs_dict.values())), np.ndarray):
                        obs_dict[agent_id] = np.zeros_like(next(iter(obs_dict.values())))
                    else:
                        obs_dict[agent_id] = obs_dict[agent_id]
            
            # 计算总奖励
            total_reward = sum(rewards_dict.values())
            episode_reward += total_reward
            
            # 检查是否结束
            if len(env.agents) == 0:
                break
        
        step_count += 1
    
    return episode_data, episode_reward


def evaluate_model(model, env_name, n_pistons=40, num_episodes=200, device='cuda', seed=None, preprocess_atari=True):
    """评估模型，返回平均奖励"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        # 创建环境
        if env_name == 'hanabi_v5':
            env = create_hanabi_env(seed=seed + episode if seed is not None else None)
        elif env_name == 'entombed_cooperative_v3':
            env = create_entombed_env(preprocess=preprocess_atari, seed=seed + episode if seed is not None else None)
        else:
            raise ValueError(f"不支持的环境: {env_name}")
        
        # 收集episode数据（不用于训练，只评估，使用确定性策略）
        _, episode_reward = collect_episode_data(env, model, env_name, device, epsilon=0.0, deterministic=True)
        episode_rewards.append(episode_reward)
        
        env.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'all_rewards': episode_rewards,
    }


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
        title = 'MAPPO训练曲线'
    else:
        mean_label = 'Mean Reward'
        std_label = 'Std Dev'
        ylabel = 'Mean Reward (200 episodes)'
        title = 'MAPPO Training Curve'
    
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
    parser = argparse.ArgumentParser(description='MAPPO训练脚本')
    
    # 环境参数
    parser.add_argument('--env_name', type=str, default='hanabi_v5',
                       choices=['hanabi_v5', 'entombed_cooperative_v3'],
                       help='环境名称')
    parser.add_argument('--no_preprocess', action='store_true',
                       help='禁用Atari环境的图像预处理（仅对entombed_cooperative_v3有效）')
    
    # 训练参数
    parser.add_argument('--num_iterations', type=int, default=1000,
                       help='训练迭代次数')
    parser.add_argument('--episodes_per_iteration', type=int, default=10,
                       help='每次迭代收集的episode数')
    parser.add_argument('--eval_episodes', type=int, default=50,
                       help='每次评估的episode数（默认50，可增加到200以获得更准确的评估）')
    parser.add_argument('--eval_freq', type=int, default=5,
                       help='评估频率（每N个iteration评估一次，默认5，减少评估次数以加快训练）')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda参数')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                       help='PPO裁剪比例')
    parser.add_argument('--value_clip', type=float, default=0.2,
                       help='价值裁剪比例')
    parser.add_argument('--entropy_coef', type=float, default=0.05,
                       help='熵系数（默认0.05，如果策略过早收敛可以增加到0.1）')
    parser.add_argument('--value_coef', type=float, default=0.5,
                       help='价值损失系数')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='PPO更新轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批量大小')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='探索率')
    
    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志保存目录')
    parser.add_argument('--experiment_name', type=str, default='mappo_training',
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
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.log_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print("=" * 60)
    print("MAPPO 训练")
    print("=" * 60)
    print(f"环境: {args.env_name}")
    print(f"设备: {device}")
    print(f"实验目录: {exp_dir}")
    print("=" * 60)
    
    # 初始化环境以获取维度信息
    print("\n初始化环境...")
    if args.env_name == 'hanabi_v5':
        env = create_hanabi_env(seed=args.seed)
    elif args.env_name == 'entombed_cooperative_v3':
        env = create_entombed_env(preprocess=not args.no_preprocess, seed=args.seed)
    else:
        raise ValueError(f"不支持的环境: {args.env_name}")
    
    obs_dim, action_dim, num_agents = get_env_info(env, args.env_name)
    env.close()
    
    print(f"观察维度: {obs_dim}")
    print(f"动作维度: {action_dim}")
    print(f"智能体数量: {num_agents}")
    
    # 创建MAPPO模型
    print("\n创建MAPPO模型...")
    model = MAPPO(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=num_agents,
        hidden_dim=args.hidden_dim,
        lr_actor=args.learning_rate,
        lr_critic=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_clip=args.value_clip,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        discrete=True,
        device=device,
    )
    
    # 评估结果记录
    eval_results = []
    
    # 训练循环
    print("\n开始训练...")
    print(f"每个iteration收集 {args.episodes_per_iteration} 个episode")
    print(f"每 {args.eval_freq} 个iteration评估一次（{args.eval_episodes} 个episode）")
    print(f"⚠️  提示: 如果训练太慢，可以增加 --eval_freq 或减少 --eval_episodes")
    print("=" * 60)
    
    for iteration in tqdm(range(1, args.num_iterations + 1), desc="训练进度"):
        # 收集episode数据
        all_episode_data = []
        all_episode_rewards = []
        
        for episode in range(args.episodes_per_iteration):
            # 创建环境
            if args.env_name == 'hanabi_v5':
                env = create_hanabi_env(seed=args.seed + iteration * 1000 + episode)
            elif args.env_name == 'entombed_cooperative_v3':
                env = create_entombed_env(preprocess=not args.no_preprocess, seed=args.seed + iteration * 1000 + episode)
            
            # 收集数据
            episode_data, episode_reward = collect_episode_data(
                env, model, args.env_name, device, epsilon=args.epsilon
            )
            all_episode_data.append(episode_data)
            all_episode_rewards.append(episode_reward)
            
            env.close()
        
        # 合并所有episode的数据
        combined_data = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'old_log_probs': [],
            'global_obs': [],
            'action_masks': [],
        }
        
        for episode_data in all_episode_data:
            combined_data['obs'].extend(episode_data['obs'])
            combined_data['actions'].extend(episode_data['actions'])
            combined_data['rewards'].extend(episode_data['rewards'])
            combined_data['dones'].extend(episode_data['dones'])
            combined_data['old_log_probs'].extend(episode_data['old_log_probs'])
            combined_data['global_obs'].extend(episode_data['global_obs'])
            combined_data['action_masks'].extend(episode_data['action_masks'])
        
        # 更新模型
        if len(combined_data['obs']) > 0:
            loss_info = model.update(
                obs_batch=combined_data['obs'],
                action_batch=combined_data['actions'],
                reward_batch=combined_data['rewards'],
                done_batch=combined_data['dones'],
                old_log_probs_batch=combined_data['old_log_probs'],
                global_obs_batch=combined_data['global_obs'],
                action_masks_batch=combined_data['action_masks'] if any(combined_data['action_masks']) else None,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
            )
            
            # 打印训练统计（前10个iteration每次都打印，之后每10个iteration打印一次）
            should_print = (iteration <= 10) or (iteration % 10 == 0)
            if should_print and loss_info is not None:
                print(f"\n[训练统计] Iteration {iteration}:")
                if 'policy_loss' in loss_info:
                    print(f"  Policy Loss: {loss_info['policy_loss']:.4f}")
                if 'value_loss' in loss_info:
                    print(f"  Value Loss: {loss_info['value_loss']:.4f}")
                if 'entropy' in loss_info:
                    entropy_val = loss_info['entropy']
                    print(f"  Entropy: {entropy_val:.4f}", end="")
                    if entropy_val < 0.001:
                        print(" ⚠️  警告: Entropy过低，策略可能失去探索能力！建议增加 --entropy_coef")
                    else:
                        print()
                print(f"  收集的数据步数: {len(combined_data['obs'])}")
                print(f"  平均episode奖励: {np.mean(all_episode_rewards):.4f} ± {np.std(all_episode_rewards):.4f}")
                print(f"  Episode奖励范围: [{np.min(all_episode_rewards):.4f}, {np.max(all_episode_rewards):.4f}]")
                if len(all_episode_rewards) > 0:
                    print(f"  详细episode奖励: {[f'{r:.2f}' for r in all_episode_rewards[:5]]}{'...' if len(all_episode_rewards) > 5 else ''}")
                    
                    # 检查奖励是否完全一样
                    if len(set(all_episode_rewards)) == 1:
                        print(f"  ⚠️  警告: 所有episode奖励完全相同！可能是环境问题或模型陷入局部最优")
        
        # 评估模型
        if iteration % args.eval_freq == 0:
            print(f"\nIteration {iteration}: 评估模型...")
            eval_stats = evaluate_model(
                model,
                args.env_name,
                num_episodes=args.eval_episodes,
                device=device,
                seed=args.seed + iteration * 10000,
                preprocess_atari=not args.no_preprocess,
            )
            
            eval_results.append({
                'iteration': iteration,
                'mean_reward': eval_stats['mean_reward'],
                'std_reward': eval_stats['std_reward'],
                'min_reward': eval_stats['min_reward'],
                'max_reward': eval_stats['max_reward'],
            })
            
            print(f"  平均奖励: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
            print(f"  范围: [{eval_stats['min_reward']:.4f}, {eval_stats['max_reward']:.4f}]")
            
            # 检查奖励是否有变化
            if len(eval_results) > 1:
                prev_reward = eval_results[-2]['mean_reward']
                curr_reward = eval_stats['mean_reward']
                change = curr_reward - prev_reward
                print(f"  与上次评估相比: {change:+.4f} ({'↑' if change > 0 else '↓' if change < 0 else '='})")
            
            # 显示部分评估奖励，帮助诊断
            if 'all_rewards' in eval_stats and len(eval_stats['all_rewards']) > 0:
                sample_rewards = eval_stats['all_rewards'][:10]
                print(f"  前10个episode奖励样本: {[f'{r:.2f}' for r in sample_rewards]}")
            
            # 保存评估结果
            eval_file = os.path.join(exp_dir, 'eval_results.json')
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
        
        # 定期保存模型
        if iteration % 100 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}_iter_{iteration}.pt")
            model.save(checkpoint_path)
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}_final.pt")
    model.save(final_checkpoint_path)
    print(f"\n最终模型已保存到: {final_checkpoint_path}")
    
    # 绘制训练曲线
    if len(eval_results) > 0:
        plot_path = os.path.join(exp_dir, 'training_curve.png')
        plot_training_curve(eval_results, plot_path)
        
        # 保存训练统计
        stats = {
            'final_mean_reward': eval_results[-1]['mean_reward'],
            'best_mean_reward': max([r['mean_reward'] for r in eval_results]),
            'total_iterations': args.num_iterations,
            'eval_episodes': args.eval_episodes,
        }
        stats_file = os.path.join(exp_dir, 'training_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n训练完成！")
        print(f"最终平均奖励: {eval_results[-1]['mean_reward']:.4f}")
        print(f"最佳平均奖励: {max([r['mean_reward'] for r in eval_results]):.4f}")
        print(f"训练曲线已保存到: {plot_path}")


if __name__ == '__main__':
    main()

