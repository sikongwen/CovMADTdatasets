"""
使用训练好的Population MADDPG模型收集离线数据
数据格式兼容CovMADT离线预训练
"""
import os
import sys
import argparse
import numpy as np
import torch
import h5py
from tqdm import tqdm
from pettingzoo.classic import hanabi_v5

# 导入Population MADDPG
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_population_maddpg import PopulationMADDPG


def collect_data_with_population_maddpg(
    checkpoint_path: str,
    num_episodes: int = 500,
    save_path: str = "./data/offline_data_population_maddpg.h5",
    epsilon: float = 0.05,
    use_best_population: bool = True,
    device: str = "cuda",
):
    """
    使用训练好的Population MADDPG模型收集离线数据
    
    参数:
        checkpoint_path: 训练好的模型检查点路径
        num_episodes: 收集的episode数量
        save_path: 数据保存路径（H5格式）
        epsilon: 探索率（0表示完全贪婪，0.05表示5%探索）
        use_best_population: 是否使用最佳population（True）或所有population（False）
        device: 设备（cuda/cpu）
    """
    print("=" * 60)
    print("Population MADDPG 数据收集")
    print("=" * 60)
    print(f"加载模型: {checkpoint_path}")
    
    # 加载检查点获取模型配置
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dim = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']
    n_agents = checkpoint['n_agents']
    population_size = checkpoint['population_size']
    best_population_idx = checkpoint.get('best_population_idx', 0)
    
    print(f"观察维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"智能体数量: {n_agents}")
    print(f"Population大小: {population_size}")
    print(f"最佳population: {best_population_idx}")
    
    # 初始化环境
    print("\n初始化环境...")
    env = hanabi_v5.env(players=4, max_life_tokens=6)
    env.reset(seed=0)
    
    # 验证环境维度
    obs, _, _, _, _ = env.last()
    obs_dim = len(obs["observation"])
    act_dim = env.action_space(env.agents[0]).n
    
    if obs_dim != state_dim or act_dim != action_dim:
        print(f"⚠️  警告: 环境维度与模型不匹配")
        print(f"  环境: obs_dim={obs_dim}, act_dim={act_dim}")
        print(f"  模型: state_dim={state_dim}, action_dim={action_dim}")
        print(f"  使用环境维度继续...")
        state_dim = obs_dim
        action_dim = act_dim
    
    # 创建模型
    print("\n创建Population MADDPG模型...")
    model = PopulationMADDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        population_size=population_size,
        device=device,
    )
    
    # 加载检查点
    model.load_checkpoint(checkpoint_path)
    
    # 选择使用的population
    if use_best_population:
        population_indices = [model.best_population_idx]
        print(f"使用最佳population: {model.best_population_idx}")
    else:
        population_indices = list(range(population_size))
        print(f"使用所有population: {population_indices}")
    
    # 数据存储
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    
    print(f"\n开始收集数据...")
    print(f"Episodes: {num_episodes}")
    print(f"探索率: {epsilon}")
    print("=" * 60)
    
    episode_count = 0
    for episode in tqdm(range(num_episodes), desc="收集数据"):
        # 选择population（轮询）
        population_idx = population_indices[episode % len(population_indices)]
        
        # 重置环境
        env.reset()
        obs, _, _, _, _ = env.last()
        state = obs["observation"]
        
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_dones = []
        
        done = False
        step_count = 0
        
        while not done:
            # 获取动作掩码（Hanabi环境提供）
            action_mask = obs.get("action_mask", None) if isinstance(obs, dict) else None
            
            # 选择动作
            actions = model.select_actions(
                state,
                epsilon=epsilon,
                population_idx=population_idx,
                action_mask=action_mask,
            )
            action = actions[0] if isinstance(actions, np.ndarray) else actions
            
            # 安全检查：确保动作有效
            if action_mask is not None:
                valid_actions = np.where(action_mask > 0)[0]
                if action not in valid_actions:
                    # 如果动作无效，从有效动作中随机选择
                    if len(valid_actions) > 0:
                        action = int(np.random.choice(valid_actions))
                    else:
                        # 如果没有有效动作（不应该发生），使用动作0
                        action = 0
            
            # 执行动作
            env.step(action)
            obs, reward, terminated, truncated, info = env.last()
            done = terminated or truncated
            
            # 获取下一个状态
            if not done:
                next_state = obs["observation"]
            else:
                next_state = state  # Episode结束，使用当前状态
            
            # 存储数据
            episode_states.append(state.copy())
            episode_actions.append(action)  # 单个动作（Hanabi是轮流行动）
            episode_rewards.append(reward)
            episode_next_states.append(next_state.copy())
            episode_dones.append(done)
            
            state = next_state
            step_count += 1
            
            # 防止无限循环
            if step_count >= 1000:
                break
        
        # 添加到总数据
        all_states.extend(episode_states)
        all_actions.extend(episode_actions)
        all_rewards.extend(episode_rewards)
        all_next_states.extend(episode_next_states)
        all_dones.extend(episode_dones)
        
        episode_count += 1
    
    # 转换为numpy数组
    print("\n处理数据...")
    states_array = np.array(all_states, dtype=np.float32)
    actions_array = np.array(all_actions, dtype=np.int64)
    rewards_array = np.array(all_rewards, dtype=np.float32)
    next_states_array = np.array(all_next_states, dtype=np.float32)
    dones_array = np.array(all_dones, dtype=np.bool_)
    
    print(f"数据统计:")
    print(f"  总步数: {len(states_array)}")
    print(f"  状态形状: {states_array.shape}")
    print(f"  动作形状: {actions_array.shape}")
    print(f"  奖励形状: {rewards_array.shape}")
    print(f"  下一状态形状: {next_states_array.shape}")
    print(f"  终止标志形状: {dones_array.shape}")
    
    # 保存为H5格式
    print(f"\n保存数据到: {save_path}")
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('states', data=states_array, compression='gzip')
        f.create_dataset('actions', data=actions_array, compression='gzip')
        f.create_dataset('rewards', data=rewards_array, compression='gzip')
        f.create_dataset('next_states', data=next_states_array, compression='gzip')
        f.create_dataset('dones', data=dones_array, compression='gzip')
        
        # 保存元数据
        f.attrs['num_episodes'] = episode_count
        f.attrs['total_steps'] = len(states_array)
        f.attrs['state_dim'] = state_dim
        f.attrs['action_dim'] = action_dim
        f.attrs['n_agents'] = n_agents
        f.attrs['epsilon'] = epsilon
        f.attrs['population_idx'] = population_idx if use_best_population else -1
        f.attrs['checkpoint_path'] = checkpoint_path
    
    print(f"✓ 数据收集完成！")
    print(f"  保存路径: {save_path}")
    print(f"  总步数: {len(states_array)}")
    print(f"  Episodes: {episode_count}")
    print(f"  平均episode长度: {len(states_array) / episode_count:.1f}")
    
    # 验证数据格式
    print(f"\n验证数据格式...")
    with h5py.File(save_path, 'r') as f:
        print(f"  ✓ states: {f['states'].shape}, dtype: {f['states'].dtype}")
        print(f"  ✓ actions: {f['actions'].shape}, dtype: {f['actions'].dtype}")
        print(f"  ✓ rewards: {f['rewards'].shape}, dtype: {f['rewards'].dtype}")
        print(f"  ✓ next_states: {f['next_states'].shape}, dtype: {f['next_states'].dtype}")
        print(f"  ✓ dones: {f['dones'].shape}, dtype: {f['dones'].dtype}")
        print(f"  ✓ 元数据: {dict(f.attrs)}")
    
    print(f"\n数据格式验证通过！可以用于CovMADT离线预训练。")
    print(f"\n使用此数据训练CovMADT:")
    print(f"  python train_covmadt.py --data_path {save_path}")


def main():
    parser = argparse.ArgumentParser(description='使用Population MADDPG模型收集离线数据')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Population MADDPG模型检查点路径（best_model.pt）')
    
    # 数据收集参数
    parser.add_argument('--num_episodes', type=int, default=500,
                       help='收集的episode数量（默认: 500）')
    parser.add_argument('--save_path', type=str, default='./data/offline_data_population_maddpg.h5',
                       help='数据保存路径（默认: ./data/offline_data_population_maddpg.h5）')
    parser.add_argument('--epsilon', type=float, default=0.05,
                       help='探索率（默认: 0.05，5%%探索）')
    
    # 其他参数
    parser.add_argument('--use_best_population', action='store_true', default=True,
                       help='使用最佳population（默认: True）')
    parser.add_argument('--use_all_populations', action='store_true',
                       help='使用所有population（与--use_best_population互斥）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu，默认: cuda）')
    
    args = parser.parse_args()
    
    # 处理population选择
    use_best = args.use_best_population and not args.use_all_populations
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    # 开始收集数据
    collect_data_with_population_maddpg(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        save_path=args.save_path,
        epsilon=args.epsilon,
        use_best_population=use_best,
        device=device,
    )


if __name__ == '__main__':
    main()

