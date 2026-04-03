"""
使用训练好的R2D2模型采集离线数据
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pettingzoo.classic import hanabi_v5
import os
import argparse
import h5py
from tqdm import tqdm
from main import R2D2Net, select_action, DEVICE


def collect_data_with_model(
    checkpoint_path: str,
    num_episodes: int = 100,
    save_path: str = "./data/offline_data.h5",
    epsilon: float = 0.05,
    env_name: str = "hanabi_v5",
):
    """
    使用训练好的模型采集离线数据
    
    参数:
        checkpoint_path: 训练好的模型检查点路径
        num_episodes: 收集的episode数量
        save_path: 数据保存路径（H5格式）
        epsilon: 探索率（0表示完全贪婪，0.05表示5%探索）
        env_name: 环境名称
    """
    print(f"加载模型: {checkpoint_path}")
    
    # 初始化环境
    if env_name == "hanabi_v5":
        env = hanabi_v5.env(players=4, max_life_tokens=6)  # 4个智能体，max_life=6
    else:
        raise ValueError(f"不支持的环境: {env_name}")
    
    env.reset(seed=0)
    
    # 获取环境信息
    obs, _, _, _, _ = env.last()
    obs_dim = len(obs["observation"])
    act_dim = env.action_space(env.agents[0]).n
    
    print(f"观察维度: {obs_dim}, 动作维度: {act_dim}")
    
    # 创建网络并加载权重
    net = R2D2Net(obs_dim, act_dim).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # 兼容不同的检查点格式
    checkpoint_keys = list(checkpoint.keys())
    print(f"检查点文件中的键: {checkpoint_keys}")
    
    if 'net_state_dict' in checkpoint:
        # R2D2格式（main.py保存的）
        net.load_state_dict(checkpoint['net_state_dict'])
        episode_info = checkpoint.get('episode', 'N/A')
        print(f"✓ 使用R2D2格式检查点")
    elif 'model_state_dict' in checkpoint:
        # 其他模型格式，尝试加载但可能不兼容
        print(f"⚠️  警告: 检查点文件格式不匹配（包含 'model_state_dict' 而不是 'net_state_dict'）")
        print(f"   这可能是其他模型的检查点，不是R2D2模型的。")
        print(f"   建议使用R2D2训练的检查点，例如:")
        print(f"   - ./checkpoints/latest_checkpoint.pt")
        print(f"   - ./checkpoints/final_model.pt")
        print(f"   - ./checkpoints/checkpoint_episode_*.pt")
        print(f"   尝试使用 strict=False 加载（可能失败或不兼容）...")
        try:
            net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            episode_info = 'N/A'
            print(f"⚠️  已使用非严格模式加载，模型可能不完整或不兼容")
        except Exception as e:
            raise ValueError(
                f"无法加载模型权重。\n"
                f"错误: {e}\n"
                f"检查点文件中的键: {checkpoint_keys}\n"
                f"请使用R2D2训练的检查点文件。"
            )
    else:
        # 尝试直接加载整个checkpoint（可能是直接保存的state_dict）
        print(f"⚠️  检查点格式未知，尝试直接加载...")
        try:
            net.load_state_dict(checkpoint, strict=False)
            episode_info = 'N/A'
            print(f"⚠️  已使用非严格模式加载")
        except Exception as e:
            raise ValueError(
                f"无法加载模型权重。\n"
                f"错误: {e}\n"
                f"检查点文件中的键: {checkpoint_keys}\n"
                f"请确保使用R2D2训练的检查点文件（应包含 'net_state_dict' 键）。"
            )
    
    net.eval()  # 设置为评估模式
    
    print(f"✓ 模型加载成功 (训练到 Episode {episode_info})")
    print(f"✓ 使用探索率: {epsilon}")
    print(f"✓ 开始收集 {num_episodes} 个episodes的数据...\n")
    
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 存储所有episode的数据
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    all_episode_rewards = []
    
    # 收集数据
    for episode in tqdm(range(num_episodes), desc="收集数据"):
        episode_data = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }
        
        hidden_states = {}
        env.reset()
        episode_reward = 0
        
        # 先收集所有状态和动作
        states_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, _ = env.last()
            done = terminated or truncated
            
            if done:
                env.step(None)
                continue
            
            # 初始化或获取LSTM隐藏状态
            if agent not in hidden_states:
                hidden_states[agent] = net.init_hidden(1)
            
            # 使用模型预测动作
            obs_vec = torch.FloatTensor(
                obs["observation"]
            ).view(1, 1, -1).to(DEVICE)
            
            with torch.no_grad():
                q, hidden = net(obs_vec, hidden_states[agent])
                hidden_states[agent] = hidden
            
            # 选择动作（使用epsilon-greedy策略）
            action = select_action(
                q.squeeze(0).squeeze(0).cpu().numpy(),
                obs["action_mask"],
                eps=epsilon
            )
            
            # 存储当前状态和动作
            states_list.append(obs["observation"])
            actions_list.append(action)
            rewards_list.append(reward)
            dones_list.append(done)
            episode_reward += reward
            
            # 执行动作
            env.step(action)
        
        # 处理next_states：下一个状态就是序列中的下一个状态
        # 最后一个状态的next_state就是自己（episode结束）
        next_states_list = []
        for i in range(len(states_list)):
            if i + 1 < len(states_list):
                next_states_list.append(states_list[i + 1])
            else:
                # 最后一个状态，next_state就是自己（episode结束）
                next_states_list.append(states_list[i])
        
        # 保存到episode_data
        episode_data["states"] = states_list
        episode_data["actions"] = actions_list
        episode_data["rewards"] = rewards_list
        episode_data["dones"] = dones_list
        episode_data["next_states"] = next_states_list
        
        # 添加到总数据
        all_states.extend(episode_data["states"])
        all_actions.extend(episode_data["actions"])
        all_rewards.extend(episode_data["rewards"])
        all_next_states.extend(episode_data["next_states"])
        all_dones.extend(episode_data["dones"])
        all_episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_episode_rewards[-10:])
            print(f"Episode {episode + 1}: 平均奖励 (最近10个) = {avg_reward:.2f}")
    
    # 转换为numpy数组
    all_states = np.array(all_states, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.int32)
    all_rewards = np.array(all_rewards, dtype=np.float32)
    all_next_states = np.array(all_next_states, dtype=np.float32)
    all_dones = np.array(all_dones, dtype=np.bool_)
    
    print(f"\n✓ 数据收集完成!")
    print(f"  - 总样本数: {len(all_states)}")
    print(f"  - 平均episode奖励: {np.mean(all_episode_rewards):.2f}")
    print(f"  - 最佳episode奖励: {np.max(all_episode_rewards):.2f}")
    print(f"  - 最差episode奖励: {np.min(all_episode_rewards):.2f}")
    
    # 保存到H5文件
    print(f"\n保存数据到: {save_path}")
    with h5py.File(save_path, 'w') as f:
        f.create_dataset("states", data=all_states, compression="gzip", compression_opts=4)
        f.create_dataset("actions", data=all_actions, compression="gzip", compression_opts=4)
        f.create_dataset("rewards", data=all_rewards, compression="gzip", compression_opts=4)
        f.create_dataset("next_states", data=all_next_states, compression="gzip", compression_opts=4)
        f.create_dataset("dones", data=all_dones, compression="gzip", compression_opts=4)
        
        # 保存元数据
        f.attrs["num_episodes"] = num_episodes
        f.attrs["epsilon"] = epsilon
        f.attrs["checkpoint_path"] = checkpoint_path
        f.attrs["env_name"] = env_name
        f.attrs["obs_dim"] = obs_dim
        f.attrs["act_dim"] = act_dim
        f.attrs["avg_episode_reward"] = float(np.mean(all_episode_rewards))
    
    print(f"✓ 数据已保存到: {save_path}")
    print(f"  文件大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    
    env.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型采集离线数据')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='训练好的模型检查点路径')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='收集的episode数量')
    parser.add_argument('--save_path', type=str, default='./data/offline_data.h5',
                       help='数据保存路径')
    parser.add_argument('--epsilon', type=float, default=0.05,
                       help='探索率 (0=完全贪婪, 0.05=5%%探索)')
    parser.add_argument('--env_name', type=str, default='hanabi_v5',
                       help='环境名称')
    
    args = parser.parse_args()
    
    collect_data_with_model(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        save_path=args.save_path,
        epsilon=args.epsilon,
        env_name=args.env_name,
    )


if __name__ == "__main__":
    main()

