"""
在线微调CovMADT模型（适配hanabi环境）
"""
import argparse
import os
import torch
import numpy as np
from pettingzoo.classic import hanabi_v5
from tqdm import tqdm

from algorithms.covmadt import CovMADT
from data.replay_buffer import ReplayBuffer
from utils.logger import Logger
from utils.metrics import MetricsTracker


def collect_episode_hanabi(env, model, device, epsilon=0.1):
    """在hanabi环境中收集一个episode的数据"""
    episode_data = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": [],
    }
    
    env.reset()
    episode_reward = 0
    episode_steps = 0
    
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        done = terminated or truncated
        
        if done:
            env.step(None)
            continue
        
        # 获取观察和动作掩码
        state = obs["observation"]
        action_mask = obs["action_mask"]
        
        # 预测动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, obs_dim]
            mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(device)  # [1, act_dim]
            
            # 使用模型预测动作
            actions, log_probs, info = model.predict_action(
                states=state_tensor,
                mask=mask_tensor,
                deterministic=False,
            )
            
            # 获取动作索引（避免NumPy deprecation warning）
            action = int(actions.cpu().numpy().item())
            
            # Epsilon-greedy探索
            if np.random.random() < epsilon:
                valid_actions = np.where(action_mask == 1)[0]
                action = int(np.random.choice(valid_actions))
            else:
                action = int(action)
        
        # 执行动作
        env.step(action)
        
        # 获取下一个观察（下一个智能体的观察）
        next_obs = None
        if env.agent_selection:  # 如果还有下一个智能体
            next_obs, _, _, _, _ = env.last()
            next_state = next_obs["observation"]
        else:
            # Episode结束，使用当前状态
            next_state = state
        
        # 存储数据
        episode_data["states"].append(state)
        episode_data["actions"].append(action)
        episode_data["rewards"].append(reward)
        episode_data["next_states"].append(next_state)
        episode_data["dones"].append(done)
        
        episode_reward += reward
        episode_steps += 1
        
        if done:
            break
    
    # 转换为numpy数组
    episode_data = {k: np.array(v) for k, v in episode_data.items()}
    
    return episode_data, episode_reward, episode_steps


def train_step(model, batch, device, optimizer):
    """执行一个训练步骤"""
    model.train()
    
    # 准备批次数据
    states = torch.FloatTensor(batch["states"]).to(device)
    actions = torch.LongTensor(batch["actions"]).to(device)
    rewards = torch.FloatTensor(batch["rewards"]).to(device)
    next_states = torch.FloatTensor(batch["next_states"]).to(device)
    
    # 将actions转换为one-hot编码
    action_dim = model.action_dim
    actions_onehot = torch.zeros(len(actions), action_dim).to(device)
    actions_onehot.scatter_(1, actions.unsqueeze(1), 1.0)
    
    # 添加序列维度 [B, 1, ...]
    states = states.unsqueeze(1)  # [B, 1, obs_dim]
    actions_onehot = actions_onehot.unsqueeze(1)  # [B, 1, action_dim]
    rewards = rewards.unsqueeze(1).unsqueeze(-1)  # [B, 1, 1]
    next_states = next_states.unsqueeze(1)  # [B, 1, obs_dim]
    
    # 前向传播
    outputs = model(
        states=states,
        actions=actions_onehot,
        rewards=rewards,
        next_states=next_states,
        compute_loss=True,
    )
    
    # 计算损失
    loss_dict = outputs.get("loss_dict", {})
    
    # 策略损失
    policy_logits = outputs["logits"]  # [B, 1, action_dim]
    policy_loss = torch.nn.functional.cross_entropy(
        policy_logits.view(-1, action_dim),
        actions.view(-1),
    )
    
    # 价值损失
    values = outputs["values"]  # [B, 1, 1] 或 [B, 1]
    with torch.no_grad():
        next_values = model.compute_value(next_states, use_target=True)
        if next_values.dim() == 3:
            next_values = next_values.squeeze(-1)
        target_values = rewards.squeeze(-1) + model.gamma * next_values
    
    if values.dim() == 3:
        values = values.squeeze(-1)
    value_loss = torch.nn.functional.mse_loss(values, target_values)
    
    # RKHS损失
    rkhs_loss = torch.tensor(0.0, device=device)
    if "next_state_pred" in outputs and outputs["next_state_pred"] is not None:
        next_state_pred = outputs["next_state_pred"]
        if next_state_pred.shape == next_states.shape:
            rkhs_loss = torch.nn.functional.mse_loss(next_state_pred, next_states)
    
    # 凸正则化损失
    convex_loss = loss_dict.get("loss", torch.tensor(0.0, device=device))
    
    # 总损失
    total_loss = (
        1.0 * policy_loss +
        0.5 * value_loss +
        0.1 * rkhs_loss +
        1.0 * convex_loss
    )
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # 更新目标网络
    model.update_target_networks()
    
    return {
        "total_loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "rkhs_loss": rkhs_loss.item() if isinstance(rkhs_loss, torch.Tensor) else rkhs_loss,
        "convex_loss": convex_loss.item() if isinstance(convex_loss, torch.Tensor) else convex_loss,
    }


def main():
    parser = argparse.ArgumentParser(description='在线微调CovMADT模型')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='预训练模型检查点路径')
    parser.add_argument('--env_name', type=str, default='hanabi_v5',
                       help='环境名称')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='微调episode数量')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率（微调时通常较小）')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='探索率')
    parser.add_argument('--use_mfvi', action='store_true',
                       help='使用MFVI Critic（默认使用标准Critic）')
    
    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志保存目录')
    parser.add_argument('--experiment_name', type=str, default='covmadt_finetune',
                       help='实验名称')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    parser.add_argument('--replay_buffer_size', type=int, default=10000,
                       help='经验回放缓冲区大小')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("CovMADT 在线微调")
    print("=" * 60)
    print(f"预训练模型: {args.checkpoint}")
    print(f"环境: {args.env_name}")
    print(f"设备: {device}")
    print(f"Episode数: {args.num_episodes}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"探索率: {args.epsilon}")
    print(f"Critic类型: {'MFVI Critic' if args.use_mfvi else '标准Critic（默认）'}")
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
    print(f"\n加载预训练模型: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件不存在: {args.checkpoint}")
    
    # 创建模型
    model = CovMADT(
        state_dim=obs_dim,
        action_dim=act_dim,
        n_agents=num_agents,
        hidden_dim=128,
        transformer_layers=2,
        transformer_heads=4,
        rkhs_embedding_dim=128,
        kernel_type="rbf",
        tau=0.1,
        gamma=0.99,
        use_mfvi=args.use_mfvi,
        device=device,
        config={'use_occupancy_measure': False},
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 尝试直接加载
        model.load_state_dict(checkpoint, strict=False)
    
    print("✓ 模型加载成功")
    
    # 创建优化器（使用较小的学习率进行微调）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 创建经验回放缓冲区
    replay_buffer = ReplayBuffer(
        capacity=args.replay_buffer_size,
        state_dim=obs_dim,
        action_dim=act_dim,
    )
    
    # 创建日志记录器
    log_dir = os.path.join(args.log_dir, args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir=log_dir)
    
    # 创建指标跟踪器
    metrics = MetricsTracker()
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 开始微调
    print("\n开始在线微调...")
    print("=" * 60)
    
    best_reward = float('-inf')
    
    for episode in tqdm(range(1, args.num_episodes + 1), desc="微调进度"):
        # 收集数据
        episode_data, episode_reward, episode_steps = collect_episode_hanabi(
            env, model, device, epsilon=args.epsilon
        )
        
        # 存储到回放缓冲区
        for i in range(len(episode_data["states"])):
            replay_buffer.add(
                state=episode_data["states"][i],
                action=episode_data["actions"][i],
                reward=episode_data["rewards"][i],
                next_state=episode_data["next_states"][i],
                done=episode_data["dones"][i],
            )
        
        # 训练（如果缓冲区有足够数据）
        train_metrics = {}
        if len(replay_buffer) >= args.batch_size:
            batch = replay_buffer.sample(args.batch_size)
            train_metrics = train_step(model, batch, device, optimizer)
        
        # 记录指标
        episode_metrics = {
            "episode": episode,
            "episode_reward": episode_reward,
            "episode_steps": episode_steps,
            **train_metrics,
        }
        
        metrics.update(episode_metrics)
        if logger:
            logger.log_metrics(episode_metrics, step=episode)
        
        # 打印进度
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{args.num_episodes}:")
            print(f"  奖励: {episode_reward:.2f}, 步数: {episode_steps}")
            if train_metrics:
                print(f"  损失: {train_metrics.get('total_loss', 0):.4f}")
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(args.checkpoint_dir, "best_online_model_4.pt")
            model.save_checkpoint(best_path)
            if episode % 10 == 0:
                print(f"  ✓ 保存最佳模型 (奖励: {best_reward:.2f})")
    
    # 保存最终模型
    final_path = os.path.join(args.checkpoint_dir, "final_online_model_4.pt")
    model.save_checkpoint(final_path)
    
    # 保存指标
    metrics_path = os.path.join(args.checkpoint_dir, "finetune_metrics.npy")
    metrics.save(metrics_path)
    
    print("\n" + "=" * 60)
    print("在线微调完成！")
    print(f"最佳episode奖励: {best_reward:.2f}")
    print(f"模型保存在: {args.checkpoint_dir}")
    print(f"日志保存在: {log_dir}")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()


