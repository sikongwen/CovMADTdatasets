"""
使用离线数据训练CovMADT模型
"""
import argparse
import os
import numpy as np
import torch

# 设置环境变量，避免XDG_RUNTIME_DIR警告
os.environ.setdefault('XDG_RUNTIME_DIR', '/tmp/runtime-root')

# 设置matplotlib使用非交互式后端，避免显示相关警告
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

import json
from datetime import datetime
from tqdm import tqdm
from pettingzoo.classic import hanabi_v5

from algorithms.covmadt import CovMADT
from algorithms.offline_trainer import OfflineTrainer
from data.dataset import OfflineDataset
from utils.logger import Logger
from config.base_config import BaseConfig


def create_hanabi_env(seed=None):
    """创建Hanabi环境"""
    env = hanabi_v5.env(players=4, max_life_tokens=6)
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    return env


def evaluate_episode_hanabi(env, model, device, deterministic=True):
    """评估Hanabi环境的一个episode"""
    env.reset()
    
    episode_stats = {
        'episode_reward': 0,
        'final_score': 0,
        'episode_steps': 0,
    }
    
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        done = termination or truncation
        
        if done:
            env.step(None)
            # 获取最终得分
            if isinstance(obs, dict) and 'observation' in obs:
                # 尝试从环境获取最终得分
                try:
                    if hasattr(env, 'env') and hasattr(env.env, 'env'):
                        raw_env = env.env.env
                        if hasattr(raw_env, '_state'):
                            state = raw_env._state
                            fireworks = getattr(state, 'fireworks', None)
                            if fireworks is not None:
                                if isinstance(fireworks, dict):
                                    episode_stats['final_score'] = sum(fireworks.values())
                                elif isinstance(fireworks, (list, np.ndarray)):
                                    episode_stats['final_score'] = sum(fireworks)
                except:
                    pass
            break
        
        # 获取观察和动作掩码
        if isinstance(obs, dict):
            state = obs["observation"]
            action_mask = obs.get("action_mask", None)
        else:
            state = obs
            action_mask = None
        
        # 预测动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            mask_tensor = None
            if action_mask is not None:
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
    
    # 如果最终得分为0，使用累计奖励
    if episode_stats['final_score'] == 0:
        episode_stats['final_score'] = int(episode_stats['episode_reward'])
    
    return episode_stats


def evaluate_model(model, env_name, num_episodes=200, device='cuda', seed=None):
    """评估模型，返回平均奖励"""
    episode_rewards = []
    episode_scores = []
    
    # 确保模型处于评估模式
    model.eval()
    
    with torch.no_grad():
        for episode in range(num_episodes):
            # 创建环境
            if env_name == 'hanabi_v5':
                env = create_hanabi_env(seed=seed + episode if seed is not None else None)
                episode_stats = evaluate_episode_hanabi(env, model, device, deterministic=True)
            else:
                raise ValueError(f"不支持的环境: {env_name}")
            
            # 确保存储的是Python标量，而不是numpy类型
            episode_rewards.append(float(episode_stats['episode_reward']))
            if 'final_score' in episode_stats:
                episode_scores.append(float(episode_stats['final_score']))
            
            env.close()
            
            # 每10个episode清理一次GPU缓存（如果使用GPU）
            if episode % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    result = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
    }
    
    if episode_scores:
        result['mean_score'] = float(np.mean(episode_scores))
        result['std_score'] = float(np.std(episode_scores))
    
    return result


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
    epochs = [r['epoch'] for r in eval_results]
    mean_rewards = [r['mean_reward'] for r in eval_results]
    std_rewards = [r['std_reward'] for r in eval_results]
    
    # 尝试设置中文字体
    has_chinese_font = setup_chinese_font()
    
    # 根据是否有中文字体选择标签
    if has_chinese_font:
        mean_label = '平均奖励'
        std_label = '标准差'
        ylabel = '平均奖励 (200 episodes)'
        title = 'CovMADT训练曲线'
    else:
        mean_label = 'Mean Reward'
        std_label = 'Std Dev'
        ylabel = 'Mean Reward (200 episodes)'
        title = 'CovMADT Training Curve'
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, mean_rewards, 'b-', label=mean_label, linewidth=2)
    plt.fill_between(epochs, 
                     np.array(mean_rewards) - np.array(std_rewards),
                     np.array(mean_rewards) + np.array(std_rewards),
                     alpha=0.3, color='blue', label=std_label)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='使用离线数据训练CovMADT模型')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True,
                       help='离线数据文件路径（H5格式）')
    parser.add_argument('--env_name', type=str, default='hanabi_v5',
                       help='环境名称（用于获取环境信息）')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--seq_len', type=int, default=100,
                       help='序列长度')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--lazy_load', action='store_true',
                       help='使用按需加载数据（节省内存，但可能稍慢）')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数（用于限制内存使用，None表示使用全部数据）')
    parser.add_argument('--data_usage_ratio', type=float, default=None,
                       help='数据集使用比例（0.0-1.0，例如0.5表示只使用50%%的数据，None表示使用全部数据）')
    parser.add_argument('--random_sample', action='store_true',
                       help='是否随机采样数据（默认False，使用前N%%的数据）')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--transformer_layers', type=int, default=2,
                       help='Transformer层数')
    parser.add_argument('--transformer_heads', type=int, default=4,
                       help='Transformer注意力头数')
    parser.add_argument('--rkhs_embedding_dim', type=int, default=128,
                       help='RKHS嵌入维度')
    parser.add_argument('--tau', type=float, default=0.1,
                       help='凸正则化系数')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    
    # 损失权重
    parser.add_argument('--lambda_policy', type=float, default=1.0,
                       help='策略损失权重')
    parser.add_argument('--lambda_value', type=float, default=0.5,
                       help='价值损失权重')
    parser.add_argument('--lambda_rkhs', type=float, default=0.1,
                       help='RKHS损失权重')
    parser.add_argument('--lambda_convex', type=float, default=1.0,
                       help='凸正则化损失权重')
    
    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志保存目录')
    parser.add_argument('--experiment_name', type=str, default='covmadt_training',
                       help='实验名称')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    parser.add_argument('--num_agents', type=int, default=None,
                       help='智能体数量（如果不指定，从环境获取）')
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练')
    parser.add_argument('--use_mfvi', action='store_true',
                       help='使用MFVI Critic（默认使用标准Critic）')
    parser.add_argument('--use_transformer_critic', action='store_true',
                       help='使用Transformer Critic（默认使用标准Critic）')
    parser.add_argument('--critic_use_action', type=str, default='True',
                       choices=['True', 'False', 'true', 'false'],
                       help='Transformer Critic是否使用动作作为输入（默认True）')
    
    # 评估参数
    parser.add_argument('--eval_episodes', type=int, default=200,
                       help='每次评估的episode数')
    parser.add_argument('--eval_freq', type=int, default=1,
                       help='评估频率（每N个epoch评估一次，默认每个epoch都评估）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("CovMADT 离线训练")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"设备: {device}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    if args.use_transformer_critic:
        critic_use_action = args.critic_use_action.lower() == 'true'
        print(f"Critic类型: Transformer Critic")
        print(f"  - 使用动作: {critic_use_action}")
        print(f"  - 最大序列长度: {args.seq_len}")
    elif args.use_mfvi:
        print(f"Critic类型: MFVI Critic")
    else:
        print(f"Critic类型: 标准Critic（默认）")
    print("=" * 60)
    
    # 初始化环境以获取维度信息
    print("\n初始化环境以获取维度信息...")
    
    # 支持不同的环境
    if args.env_name == "hanabi_v5":
        from pettingzoo.classic import hanabi_v5
        env = hanabi_v5.env(players=4, max_life_tokens=6)  # 4个智能体，max_life=6
        env.reset(seed=0)
        obs, _, _, _, _ = env.last()
        obs_dim = len(obs["observation"])
        act_dim = env.action_space(env.agents[0]).n
        num_agents = args.num_agents if args.num_agents is not None else len(env.agents)
    else:
        raise ValueError(f"不支持的环境: {args.env_name}")
    
    print(f"观察维度: {obs_dim}")
    print(f"动作维度: {act_dim}")
    print(f"智能体数量: {num_agents}")
    
    env.close()
    
    # 加载数据
    print(f"\n加载离线数据: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"数据文件不存在: {args.data_path}")
    
    # 检查数据格式并从数据文件读取维度信息（优先使用数据文件中的信息）
    import h5py
    with h5py.File(args.data_path, 'r') as f:
        actions_shape = f['actions'].shape
        states_shape = f['states'].shape
        print(f"数据中的states形状: {states_shape}")
        print(f"数据中的actions形状: {actions_shape}")
        
        # 从数据文件读取观察维度（如果存在元数据）
        if 'obs_dim' in f.attrs:
            data_obs_dim = int(f.attrs['obs_dim'])
            print(f"从数据文件读取观察维度: {data_obs_dim}")
            # 使用数据文件中的观察维度（更准确）
            if states_shape[1] != obs_dim:
                print(f"⚠️  警告: 环境观察维度 ({obs_dim}) 与数据文件中的观察维度 ({data_obs_dim}) 不匹配")
                print(f"   使用数据文件中的观察维度: {data_obs_dim}")
                obs_dim = data_obs_dim
        elif len(states_shape) >= 2:
            # 从实际数据形状推断观察维度
            data_obs_dim = int(states_shape[1])
            print(f"从数据形状推断观察维度: {data_obs_dim}")
            if states_shape[1] != obs_dim:
                print(f"⚠️  警告: 环境观察维度 ({obs_dim}) 与数据中的观察维度 ({data_obs_dim}) 不匹配")
                print(f"   使用数据中的观察维度: {data_obs_dim}")
                obs_dim = data_obs_dim
        
        # 从数据文件读取动作维度（如果存在元数据）
        if 'act_dim' in f.attrs:
            data_act_dim = int(f.attrs['act_dim'])
            print(f"从数据文件读取动作维度: {data_act_dim}")
            if data_act_dim != act_dim:
                print(f"⚠️  警告: 环境动作维度 ({act_dim}) 与数据文件中的动作维度 ({data_act_dim}) 不匹配")
                print(f"   使用数据文件中的动作维度: {data_act_dim}")
                act_dim = data_act_dim
        
        # 从数据文件读取智能体数量（如果存在元数据）
        if 'num_agents' in f.attrs:
            data_num_agents = int(f.attrs['num_agents'])
            print(f"从数据文件读取智能体数量: {data_num_agents}")
            if data_num_agents != num_agents:
                print(f"⚠️  警告: 环境智能体数量 ({num_agents}) 与数据文件中的智能体数量 ({data_num_agents}) 不匹配")
                print(f"   使用数据文件中的智能体数量: {data_num_agents}")
                num_agents = data_num_agents
        
        # 数据集类会自动处理，不需要在这里加载所有数据（避免内存溢出）
        if len(actions_shape) == 1:
            print("检测到1D actions格式（这是正常的，数据集类会自动处理）")
            # 注意：不需要在这里加载数据，数据集类会按需加载（lazy_load）
    
    print(f"\n最终使用的维度信息:")
    print(f"  观察维度: {obs_dim}")
    print(f"  动作维度: {act_dim}")
    print(f"  智能体数量: {num_agents}")
    
    # 创建训练和验证数据集
    print("创建数据集...")
    # 对于大数据集，默认使用lazy_load以节省内存
    # 检查数据大小，如果超过100万样本，自动启用lazy_load
    import h5py
    with h5py.File(args.data_path, 'r') as f:
        total_samples = len(f['states'])
    # 如果数据量很大（>100万样本）且未明确指定lazy_load，自动启用
    use_lazy_load = args.lazy_load if args.lazy_load else (total_samples > 1000000)
    if use_lazy_load:
        print(f"⚠️  使用按需加载模式（节省内存，数据量: {total_samples:,} 样本）")
    else:
        print(f"⚠️  使用全量加载模式（数据量: {total_samples:,} 样本）")
    if args.max_samples is not None:
        print(f"⚠️  限制最大样本数: {args.max_samples}")
    if args.data_usage_ratio is not None:
        sample_type = "随机采样" if args.random_sample else "顺序采样（前N%）"
        print(f"⚠️  使用数据集比例: {args.data_usage_ratio*100:.1f}% ({sample_type})")
    # 注意：对于hanabi，每个时间步只有一个智能体行动
    # 所以single_agent_action_dim就是act_dim，num_agents用于处理多智能体情况
    train_dataset = OfflineDataset(
        data_path=args.data_path,
        split="train",
        train_ratio=args.train_ratio,
        seq_len=args.seq_len,
        single_agent_action_dim=act_dim,
        num_agents=num_agents,
        lazy_load=use_lazy_load,
        max_samples=args.max_samples,
        data_usage_ratio=args.data_usage_ratio,
        random_sample=args.random_sample,
    )
    
    val_dataset = OfflineDataset(
        data_path=args.data_path,
        split="val",
        train_ratio=args.train_ratio,
        seq_len=args.seq_len,
        single_agent_action_dim=act_dim,
        num_agents=num_agents,
        lazy_load=use_lazy_load,
        max_samples=args.max_samples,
        data_usage_ratio=args.data_usage_ratio,
        random_sample=args.random_sample,
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    print("\n创建CovMADT模型...")
    
    # 准备config
    model_config = {
        'use_occupancy_measure': False,  # 节省内存
        'max_seq_len': args.seq_len,  # 主Transformer策略网络也需要max_seq_len
    }
    
    # 如果使用Transformer Critic，添加相关配置
    if args.use_transformer_critic:
        # 将字符串转换为布尔值
        critic_use_action = args.critic_use_action.lower() == 'true'
        model_config['critic_use_action'] = critic_use_action
    
    model = CovMADT(
        state_dim=obs_dim,
        action_dim=act_dim,
        n_agents=num_agents,
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        rkhs_embedding_dim=args.rkhs_embedding_dim,
        kernel_type="rbf",
        tau=args.tau,
        gamma=args.gamma,
        use_mfvi=args.use_mfvi,
        use_transformer_critic=args.use_transformer_critic,
        device=device,
        config=model_config,
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建日志记录器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir=log_dir)
    
    # 评估结果记录（限制大小，防止内存泄漏）
    eval_results = []
    max_eval_results = 1000  # 最多保存1000个评估结果
    
    # 创建训练配置
    config = {
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "lambda_policy": args.lambda_policy,
        "lambda_value": args.lambda_value,
        "lambda_rkhs": args.lambda_rkhs,
        "lambda_convex": args.lambda_convex,
        "checkpoint_dir": args.checkpoint_dir,
        "use_amp": args.use_amp,
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 1,
    }
    
    # 创建训练器
    print("\n创建训练器...")
    trainer = OfflineTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        logger=logger,
        device=device,
    )
    
    # 开始训练
    print("\n开始训练...")
    print(f"每个epoch后评估 {args.eval_episodes} 个episode")
    print(f"评估频率: 每 {args.eval_freq} 个epoch评估一次")
    print("=" * 60)
    
    # 修改训练器以支持评估回调
    original_train = trainer.train
    
    def train_with_eval():
        """带评估的训练函数"""
        nonlocal eval_results  # 声明使用外部作用域的eval_results变量
        
        print("Starting offline training...")
        print(f"Training dataset size: {len(trainer.train_dataset)}")
        if trainer.val_loader:
            print(f"Validation dataset size: {len(trainer.val_dataset)}")
        print(f"Number of epochs: {trainer.num_epochs}")
        print(f"Batch size: {trainer.batch_size}")
        print(f"Learning rate: {trainer.learning_rate}")
        
        # 初始化最佳平均奖励（用于保存最佳模型）
        best_mean_reward = -float('inf')
        best_epoch = 0
        
        # 限制eval_results大小，防止内存泄漏
        max_eval_results_local = 1000  # 最多保存1000个评估结果
        
        # 训练循环
        for epoch in range(1, trainer.num_epochs + 1):
            # 训练一个epoch
            train_metrics = trainer.train_epoch(epoch)
            
            # 验证
            val_metrics = trainer.validate()
            
            # 更新学习率
            trainer.scheduler.step()
            
            # 合并指标
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics["epoch"] = epoch
            all_metrics["learning_rate"] = trainer.scheduler.get_last_lr()[0]
            
            # 记录指标
            trainer.metrics.update(all_metrics)
            if trainer.logger:
                trainer.logger.log_metrics(all_metrics, step=epoch)
            
            # 打印指标
            print(f"\nEpoch {epoch}/{trainer.num_epochs}:")
            for key, value in all_metrics.items():
                if key not in ["epoch", "learning_rate"]:
                    if "rkhs_loss" in key:
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value:.4f}")
            
            # 评估模型（如果达到评估频率）
            if epoch % args.eval_freq == 0:
                print(f"\nEpoch {epoch}: 评估模型...")
                model.eval()
                # 每个epoch使用不同的随机种子，确保评估结果能反映模型变化
                eval_seed = (args.seed if hasattr(args, 'seed') else 42) + epoch * 10000
                eval_stats = evaluate_model(
                    model, args.env_name,
                    num_episodes=args.eval_episodes, device=device,
                    seed=eval_seed,
                )
                
                eval_results.append({
                    'epoch': epoch,
                    'mean_reward': eval_stats['mean_reward'],
                    'std_reward': eval_stats['std_reward'],
                    'min_reward': eval_stats['min_reward'],
                    'max_reward': eval_stats['max_reward'],
                })
                
                if 'mean_score' in eval_stats:
                    eval_results[-1]['mean_score'] = eval_stats['mean_score']
                    eval_results[-1]['std_score'] = eval_stats['std_score']
                    print(f"  平均奖励: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
                    print(f"  平均得分: {eval_stats['mean_score']:.4f} ± {eval_stats['std_score']:.4f}")
                else:
                    print(f"  平均奖励: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
                
                # 保存评估结果
                eval_file = os.path.join(log_dir, 'eval_results.json')
                with open(eval_file, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                
                # 限制eval_results大小，防止内存泄漏
                if len(eval_results) > max_eval_results_local:
                    # 只保留最近的一半结果
                    eval_results = eval_results[-max_eval_results_local//2:]
                
                # 根据平均奖励保存最佳模型
                current_mean_reward = eval_stats['mean_reward']
                if current_mean_reward > best_mean_reward:
                    best_mean_reward = current_mean_reward
                    best_epoch = epoch
                    best_path = os.path.join(trainer.checkpoint_dir, "best_model.pt")
                    trainer.model.save_checkpoint(best_path)
                    print(f"  ✓ 最佳模型已保存（Epoch {epoch}，平均奖励: {best_mean_reward:.4f}）")
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                model.train()
            
            # 保存检查点
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(
                    trainer.checkpoint_dir,
                    f"checkpoint_epoch_{epoch}.pt",
                )
                trainer.model.save_checkpoint(checkpoint_path)
                # 定期清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 保存最终模型
        final_path = os.path.join(trainer.checkpoint_dir, "final_model.pt")
        trainer.model.save_checkpoint(final_path)
        
        # 保存指标
        metrics_path = os.path.join(trainer.checkpoint_dir, "training_metrics.npy")
        trainer.metrics.save(metrics_path)
        
        print("\nTraining completed!")
        if best_mean_reward > -float('inf'):
            print(f"Best mean reward: {best_mean_reward:.4f} (Epoch {best_epoch})")
        if hasattr(trainer, 'best_val_loss'):
            print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Models saved to: {trainer.checkpoint_dir}")
        
        return trainer.metrics
    
    # 替换训练方法
    trainer.train = train_with_eval
    
    # 执行训练
    metrics = trainer.train()
    
    # 绘制训练曲线
    if len(eval_results) > 0:
        plot_path = os.path.join(log_dir, 'training_curve.png')
        plot_training_curve(eval_results, plot_path)
        
        # 保存训练统计
        stats = {
            'final_mean_reward': eval_results[-1]['mean_reward'],
            'best_mean_reward': max([r['mean_reward'] for r in eval_results]),
            'total_epochs': args.num_epochs,
            'eval_episodes': args.eval_episodes,
        }
        if 'mean_score' in eval_results[-1]:
            stats['final_mean_score'] = eval_results[-1]['mean_score']
            stats['best_mean_score'] = max([r.get('mean_score', 0) for r in eval_results])
        
        stats_file = os.path.join(log_dir, 'training_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n训练完成！")
        print(f"最终平均奖励: {eval_results[-1]['mean_reward']:.4f}")
        print(f"最佳平均奖励: {max([r['mean_reward'] for r in eval_results]):.4f}")
        if 'mean_score' in eval_results[-1]:
            print(f"最终平均得分: {eval_results[-1]['mean_score']:.4f}")
            print(f"最佳平均得分: {max([r.get('mean_score', 0) for r in eval_results]):.4f}")
        print(f"训练曲线已保存到: {plot_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"模型保存在: {args.checkpoint_dir}")
    print(f"日志保存在: {log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

