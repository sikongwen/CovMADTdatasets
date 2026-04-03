"""
OMIGA算法训练脚本

使用离线数据训练OMIGA模型
"""
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pettingzoo.classic import hanabi_v5

from algorithms.omiga import OMIGA
from algorithms.omiga_trainer import OMIGATrainer
from data.dataset import OfflineDataset
from utils.logger import Logger


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
        'final_score': 0,
        'episode_reward': 0,
        'episode_steps': 0,
    }
    
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        done = terminated or truncated
        
        if done:
            env.step(None)
            # 尝试获取最终得分
            try:
                if hasattr(env, 'env') and hasattr(env.env, 'env'):
                    raw_env = env.env.env
                    if hasattr(raw_env, '_state'):
                        state = raw_env._state
                        fireworks = getattr(state, 'fireworks', None)
                        if fireworks is not None:
                            if isinstance(fireworks, dict):
                                score = sum(fireworks.values())
                            elif isinstance(fireworks, (list, np.ndarray)):
                                score = sum(fireworks)
                            else:
                                score = 0
                            if score > 0:
                                episode_stats['final_score'] = score
            except Exception:
                pass
            
            if episode_stats['final_score'] == 0:
                episode_stats['final_score'] = int(episode_stats['episode_reward'])
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
        
        # 执行动作
        env.step(action)
        episode_stats['episode_reward'] += reward
        episode_stats['episode_steps'] += 1
    
    return episode_stats


def evaluate_model(model, env_name, num_episodes=200, device='cuda', seed=None):
    """评估模型，返回平均奖励"""
    episode_rewards = []
    episode_scores = []
    
    for episode in range(num_episodes):
        # 创建环境
        if env_name == 'hanabi_v5':
            env = create_hanabi_env(seed=seed + episode if seed is not None else None)
            episode_stats = evaluate_episode_hanabi(env, model, device, deterministic=True)
        else:
            raise ValueError(f"不支持的环境: {env_name}")
        
        episode_rewards.append(episode_stats['episode_reward'])
        if 'final_score' in episode_stats:
            episode_scores.append(episode_stats['final_score'])
        
        env.close()
    
    result = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
    }
    
    if episode_scores:
        result['mean_score'] = np.mean(episode_scores)
        result['std_score'] = np.std(episode_scores)
    
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
        title = 'OMIGA训练曲线'
    else:
        mean_label = 'Mean Reward'
        std_label = 'Std Dev'
        ylabel = 'Mean Reward (200 episodes)'
        title = 'OMIGA Training Curve'
    
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
    parser = argparse.ArgumentParser(description='使用离线数据训练OMIGA模型')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True,
                       help='离线数据文件路径（H5格式）')
    parser.add_argument('--env_name', type=str, default='hanabi_v5',
                       help='环境名称（用于获取环境信息）')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批量大小（默认64，提高训练速度）')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--seq_len', type=int, default=20,
                       help='序列长度（默认20，减少计算量）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='网络层数')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    parser.add_argument('--lambda_reg', type=float, default=0.1,
                       help='正则化系数')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='保守性系数')
    
    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志保存目录')
    parser.add_argument('--experiment_name', type=str, default='omiga_training',
                       help='实验名称')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    parser.add_argument('--num_agents', type=int, default=None,
                       help='智能体数量（如果不指定，从环境获取）')
    
    # 评估参数
    parser.add_argument('--eval_episodes', type=int, default=50,
                       help='每次评估的episode数（默认50，减少评估时间）')
    parser.add_argument('--eval_freq', type=int, default=5,
                       help='评估频率（每N个epoch评估一次，默认每5个epoch评估一次）')
    parser.add_argument('--data_usage_ratio', type=float, default=None,
                       help='数据集使用比例（0.0-1.0，例如0.5表示只使用50%的数据，None表示使用全部数据）')
    parser.add_argument('--random_sample', action='store_true',
                       help='是否随机采样（True时随机选择，False时使用前N%的数据）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("OMIGA 离线训练")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"设备: {device}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"正则化系数: {args.lambda_reg}")
    print(f"保守性系数: {args.alpha}")
    print("=" * 60)
    
    # 初始化环境以获取维度信息
    print("\n初始化环境以获取维度信息...")
    if args.env_name == 'hanabi_v5':
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
    
    # 创建训练和验证数据集
    print("创建数据集...")
    if args.data_usage_ratio is not None:
        if args.random_sample:
            print(f"随机采样 {args.data_usage_ratio*100:.1f}% 的数据")
        else:
            print(f"使用前 {args.data_usage_ratio*100:.1f}% 的数据")
    else:
        print("使用全部数据")
    
    train_dataset = OfflineDataset(
        data_path=args.data_path,
        split="train",
        train_ratio=args.train_ratio,
        seq_len=args.seq_len,
        single_agent_action_dim=act_dim,
        num_agents=num_agents,
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
        data_usage_ratio=args.data_usage_ratio,
        random_sample=args.random_sample,
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    print("\n创建OMIGA模型...")
    model = OMIGA(
        state_dim=obs_dim,
        action_dim=act_dim,
        n_agents=num_agents,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gamma=args.gamma,
        lambda_reg=args.lambda_reg,
        alpha=args.alpha,
        device=device,
    ).to(device)
    
    print(f"模型参数量: {model.get_num_params():,}")
    
    # 创建日志记录器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir=log_dir, experiment_name=args.experiment_name)
    
    # 评估结果记录
    eval_results = []
    
    # 创建训练配置
    config = {
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "checkpoint_dir": args.checkpoint_dir,
    }
    
    # 创建训练器
    print("\n创建训练器...")
    trainer = OMIGATrainer(
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
    print(f"每个epoch都会保存模型")
    print("=" * 60)
    
    # 修改训练器以支持评估回调
    original_train = trainer.train
    
    def train_with_eval():
        """带评估的训练函数"""
        print("Starting OMIGA training...")
        print(f"Training dataset size: {len(trainer.train_dataset)}")
        if trainer.val_loader:
            print(f"Validation dataset size: {len(trainer.val_dataset)}")
        print(f"Number of epochs: {trainer.num_epochs}")
        print(f"Batch size: {trainer.batch_size}")
        print(f"Learning rate: {trainer.learning_rate}")
        
        # 训练循环
        for epoch in range(trainer.num_epochs):
            # 训练一个epoch
            train_metrics = trainer.train_epoch(epoch)
            
            # 验证
            val_metrics = trainer.validate()
            
            # 更新学习率
            trainer.scheduler.step()
            
            # 合并指标
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics["epoch"] = epoch + 1
            all_metrics["learning_rate"] = trainer.scheduler.get_last_lr()[0]
            
            # 记录指标
            trainer.metrics.update(all_metrics)
            if trainer.logger:
                trainer.logger.log_metrics(all_metrics, step=epoch + 1)
            
            # 打印指标
            print(f"\nEpoch {epoch+1}/{trainer.num_epochs}:")
            for key, value in all_metrics.items():
                if key not in ["epoch", "learning_rate"]:
                    print(f"  {key}: {value:.4f}")
            
            # 评估模型（每个epoch都评估）
            print(f"\nEpoch {epoch+1}: 评估模型...")
            model.eval()
            eval_stats = evaluate_model(
                model, args.env_name,
                num_episodes=args.eval_episodes, device=device,
                seed=args.seed + (epoch + 1) * 10000,
            )
            
            eval_results.append({
                'epoch': epoch + 1,
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
            
            model.train()
            
            # 保存检查点（每个epoch都保存）
            checkpoint_path = os.path.join(
                trainer.checkpoint_dir,
                f"checkpoint_epoch_{epoch+1}.pt",
            )
            trainer.model.save_checkpoint(checkpoint_path)
            print(f"  模型已保存: {checkpoint_path}")
            
            # 保存最佳模型
            if "val_total_loss" in val_metrics:
                if val_metrics["val_total_loss"] < trainer.best_val_loss:
                    trainer.best_val_loss = val_metrics["val_total_loss"]
                    best_path = os.path.join(trainer.checkpoint_dir, "best_omiga_model.pt")
                    trainer.model.save_checkpoint(best_path)
                    print(f"New best model saved with val_loss: {trainer.best_val_loss:.4f}")
        
        # 保存最终模型
        final_path = os.path.join(trainer.checkpoint_dir, "final_omiga_model.pt")
        trainer.model.save_checkpoint(final_path)
        
        # 保存指标
        metrics_path = os.path.join(trainer.checkpoint_dir, "training_metrics.npy")
        trainer.metrics.save(metrics_path)
        
        print("\nTraining completed!")
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


















