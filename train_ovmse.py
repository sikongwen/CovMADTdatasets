"""
OVMSE算法训练脚本

使用示例：
    python train_ovmse.py --env_name "Hanabi-Full" --num_episodes 1000
"""
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm

from algorithms.ovmse import OVMSE
from algorithms.ovmse_trainer import OVMSETrainer
from utils.logger import Logger


def create_env(env_name: str, players: int = 4, max_life_tokens: int = 6, seed=None):
    """创建环境"""
    # 检查是否是Hanabi环境
    if 'hanabi' in env_name.lower():
        try:
            from pettingzoo.classic import hanabi_v5
            env = hanabi_v5.env(players=players, max_life_tokens=max_life_tokens)
            if seed is not None:
                env.reset(seed=seed)
            else:
                env.reset()
            return env
        except ImportError:
            print("错误: 无法导入Hanabi环境，请安装pettingzoo: pip install pettingzoo")
            raise
        except Exception as e:
            print(f"错误: 无法创建Hanabi环境: {e}")
            raise
    else:
        # 其他Gym环境
        try:
            import gymnasium as gym
            env = gym.make(env_name)
            if seed is not None:
                env.reset(seed=seed)
            else:
                env.reset()
            return env
        except Exception as e:
            print(f"Warning: Could not create gym environment: {e}")
            # 返回一个简单的mock环境用于测试
            class MockEnv:
                def __init__(self):
                    self.state_dim = 128
                    self.action_dim = 4
                    self.observation_space = type('obj', (object,), {'shape': (128,)})
                    self.action_space = type('obj', (object,), {'n': 4})
                
                def reset(self, seed=None):
                    return np.random.randn(128), {}
                
                def step(self, action):
                    next_obs = np.random.randn(128)
                    reward = np.random.randn()
                    done = False
                    truncated = False
                    info = {}
                    return next_obs, reward, done, truncated, info
            
            return MockEnv()


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
        
        # 执行动作
        env.step(action)
        episode_stats['episode_reward'] += reward
        episode_stats['episode_steps'] += 1
    
    return episode_stats


def evaluate_episode_gym(env, model, device, deterministic=True):
    """评估Gym环境的一个episode"""
    obs, info = env.reset()
    
    episode_stats = {
        'episode_reward': 0,
        'episode_steps': 0,
    }
    
    done = False
    truncated = False
    
    while not (done or truncated):
        # 预测动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            actions, log_probs, info = model.predict_action(
                states=state_tensor,
                deterministic=deterministic,
            )
            action = actions.cpu().numpy().item()
        
        # 执行动作
        next_obs, reward, done, truncated, info = env.step(action)
        
        # 记录统计
        episode_stats['episode_reward'] += reward
        episode_stats['episode_steps'] += 1
        
        obs = next_obs
    
    return episode_stats


def evaluate_model(model, env_name, players=4, max_life_tokens=6, num_episodes=200, device='cuda', seed=None):
    """评估模型，返回平均奖励"""
    episode_rewards = []
    episode_scores = []
    
    is_hanabi = 'hanabi' in env_name.lower()
    
    for episode in range(num_episodes):
        # 创建环境
        env = create_env(env_name, players=players, max_life_tokens=max_life_tokens, seed=seed + episode if seed is not None else None)
        
        # 评估episode
        if is_hanabi:
            episode_stats = evaluate_episode_hanabi(env, model, device, deterministic=True)
        else:
            episode_stats = evaluate_episode_gym(env, model, device, deterministic=True)
        
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
    episodes = [r['episode'] for r in eval_results]
    mean_rewards = [r['mean_reward'] for r in eval_results]
    std_rewards = [r['std_reward'] for r in eval_results]
    
    # 尝试设置中文字体
    has_chinese_font = setup_chinese_font()
    
    # 根据是否有中文字体选择标签
    if has_chinese_font:
        mean_label = '平均奖励'
        std_label = '标准差'
        ylabel = '平均奖励 (200 episodes)'
        title = 'OVMSE训练曲线'
    else:
        mean_label = 'Mean Reward'
        std_label = 'Std Dev'
        ylabel = 'Mean Reward (200 episodes)'
        title = 'OVMSE Training Curve'
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, mean_rewards, 'b-', label=mean_label, linewidth=2)
    plt.fill_between(episodes, 
                     np.array(mean_rewards) - np.array(std_rewards),
                     np.array(mean_rewards) + np.array(std_rewards),
                     alpha=0.3, color='blue', label=std_label)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='OVMSE算法训练')
    parser.add_argument('--env_name', type=str, default='hanabi_v5',
                       help='环境名称（hanabi_v5 或 Gym环境名称）')
    parser.add_argument('--state_dim', type=int, default=None,
                       help='状态维度（如果环境不支持自动获取）')
    parser.add_argument('--action_dim', type=int, default=None,
                       help='动作维度（如果环境不支持自动获取）')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='网络层数')
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='训练episode数量')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    parser.add_argument('--kl_weight', type=float, default=0.01,
                       help='KL散度权重')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='计算设备')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志保存目录')
    parser.add_argument('--players', type=int, default=4,
                       help='Hanabi环境玩家数量（仅Hanabi环境）')
    parser.add_argument('--max_life_tokens', type=int, default=6,
                       help='Hanabi环境最大生命令牌数（仅Hanabi环境）')
    
    # 评估参数
    parser.add_argument('--eval_episodes', type=int, default=200,
                       help='每次评估的episode数')
    parser.add_argument('--eval_freq', type=int, default=1,
                       help='评估频率（每N个episode评估一次，默认每个episode都评估）')
    parser.add_argument('--experiment_name', type=str, default='ovmse_training',
                       help='实验名称')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 创建环境
    print(f"Creating environment: {args.env_name}")
    is_hanabi = 'hanabi' in args.env_name.lower()
    
    if is_hanabi:
        env = create_env(args.env_name, players=args.players, max_life_tokens=args.max_life_tokens)
        # 获取Hanabi环境的维度
        obs, _, _, _, _ = env.last()
        state_dim = len(obs["observation"])
        action_dim = env.action_space(env.agents[0]).n
        num_agents = len(env.agents)
        print(f"Hanabi环境配置: {num_agents}个智能体, max_life_tokens={args.max_life_tokens}")
    else:
        env = create_env(args.env_name)
        # 获取Gym环境的维度
        if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'shape'):
            state_dim = env.observation_space.shape[0] if len(env.observation_space.shape) > 0 else args.state_dim
        elif hasattr(env, 'state_dim'):
            state_dim = env.state_dim
        else:
            state_dim = args.state_dim or 128
        
        if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
            action_dim = env.action_space.n
        elif hasattr(env, 'action_dim'):
            action_dim = env.action_dim
        else:
            action_dim = args.action_dim or 4
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # 创建模型
    print("Creating OVMSE model...")
    model = OVMSE(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gamma=args.gamma,
        kl_weight=args.kl_weight,
        device=args.device,
    )
    
    print(f"Model created with {model.get_num_params() if hasattr(model, 'get_num_params') else 'unknown'} parameters")
    
    # 创建日志记录器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir=log_dir, experiment_name='ovmse')
    
    # 评估结果记录
    eval_results = []
    
    # 创建训练器
    print("Creating trainer...")
    trainer = OVMSETrainer(
        model=model,
        env=env,
        config={
            'num_episodes': args.num_episodes,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'gamma': args.gamma,
            'checkpoint_dir': args.checkpoint_dir,
        },
        logger=logger,
        device=args.device,
    )
    
    # 开始训练
    print("Starting training...")
    print(f"每个episode后评估 {args.eval_episodes} 个episode")
    print(f"评估频率: 每 {args.eval_freq} 个episode评估一次")
    print("=" * 60)
    
    # 修改训练器以支持评估回调
    original_train = trainer.train
    
    def train_with_eval():
        """带评估的训练函数"""
        print("Starting OVMSE training...")
        print(f"Number of episodes: {trainer.num_episodes}")
        print(f"Batch size: {trainer.batch_size}")
        print(f"Learning rate: {trainer.learning_rate}")
        
        best_reward = float('-inf')
        
        for episode in range(1, trainer.num_episodes + 1):
            # 收集数据
            episode_data, episode_reward, episode_steps = trainer.collect_episode()
            
            # 存储到回放缓冲区
            for i in range(len(episode_data["states"])):
                trainer.replay_buffer.add(
                    state=episode_data["states"][i],
                    action=episode_data["actions"][i],
                    reward=episode_data["rewards"][i],
                    next_state=episode_data["next_states"][i],
                    done=episode_data["dones"][i],
                )
            
            # 训练（如果缓冲区有足够数据）
            if len(trainer.replay_buffer) >= trainer.batch_size:
                batch = trainer.replay_buffer.sample(trainer.batch_size)
                # 确保所有tensor都是float32类型
                batch = {
                    k: torch.tensor(v, dtype=torch.float32 if k in ['states', 'rewards', 'next_states', 'dones'] else torch.long if k == 'actions' else torch.float32).to(trainer.device)
                    for k, v in batch.items()
                }
                
                train_metrics = trainer.train_step(batch)
            else:
                train_metrics = {}
            
            # 记录指标
            metrics = {
                "episode": episode,
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                **train_metrics,
            }
            
            trainer.metrics.update(metrics)
            if trainer.logger:
                trainer.logger.log_metrics(metrics, step=episode)
            
            # 打印进度
            if episode % 10 == 0:
                print(f"Episode {episode}/{trainer.num_episodes}: "
                      f"Reward: {episode_reward:.2f}, Steps: {episode_steps}, "
                      f"Loss: {train_metrics.get('loss', 0):.4f}")
            
            # 评估模型（如果达到评估频率）
            if episode % args.eval_freq == 0:
                print(f"\nEpisode {episode}: 评估模型...")
                model.eval()
                eval_stats = evaluate_model(
                    model, args.env_name, args.players, args.max_life_tokens,
                    num_episodes=args.eval_episodes, device=args.device, 
                    seed=args.seed + episode * 10000
                )
                
                eval_results.append({
                    'episode': episode,
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
            
            # 保存最佳模型
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_path = os.path.join(trainer.checkpoint_dir, "best_ovmse_model.pt")
                trainer.model.save_checkpoint(best_path)
        
        print("\nOVMSE training completed!")
        print(f"Best episode reward: {best_reward:.2f}")
        
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
            'total_episodes': args.num_episodes,
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
    
    print("\nTraining completed!")
    print(f"模型保存在: {args.checkpoint_dir}")
    print(f"日志保存在: {log_dir}")


if __name__ == '__main__':
    main()

