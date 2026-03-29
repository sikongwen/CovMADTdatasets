"""
Population MADDPG训练脚本 - Hanabi环境
Population MADDPG维护多个策略（population）来增加多样性，提高训练稳定性
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
import json
from datetime import datetime
from typing import Optional

from pettingzoo.classic import hanabi_v5
from environments.hanabi_env import HanabiEnvironment
from algorithms.maddpg import MADDPG


class PopulationMADDPG:
    """Population MADDPG算法 - 维护多个策略来增加多样性"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        population_size: int = 3,
        hidden_dim: int = 256,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        device: str = "cuda",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.population_size = population_size
        self.device = device
        
        # 为每个population创建MADDPG实例
        self.populations = []
        for i in range(population_size):
            maddpg = MADDPG(
                state_dim=state_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                single_agent_action_dim=action_dim,
                hidden_dim=hidden_dim,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                discrete=True,
                device=device,
            )
            self.populations.append(maddpg)
        
        # 选择最佳population的索引（基于平均奖励）
        self.best_population_idx = 0
        self.population_scores = [0.0] * population_size
        self.population_episodes = [0] * population_size
    
    def select_actions(
        self,
        states: np.ndarray,
        epsilon: float = 0.0,
        population_idx: Optional[int] = None,
        action_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """选择动作（从指定population或最佳population）"""
        if population_idx is None:
            population_idx = self.best_population_idx
        
        return self.populations[population_idx].select_actions(states, epsilon=epsilon, action_mask=action_mask)
    
    def push_transition(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        population_idx: Optional[int] = None,
    ):
        """添加经验到回放缓冲区（所有population共享经验）"""
        # 将经验添加到所有population的回放缓冲区
        for maddpg in self.populations:
            maddpg.push_transition(state, actions, reward, next_state, done)
    
    def update(self, batch_size: int = 64):
        """更新所有population"""
        for maddpg in self.populations:
            maddpg.update(batch_size)
    
    def update_best_population(self, episode_reward: float, population_idx: int):
        """更新最佳population"""
        self.population_scores[population_idx] = episode_reward
        self.population_episodes[population_idx] += 1
        
        # 选择平均奖励最高的population
        avg_scores = [
            self.population_scores[i] / max(1, self.population_episodes[i])
            for i in range(self.population_size)
        ]
        self.best_population_idx = np.argmax(avg_scores)
    
    def save_checkpoint(self, checkpoint_dir: str, episode: int, best: bool = False):
        """保存检查点"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'population_size': self.population_size,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'n_agents': self.n_agents,
            'best_population_idx': self.best_population_idx,
            'population_scores': self.population_scores,
            'population_episodes': self.population_episodes,
        }
        
        # 保存每个population的模型
        for i, maddpg in enumerate(self.populations):
            checkpoint[f'population_{i}_actors'] = [
                actor.state_dict() for actor in maddpg.actors
            ]
            checkpoint[f'population_{i}_critics'] = [
                critic.state_dict() for critic in maddpg.critics
            ]
        
        if best:
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_episode_{episode}.pt')
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.best_population_idx = checkpoint.get('best_population_idx', 0)
        self.population_scores = checkpoint.get('population_scores', [0.0] * self.population_size)
        self.population_episodes = checkpoint.get('population_episodes', [0] * self.population_size)
        
        # 加载每个population的模型
        for i, maddpg in enumerate(self.populations):
            if f'population_{i}_actors' in checkpoint:
                for j, actor in enumerate(maddpg.actors):
                    actor.load_state_dict(checkpoint[f'population_{i}_actors'][j])
            if f'population_{i}_critics' in checkpoint:
                for j, critic in enumerate(maddpg.critics):
                    critic.load_state_dict(checkpoint[f'population_{i}_critics'][j])
        
        print(f"✓ 已加载检查点: {checkpoint_path}")
        print(f"  最佳population: {self.best_population_idx}")
        print(f"  Population分数: {self.population_scores}")


def evaluate_episode(env, model, epsilon: float = 0.0, population_idx: Optional[int] = None):
    """评估一个episode"""
    env.reset()
    episode_reward = 0.0
    episode_steps = 0
    done = False
    
    # 获取初始观察
    obs, _, _, _, _ = env.last()
    state = obs["observation"]
    
    while not done:
        # 获取动作掩码
        action_mask = obs.get("action_mask", None) if isinstance(obs, dict) else None
        
        # 选择动作
        actions = model.select_actions(state, epsilon=epsilon, population_idx=population_idx, action_mask=action_mask)
        
        # 执行动作（Hanabi是轮流行动，只执行第一个智能体的动作）
        action = actions[0] if isinstance(actions, np.ndarray) else actions
        
        # 安全检查
        if action_mask is not None:
            valid_actions = np.where(action_mask > 0)[0]
            if action not in valid_actions:
                if len(valid_actions) > 0:
                    action = int(np.random.choice(valid_actions))
                else:
                    action = 0
        
        env.step(action)
        
        # 获取奖励和下一个观察
        obs, reward, terminated, truncated, info = env.last()
        done = terminated or truncated
        episode_reward += reward
        episode_steps += 1
        
        if not done:
            next_state = obs["observation"]
            state = next_state
    
    # 获取最终得分
    final_score = info.get('score', episode_reward) if isinstance(info, dict) else episode_reward
    
    return {
        'episode_reward': episode_reward,
        'final_score': final_score,
        'episode_steps': episode_steps,
    }


def train_population_maddpg(
    num_episodes: int = 10000,
    population_size: int = 3,
    hidden_dim: int = 256,
    lr_actor: float = 1e-4,
    lr_critic: float = 1e-3,
    batch_size: int = 64,
    gamma: float = 0.99,
    tau: float = 0.01,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    train_freq: int = 1,
    target_update_freq: int = 100,
    min_buffer_size: int = 1000,
    checkpoint_dir: str = "./checkpoints/population_maddpg",
    save_freq: int = 100,
    eval_freq: int = 100,
    num_eval_episodes: int = 10,
    device: str = "cuda",
    resume_from: Optional[str] = None,
):
    """训练Population MADDPG"""
    
    print("=" * 60)
    print("Population MADDPG 训练 - Hanabi环境")
    print("=" * 60)
    
    # 初始化环境
    print("\n初始化环境...")
    env = hanabi_v5.env(players=4, max_life_tokens=6)
    env.reset(seed=0)
    
    obs, _, _, _, _ = env.last()
    obs_dim = len(obs["observation"])
    act_dim = env.action_space(env.agents[0]).n
    num_agents = len(env.agents)
    
    print(f"观察维度: {obs_dim}")
    print(f"动作维度: {act_dim}")
    print(f"智能体数量: {num_agents}")
    print(f"Population大小: {population_size}")
    
    # 创建模型
    print("\n创建Population MADDPG模型...")
    model = PopulationMADDPG(
        state_dim=obs_dim,
        action_dim=act_dim,
        n_agents=num_agents,
        population_size=population_size,
        hidden_dim=hidden_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau,
        device=device,
    )
    
    # 加载检查点（如果指定）
    start_episode = 0
    if resume_from:
        model.load_checkpoint(resume_from)
        start_episode = torch.load(resume_from, map_location=device, weights_only=False).get('episode', 0)
        print(f"从episode {start_episode}继续训练...")
    
    # 训练统计
    epsilon = epsilon_start
    episode_rewards = []
    episode_scores = []
    best_score = -np.inf
    
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\n开始训练...")
    print(f"总episodes: {num_episodes}")
    print(f"探索率: {epsilon_start} -> {epsilon_end} (decay: {epsilon_decay})")
    print(f"训练频率: 每{train_freq}个episode")
    print(f"目标网络更新频率: 每{target_update_freq}步")
    print("=" * 60)
    
    for episode in tqdm(range(start_episode, num_episodes), desc="训练进度"):
        # 选择population（轮询或随机）
        population_idx = episode % population_size
        
        # 重置环境
        env.reset()
        obs, _, _, _, _ = env.last()
        # 对于Hanabi，使用当前智能体的观察（不是合并的观察）
        # 因为Hanabi是轮流行动，每个时间步只有一个智能体行动
        current_agent = env.agent_selection
        state = obs["observation"]  # 这是当前智能体的观察
        
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        while not done:
            # 获取动作掩码（Hanabi环境提供）
            action_mask = obs.get("action_mask", None) if isinstance(obs, dict) else None
            
            # 选择动作
            # 对于Hanabi，每个时间步只有一个智能体行动，所以只需要一个动作
            # 但MADDPG期望所有智能体的动作，我们只使用第一个
            actions = model.select_actions(state, epsilon=epsilon, population_idx=population_idx, action_mask=action_mask)
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
            episode_reward += reward
            episode_steps += 1
            
            # 获取下一个状态
            if not done:
                next_state = obs["observation"]
            else:
                next_state = state  # Episode结束，使用当前状态
            
            # 存储经验（所有population共享）
            model.push_transition(
                state=state,
                actions=actions,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            
            state = next_state if not done else state
        
        # 获取最终得分
        final_score = info.get('score', episode_reward) if isinstance(info, dict) else episode_reward
        
        # 更新population分数
        model.update_best_population(episode_reward, population_idx)
        
        # 训练（如果缓冲区有足够样本）
        if len(model.populations[0].replay_buffer) >= min_buffer_size:
            if episode % train_freq == 0:
                model.update(batch_size)
        
        # 更新目标网络
        if episode % target_update_freq == 0:
            for maddpg in model.populations:
                maddpg.train_step += 1
        
        # 记录统计
        episode_rewards.append(episode_reward)
        episode_scores.append(final_score)
        
        # 更新探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 评估
        if episode % eval_freq == 0 and episode > 0:
            eval_scores = []
            for _ in range(num_eval_episodes):
                eval_stats = evaluate_episode(env, model, epsilon=0.0)  # 评估时使用贪婪策略
                eval_scores.append(eval_stats['final_score'])
            
            avg_eval_score = np.mean(eval_scores)
            avg_reward = np.mean(episode_rewards[-eval_freq:])
            avg_score = np.mean(episode_scores[-eval_freq:])
            
            print(f"\nEpisode {episode}:")
            print(f"  平均奖励: {avg_reward:.2f}")
            print(f"  平均得分: {avg_score:.2f}")
            print(f"  评估得分: {avg_eval_score:.2f} (贪婪策略, {num_eval_episodes} episodes)")
            print(f"  探索率: {epsilon:.4f}")
            print(f"  最佳population: {model.best_population_idx}")
            print(f"  缓冲区大小: {len(model.populations[0].replay_buffer)}")
            
            # 保存最佳模型
            if avg_eval_score > best_score:
                best_score = avg_eval_score
                checkpoint_path = model.save_checkpoint(checkpoint_dir, episode, best=True)
                print(f"  ✓ 保存最佳模型: {checkpoint_path} (得分: {best_score:.2f})")
        
        # 定期保存检查点
        if episode % save_freq == 0 and episode > 0:
            checkpoint_path = model.save_checkpoint(checkpoint_dir, episode)
            print(f"\n保存检查点: {checkpoint_path}")
    
    # 保存最终模型
    final_checkpoint_path = model.save_checkpoint(checkpoint_dir, num_episodes - 1, best=False)
    print(f"\n训练完成！")
    print(f"最终检查点: {final_checkpoint_path}")
    print(f"最佳得分: {best_score:.2f}")
    print(f"最佳模型: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    
    # 保存训练统计
    stats = {
        'episode_rewards': episode_rewards,
        'episode_scores': episode_scores,
        'best_score': best_score,
        'final_epsilon': epsilon,
    }
    stats_path = os.path.join(checkpoint_dir, 'training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"训练统计已保存: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Population MADDPG训练 - Hanabi环境')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=10000,
                       help='训练episode数量（默认: 10000）')
    parser.add_argument('--population_size', type=int, default=3,
                       help='Population大小（默认: 3）')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='隐藏层维度（默认: 256）')
    parser.add_argument('--lr_actor', type=float, default=1e-4,
                       help='Actor学习率（默认: 1e-4）')
    parser.add_argument('--lr_critic', type=float, default=1e-3,
                       help='Critic学习率（默认: 1e-3）')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批量大小（默认: 64）')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子（默认: 0.99）')
    parser.add_argument('--tau', type=float, default=0.01,
                       help='软更新系数（默认: 0.01）')
    
    # 探索参数
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                       help='初始探索率（默认: 1.0）')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                       help='最终探索率（默认: 0.01）')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                       help='探索率衰减（默认: 0.995）')
    
    # 训练频率
    parser.add_argument('--train_freq', type=int, default=1,
                       help='训练频率（每N个episode训练一次，默认: 1）')
    parser.add_argument('--target_update_freq', type=int, default=100,
                       help='目标网络更新频率（每N步，默认: 100）')
    parser.add_argument('--min_buffer_size', type=int, default=1000,
                       help='开始训练的最小缓冲区大小（默认: 1000）')
    
    # 保存和评估
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/population_maddpg',
                       help='检查点保存目录（默认: ./checkpoints/population_maddpg）')
    parser.add_argument('--save_freq', type=int, default=100,
                       help='保存检查点频率（每N个episode，默认: 100）')
    parser.add_argument('--eval_freq', type=int, default=100,
                       help='评估频率（每N个episode，默认: 100）')
    parser.add_argument('--num_eval_episodes', type=int, default=10,
                       help='评估episode数量（默认: 10）')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu，默认: cuda）')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从检查点恢复训练（默认: None）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    # 开始训练
    train_population_maddpg(
        num_episodes=args.num_episodes,
        population_size=args.population_size,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        train_freq=args.train_freq,
        target_update_freq=args.target_update_freq,
        min_buffer_size=args.min_buffer_size,
        checkpoint_dir=args.checkpoint_dir,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        num_eval_episodes=args.num_eval_episodes,
        device=device,
        resume_from=args.resume_from,
    )


if __name__ == '__main__':
    main()

