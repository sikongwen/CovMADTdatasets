"""
R2D2模型评估器模块
提供模块化的R2D2模型评估功能
"""
import os
import torch
import numpy as np
import json
import csv
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

from main import R2D2Net, select_action, DEVICE

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


class R2D2Evaluator:
    """R2D2模型评估器"""
    
    def __init__(
        self,
        model: R2D2Net,
        env,
        device: torch.device,
        epsilon: float = 0.0,
    ):
        """
        初始化评估器
        
        参数:
            model: R2D2模型
            env: 环境实例
            device: 设备
            epsilon: 探索率（0表示完全贪婪）
        """
        self.model = model
        self.env = env
        self.device = device
        self.epsilon = epsilon
        
        # 设置模型为评估模式
        self.model.eval()
    
    def get_game_statistics(self, env, cumulative_reward: float = 0) -> Dict[str, Any]:
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
                    
                    # 获取烟花得分（最终得分）
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
                    if information_tokens is not None:
                        stats['information_tokens_used'] = max(0, max_information_tokens - information_tokens)
                    
                    # 尝试获取动作历史
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
        except Exception:
            pass
        
        return stats
    
    def evaluate_episode(self) -> Dict[str, Any]:
        """
        评估一个episode并返回指标
        
        返回:
            episode_stats: 包含所有指标的字典
        """
        self.env.reset()
        
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
        
        hidden_states = {}
        
        for agent in self.env.agent_iter():
            obs, reward, terminated, truncated, _ = self.env.last()
            done = terminated or truncated
            
            if done:
                self.env.step(None)
                # 获取最终统计信息
                final_stats = self.get_game_statistics(
                    self.env, 
                    cumulative_reward=episode_stats['episode_reward']
                )
                episode_stats.update(final_stats)
                break
            
            # 初始化或获取LSTM隐藏状态
            if agent not in hidden_states:
                hidden_states[agent] = self.model.init_hidden(1)
            
            # 使用模型预测动作
            obs_vec = torch.FloatTensor(
                obs["observation"]
            ).view(1, 1, -1).to(self.device)
            
            with torch.no_grad():
                q, hidden = self.model(obs_vec, hidden_states[agent])
                hidden_states[agent] = hidden
            
            # 选择动作（使用epsilon-greedy策略）
            action = select_action(
                q.squeeze(0).squeeze(0).cpu().numpy(),
                obs["action_mask"],
                eps=self.epsilon
            )
            
            # 记录动作
            episode_stats['actions_taken'].append(action)
            
            # 执行动作
            self.env.step(action)
            episode_stats['episode_reward'] += reward
            episode_stats['episode_steps'] += 1
        
        # 如果episode结束时没有获取到统计信息，再次尝试
        if episode_stats['final_score'] == 0:
            final_stats = self.get_game_statistics(
                self.env, 
                cumulative_reward=episode_stats['episode_reward']
            )
            episode_stats.update(final_stats)
        
        # 如果仍然无法获取，使用累计奖励作为得分
        if episode_stats['final_score'] == 0 and episode_stats['episode_reward'] > 0:
            episode_stats['final_score'] = int(episode_stats['episode_reward'])
            episode_stats['is_perfect_game'] = (episode_stats['final_score'] == 25)
        
        # 计算信息效率
        if episode_stats['information_tokens_used'] > 0:
            episode_stats['information_efficiency'] = (
                episode_stats['final_score'] / episode_stats['information_tokens_used']
            )
        else:
            episode_stats['information_efficiency'] = episode_stats['final_score']
        
        # 计算风险控制能力
        max_life = 6  # 4个智能体环境使用max_life=6
        episode_stats['life_loss_rate'] = (
            episode_stats['life_tokens_lost'] / max_life if max_life > 0 else 0
        )
        episode_stats['risk_control_score'] = 1.0 - episode_stats['life_loss_rate']
        
        return episode_stats
    
    def evaluate(self, num_episodes: int, verbose: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        评估多个episodes
        
        参数:
            num_episodes: episode数量
            verbose: 是否显示进度
            
        返回:
            all_episode_stats: 所有episode的统计信息列表
            stats_summary: 统计摘要
        """
        all_episode_stats = []
        
        iterator = tqdm(range(1, num_episodes + 1), desc="评估进度") if verbose else range(1, num_episodes + 1)
        
        for episode in iterator:
            episode_stats = self.evaluate_episode()
            all_episode_stats.append(episode_stats)
            
            # 每10个episode打印一次进度
            if verbose and episode % 10 == 0:
                recent_scores = [s['final_score'] for s in all_episode_stats[-10:]]
                avg_score = np.mean(recent_scores)
                print(f"\nEpisode {episode}: 最近10个episode平均得分 = {avg_score:.2f}")
        
        # 计算统计摘要
        stats_summary = self.compute_statistics(all_episode_stats)
        
        return all_episode_stats, stats_summary
    
    def compute_statistics(self, all_episode_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算统计摘要
        
        参数:
            all_episode_stats: 所有episode的统计信息列表
            
        返回:
            stats_summary: 统计摘要字典
        """
        # 提取所有指标
        final_scores = [s['final_score'] for s in all_episode_stats]
        perfect_games = [s['is_perfect_game'] for s in all_episode_stats]
        information_efficiencies = [s['information_efficiency'] for s in all_episode_stats]
        life_loss_rates = [s['life_loss_rate'] for s in all_episode_stats]
        risk_control_scores = [s['risk_control_score'] for s in all_episode_stats]
        episode_rewards = [s['episode_reward'] for s in all_episode_stats]
        
        # 计算统计信息
        stats_summary = {
            'num_episodes': len(all_episode_stats),
            
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
        
        return stats_summary
    
    def save_results(
        self,
        all_episode_stats: List[Dict[str, Any]],
        stats_summary: Dict[str, Any],
        output_dir: str,
        save_detailed: bool = False,
    ) -> Dict[str, str]:
        """
        保存评估结果
        
        参数:
            all_episode_stats: 所有episode的统计信息列表
            stats_summary: 统计摘要
            output_dir: 输出目录
            save_detailed: 是否保存详细结果
            
        返回:
            saved_files: 保存的文件路径字典
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        # 保存统计摘要
        summary_path = os.path.join(output_dir, 'r2d2_evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(stats_summary, f, indent=2)
        saved_files['summary'] = summary_path
        
        # 保存详细结果（如果启用）
        if save_detailed:
            detailed_path = os.path.join(output_dir, 'r2d2_evaluation_detailed.json')
            with open(detailed_path, 'w') as f:
                json.dump(all_episode_stats, f, indent=2)
            saved_files['detailed'] = detailed_path
        
        # 保存CSV格式
        csv_path = os.path.join(output_dir, 'r2d2_evaluation_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'episode', 'final_score', 'is_perfect_game', 'information_efficiency',
                'life_loss_rate', 'risk_control_score', 'episode_reward', 'episode_steps'
            ])
            writer.writeheader()
            for i, stats in enumerate(all_episode_stats):
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
        saved_files['csv'] = csv_path
        
        # 保存NPY格式（包含所有指标）
        npy_path = os.path.join(output_dir, 'r2d2_evaluation_results.npy')
        # 提取所有可能的指标字段
        all_keys = set()
        for stats in all_episode_stats:
            all_keys.update(stats.keys())
        
        # 定义数据类型（排除列表类型字段如actions_taken）
        exclude_keys = {'actions_taken'}  # 排除列表类型字段
        numeric_keys = sorted([k for k in all_keys if k not in exclude_keys])
        
        # 创建结构化数组
        dtype_list = []
        for key in numeric_keys:
            # 根据第一个episode的值推断类型
            sample_value = all_episode_stats[0].get(key, 0)
            if isinstance(sample_value, bool):
                dtype_list.append((key, np.bool_))
            elif isinstance(sample_value, (int, np.integer)):
                dtype_list.append((key, np.int64))
            elif isinstance(sample_value, (float, np.floating)):
                dtype_list.append((key, np.float64))
            else:
                dtype_list.append((key, np.float64))  # 默认使用float64
        
        # 创建结构化数组
        npy_array = np.zeros(len(all_episode_stats), dtype=dtype_list)
        
        # 填充数据
        for i, stats in enumerate(all_episode_stats):
            for key in numeric_keys:
                value = stats.get(key, 0)
                # 处理布尔值
                if isinstance(value, bool):
                    npy_array[i][key] = value
                else:
                    npy_array[i][key] = float(value) if value is not None else 0.0
        
        # 保存NPY文件
        np.save(npy_path, npy_array)
        saved_files['npy'] = npy_path
        print(f"✓ NPY结果已保存到: {npy_path}")
        print(f"  包含 {len(numeric_keys)} 个指标字段: {', '.join(numeric_keys)}")
        
        return saved_files
    
    def plot_results(
        self,
        all_episode_stats: List[Dict[str, Any]],
        output_dir: str,
    ) -> Optional[Dict[str, str]]:
        """
        绘制评估结果图表
        
        参数:
            all_episode_stats: 所有episode的统计信息列表
            output_dir: 输出目录
            
        返回:
            plot_files: 生成的图表文件路径字典（如果绘图可用）
        """
        if not HAS_PLOTTING:
            print("跳过绘图（matplotlib/seaborn未安装）")
            return None
        
        print("\n生成评估图表...")
        plot_files = {}
        
        # 提取数据
        episodes = list(range(1, len(all_episode_stats) + 1))
        final_scores = [s['final_score'] for s in all_episode_stats]
        information_efficiencies = [s['information_efficiency'] for s in all_episode_stats]
        risk_control_scores = [s['risk_control_score'] for s in all_episode_stats]
        life_loss_rates = [s['life_loss_rate'] for s in all_episode_stats]
        episode_rewards = [s['episode_reward'] for s in all_episode_stats]
        
        # 创建综合图表
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Final Score趋势图
        ax1 = plt.subplot(3, 3, 1)
        plt.plot(episodes, final_scores, alpha=0.6, linewidth=1.5, color='#2E86AB')
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
            [e * 10 for e in information_efficiencies],
            [s * 25 for s in risk_control_scores],
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
        
        # 8. Episode Reward趋势
        ax8 = plt.subplot(3, 3, 8)
        plt.plot(episodes, episode_rewards, alpha=0.6, linewidth=1.5, color='#8B5A3C')
        if window > 1:
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            plt.plot(moving_episodes, moving_avg, linewidth=2, color='#A23B72', label=f'Moving Avg ({window})')
            plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title('Episode Reward Over Episodes')
        plt.grid(True, alpha=0.3)
        
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
        
        # 保存综合图表
        plot_path = os.path.join(output_dir, 'r2d2_evaluation_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plot_files['plots'] = plot_path
        plt.close()
        print(f"✓ 评估图表已保存到: {plot_path}")
        
        # 创建详细的Final Score趋势图
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
            ax.legend()
        else:
            ax.plot(episodes, final_scores, linewidth=1.5, color='#2E86AB')
        
        ax.axhline(np.mean(final_scores), color='red', linestyle='--', linewidth=2, 
                   label=f'Overall Mean: {np.mean(final_scores):.2f}')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Final Score', fontsize=12)
        ax.set_title('R2D2 Final Score Trend with Confidence Interval', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        detailed_path = os.path.join(output_dir, 'r2d2_final_score_trend.png')
        plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
        plot_files['detailed_trend'] = detailed_path
        plt.close()
        print(f"✓ 详细趋势图已保存到: {detailed_path}")
        
        return plot_files
    
    def print_summary(self, stats_summary: Dict[str, Any]):
        """
        打印统计摘要
        
        参数:
            stats_summary: 统计摘要字典
        """
        print("\n" + "=" * 60)
        print("评估结果统计")
        print("=" * 60)
        
        print(f"\n1. Final Score（最终得分）:")
        print(f"   Mean ± Std: {stats_summary['final_score_mean']:.2f} ± {stats_summary['final_score_std']:.2f}")
        print(f"   Range: [{stats_summary['final_score_min']:.0f}, {stats_summary['final_score_max']:.0f}]")
        
        print(f"\n2. Perfect Game Rate（满分率）:")
        print(f"   Perfect Games: {stats_summary['perfect_game_count']}/{stats_summary['num_episodes']}")
        print(f"   Rate: {stats_summary['perfect_game_rate']*100:.2f}%")
        
        print(f"\n3. Information Efficiency（信息效率）:")
        print(f"   Mean ± Std: {stats_summary['information_efficiency_mean']:.3f} ± {stats_summary['information_efficiency_std']:.3f}")
        
        print(f"\n4. Life Loss Rate / Risk Control（风险控制能力）:")
        print(f"   Life Loss Rate (Mean ± Std): {stats_summary['life_loss_rate_mean']:.3f} ± {stats_summary['life_loss_rate_std']:.3f}")
        print(f"   Risk Control Score (Mean ± Std): {stats_summary['risk_control_score_mean']:.3f} ± {stats_summary['risk_control_score_std']:.3f}")
        
        print(f"\n5. Episode Reward（Episode奖励）:")
        print(f"   Mean ± Std: {stats_summary['episode_reward_mean']:.2f} ± {stats_summary['episode_reward_std']:.2f}")
        
        print("\n" + "=" * 60)


def load_r2d2_model(checkpoint_path: str, obs_dim: int, act_dim: int, device: torch.device) -> R2D2Net:
    """
    加载R2D2模型
    
    参数:
        checkpoint_path: 检查点路径
        obs_dim: 观察维度
        act_dim: 动作维度
        device: 设备
        
    返回:
        model: 加载的R2D2模型
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 创建网络
    net = R2D2Net(obs_dim, act_dim).to(device)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 兼容不同的检查点格式
    checkpoint_keys = list(checkpoint.keys())
    
    # 检测检查点类型
    is_covmadt = 'transformer_state_dict' in checkpoint_keys or 'rkhs_model_state_dict' in checkpoint_keys
    is_r2d2 = 'net_state_dict' in checkpoint_keys
    
    if is_covmadt:
        raise ValueError(
            f"❌ 错误: 检查点文件是CovMADT模型，不是R2D2模型！\n"
            f"   检查点包含的键: {checkpoint_keys}\n"
            f"   请使用R2D2训练的检查点文件（应包含 'net_state_dict' 键）\n"
            f"   正确的R2D2检查点通常来自: python main.py --mode train_r2d2"
        )
    
    if 'net_state_dict' in checkpoint:
        # 标准R2D2检查点格式
        net.load_state_dict(checkpoint['net_state_dict'])
        episode_info = checkpoint.get('episode', 'N/A')
        print(f"✓ R2D2模型加载成功 (训练到 Episode {episode_info})")
    elif 'model_state_dict' in checkpoint:
        # 可能是其他格式的检查点，尝试兼容加载
        print(f"⚠️  警告: 检查点格式不匹配（包含 'model_state_dict' 而不是 'net_state_dict'）")
        print(f"   尝试使用非严格模式加载...")
        try:
            net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            episode_info = checkpoint.get('episode', 'N/A')
            print(f"✓ 模型加载成功（非严格模式） (训练到 Episode {episode_info})")
            print(f"⚠️  注意: 部分权重可能未加载，建议使用正确的R2D2检查点格式")
        except Exception as e:
            raise ValueError(
                f"❌ 无法加载模型权重。错误: {e}\n"
                f"   请确保检查点文件是R2D2模型格式（应包含 'net_state_dict' 键）"
            )
    else:
        # 尝试直接加载（可能是旧格式）
        try:
            print(f"⚠️  警告: 检查点格式未知，尝试直接加载...")
            net.load_state_dict(checkpoint, strict=False)
            episode_info = 'N/A'
            print(f"✓ 模型加载成功（非严格模式）")
        except Exception as e:
            raise ValueError(
                f"❌ 无法加载模型权重。错误: {e}\n"
                f"   检查点包含的键: {checkpoint_keys}\n"
                f"   请确保检查点文件是R2D2模型格式（应包含 'net_state_dict' 键）"
            )
    
    net.eval()
    return net

