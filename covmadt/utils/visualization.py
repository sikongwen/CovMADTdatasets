import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import os
import seaborn as sns

sns.set_style("whitegrid")


class Visualization:
    """可视化工具"""
    
    def __init__(self, save_dir: str = "./plots"):
        """
        初始化可视化工具
        
        参数:
            save_dir: 保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_curves(self, metrics: Dict[str, List[Any]], save_path: Optional[str] = None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 损失曲线
        if "total_loss" in metrics:
            axes[0, 0].plot(metrics["total_loss"])
            axes[0, 0].set_title("Total Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].grid(True)
        
        # 策略损失
        if "policy_loss" in metrics:
            axes[0, 1].plot(metrics["policy_loss"])
            axes[0, 1].set_title("Policy Loss")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].grid(True)
        
        # 价值损失
        if "value_loss" in metrics:
            axes[1, 0].plot(metrics["value_loss"])
            axes[1, 0].set_title("Value Loss")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].grid(True)
        
        # 验证损失
        if "val_total_loss" in metrics:
            axes[1, 1].plot(metrics["val_total_loss"])
            axes[1, 1].set_title("Validation Loss")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, "training_curves.png")
        
        plt.savefig(save_path)
        plt.close()
        print(f"Training curves saved to {save_path}")
    
    def plot_evaluation_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """绘制评估结果"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 奖励分布
        if "rewards" in results:
            rewards = results["rewards"]
            axes[0].hist(rewards, bins=20, edgecolor='black')
            axes[0].axvline(np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
            axes[0].set_title("Reward Distribution")
            axes[0].set_xlabel("Reward")
            axes[0].set_ylabel("Frequency")
            axes[0].legend()
            axes[0].grid(True)
        
        # 步数分布
        if "steps" in results:
            steps = results["steps"]
            axes[1].hist(steps, bins=20, edgecolor='black')
            axes[1].axvline(np.mean(steps), color='r', linestyle='--', label=f'Mean: {np.mean(steps):.2f}')
            axes[1].set_title("Episode Length Distribution")
            axes[1].set_xlabel("Steps")
            axes[1].set_ylabel("Frequency")
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, "evaluation_results.png")
        
        plt.savefig(save_path)
        plt.close()
        print(f"Evaluation results saved to {save_path}")


