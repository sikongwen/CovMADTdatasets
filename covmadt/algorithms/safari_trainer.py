"""
SAFARI训练器

用于训练SAFARI算法的离线训练器
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os

from algorithms.safari import SAFARI


class SAFARITrainer:
    """SAFARI离线训练器"""
    
    def __init__(
        self,
        model: SAFARI,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda",
        log_dir: Optional[str] = None,
    ):
        self.model = model
        self.device = device
        self.log_dir = log_dir
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
        )
        
        # 训练历史
        self.train_history = {
            "loss": [],
            "value_loss": [],
            "policy_loss": [],
            "conservative_loss": [],
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        total_conservative_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 解包批次数据
            states, actions, rewards, next_states, dones = batch
            
            # 移动到设备
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            if dones is not None:
                dones = dones.to(self.device)
            
            # 前向传播
            loss_dict = self.model.compute_loss(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                empirical_dist=None,  # 可以后续添加经验分布采样
            )
            
            # 反向传播
            loss = loss_dict["loss"]
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # 更新目标网络
            if batch_idx % 10 == 0:
                self.model.update_target_networks(tau=0.005)
            
            # 累积损失
            total_loss += loss.item()
            total_value_loss += loss_dict["value_loss"].item()
            total_policy_loss += loss_dict["policy_loss"].item()
            total_conservative_loss += loss_dict["conservative_loss"].item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "v_loss": f"{loss_dict['value_loss'].item():.4f}",
                "p_loss": f"{loss_dict['policy_loss'].item():.4f}",
            })
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_conservative_loss = total_conservative_loss / num_batches
        
        # 更新学习率
        self.scheduler.step(avg_loss)
        
        metrics = {
            "loss": avg_loss,
            "value_loss": avg_value_loss,
            "policy_loss": avg_policy_loss,
            "conservative_loss": avg_conservative_loss,
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_dir: Optional[str] = None,
        save_freq: int = 10,
    ) -> Dict[str, Any]:
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            num_epochs: 训练轮数
            save_dir: 模型保存目录
            save_freq: 保存频率
        """
        best_val_loss = float('inf')
        
        print("=" * 60)
        print("Starting SAFARI training...")
        print(f"Number of epochs: {num_epochs}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 记录训练历史
            for key, value in train_metrics.items():
                self.train_history[key].append(value)
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics["loss"]
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_dir is not None:
                        self.save_checkpoint(
                            os.path.join(save_dir, "best_safari_model.pt"),
                            epoch,
                            train_metrics,
                            val_metrics,
                        )
            
            # 定期保存
            if save_dir is not None and epoch % save_freq == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f"safari_model_epoch_{epoch}.pt"),
                    epoch,
                    train_metrics,
                    val_metrics if val_loader is not None else None,
                )
            
            # 打印训练信息
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"    Value Loss: {train_metrics['value_loss']:.4f}")
            print(f"    Policy Loss: {train_metrics['policy_loss']:.4f}")
            print(f"    Conservative Loss: {train_metrics['conservative_loss']:.4f}")
            
            if val_loader is not None:
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        
        return {
            "train_history": self.train_history,
            "best_val_loss": best_val_loss,
        }
    
    def validate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        total_conservative_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                states, actions, rewards, next_states, dones = batch
                
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                if dones is not None:
                    dones = dones.to(self.device)
                
                loss_dict = self.model.compute_loss(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=dones,
                    empirical_dist=None,
                )
                
                total_loss += loss_dict["loss"].item()
                total_value_loss += loss_dict["value_loss"].item()
                total_policy_loss += loss_dict["policy_loss"].item()
                total_conservative_loss += loss_dict["conservative_loss"].item()
                num_batches += 1
        
        metrics = {
            "loss": total_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "policy_loss": total_policy_loss / num_batches,
            "conservative_loss": total_conservative_loss / num_batches,
        }
        
        return metrics
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_history": self.train_history,
            "config": {
                "state_dim": self.model.state_dim,
                "action_dim": self.model.action_dim,
                "n_agents": self.model.n_agents,
                "hidden_dim": self.model.hidden_dim,
                "embedding_dim": self.model.embedding_dim,
                "gamma": self.model.gamma,
                "beta": self.model.beta,
                "alpha": self.model.alpha,
            },
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_history = checkpoint.get("train_history", self.train_history)
        
        print(f"Checkpoint loaded from {path}")
        print(f"Epoch: {checkpoint['epoch']}")
        
        return checkpoint
















