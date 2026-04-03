"""
OMIGA训练器

用于训练OMIGA算法的离线训练器
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
from tqdm import tqdm

from algorithms.omiga import OMIGA
from data.dataset import OfflineDataset
from utils.logger import Logger
from utils.metrics import MetricsTracker


class OMIGATrainer:
    """OMIGA离线训练器"""
    
    def __init__(
        self,
        model: OMIGA,
        train_dataset: OfflineDataset,
        val_dataset: Optional[OfflineDataset] = None,
        config: Dict[str, Any] = None,
        logger: Optional[Logger] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or {}
        self.logger = logger
        self.device = device
        
        # 训练参数
        self.batch_size = self.config.get("batch_size", 32)
        self.num_epochs = self.config.get("num_epochs", 100)
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.weight_decay = self.config.get("weight_decay", 1e-5)
        self.grad_clip = self.config.get("grad_clip", 1.0)
        
        # 创建数据加载器
        # 使用num_workers=2加速数据加载（如果系统支持）
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True if 2 > 0 else False,
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
        else:
            self.val_loader = None
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
        )
        
        # 指标跟踪器
        self.metrics = MetricsTracker()
        
        # 检查点目录
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 最佳验证损失
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_global_value_loss = 0.0
        total_local_value_loss = 0.0
        total_regularization_loss = 0.0
        total_conservative_loss = 0.0
        total_policy_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch in pbar:
            # 准备数据
            states = batch["states"].float().to(self.device)
            actions = batch["actions"].long().to(self.device)
            rewards = batch["rewards"].float().to(self.device)
            next_states = batch["next_states"].float().to(self.device)
            dones = batch.get("dones", None)
            
            if dones is not None:
                dones = dones.float().to(self.device)
            
            # 确保维度正确
            if states.dim() == 2:
                states = states.unsqueeze(1)  # [B, 1, S]
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)  # [B, 1]
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(1).unsqueeze(-1)  # [B, 1, 1]
            if next_states.dim() == 2:
                next_states = next_states.unsqueeze(1)  # [B, 1, S]
            if dones is not None and dones.dim() == 1:
                dones = dones.unsqueeze(1).unsqueeze(-1)  # [B, 1, 1]
            
            # 前向传播
            outputs = self.model(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                compute_loss=True,
            )
            
            # 获取损失
            loss_dict = outputs.get("loss_dict", {})
            total_loss_batch = loss_dict.get("loss", torch.tensor(0.0))
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            # 更新目标网络
            self.model.update_target_networks()
            
            # 累计损失
            total_loss += total_loss_batch.item()
            total_global_value_loss += loss_dict.get("global_value_loss", 0).item() if isinstance(loss_dict.get("global_value_loss", 0), torch.Tensor) else loss_dict.get("global_value_loss", 0)
            total_local_value_loss += loss_dict.get("local_value_loss", 0).item() if isinstance(loss_dict.get("local_value_loss", 0), torch.Tensor) else loss_dict.get("local_value_loss", 0)
            total_regularization_loss += loss_dict.get("regularization_loss", 0).item() if isinstance(loss_dict.get("regularization_loss", 0), torch.Tensor) else loss_dict.get("regularization_loss", 0)
            total_conservative_loss += loss_dict.get("conservative_loss", 0).item() if isinstance(loss_dict.get("conservative_loss", 0), torch.Tensor) else loss_dict.get("conservative_loss", 0)
            total_policy_loss += loss_dict.get("policy_loss", 0).item() if isinstance(loss_dict.get("policy_loss", 0), torch.Tensor) else loss_dict.get("policy_loss", 0)
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{total_loss_batch.item():.4f}',
                'global': f'{loss_dict.get("global_value_loss", 0).item() if isinstance(loss_dict.get("global_value_loss", 0), torch.Tensor) else loss_dict.get("global_value_loss", 0):.4f}',
                'local': f'{loss_dict.get("local_value_loss", 0).item() if isinstance(loss_dict.get("local_value_loss", 0), torch.Tensor) else loss_dict.get("local_value_loss", 0):.4f}',
            })
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "global_value_loss": total_global_value_loss / num_batches if num_batches > 0 else 0.0,
            "local_value_loss": total_local_value_loss / num_batches if num_batches > 0 else 0.0,
            "regularization_loss": total_regularization_loss / num_batches if num_batches > 0 else 0.0,
            "conservative_loss": total_conservative_loss / num_batches if num_batches > 0 else 0.0,
            "policy_loss": total_policy_loss / num_batches if num_batches > 0 else 0.0,
        }
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 准备数据
                states = batch["states"].float().to(self.device)
                actions = batch["actions"].long().to(self.device)
                rewards = batch["rewards"].float().to(self.device)
                next_states = batch["next_states"].float().to(self.device)
                dones = batch.get("dones", None)
                
                if dones is not None:
                    dones = dones.float().to(self.device)
                
                # 确保维度正确
                if states.dim() == 2:
                    states = states.unsqueeze(1)
                if actions.dim() == 1:
                    actions = actions.unsqueeze(1)
                if rewards.dim() == 1:
                    rewards = rewards.unsqueeze(1).unsqueeze(-1)
                if next_states.dim() == 2:
                    next_states = next_states.unsqueeze(1)
                if dones is not None and dones.dim() == 1:
                    dones = dones.unsqueeze(1).unsqueeze(-1)
                
                # 前向传播
                outputs = self.model(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=dones,
                    compute_loss=True,
                )
                
                # 获取损失
                loss_dict = outputs.get("loss_dict", {})
                total_loss_batch = loss_dict.get("loss", torch.tensor(0.0))
                
                total_loss += total_loss_batch.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {"val_loss": avg_loss}
    
    def train(self):
        """完整训练过程"""
        print("Starting OMIGA training...")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        
        for epoch in range(self.num_epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录指标
            metrics = {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics,
                "learning_rate": self.scheduler.get_last_lr()[0],
            }
            
            self.metrics.update(metrics)
            if self.logger:
                self.logger.log_metrics(metrics, step=epoch + 1)
            
            # 打印进度
            print(f"\nEpoch {epoch+1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"    Global Value Loss: {train_metrics['global_value_loss']:.4f}")
            print(f"    Local Value Loss: {train_metrics['local_value_loss']:.4f}")
            print(f"    Regularization Loss: {train_metrics['regularization_loss']:.4f}")
            print(f"    Conservative Loss: {train_metrics['conservative_loss']:.4f}")
            print(f"    Policy Loss: {train_metrics['policy_loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            
            # 保存最佳模型
            val_loss = val_metrics.get("val_loss", train_metrics['loss'])
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.checkpoint_dir, "best_omiga_model.pt")
                self.model.save_checkpoint(best_path)
                print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
        
        # 保存最终模型
        final_path = os.path.join(self.checkpoint_dir, "final_omiga_model.pt")
        self.model.save_checkpoint(final_path)
        print(f"\n✓ 最终模型已保存到: {final_path}")
        
        print("\nOMIGA training completed!")
        
        return self.metrics


















