"""
ComaDICE训练器

用于训练ComaDICE算法的离线训练器
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
from tqdm import tqdm

from algorithms.comadice import ComaDICE
from data.dataset import OfflineDataset
from utils.logger import Logger
from utils.metrics import MetricsTracker


class ComaDICETrainer:
    """ComaDICE离线训练器"""
    
    def __init__(
        self,
        model: ComaDICE,
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
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
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
        total_td_loss = 0.0
        total_conservative_loss = 0.0
        total_weight_reg = 0.0
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
            total_td_loss += loss_dict.get("td_loss", 0).item() if isinstance(loss_dict.get("td_loss", 0), torch.Tensor) else loss_dict.get("td_loss", 0)
            total_conservative_loss += loss_dict.get("conservative_loss", 0).item() if isinstance(loss_dict.get("conservative_loss", 0), torch.Tensor) else loss_dict.get("conservative_loss", 0)
            total_weight_reg += loss_dict.get("weight_reg", 0).item() if isinstance(loss_dict.get("weight_reg", 0), torch.Tensor) else loss_dict.get("weight_reg", 0)
            total_policy_loss += loss_dict.get("policy_loss", 0).item() if isinstance(loss_dict.get("policy_loss", 0), torch.Tensor) else loss_dict.get("policy_loss", 0)
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{total_loss_batch.item():.4f}',
                'td': f'{loss_dict.get("td_loss", 0).item() if isinstance(loss_dict.get("td_loss", 0), torch.Tensor) else loss_dict.get("td_loss", 0):.4f}',
            })
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "td_loss": total_td_loss / num_batches if num_batches > 0 else 0.0,
            "conservative_loss": total_conservative_loss / num_batches if num_batches > 0 else 0.0,
            "weight_reg": total_weight_reg / num_batches if num_batches > 0 else 0.0,
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
        print("Starting ComaDICE training...")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        
        # 首先训练行为策略（行为克隆）
        print("\n训练行为策略网络（行为克隆）...")
        # 从数据集中提取所有状态和动作
        all_states = []
        all_actions = []
        for batch in self.train_loader:
            states = batch["states"].float()
            actions = batch["actions"]
            
            # 处理 states
            if states.dim() == 3:
                states = states.view(-1, states.shape[-1])  # [B*T, S]
            
            # 处理 actions：可能是 one-hot 格式 [B, T, A] 或索引格式 [B, T]
            action_dim = self.model.action_dim
            if actions.dim() == 3:
                # [B, T, A] - one-hot 格式，转换为索引
                if actions.shape[-1] == action_dim:
                    actions = torch.argmax(actions, dim=-1)  # [B, T]
                    actions = actions.view(-1)  # [B*T]
                else:
                    # [B, T, 1] -> [B, T]
                    actions = actions.squeeze(-1).view(-1)
            elif actions.dim() == 2:
                # [B, T] 或 [B, T*A]
                if actions.shape[-1] == action_dim:
                    # [B, T, A] 被当作 [B, T*A] 处理，需要reshape
                    # 从 states 推断 batch_size 和 seq_len
                    if states.dim() == 3:
                        batch_size, seq_len = states.shape[0], states.shape[1]
                    else:
                        # states 已经展平，需要从原始形状推断
                        # 这里假设 states 是 [B*T, S]
                        batch_size = len(states) // (states.shape[-1] if states.dim() == 2 else 1)
                        seq_len = 1
                    actions = actions.view(batch_size, seq_len, action_dim)
                    actions = torch.argmax(actions, dim=-1).view(-1)  # [B*T]
                else:
                    # [B, T] - 索引格式
                    actions = actions.view(-1)  # [B*T]
            elif actions.dim() == 1:
                # 已经是展平格式
                pass
            
            all_states.append(states)
            all_actions.append(actions.long())
        
        all_states = torch.cat(all_states, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        
        self.model.update_behavior_policy(
            states=all_states.to(self.device),
            actions=all_actions.to(self.device),
            num_epochs=10,
            batch_size=self.batch_size,
            lr=1e-3,
        )
        
        print("\n开始ComaDICE训练...")
        
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
            print(f"    TD Loss: {train_metrics['td_loss']:.4f}")
            print(f"    Conservative Loss: {train_metrics['conservative_loss']:.4f}")
            print(f"    Weight Reg: {train_metrics['weight_reg']:.4f}")
            print(f"    Policy Loss: {train_metrics['policy_loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            
            # 保存最佳模型
            val_loss = val_metrics.get("val_loss", train_metrics['loss'])
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.checkpoint_dir, "best_comadice_model.pt")
                self.model.save_checkpoint(best_path)
                print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
        
        # 保存最终模型
        final_path = os.path.join(self.checkpoint_dir, "final_comadice_model.pt")
        self.model.save_checkpoint(final_path)
        print(f"\n✓ 最终模型已保存到: {final_path}")
        
        print("\nComaDICE training completed!")
        
        return self.metrics





