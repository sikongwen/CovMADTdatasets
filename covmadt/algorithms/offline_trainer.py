import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
from tqdm import tqdm

from algorithms.covmadt import CovMADT
from data.dataset import OfflineDataset
from utils.logger import Logger
from utils.metrics import MetricsTracker


class OfflineTrainer:
    """离线训练器"""
    
    def __init__(
        self,
        model: CovMADT,
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
        self.num_epochs = self.config.get("num_epochs", 10)
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.weight_decay = self.config.get("weight_decay", 1e-5)
        self.grad_clip = self.config.get("grad_clip", 1.0)
        
        # 损失权重
        self.lambda_policy = self.config.get("lambda_policy", 1.0)
        self.lambda_value = self.config.get("lambda_value", 0.5)
        self.lambda_rkhs = self.config.get("lambda_rkhs", 0.1)
        self.lambda_convex = self.config.get("lambda_convex", 1.0)
        
        # 内存优化选项
        self.use_amp = self.config.get("use_amp", False)
        self.gradient_checkpointing = self.config.get("gradient_checkpointing", False)
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        
        # 混合精度训练的scaler
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # 创建数据加载器（使用pin_memory=False以节省内存）
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Windows兼容性
            pin_memory=True,
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,  # 节省内存
                persistent_workers=False,
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
        epoch_metrics = {
            "total_loss": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "rkhs_loss": 0,
            "convex_loss": 0,
            "utility": 0,
            "kl_term": 0,
            "exploitability": 0,
        }
        
        with tqdm(self.train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # 移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 清理CUDA缓存（更频繁地清理以节省内存）
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                
                # 前向传播和损失计算
                # 调试：在第一个batch打印
                if batch_idx == 0 and epoch == 1:
                    print(f"\n=== RKHS Debug Start (Epoch {epoch}, Batch {batch_idx}) ===")
                    print(f"Using AMP: {self.use_amp}")
                    print(f"Batch actions shape: {batch['actions'].shape}")
                    print(f"Batch states shape: {batch['states'].shape}")
                    print(f"Batch next_states shape: {batch['next_states'].shape}")
                
                if self.use_amp:
                    # 在 autocast 外初始化 rkhs_loss_for_accumulation
                    rkhs_loss_for_accumulation = 0.0
                    
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            states=batch["states"],
                            actions=batch["actions"],
                            agent_ids=batch.get("agent_ids"),
                            rewards=batch["rewards"],
                            next_states=batch["next_states"],
                            action_mask=batch.get("action_mask"),
                            compute_loss=True,
                        )
                        
                        # 调试：立即检查 outputs
                        if batch_idx == 0 and epoch == 1:
                            print(f"Outputs keys (AMP): {list(outputs.keys())}")
                            print(f"Has next_state_pred: {'next_state_pred' in outputs}")
                            if "next_state_pred" in outputs:
                                print(f"next_state_pred type: {type(outputs['next_state_pred'])}")
                                if outputs["next_state_pred"] is not None:
                                    print(f"next_state_pred shape: {outputs['next_state_pred'].shape}")
                        
                        # 在autocast内计算所有损失
                        loss_dict = outputs.get("loss_dict", {})
                        
                        # 1. 策略损失
                        policy_logits = outputs["logits"]
                        if batch["actions"].dim() == 3 and batch["actions"].shape[-1] > 1:
                            action_targets = torch.argmax(batch["actions"], dim=-1)
                        else:
                            action_targets = batch["actions"].long()
                        
                        policy_loss = nn.functional.cross_entropy(
                            policy_logits.view(-1, policy_logits.shape[-1]),
                            action_targets.view(-1),
                        )
                        
                        # 2. 价值损失
                        values = outputs["values"]
                        with torch.no_grad():
                            next_values = self.model.compute_value(
                                batch["next_states"],
                                use_target=True,
                            )
                            rewards = batch["rewards"]
                            if rewards.dim() == 3 and rewards.shape[-1] == 1:
                                rewards = rewards.squeeze(-1)
                            
                            if next_values.dim() == 3:
                                next_values = next_values.squeeze(-1)
                            elif next_values.dim() == 2 and next_values.shape[-1] == 1:
                                next_values = next_values.squeeze(-1).unsqueeze(-1).expand(-1, rewards.shape[1])
                            elif next_values.dim() == 1:
                                next_values = next_values.unsqueeze(-1).expand(-1, rewards.shape[1])
                            
                            target_values = rewards + self.model.gamma * next_values
                        
                        if values.dim() == 3:
                            values = values.squeeze(-1)
                        value_loss = nn.functional.mse_loss(values, target_values)
                        
                        # 3. RKHS损失
                        rkhs_loss = torch.tensor(0.0, device=values.device, dtype=values.dtype)
                        # 调试：强制检查第一个batch
                        if batch_idx == 0 and epoch == 1:
                            print(f"\n=== RKHS Loss Debug (Epoch {epoch}, Batch {batch_idx}) - AMP path ===")
                            print(f"Outputs keys: {list(outputs.keys())}")
                            print(f"Has 'next_state_pred' key: {'next_state_pred' in outputs}")
                            if "next_state_pred" in outputs:
                                print(f"next_state_pred is None: {outputs['next_state_pred'] is None}")
                                if outputs["next_state_pred"] is not None:
                                    print(f"next_state_pred shape: {outputs['next_state_pred'].shape}")
                                    print(f"next_states shape: {batch['next_states'].shape}")
                            print(f"actions shape: {batch['actions'].shape}")
                            print(f"states shape: {batch['states'].shape}")
                            print("=" * 50)
                        
                        if "next_state_pred" in outputs:
                            if outputs["next_state_pred"] is not None:
                                next_state_pred = outputs["next_state_pred"]
                                next_states = batch["next_states"]
                                # 确保形状匹配
                                if next_state_pred.shape != next_states.shape:
                                    # 如果形状不匹配，尝试调整
                                    if next_state_pred.dim() == 2 and next_states.dim() == 3:
                                        # [B*T, S] -> [B, T, S]
                                        seq_len = batch["states"].shape[1]
                                        batch_size = batch["states"].shape[0]
                                        next_state_pred = next_state_pred.view(batch_size, seq_len, -1)
                                if next_state_pred.shape == next_states.shape:
                                    rkhs_loss = nn.functional.mse_loss(
                                        next_state_pred,
                                        next_states,
                                    )
                                    if batch_idx == 0 and epoch == 1:
                                        print(f"RKHS loss calculated: {rkhs_loss.item():.6f}")
                                else:
                                    # 调试信息：只在第一个batch打印一次
                                    if batch_idx == 0 and epoch == 1:
                                        print(f"Warning: RKHS loss shape mismatch - pred: {next_state_pred.shape}, target: {next_states.shape}")
                            else:
                                # 调试信息：检查为什么 next_state_pred 是 None
                                if batch_idx == 0 and epoch == 1:
                                    print(f"Warning: next_state_pred is None in outputs. Keys: {outputs.keys()}")
                                    print(f"  actions shape: {batch['actions'].shape}, actions is None: {batch['actions'] is None}")
                        else:
                            if batch_idx == 0 and epoch == 1:
                                print(f"Warning: 'next_state_pred' key not found in outputs!")
                        
                        # 4. 凸正则化损失
                        convex_loss = loss_dict.get("loss", torch.tensor(0.0, device=values.device, dtype=values.dtype))
                        
                        # 总损失
                        total_loss = (
                            self.lambda_policy * policy_loss +
                            self.lambda_value * value_loss +
                            self.lambda_rkhs * rkhs_loss +
                            self.lambda_convex * convex_loss
                        ) / self.gradient_accumulation_steps
                        
                        # 保存 rkhs_loss 用于后续累积（在 autocast 外）
                        rkhs_loss_for_accumulation = rkhs_loss.item() if isinstance(rkhs_loss, torch.Tensor) else float(rkhs_loss)
                    
                    # backward在autocast外
                    self.scaler.scale(total_loss).backward()
                else:
                    outputs = self.model(
                        states=batch["states"],
                        actions=batch["actions"],
                        agent_ids=batch.get("agent_ids"),
                        rewards=batch["rewards"],
                        next_states=batch["next_states"],
                        action_mask=batch.get("action_mask"),
                        compute_loss=True,
                    )
                    
                    # 调试：立即检查 outputs
                    if batch_idx == 0 and epoch == 1:
                        print(f"Outputs keys (Non-AMP): {list(outputs.keys())}")
                        print(f"Has next_state_pred: {'next_state_pred' in outputs}")
                        if "next_state_pred" in outputs:
                            print(f"next_state_pred type: {type(outputs['next_state_pred'])}")
                            if outputs["next_state_pred"] is not None:
                                print(f"next_state_pred shape: {outputs['next_state_pred'].shape}")
                    
                    loss_dict = outputs.get("loss_dict", {})
                    
                    policy_logits = outputs["logits"]
                    if batch["actions"].dim() == 3 and batch["actions"].shape[-1] > 1:
                        action_targets = torch.argmax(batch["actions"], dim=-1)
                    else:
                        action_targets = batch["actions"].long()
                    
                    policy_loss = nn.functional.cross_entropy(
                        policy_logits.view(-1, policy_logits.shape[-1]),
                        action_targets.view(-1),
                    )
                    
                    values = outputs["values"]
                    with torch.no_grad():
                        next_values = self.model.compute_value(
                            batch["next_states"],
                            use_target=True,
                        )
                        rewards = batch["rewards"]
                        if rewards.dim() == 3 and rewards.shape[-1] == 1:
                            rewards = rewards.squeeze(-1)
                        
                        if next_values.dim() == 3:
                            next_values = next_values.squeeze(-1)
                        elif next_values.dim() == 2 and next_values.shape[-1] == 1:
                            next_values = next_values.squeeze(-1).unsqueeze(-1).expand(-1, rewards.shape[1])
                        elif next_values.dim() == 1:
                            next_values = next_values.unsqueeze(-1).expand(-1, rewards.shape[1])
                        
                        target_values = rewards + self.model.gamma * next_values
                    
                    if values.dim() == 3:
                        values = values.squeeze(-1)
                    value_loss = nn.functional.mse_loss(values, target_values)
                    
                    rkhs_loss = torch.tensor(0.0, device=values.device, dtype=values.dtype)
                    # 调试：强制检查第一个batch
                    if batch_idx == 0 and epoch == 1:
                        print(f"\n=== RKHS Loss Debug (Epoch {epoch}, Batch {batch_idx}) - Non-AMP path ===")
                        print(f"Outputs keys: {list(outputs.keys())}")
                        print(f"Has 'next_state_pred' key: {'next_state_pred' in outputs}")
                        if "next_state_pred" in outputs:
                            print(f"next_state_pred is None: {outputs['next_state_pred'] is None}")
                            if outputs["next_state_pred"] is not None:
                                print(f"next_state_pred shape: {outputs['next_state_pred'].shape}")
                                print(f"next_states shape: {batch['next_states'].shape}")
                        print(f"actions shape: {batch['actions'].shape}")
                        print(f"states shape: {batch['states'].shape}")
                        print("=" * 50)
                    
                    if "next_state_pred" in outputs:
                        if outputs["next_state_pred"] is not None:
                            next_state_pred = outputs["next_state_pred"]
                            next_states = batch["next_states"]
                            # 确保形状匹配
                            if next_state_pred.shape != next_states.shape:
                                # 如果形状不匹配，尝试调整
                                if next_state_pred.dim() == 2 and next_states.dim() == 3:
                                    # [B*T, S] -> [B, T, S]
                                    seq_len = batch["states"].shape[1]
                                    batch_size = batch["states"].shape[0]
                                    next_state_pred = next_state_pred.view(batch_size, seq_len, -1)
                            if next_state_pred.shape == next_states.shape:
                                rkhs_loss = nn.functional.mse_loss(
                                    next_state_pred,
                                    next_states,
                                )
                                if batch_idx == 0 and epoch == 1:
                                    print(f"RKHS loss calculated: {rkhs_loss.item():.6f}")
                            else:
                                # 调试信息：只在第一个batch打印一次
                                if batch_idx == 0 and epoch == 1:
                                    print(f"Warning: RKHS loss shape mismatch - pred: {next_state_pred.shape}, target: {next_states.shape}")
                        else:
                            # 调试信息：检查为什么 next_state_pred 是 None
                            if batch_idx == 0 and epoch == 1:
                                print(f"Warning: next_state_pred is None in outputs. Keys: {outputs.keys()}")
                                print(f"  actions shape: {batch['actions'].shape}, actions is None: {batch['actions'] is None}")
                    else:
                        if batch_idx == 0 and epoch == 1:
                            print(f"Warning: 'next_state_pred' key not found in outputs!")
                    
                    convex_loss = loss_dict.get("loss", torch.tensor(0.0))
                    
                    total_loss = (
                        self.lambda_policy * policy_loss +
                        self.lambda_value * value_loss +
                        self.lambda_rkhs * rkhs_loss +
                        self.lambda_convex * convex_loss
                    ) / self.gradient_accumulation_steps
                    
                    total_loss.backward()
                
                # 只在累积步数达到时才更新参数
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        # 梯度裁剪
                        if self.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.grad_clip,
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # 梯度裁剪
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.grad_clip,
                            )
                        self.optimizer.step()
                    
                    # 清零梯度
                    self.optimizer.zero_grad()
                    
                    # 清理缓存
                    torch.cuda.empty_cache()
                
                # 软更新目标网络
                self.model.update_target_networks()
                
                # 记录指标
                epoch_metrics["total_loss"] += total_loss.item()
                epoch_metrics["policy_loss"] += policy_loss.item()
                epoch_metrics["value_loss"] += value_loss.item()
                
                # 处理 rkhs_loss：在 AMP 路径中可能已经提取了值
                if self.use_amp:
                    # AMP 路径：使用之前保存的值
                    rkhs_loss_val = rkhs_loss_for_accumulation
                else:
                    # 非 AMP 路径：从 tensor 提取值
                    if isinstance(rkhs_loss, torch.Tensor):
                        rkhs_loss_val = rkhs_loss.item()
                    else:
                        rkhs_loss_val = float(rkhs_loss)
                
                epoch_metrics["rkhs_loss"] += rkhs_loss_val
                
                # 调试：统计非零 rkhs_loss 的 batch 数量
                if epoch == 1 and batch_idx < 10:
                    if rkhs_loss_val > 1e-6:
                        if not hasattr(self, '_rkhs_nonzero_count'):
                            self._rkhs_nonzero_count = 0
                            self._rkhs_sum = 0.0
                        self._rkhs_nonzero_count += 1
                        self._rkhs_sum += rkhs_loss_val
                        if batch_idx == 9:
                            print(f"RKHS loss: {self._rkhs_nonzero_count}/10 batches have non-zero loss, sum: {self._rkhs_sum:.6f}, avg: {self._rkhs_sum/10:.6f}")
                            print(f"epoch_metrics['rkhs_loss'] after 10 batches: {epoch_metrics['rkhs_loss']:.6f}")
                epoch_metrics["convex_loss"] += convex_loss.item() if isinstance(convex_loss, torch.Tensor) else convex_loss
                epoch_metrics["utility"] += loss_dict.get("utility", 0)
                epoch_metrics["kl_term"] += loss_dict.get("kl_term", 0)
                epoch_metrics["exploitability"] += loss_dict.get("exploitability_bound", 0)
                
                # 更新进度条
                pbar.set_postfix({
                    "loss": total_loss.item(),
                    "policy": policy_loss.item(),
                    "value": value_loss.item(),
                })
        
        # 计算平均指标
        num_batches = len(self.train_loader)
        
        # 调试：在第一个 epoch 打印累积值和平均值
        if epoch == 1:
            print(f"\nBefore averaging - rkhs_loss sum: {epoch_metrics['rkhs_loss']:.6f}, num_batches: {num_batches}")
        
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # 调试：在第一个 epoch 打印平均值
        if epoch == 1:
            print(f"After averaging - rkhs_loss: {epoch_metrics['rkhs_loss']:.6f}\n")
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_metrics = {
            "val_total_loss": 0,
            "val_policy_loss": 0,
            "val_value_loss": 0,
            "val_rkhs_loss": 0,
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(
                    states=batch["states"],
                    actions=batch["actions"],
                    agent_ids=batch.get("agent_ids"),
                    rewards=batch["rewards"],
                    next_states=batch["next_states"],
                    action_mask=batch.get("action_mask"),
                    compute_loss=False,
                )
                
                # 计算损失
                policy_logits = outputs["logits"]
                if batch["actions"].dim() == 3 and batch["actions"].shape[-1] > 1:
                    action_targets = torch.argmax(batch["actions"], dim=-1)
                else:
                    action_targets = batch["actions"].long()
                
                policy_loss = nn.functional.cross_entropy(
                    policy_logits.view(-1, policy_logits.shape[-1]),
                    action_targets.view(-1),
                )
                
                values = outputs["values"]
                with torch.no_grad():
                    next_values = self.model.compute_value(
                        batch["next_states"],
                        use_target=True,
                    )
                    # 确保维度匹配（与训练函数一致）
                    rewards = batch["rewards"]
                    
                    # 处理rewards维度：可能是[B, T, 1]、[B, T]或[B, 1]
                    if rewards.dim() == 3:
                        if rewards.shape[-1] == 1:
                            rewards = rewards.squeeze(-1)  # [B, T]
                        else:
                            # rewards是[B, T, ...]，取最后一个维度
                            rewards = rewards.squeeze(-1) if rewards.shape[-1] == 1 else rewards.mean(dim=-1)
                    elif rewards.dim() == 2:
                        # rewards是[B, T]或[B, 1]
                        if rewards.shape[-1] == 1:
                            # 如果是[B, 1]，需要扩展到[B, T]
                            seq_len = batch["states"].shape[1]
                            rewards = rewards.expand(-1, seq_len)  # [B, T]
                        # 否则已经是[B, T]
                    elif rewards.dim() == 1:
                        # rewards是[B]，需要扩展到[B, T]
                        seq_len = batch["states"].shape[1]
                        rewards = rewards.unsqueeze(-1).expand(-1, seq_len)  # [B, T]
                    
                    # next_values可能是[B, T, 1]、[B, T]、[B, 1]或[B]
                    if next_values.dim() == 3:
                        next_values = next_values.squeeze(-1)  # [B, T]
                    elif next_values.dim() == 2:
                        if next_values.shape[-1] == 1:
                            # [B, 1] -> [B, T]
                            next_values = next_values.expand(-1, rewards.shape[1])
                        # 否则已经是[B, T]
                    elif next_values.dim() == 1:
                        # [B] -> [B, T]
                        next_values = next_values.unsqueeze(-1).expand(-1, rewards.shape[1])
                    
                    # 确保rewards和next_values都是[B, T]
                    assert rewards.shape == next_values.shape, \
                        f"rewards shape {rewards.shape} != next_values shape {next_values.shape}"
                    
                    target_values = rewards + self.model.gamma * next_values
                
                # values可能是[B, T, 1]或[B, T]
                if values.dim() == 3:
                    values = values.squeeze(-1)
                
                # 确保values和target_values都是[B, T]
                if values.shape != target_values.shape:
                    # 如果维度不匹配，尝试调整
                    if values.dim() == 1 and target_values.dim() == 2:
                        values = values.unsqueeze(-1).expand(-1, target_values.shape[1])
                    elif values.dim() == 2 and values.shape[1] == 1:
                        values = values.expand(-1, target_values.shape[1])
                
                value_loss = nn.functional.mse_loss(values, target_values)
                
                # RKHS损失
                rkhs_loss = torch.tensor(0.0, device=batch["states"].device, dtype=batch["states"].dtype)
                if "next_state_pred" in outputs and outputs["next_state_pred"] is not None:
                    next_state_pred = outputs["next_state_pred"]
                    next_states = batch["next_states"]
                    # 确保形状匹配
                    if next_state_pred.shape != next_states.shape:
                        # 如果形状不匹配，尝试调整
                        if next_state_pred.dim() == 2 and next_states.dim() == 3:
                            # [B*T, S] -> [B, T, S]
                            seq_len = batch["states"].shape[1]
                            batch_size = batch["states"].shape[0]
                            next_state_pred = next_state_pred.view(batch_size, seq_len, -1)
                    if next_state_pred.shape == next_states.shape:
                        rkhs_loss = nn.functional.mse_loss(
                            next_state_pred,
                            next_states,
                        )
                
                total_loss = policy_loss + 0.5 * value_loss + 0.1 * rkhs_loss
                
                # 累积指标
                val_metrics["val_total_loss"] += total_loss.item()
                val_metrics["val_policy_loss"] += policy_loss.item()
                val_metrics["val_value_loss"] += value_loss.item()
                # 确保 rkhs_loss 是 tensor 才能调用 .item()
                if isinstance(rkhs_loss, torch.Tensor):
                    rkhs_loss_val = rkhs_loss.item()
                else:
                    rkhs_loss_val = float(rkhs_loss)
                val_metrics["val_rkhs_loss"] += rkhs_loss_val
        
        # 计算平均指标
        num_batches = len(self.val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def train(self):
        """完整训练过程"""
        print("Starting offline training...")
        print(f"Training dataset size: {len(self.train_dataset)}")
        if self.val_loader:
            print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        
        # 训练循环
        for epoch in range(1, self.num_epochs + 1):
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 合并指标
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics["epoch"] = epoch
            all_metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
            
            # 记录指标
            self.metrics.update(all_metrics)
            if self.logger:
                self.logger.log_metrics(all_metrics, step=epoch)
            
            # 打印指标
            print(f"\nEpoch {epoch}/{self.num_epochs}:")
            for key, value in all_metrics.items():
                if key not in ["epoch", "learning_rate"]:
                    # 对于rkhs_loss，使用更多小数位显示
                    if "rkhs_loss" in key:
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value:.4f}")
            
            # 保存检查点
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_epoch_{epoch}.pt",
                )
                self.model.save_checkpoint(checkpoint_path)
            
            # 保存最佳模型
            if "val_total_loss" in val_metrics:
                if val_metrics["val_total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_total_loss"]
                    best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                    self.model.save_checkpoint(best_path)
                    print(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
        
        # 保存最终模型
        final_path = os.path.join(self.checkpoint_dir, "final_model.pt")
        self.model.save_checkpoint(final_path)
        
        # 保存指标
        metrics_path = os.path.join(self.checkpoint_dir, "training_metrics.npy")
        self.metrics.save(metrics_path)
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Models saved to: {self.checkpoint_dir}")
        
        return self.metrics


