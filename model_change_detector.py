"""
模型变化检测器
用于全面检测微调过程中模型的变化
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import os
import json
from collections import defaultdict


class ModelChangeDetector:
    """模型变化检测器"""
    
    def __init__(self, model, device, save_dir: str = "./model_change_logs"):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存初始参数
        self.initial_params = self._extract_params()
        self.initial_param_norms = self._compute_param_norms()
        
        # 跟踪历史（只保留最近的数据，节省内存）
        self.update_count = 0
        self.param_history = []  # 只保留最近1次，用于增量计算
        self.grad_history = []   # 不保存梯度历史（训练不需要）
        self.loss_history = []   # 不保存损失历史（训练不需要）
        self.change_history = []  # 只保留最近100次，用于最终报告
        self.max_history_size = 100  # 最大历史记录数
        
        print(f"✓ 模型变化检测器已初始化")
        print(f"  初始参数数量: {len(self.initial_params)}")
        print(f"  总参数量: {sum(p.numel() for p in self.initial_params.values()):,}")
    
    def _extract_params(self) -> Dict[str, torch.Tensor]:
        """提取模型参数"""
        params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params[name] = param.data.clone().cpu()
        return params
    
    def _compute_param_norms(self) -> Dict[str, float]:
        """计算参数范数"""
        norms = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                norms[name] = param.data.norm(2).item()
        return norms
    
    def _compute_grad_norms(self) -> Dict[str, float]:
        """计算梯度范数"""
        grad_norms = {}
        # 优先使用保存的梯度信息（在 zero_grad 之前保存的）
        if hasattr(self, '_last_grad_info') and self._last_grad_info:
            for name, grad_tensor in self._last_grad_info.items():
                if grad_tensor is not None:
                    grad_norms[name] = grad_tensor.norm(2).item()
                else:
                    grad_norms[name] = 0.0
        else:
            # 如果没有保存的梯度信息，尝试从模型中读取（可能已经被清零）
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norms[name] = param.grad.data.norm(2).item()
                else:
                    grad_norms[name] = 0.0
        return grad_norms
    
    def check_update(self, optimizer, loss_dict: Dict[str, float], step: int = 0):
        """检查一次更新后的变化"""
        self.update_count += 1
        
        # 获取当前参数
        current_params = self._extract_params()
        current_param_norms = self._compute_param_norms()
        current_grad_norms = self._compute_grad_norms()
        
        # 计算参数变化
        param_changes = {}
        total_change = 0.0
        max_change = 0.0
        max_change_name = None
        
        for name in current_params:
            if name in self.initial_params:
                change = (current_params[name] - self.initial_params[name]).norm(2).item()
                param_changes[name] = change
                total_change += change
                if change > max_change:
                    max_change = change
                    max_change_name = name
        
        # 计算相对变化（相对于初始参数范数）
        relative_changes = {}
        for name in param_changes:
            if name in self.initial_param_norms and self.initial_param_norms[name] > 0:
                relative_changes[name] = param_changes[name] / self.initial_param_norms[name]
            else:
                relative_changes[name] = 0.0
        
        # 计算增量变化（相对于上一次）
        incremental_changes = {}
        if len(self.param_history) > 0:
            last_params = self.param_history[-1]
            for name in current_params:
                if name in last_params:
                    inc_change = (current_params[name] - last_params[name]).norm(2).item()
                    incremental_changes[name] = inc_change
        
        # 统计有梯度的参数
        params_with_grad = sum(1 for v in current_grad_norms.values() if v > 0)
        total_params = len(current_grad_norms)
        
        # 获取学习率
        current_lr = optimizer.param_groups[0]['lr'] if optimizer else 0.0
        
        # 保存历史（只保留必要的数据，节省内存）
        # 只保留最近1次参数用于增量计算
        if len(self.param_history) >= 1:
            self.param_history.pop(0)  # 移除最旧的
        self.param_history.append(current_params)
        
        # 不保存梯度和损失历史（训练不需要）
        # self.grad_history.append(current_grad_norms)  # 已移除
        # self.loss_history.append(loss_dict)  # 已移除
        
        change_info = {
            'step': step,
            'update_count': self.update_count,
            'total_param_change': total_change,
            'max_param_change': max_change,
            'max_change_param': max_change_name,
            'params_with_grad': params_with_grad,
            'total_params': total_params,
            'grad_ratio': params_with_grad / total_params if total_params > 0 else 0.0,
            'learning_rate': current_lr,
            'loss': loss_dict,
        }
        # 只保留最近N次变化历史
        self.change_history.append(change_info)
        if len(self.change_history) > self.max_history_size:
            self.change_history.pop(0)  # 移除最旧的
        
        # 每10次更新打印一次详细报告
        if self.update_count % 10 == 0:
            self._print_report(change_info, param_changes, relative_changes, incremental_changes, current_grad_norms)
        
        # 不再每100次更新保存检查点（用户要求）
        
        return change_info
    
    def _print_report(self, change_info, param_changes, relative_changes, incremental_changes, grad_norms):
        """打印详细报告"""
        # 区分应该训练的参数和不应该训练的参数（目标网络）
        trainable_params = []
        target_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'target_' in name:
                    target_params.append(name)
                else:
                    trainable_params.append(name)
        
        # 统计应该训练的参数中有多少有梯度
        trainable_with_grad = sum(1 for name in trainable_params if grad_norms.get(name, 0.0) > 0)
        trainable_total = len(trainable_params)
        trainable_grad_ratio = trainable_with_grad / trainable_total if trainable_total > 0 else 0.0
        
        print(f"\n{'='*80}")
        print(f"📊 模型变化检测报告 (更新 #{self.update_count})")
        print(f"{'='*80}")
        
        # 总体变化
        print(f"\n📈 总体变化:")
        print(f"  总参数变化: {change_info['total_param_change']:.6f}")
        print(f"  最大参数变化: {change_info['max_param_change']:.6f} ({change_info['max_change_param']})")
        print(f"  有梯度参数: {change_info['params_with_grad']}/{change_info['total_params']} ({change_info['grad_ratio']:.1%})")
        if trainable_total > 0:
            print(f"  ✓ 可训练参数有梯度: {trainable_with_grad}/{trainable_total} ({trainable_grad_ratio:.1%})")
            print(f"  ℹ️  目标网络参数: {len(target_params)} (不参与训练，通过软更新)")
        print(f"  学习率: {change_info['learning_rate']:.2e}")
        
        # 损失信息
        if change_info['loss']:
            print(f"\n📉 损失信息:")
            for key, value in change_info['loss'].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.6f}")
        
        # 参数变化 Top 10
        if param_changes:
            sorted_changes = sorted(param_changes.items(), key=lambda x: x[1], reverse=True)
            print(f"\n🔝 参数变化 Top 10 (绝对变化):")
            for i, (name, change) in enumerate(sorted_changes[:10], 1):
                rel_change = relative_changes.get(name, 0.0)
                grad_norm = grad_norms.get(name, 0.0)
                print(f"  {i:2d}. {name[:60]:60s} | 变化: {change:8.6f} | 相对: {rel_change:6.2%} | 梯度: {grad_norm:8.6f}")
        
        # 增量变化 Top 5（本次更新相对于上次）
        if incremental_changes:
            sorted_inc = sorted(incremental_changes.items(), key=lambda x: x[1], reverse=True)
            print(f"\n🔄 本次更新变化 Top 5 (增量变化):")
            for i, (name, change) in enumerate(sorted_inc[:5], 1):
                print(f"  {i:2d}. {name[:60]:60s} | 增量: {change:8.6f}")
        
        # 梯度信息 Top 10
        if grad_norms:
            sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
            print(f"\n📊 梯度范数 Top 10:")
            for i, (name, grad_norm) in enumerate(sorted_grads[:10], 1):
                if grad_norm > 0:
                    print(f"  {i:2d}. {name[:60]:60s} | 梯度: {grad_norm:8.6f}")
        
        # 警告
        warnings = []
        if change_info['total_param_change'] < 1e-6:
            warnings.append("⚠️  总参数变化极小，可能没有更新")
        # 只对可训练参数检查梯度比例
        if trainable_total > 0 and trainable_grad_ratio < 0.5:
            warnings.append(f"⚠️  只有 {trainable_grad_ratio:.1%} 的可训练参数有梯度（目标网络参数不计入）")
        if change_info['max_param_change'] < 1e-8:
            warnings.append("⚠️  最大参数变化极小")
        
        if warnings:
            print(f"\n⚠️  警告:")
            for warning in warnings:
                print(f"  {warning}")
        
        print(f"{'='*80}\n")
    
    def _save_checkpoint(self, step: int):
        """保存检查点"""
        checkpoint = {
            'step': step,
            'update_count': self.update_count,
            'change_history': self.change_history[-100:],  # 只保存最近100次
        }
        
        checkpoint_path = os.path.join(self.save_dir, f"change_checkpoint_{step}.json")
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  保存检查点失败: {e}")
    
    def save_final_report(self, step: int):
        """保存最终报告"""
        # 计算最终参数变化
        final_params = self._extract_params()
        final_changes = {}
        for name in final_params:
            if name in self.initial_params:
                change = (final_params[name] - self.initial_params[name]).norm(2).item()
                final_changes[name] = change
        
        report = {
            'total_updates': self.update_count,
            'final_step': step,
            'total_param_change': sum(final_changes.values()),
            'max_param_change': max(final_changes.values()) if final_changes else 0.0,
            'param_changes': {k: float(v) for k, v in final_changes.items()},
            'change_history': self.change_history,
        }
        
        report_path = os.path.join(self.save_dir, "final_change_report.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  保存最终报告失败: {e}")
            return
        
        print(f"\n✓ 最终变化报告已保存到: {report_path}")
        
        # 打印总结
        print(f"\n{'='*80}")
        print(f"📊 最终模型变化总结")
        print(f"{'='*80}")
        print(f"  总更新次数: {self.update_count}")
        print(f"  总参数变化: {report['total_param_change']:.6f}")
        print(f"  最大参数变化: {report['max_param_change']:.6f}")
        print(f"{'='*80}\n")

