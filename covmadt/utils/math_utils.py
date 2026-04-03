import torch
import numpy as np
from typing import Tuple


def normalize(x: torch.Tensor, mean: torch.Tensor = None, std: torch.Tensor = None) -> torch.Tensor:
    """归一化张量"""
    if mean is None:
        mean = x.mean(dim=0, keepdim=True)
    if std is None:
        std = x.std(dim=0, keepdim=True) + 1e-8
    
    return (x - mean) / std


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """反归一化张量"""
    return x * std + mean


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算广义优势估计（GAE）
    
    参数:
        rewards: [T, B] 奖励
        values: [T, B] 价值
        dones: [T, B] 完成标志
        gamma: 折扣因子
        lambda_: GAE参数
        
    返回:
        advantages: [T, B] 优势
        returns: [T, B] 回报
    """
    T, B = rewards.shape
    
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * lambda_ * (1 - dones[t]) * last_gae
        last_gae = advantages[t]
    
    returns = advantages + values
    
    return advantages, returns


