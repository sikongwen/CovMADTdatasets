import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import math

from .base_models import BaseModule


class BaseKernel(nn.Module):
    """基核函数类"""
    
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算核矩阵"""
        raise NotImplementedError
        
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """特征映射（如果可用）"""
        raise NotImplementedError


class RBFKernel(BaseKernel):
    """RBF（高斯）核函数"""
    
    def __init__(self, bandwidth: float = 1.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.bandwidth = nn.Parameter(torch.tensor(bandwidth))
        else:
            self.register_buffer("bandwidth", torch.tensor(bandwidth))
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算RBF核矩阵
        
        参数:
            x: [N, D] 张量
            y: [M, D] 张量
            
        返回:
            K: [N, M] 核矩阵
        """
        x_norm = torch.sum(x**2, dim=-1, keepdim=True)
        y_norm = torch.sum(y**2, dim=-1, keepdim=True).transpose(-2, -1)
        
        dist = x_norm + y_norm - 2.0 * torch.matmul(x, y.transpose(-2, -1))
        K = torch.exp(-dist / (2.0 * self.bandwidth**2))
        
        return K
    
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """RBF核的特征映射（随机傅里叶特征）"""
        n_features = x.shape[-1]
        # 随机傅里叶特征近似
        W = torch.randn(n_features, n_features, device=x.device) / self.bandwidth
        b = torch.rand(n_features, device=x.device) * 2 * math.pi
        
        projections = torch.matmul(x, W) + b
        features = torch.sqrt(2.0 / n_features) * torch.cat([
            torch.cos(projections),
            torch.sin(projections)
        ], dim=-1)
        
        return features


class LinearKernel(BaseKernel):
    """线性核函数"""
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.transpose(-2, -1))
    
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PolynomialKernel(BaseKernel):
    """多项式核函数"""
    
    def __init__(self, degree: int = 3, coef0: float = 1.0):
        super().__init__()
        self.degree = degree
        self.coef0 = coef0
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        linear = torch.matmul(x, y.transpose(-2, -1))
        return (linear + self.coef0) ** self.degree


class MaternKernel(BaseKernel):
    """Matern核函数"""
    
    def __init__(self, nu: float = 1.5, length_scale: float = 1.0):
        super().__init__()
        self.nu = nu
        self.length_scale = length_scale
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x, y, p=2) / self.length_scale
        
        if self.nu == 0.5:
            # Matern 1/2
            K = torch.exp(-dist)
        elif self.nu == 1.5:
            # Matern 3/2
            sqrt3_dist = math.sqrt(3) * dist
            K = (1 + sqrt3_dist) * torch.exp(-sqrt3_dist)
        elif self.nu == 2.5:
            # Matern 5/2
            sqrt5_dist = math.sqrt(5) * dist
            K = (1 + sqrt5_dist + 5/3 * dist**2) * torch.exp(-sqrt5_dist)
        else:
            raise ValueError(f"Unsupported nu value: {self.nu}")
            
        return K


class KernelFactory:
    """核函数工厂"""
    
    @staticmethod
    def create_kernel(kernel_type: str, **kwargs) -> BaseKernel:
        """创建核函数"""
        kernels = {
            "rbf": RBFKernel,
            "linear": LinearKernel,
            "poly": PolynomialKernel,
            "matern": MaternKernel,
        }
        
        if kernel_type not in kernels:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
            
        return kernels[kernel_type](**kwargs)


class RKHSEmbedding(BaseModule):
    """
    RKHS均值嵌入模块
    用于近似状态转移概率
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int = 128,
        kernel_type: str = "rbf",
        kernel_params: Optional[Dict[str, Any]] = None,
        use_neural_features: bool = True,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.use_neural_features = use_neural_features
        
        # 核函数
        kernel_params = kernel_params or {}
        self.kernel = KernelFactory.create_kernel(kernel_type, **kernel_params)
        
        if use_neural_features:
            # 神经特征网络
            self.feature_net = nn.Sequential(
                nn.Linear(state_dim + action_dim, embedding_dim * 2),
                nn.LayerNorm(embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            # 使用核函数的显式特征映射
            self.feature_net = None
        
        # 转移权重
        self.transition_weights = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 下一个状态预测器
        self.next_state_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, state_dim)
        )
        
        # 转移概率预测器
        self.transition_prob_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()  # 输出概率值
        )
        
    def compute_mean_embedding(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        empirical_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算状态-动作对的均值嵌入
        
        参数:
            states: [..., state_dim] 状态
            actions: [..., action_dim] 动作
            empirical_dist: [..., K, state_dim] 经验分布样本
            
        返回:
            embeddings: [..., embedding_dim] 均值嵌入
        """
        # 确保所有输入都是 float32
        if states.dtype != torch.float32:
            states = states.float()
        if actions.dtype != torch.float32:
            actions = actions.float()
        if empirical_dist is not None and empirical_dist.dtype != torch.float32:
            empirical_dist = empirical_dist.float()
        
        batch_shape = states.shape[:-1]
        state_dim = states.shape[-1]
        
        # 检查 actions 的形状和维度
        if actions.shape[-1] != self.action_dim:
            # actions 的形状不匹配，可能是索引格式或其他格式
            # 如果 actions 是 [B, 1] 或 [B, num_agents]，无法直接使用
            # 返回零嵌入或抛出错误
            raise ValueError(f"Actions shape mismatch: expected last dimension {self.action_dim}, got {actions.shape[-1]}. "
                           f"Actions shape: {actions.shape}, States shape: {states.shape}")
        
        action_dim = actions.shape[-1]
        
        # 确保actions的形状正确 [B, T, A] 或 [B, A]
        # 检查批次维度是否匹配
        if states.dim() == 2 and actions.dim() == 2:
            # 都是 [B, ...] 格式，批次维度应该匹配
            if states.shape[0] != actions.shape[0]:
                raise ValueError(f"Batch dimension mismatch: states {states.shape[0]}, actions {actions.shape[0]}")
        elif states.dim() == 3 and actions.dim() == 2:
            # states 是 [B, T, S]，actions 是 [B, A]，需要扩展
            if states.shape[0] != actions.shape[0]:
                raise ValueError(f"Batch dimension mismatch: states {states.shape[0]}, actions {actions.shape[0]}")
            actions = actions.unsqueeze(1).expand(-1, states.shape[1], -1)
        elif states.dim() == 2 and actions.dim() == 3:
            # states 是 [B, S]，actions 是 [B, T, A]，需要压缩
            if states.shape[0] != actions.shape[0]:
                raise ValueError(f"Batch dimension mismatch: states {states.shape[0]}, actions {actions.shape[0]}")
            if actions.shape[1] == 1:
                actions = actions.squeeze(1)
            else:
                # 取最后一个时间步
                actions = actions[:, -1, :]
        
        # 展平批次维度（使用reshape而不是view以避免stride问题）
        states_flat = states.reshape(-1, state_dim)
        actions_flat = actions.reshape(-1, action_dim)
        
        # 确保展平后的批次维度匹配
        if states_flat.shape[0] != actions_flat.shape[0]:
            raise ValueError(f"Flattened batch dimension mismatch: states_flat {states_flat.shape[0]}, "
                           f"actions_flat {actions_flat.shape[0]}")
        
        if self.use_neural_features:
            # 使用神经特征网络
            sa_pairs = torch.cat([states_flat, actions_flat], dim=-1)
            # 再次确保 sa_pairs 是 float32
            if sa_pairs.dtype != torch.float32:
                sa_pairs = sa_pairs.float()
            features = self.feature_net(sa_pairs)
        else:
            # 使用核特征映射
            features = self.kernel.feature_map(sa_pairs)
        
        if empirical_dist is not None:
            # 对经验分布进行平均
            empirical_flat = empirical_dist.view(-1, empirical_dist.shape[-2], state_dim)
            empirical_flat = empirical_flat.mean(dim=-2)  # 平均经验分布
            
            # 计算与经验分布的交互
            if self.use_neural_features:
                # 为经验分布计算特征
                empirical_features = self.feature_net(
                    torch.cat([empirical_flat, torch.zeros_like(actions_flat)], dim=-1)
                )
                features = features + empirical_features  # 融合特征
            else:
                # 使用核计算交互
                kernel_vals = self.kernel(sa_pairs, empirical_flat)
                features = torch.cat([features, kernel_vals.mean(dim=-1, keepdim=True)], dim=-1)
        
        # 恢复批次形状
        embedding_shape = batch_shape + (self.embedding_dim,)
        embeddings = features.view(embedding_shape)
        
        return embeddings
    
    def predict_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: Optional[torch.Tensor] = None,
        empirical_dist: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        预测转移
        
        参数:
            states: [..., state_dim] 当前状态
            actions: [..., action_dim] 动作
            next_states: [..., state_dim] 下一个状态（可选，用于训练）
            empirical_dist: [..., K, state_dim] 经验分布
            
        返回:
            next_state_pred: [..., state_dim] 预测的下一个状态
            transition_prob: [..., 1] 转移概率（如果提供了next_states）
        """
        # 计算均值嵌入
        embeddings = self.compute_mean_embedding(states, actions, empirical_dist)
        
        # 转移变换
        transformed = self.transition_weights(embeddings)
        
        # 预测下一个状态
        next_state_pred = self.next_state_predictor(transformed)
        
        # 如果提供了下一个状态，计算转移概率
        if next_states is not None:
            # 计算下一个状态的嵌入
            if self.use_neural_features:
                next_embeddings = self.feature_net(
                    torch.cat([next_states, torch.zeros_like(actions)], dim=-1)
                )
            else:
                next_embeddings = self.kernel.feature_map(
                    torch.cat([next_states, torch.zeros_like(actions)], dim=-1)
                )
            
            # 计算转移概率
            transition_features = torch.cat([transformed, next_embeddings], dim=-1)
            transition_prob = self.transition_prob_predictor(transition_features)
            
            return next_state_pred, transition_prob
        
        return next_state_pred, None
    
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: Optional[torch.Tensor] = None,
        empirical_dist: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        返回:
            dict包含:
                - next_state_pred: 预测的下一个状态
                - transition_prob: 转移概率（如果提供了next_states）
                - embeddings: 均值嵌入（如果return_embeddings=True）
        """
        # 确保所有输入都是 float32
        if states.dtype != torch.float32:
            states = states.float()
        if actions.dtype != torch.float32:
            actions = actions.float()
        if next_states is not None and next_states.dtype != torch.float32:
            next_states = next_states.float()
        if empirical_dist is not None and empirical_dist.dtype != torch.float32:
            empirical_dist = empirical_dist.float()
        
        embeddings = self.compute_mean_embedding(states, actions, empirical_dist)
        transformed = self.transition_weights(embeddings)
        next_state_pred = self.next_state_predictor(transformed)
        
        result = {
            "next_state_pred": next_state_pred,
            "embeddings": embeddings if return_embeddings else None,
        }
        
        if next_states is not None:
            if self.use_neural_features:
                next_embeddings = self.feature_net(
                    torch.cat([next_states, torch.zeros_like(actions)], dim=-1)
                )
            else:
                next_embeddings = self.kernel.feature_map(
                    torch.cat([next_states, torch.zeros_like(actions)], dim=-1)
                )
            
            transition_features = torch.cat([transformed, next_embeddings], dim=-1)
            transition_prob = self.transition_prob_predictor(transition_features)
            result["transition_prob"] = transition_prob
        
        return result


