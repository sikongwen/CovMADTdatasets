from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import torch

@dataclass
class BaseConfig:
    """基础配置类"""
    # 实验设置
    experiment_name: str = "covmadt_experiment"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    data_dir: str = "./data"
    
    # 训练设置
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    num_epochs: int = 100
    eval_freq: int = 10
    
    # 环境设置
    env_name: str = "quadrapong_v4"
    max_steps_per_episode: int = 1000
    num_agents: int = 4
    
    # 保存设置
    save_freq: int = 50
    save_best_only: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """从字典创建配置"""
        return cls(**config_dict)

@dataclass
class RKHSConfig:
    """RKHS嵌入配置"""
    kernel_type: str = "rbf"  # "rbf", "linear", "poly", "matern"
    bandwidth: float = 1.0  # RBF核的带宽
    feature_dim: int = 128  # 特征维度
    use_explicit_kernel: bool = False  # 是否使用显式核函数
    
    # 核函数参数
    kernel_params: Dict[str, Any] = field(default_factory=lambda: {
        "length_scale": 1.0,
        "nu": 1.5,  # Matern核的nu参数
        "degree": 3,  # 多项式核的度
    })
    
@dataclass
class TransformerConfig:
    """Transformer配置"""
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    activation: str = "relu"
    max_seq_len: int = 100
    
    # 注意力配置
    attention_type: str = "standard"  # "standard", "performer", "linformer"
    use_causal_mask: bool = True
    
@dataclass
class LossConfig:
    """损失函数配置"""
    # 凸正则化参数
    tau: float = 0.1  # KL散度正则化系数
    lambda_kl: float = 1.0  # KL损失权重
    lambda_exploit: float = 0.1  # 利用性损失权重
    
    # 其他损失权重
    lambda_value: float = 0.5
    lambda_rkhs: float = 0.1
    lambda_entropy: float = 0.01
    
    # 裁剪参数
    clip_ratio: float = 0.2
    clip_value: float = 0.5

@dataclass
class TrainingConfig:
    """训练配置"""
    # 通用训练参数
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE参数
    num_epochs_per_update: int = 4
    
    # 离线训练参数
    offline_batch_size: int = 64
    offline_num_epochs: int = 20
    offline_warmup_steps: int = 1000
    
    # 在线微调参数
    online_batch_size: int = 32
    online_num_epochs: int = 100
    online_update_freq: int = 10
    
    # 优化器参数
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "cosine", "linear", "plateau"
    warmup_steps: int = 1000


