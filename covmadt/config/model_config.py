from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """模型配置"""
    state_dim: int = 128
    action_dim: int = 4
    n_agents: int = 4
    hidden_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
    rkhs_embedding_dim: int = 128
    kernel_type: str = "rbf"
    use_mfvi: bool = True


