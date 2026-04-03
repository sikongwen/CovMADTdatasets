from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class EnvConfig:
    """环境配置"""
    env_name: str = "quadrapong_v4"
    max_steps: int = 1000
    num_agents: int = 4
    render_mode: str = "rgb_array"
    seed: int = None


