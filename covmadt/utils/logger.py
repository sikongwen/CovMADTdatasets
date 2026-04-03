import os
import json
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import wandb
    WANDB_AVAILABLE = True
except (ImportError, Exception):
    # 捕获所有异常，包括版本不兼容等问题
    WANDB_AVAILABLE = False
    wandb = None


def convert_to_serializable(obj: Any) -> Any:
    """
    将对象转换为可 JSON 序列化的格式
    
    参数:
        obj: 要转换的对象（可能是张量、numpy数组等）
        
    返回:
        可序列化的对象
    """
    if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        # PyTorch 张量转换为 Python 类型
        if obj.numel() == 1:
            # 标量张量
            return obj.item()
        else:
            # 多维张量转换为列表
            return obj.detach().cpu().tolist()
    elif NUMPY_AVAILABLE and isinstance(obj, np.ndarray):
        # NumPy 数组转换为列表
        if obj.size == 1:
            return obj.item()
        else:
            return obj.tolist()
    elif NUMPY_AVAILABLE and isinstance(obj, (np.integer, np.floating)):
        # NumPy 标量类型
        return obj.item()
    elif isinstance(obj, dict):
        # 递归处理字典
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # 递归处理列表和元组
        return [convert_to_serializable(item) for item in obj]
    else:
        # 其他类型直接返回
        return obj


class Logger:
    """日志记录器"""
    
    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: str = "experiment",
        use_wandb: bool = False,
        wandb_project: str = "covmadt",
    ):
        """
        初始化日志记录器
        
        参数:
            log_dir: 日志目录
            experiment_name: 实验名称
            use_wandb: 是否使用Weights & Biases
            wandb_project: WandB项目名称
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化WandB
        if use_wandb:
            if not WANDB_AVAILABLE:
                raise ImportError(
                    "wandb is not installed. Please install it with: pip install wandb"
                )
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                dir=log_dir,
            )
        
        # 日志文件
        self.log_file = os.path.join(log_dir, f"{experiment_name}_log.txt")
        
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """记录指标"""
        # 转换为可序列化格式
        serializable_metrics = convert_to_serializable(metrics)
        
        # 写入文件
        with open(self.log_file, 'a') as f:
            f.write(f"Step {step}: {json.dumps(serializable_metrics)}\n")
        
        # 记录到WandB（WandB可以处理张量，所以使用原始metrics）
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
    
    def log_message(self, message: str):
        """记录消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def close(self):
        """关闭日志记录器"""
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


