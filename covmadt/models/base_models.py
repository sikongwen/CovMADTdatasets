import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class BaseModule(nn.Module):
    """基础模块类"""
    
    def __init__(self):
        super().__init__()
    
    def save(self, path: str):
        """保存模型"""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str, device: str = "cpu"):
        """加载模型"""
        self.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    
    def get_num_params(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


