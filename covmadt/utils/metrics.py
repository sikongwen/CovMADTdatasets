import numpy as np
from typing import Dict, List, Any
import os


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self, max_history: int = 1000):
        """
        初始化指标跟踪器
        
        参数:
            max_history: 每个指标最多保存的历史记录数（默认1000，防止内存泄漏）
        """
        self.data: Dict[str, List[Any]] = {}
        self.max_history = max_history
    
    def update(self, metrics: Dict[str, Any]):
        """更新指标"""
        for key, value in metrics.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
            # 限制历史记录数量，防止内存泄漏
            if len(self.data[key]) > self.max_history:
                self.data[key] = self.data[key][-self.max_history:]
    
    def get(self, key: str, default: Any = None) -> List[Any]:
        """获取指标"""
        return self.data.get(key, default)
    
    def get_mean(self, key: str) -> float:
        """获取指标平均值"""
        values = self.get(key, [])
        if len(values) == 0:
            return 0.0
        return float(np.mean(values))
    
    def get_std(self, key: str) -> float:
        """获取指标标准差"""
        values = self.get(key, [])
        if len(values) == 0:
            return 0.0
        return float(np.std(values))
    
    def save(self, path: str):
        """保存指标"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.data)
    
    @staticmethod
    def load(path: str) -> "MetricsTracker":
        """加载指标"""
        tracker = MetricsTracker()
        tracker.data = np.load(path, allow_pickle=True).item()
        return tracker


