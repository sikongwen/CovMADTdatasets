import numpy as np
from typing import Dict, Any, Tuple, Optional
import gymnasium as gym


class EnvWrapper:
    """通用环境包装器"""
    
    def __init__(
        self,
        env,
        normalize_obs: bool = True,
        normalize_reward: bool = False,
    ):
        """
        初始化环境包装器
        
        参数:
            env: 环境对象
            normalize_obs: 是否归一化观察
            normalize_reward: 是否归一化奖励
        """
        self.env = env
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        
        # 观察和奖励统计
        self.obs_mean = None
        self.obs_std = None
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        obs, info = self.env.reset(**kwargs)
        
        if self.normalize_obs:
            obs = self._normalize_obs(obs)
        
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        obs, reward, done, truncated, info = self.env.step(action)
        
        if self.normalize_obs:
            obs = self._normalize_obs(obs)
        
        if self.normalize_reward:
            reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        
        return obs, reward, done, truncated, info
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """归一化观察"""
        if self.obs_mean is None or self.obs_std is None:
            return obs
        
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)
    
    def close(self):
        """关闭环境"""
        self.env.close()
    
    @property
    def state_dim(self) -> int:
        """获取状态维度"""
        if hasattr(self.env, 'observation_space'):
            return int(np.prod(self.env.observation_space.shape))
        return 128  # 默认值
    
    @property
    def action_dim(self) -> int:
        """获取动作维度"""
        if hasattr(self.env, 'action_space'):
            if hasattr(self.env.action_space, 'n'):
                return self.env.action_space.n
            return int(np.prod(self.env.action_space.shape))
        return 4  # 默认值
    
    @property
    def num_agents(self) -> int:
        """获取智能体数量"""
        if hasattr(self.env, 'num_agents'):
            return self.env.num_agents
        return 1


