import numpy as np
from typing import Dict, Any, Tuple, Optional
import gymnasium as gym
from pettingzoo.atari import quadrapong_v4


class QuadrapongEnvironment:
    """Quadrapong环境封装"""
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
    ):
        """
        初始化Quadrapong环境
        
        参数:
            render_mode: 渲染模式
            max_steps: 最大步数
        """
        self.max_steps = max_steps
        self.step_count = 0
        
        # 创建环境
        self.env = quadrapong_v4.env(render_mode=render_mode)
        self.env.reset()
        
        # 获取环境信息
        self.num_agents = len(self.env.agents)
        
        # 获取观察和动作空间
        first_agent = self.env.agents[0]
        obs_space = self.env.observation_space(first_agent)
        action_space = self.env.action_space(first_agent)
        
        # 单个智能体的观察维度
        single_agent_obs_dim = int(np.prod(obs_space.shape)) if hasattr(obs_space, 'shape') else 128
        # 合并所有智能体观察后的总维度
        self.state_dim = single_agent_obs_dim * self.num_agents
        # 单个智能体的动作空间大小
        single_agent_action_dim = action_space.n if hasattr(action_space, 'n') else 4
        # 所有智能体的动作维度总和（用于one-hot编码）
        self.action_dim = single_agent_action_dim * self.num_agents
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        self.env.reset()
        self.step_count = 0
        
        # 获取所有智能体的观察
        observations = []
        for agent in self.env.agents:
            obs = self.env.observe(agent)
            if isinstance(obs, np.ndarray):
                observations.append(obs.flatten())
            else:
                observations.append(np.array([obs]))
        
        # 合并所有观察
        combined_obs = np.concatenate(observations) if len(observations) > 0 else np.zeros(self.state_dim)
        
        return combined_obs, {}
    
    def step(self, actions) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        self.step_count += 1
        
        # 将动作数组转换为字典，按智能体顺序
        if isinstance(actions, np.ndarray):
            action_dict = {}
            for i, agent in enumerate(self.env.agents):
                if i < len(actions):
                    action_val = int(actions[i])
                    # 确保动作在有效范围内
                    action_space = self.env.action_space(agent)
                    if hasattr(action_space, 'n'):
                        action_val = max(0, min(action_val, action_space.n - 1))
                    action_dict[agent] = action_val
                else:
                    action_dict[agent] = 0
        elif isinstance(actions, dict):
            action_dict = actions
        else:
            # 单个标量动作，只给第一个智能体
            action_val = int(actions)
            action_space = self.env.action_space(self.env.agents[0])
            if hasattr(action_space, 'n'):
                action_val = max(0, min(action_val, action_space.n - 1))
            action_dict = {self.env.agents[0]: action_val}
        
        # 使用 agent_iter 模式执行动作
        # pettingzoo 需要逐个处理智能体
        observations = []
        rewards = []
        dones = []
        infos = []
        
        # 保存当前所有智能体的观察（在执行动作前）
        for agent in self.env.agents:
            obs = self.env.observe(agent)
            if isinstance(obs, np.ndarray):
                observations.append(obs.flatten())
            else:
                observations.append(np.array([obs]))
        
        # 保存执行动作前的智能体列表
        agents_before = list(self.env.agents)
        
        # 使用 agent_iter 执行一步
        for agent in self.env.agent_iter():
            obs, reward, termination, truncation, info = self.env.last()
            
            if termination or truncation:
                action = None
            else:
                # 从动作字典中获取该智能体的动作
                action = action_dict.get(agent, 0)
                # 确保动作类型正确
                if isinstance(action, np.integer):
                    action = int(action)
            
            self.env.step(action)
        
        # 获取所有智能体的奖励和状态（使用可能的智能体列表）
        # 注意：执行后 agents 列表可能改变，所以使用 all_agents 或可能的智能体
        all_agents = getattr(self.env, 'possible_agents', agents_before)
        if not all_agents:
            all_agents = agents_before
        
        # 重新获取所有智能体的观察（执行动作后）
        observations = []
        for agent in all_agents:
            if agent in self.env.agents:
                obs = self.env.observe(agent)
                if isinstance(obs, np.ndarray):
                    observations.append(obs.flatten())
                else:
                    observations.append(np.array([obs]))
            else:
                # 如果智能体不在活跃列表中，使用零观察
                observations.append(np.zeros(int(np.prod(self.env.observation_space(all_agents[0]).shape))))
        
        # 获取所有智能体的奖励和状态
        for agent in all_agents:
            reward = self.env.rewards.get(agent, 0.0)
            done = self.env.terminations.get(agent, False)
            trunc = self.env.truncations.get(agent, False)
            info = self.env.infos.get(agent, {})
            rewards.append(reward)
            dones.append(done or trunc)
            infos.append(info)
        
        # 合并观察
        combined_obs = np.concatenate(observations) if len(observations) > 0 else np.zeros(self.state_dim)
        
        # 总奖励
        total_reward = sum(rewards)
        
        # 是否结束
        done = all(dones) or self.step_count >= self.max_steps
        truncated = self.step_count >= self.max_steps
        
        # 合并信息
        combined_info = {
            "rewards": rewards,
            "dones": dones,
            "infos": infos,
        }
        
        return combined_obs, total_reward, done, truncated, combined_info
    
    def close(self):
        """关闭环境"""
        self.env.close()
    
    @property
    def observation_space(self):
        """获取观察空间"""
        # 使用 possible_agents 而不是 agents
        agents = getattr(self.env, 'possible_agents', None) or self.env.agents
        if not agents:
            # 如果都没有，创建一个临时环境来获取
            temp_env = quadrapong_v4.env()
            temp_env.reset()
            return temp_env.observation_space(temp_env.agents[0])
        return self.env.observation_space(agents[0])
    
    @property
    def action_space(self):
        """获取动作空间"""
        # 使用 possible_agents 而不是 agents，因为 agents 可能在执行过程中变空
        agents = getattr(self.env, 'possible_agents', None) or self.env.agents
        if not agents:
            # 如果都没有，创建一个临时环境来获取
            temp_env = quadrapong_v4.env()
            temp_env.reset()
            return temp_env.action_space(temp_env.agents[0])
        return self.env.action_space(agents[0])


