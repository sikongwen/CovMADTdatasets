import numpy as np
from typing import Dict, Any, Tuple, Optional
import gymnasium as gym
from pettingzoo.classic import hanabi_v5
import os
import sys
from contextlib import contextmanager
import os
import sys
import warnings


class HanabiEnvironment:
    """Hanabi环境封装"""
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        colors: int = 5,
        ranks: int = 5,
        players: int = 4,
        hand_size: int = 5,
        max_information_tokens: int = 8,
        max_life_tokens: int = 10,
        observation_type: str = 'minimal',
        seed: Optional[int] = None,
    ):
        """
        初始化Hanabi环境
        
        参数:
            render_mode: 渲染模式
            max_steps: 最大步数
            colors: 颜色数量
            ranks: 等级数量
            players: 玩家数量
            hand_size: 手牌大小
            max_information_tokens: 最大信息令牌数
            max_life_tokens: 最大生命令牌数
            observation_type: 观察类型 ('minimal' 或 'card_knowledge')
            seed: 随机种子
        """
        self.max_steps = max_steps
        self.step_count = 0
        self.seed = seed
        self.max_life_tokens = max_life_tokens
        self.max_information_tokens = max_information_tokens
        
        # 禁用Hanabi环境的调试输出
        # 重定向stdout以捕获底层库的输出
        self._original_stdout = sys.stdout
        self._suppress_output = True
        
        # 创建环境（在抑制输出的上下文中）
        if self._suppress_output:
            # 临时重定向stdout到/dev/null
            sys.stdout = open(os.devnull, 'w')
        
        try:
            self.env = hanabi_v5.env(
            colors=colors,
            ranks=ranks,
            players=players,
            hand_size=hand_size,
            max_information_tokens=max_information_tokens,
            max_life_tokens=max_life_tokens,
                observation_type=observation_type,
                render_mode=render_mode
            )
        finally:
            # 恢复stdout
            if self._suppress_output:
                sys.stdout.close()
                sys.stdout = self._original_stdout
        
        # 重置环境以获取智能体信息（在抑制输出的上下文中）
        if self._suppress_output:
            sys.stdout = open(os.devnull, 'w')
        try:
            if seed is not None:
                self.env.reset(seed=seed)
            else:
                self.env.reset()
        finally:
            if self._suppress_output:
                sys.stdout.close()
                sys.stdout = self._original_stdout
        
        # 获取环境信息
        self.num_agents = len(self.env.agents)
        
        # 获取观察和动作空间
        first_agent = self.env.agents[0]
        obs_space = self.env.observation_space(first_agent)
        action_space = self.env.action_space(first_agent)
        
        # Hanabi的观察是字典格式，需要提取特征
        # 对于minimal观察类型，需要将字典转换为向量
        if isinstance(obs_space, gym.spaces.Dict):
            # 计算字典观察的总维度
            total_dim = 0
            for key, space in obs_space.spaces.items():
                if hasattr(space, 'shape'):
                    total_dim += int(np.prod(space.shape))
                elif hasattr(space, 'n'):
                    total_dim += space.n
                else:
                    # 默认维度
                    total_dim += 1
            single_agent_obs_dim = total_dim
        else:
            single_agent_obs_dim = int(np.prod(obs_space.shape)) if hasattr(obs_space, 'shape') else 128
        
        # 合并所有智能体观察后的总维度
        self.state_dim = single_agent_obs_dim * self.num_agents
        
        # 单个智能体的动作空间大小
        self.single_agent_action_dim = action_space.n if hasattr(action_space, 'n') else 20
        # 所有智能体的动作维度总和（用于one-hot编码）
        self.action_dim = self.single_agent_action_dim * self.num_agents
        
        # 保存观察空间和动作空间信息
        self._obs_space = obs_space
        self._action_space = action_space
    
    @contextmanager
    def _suppress_stdout(self):
        """上下文管理器：临时抑制stdout输出"""
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
        
    def _flatten_observation(self, obs: Dict) -> np.ndarray:
        """将字典观察展平为向量"""
        if not isinstance(obs, dict):
            # 如果已经是数组，直接展平
            if isinstance(obs, np.ndarray):
                return obs.flatten()
            else:
                return np.array([obs])
        
        # 展平字典中的所有值
        flattened = []
        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, np.ndarray):
                flattened.append(value.flatten())
            elif isinstance(value, (int, float)):
                flattened.append(np.array([value]))
            elif isinstance(value, (list, tuple)):
                flattened.append(np.array(value).flatten())
            else:
                # 其他类型，尝试转换
                flattened.append(np.array([float(value)]))
        
        return np.concatenate(flattened) if len(flattened) > 0 else np.zeros(1)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        with self._suppress_stdout():
            if self.seed is not None:
                self.env.reset(seed=self.seed)
            else:
                self.env.reset()
        self.step_count = 0
        
        # 获取所有智能体的观察
        observations = []
        for agent in self.env.agents:
            obs = self.env.observe(agent)
            flattened_obs = self._flatten_observation(obs)
            observations.append(flattened_obs)
        
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
            flattened_obs = self._flatten_observation(obs)
            observations.append(flattened_obs)
        
        # 保存执行动作前的智能体列表
        agents_before = list(self.env.agents)
        
        # 使用 agent_iter 执行一步（抑制输出）
        with self._suppress_stdout():
            for agent in self.env.agent_iter():
                obs_dict, reward, termination, truncation, info = self.env.last()
                
                if termination or truncation:
                    action = None
                else:
                    # 从动作字典中获取该智能体的动作
                    action = action_dict.get(agent, 0)
                    # 确保动作类型正确
                    if isinstance(action, np.integer):
                        action = int(action)
                    
                    # 如果有action_mask，确保动作是有效的
                    if isinstance(obs_dict, dict) and "action_mask" in obs_dict:
                        action_mask = obs_dict["action_mask"]
                        # 如果选择的动作无效，从有效动作中随机选择
                        if action >= len(action_mask) or action_mask[action] == 0:
                            valid_actions = np.where(action_mask == 1)[0]
                            if len(valid_actions) > 0:
                                action = int(np.random.choice(valid_actions))
                            else:
                                action = 0
                
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
                flattened_obs = self._flatten_observation(obs)
                observations.append(flattened_obs)
            else:
                # 如果智能体不在活跃列表中，使用零观察
                obs_dim = self.state_dim // self.num_agents
                observations.append(np.zeros(obs_dim))
        
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
    
    def get_game_state(self):
        """
        获取当前游戏状态（用于评估）
        
        返回:
            包含得分、生命令牌、信息令牌等信息的字典，如果无法获取则返回None
        """
        try:
            # 尝试访问底层环境的状态
            if hasattr(self.env, 'env') and hasattr(self.env.env, 'env'):
                raw_env = self.env.env.env
                if hasattr(raw_env, '_state'):
                    state = raw_env._state
                    fireworks = getattr(state, 'fireworks', None)
                    life_tokens = getattr(state, 'life_tokens', None)
                    information_tokens = getattr(state, 'information_tokens', None)
                    
                    # 获取最大令牌数（从环境配置）
                    max_life_tokens = getattr(self, 'max_life_tokens', None)
                    if max_life_tokens is None:
                        max_life_tokens = getattr(raw_env, 'max_life_tokens', 10)
                    
                    max_information_tokens = getattr(self, 'max_information_tokens', None)
                    if max_information_tokens is None:
                        max_information_tokens = getattr(raw_env, 'max_information_tokens', 8)
                    
                    # 计算得分
                    score = 0
                    if fireworks is not None:
                        if isinstance(fireworks, dict):
                            score = sum(fireworks.values())
                        elif isinstance(fireworks, (list, np.ndarray)):
                            score = sum(fireworks)
                    
                    return {
                        "score": score,
                        "life_tokens": life_tokens,
                        "information_tokens": information_tokens,
                        "max_life_tokens": max_life_tokens,
                        "max_information_tokens": max_information_tokens,
                        "fireworks": fireworks,
                    }
        except Exception as e:
            pass
        
        return None
    
    def close(self):
        """关闭环境"""
        self.env.close()
    
    @property
    def observation_space(self):
        """获取观察空间"""
        # 使用 possible_agents 而不是 agents
        agents = getattr(self.env, 'possible_agents', None) or self.env.agents
        if not agents:
            # 如果都没有，创建一个临时环境来获取（抑制输出）
            with self._suppress_stdout():
                temp_env = hanabi_v5.env()
                temp_env.reset()
            return temp_env.observation_space(temp_env.agents[0])
        return self.env.observation_space(agents[0])
    
    @property
    def action_space(self):
        """获取动作空间"""
        # 使用 possible_agents 而不是 agents，因为 agents 可能在执行过程中变空
        agents = getattr(self.env, 'possible_agents', None) or self.env.agents
        if not agents:
            # 如果都没有，创建一个临时环境来获取（抑制输出）
            with self._suppress_stdout():
                temp_env = hanabi_v5.env()
                temp_env.reset()
            return temp_env.action_space(temp_env.agents[0])
        return self.env.action_space(agents[0])

