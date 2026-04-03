import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
from tqdm import tqdm

from algorithms.covmadt import CovMADT
from data.replay_buffer import ReplayBuffer
from utils.logger import Logger
from utils.metrics import MetricsTracker


class OnlineTrainer:
    """在线训练器（微调）"""
    
    def __init__(
        self,
        model: CovMADT,
        env,
        config: Dict[str, Any] = None,
        logger: Optional[Logger] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.env = env
        self.config = config or {}
        self.logger = logger
        self.device = device
        
        # 训练参数
        self.num_episodes = self.config.get("num_episodes", 100)
        self.batch_size = self.config.get("batch_size", 32)
        self.learning_rate = self.config.get("learning_rate", 1e-4)
        self.gamma = self.config.get("gamma", 0.99)
        self.clip_ratio = self.config.get("clip_ratio", 0.2)
        self.eval_freq = self.config.get("eval_freq", 10)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(
            capacity=10000,
            state_dim=env.state_dim if hasattr(env, 'state_dim') else 128,
            action_dim=env.action_dim if hasattr(env, 'action_dim') else 4,
        )
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
        )
        
        # 指标跟踪器
        self.metrics = MetricsTracker()
        
        # 检查点目录
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def collect_episode(self) -> Dict[str, Any]:
        """收集一个episode的数据"""
        obs, info = self.env.reset()
        episode_data = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }
        
        episode_reward = 0
        episode_steps = 0
        
        while True:
            # 预测动作
            with torch.no_grad():
                states_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                actions, log_probs, info = self.model.predict_action(states_tensor)
                # 提取单个元素（避免NumPy deprecation warning）
                actions = actions.cpu().numpy().item()
            
            # 执行动作
            next_obs, reward, done, truncated, info = self.env.step(actions)
            
            # 存储数据
            episode_data["states"].append(obs)
            episode_data["actions"].append(actions)
            episode_data["rewards"].append(reward)
            episode_data["next_states"].append(next_obs)
            episode_data["dones"].append(done or truncated)
            
            episode_reward += reward
            episode_steps += 1
            obs = next_obs
            
            if done or truncated:
                break
        
        # 转换为numpy数组
        episode_data = {k: np.array(v) for k, v in episode_data.items()}
        
        return episode_data, episode_reward, episode_steps
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """执行一个训练步骤"""
        self.model.train()
        
        # 前向传播
        outputs = self.model(
            states=batch["states"],
            actions=batch["actions"],
            rewards=batch["rewards"],
            next_states=batch["next_states"],
            compute_loss=True,
        )
        
        # 获取损失
        loss_dict = outputs.get("loss_dict", {})
        total_loss = loss_dict.get("loss", torch.tensor(0.0))
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.model.update_target_networks()
        
        return {
            "loss": total_loss.item(),
            "utility": loss_dict.get("utility", 0),
            "kl_term": loss_dict.get("kl_term", 0),
        }
    
    def train(self):
        """完整训练过程"""
        print("Starting online fine-tuning...")
        print(f"Number of episodes: {self.num_episodes}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        
        best_reward = float('-inf')
        
        for episode in range(1, self.num_episodes + 1):
            # 收集数据
            episode_data, episode_reward, episode_steps = self.collect_episode()
            
            # 存储到回放缓冲区
            for i in range(len(episode_data["states"])):
                self.replay_buffer.add(
                    state=episode_data["states"][i],
                    action=episode_data["actions"][i],
                    reward=episode_data["rewards"][i],
                    next_state=episode_data["next_states"][i],
                    done=episode_data["dones"][i],
                )
            
            # 训练（如果缓冲区有足够数据）
            if len(self.replay_buffer) >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                batch = {k: torch.tensor(v).to(self.device) for k, v in batch.items()}
                
                train_metrics = self.train_step(batch)
            else:
                train_metrics = {}
            
            # 记录指标
            metrics = {
                "episode": episode,
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                **train_metrics,
            }
            
            self.metrics.update(metrics)
            if self.logger:
                self.logger.log_metrics(metrics, step=episode)
            
            # 打印进度
            if episode % 10 == 0:
                print(f"Episode {episode}/{self.num_episodes}: "
                      f"Reward: {episode_reward:.2f}, Steps: {episode_steps}")
            
            # 保存最佳模型
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_path = os.path.join(self.checkpoint_dir, "best_online_model.pt")
                self.model.save_checkpoint(best_path)
        
        print("\nOnline fine-tuning completed!")
        print(f"Best episode reward: {best_reward:.2f}")
        
        return self.metrics


