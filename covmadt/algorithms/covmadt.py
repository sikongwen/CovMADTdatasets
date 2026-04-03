import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
import math
import copy

from models.transformer_models import MultiAgentTransformer
from models.rkhs_models import RKHSEmbedding
from models.convex_loss import ConvexRegularizationLoss
from .mfvi_critic import MFVICritic
from .standard_critic import StandardCritic
from .transformer_critic import TransformerCritic


class CovMADT(nn.Module):
    """
    CovMADT主算法实现
    
    集成:
    1. Transformer-based policy network
    2. RKHS state transition modeling
    3. Convex regularization with KL divergence
    4. Mean-Field Value Iteration critic
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        rkhs_embedding_dim: int = 128,
        kernel_type: str = "rbf",
        tau: float = 0.1,
        gamma: float = 0.99,
        use_mfvi: bool = False,
        use_transformer_critic: bool = False,
        device: str = "cuda",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.rkhs_embedding_dim = rkhs_embedding_dim
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.config = config  # 保存 config 以便 save_checkpoint 使用
        
        # Transformer策略网络
        # 从config获取max_seq_len，如果没有则使用默认值100
        max_seq_len = config.get('max_seq_len', 100) if config else 100
        self.transformer = MultiAgentTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            max_seq_len=max_seq_len,
        )
        
        # RKHS转移模型
        self.rkhs_model = RKHSEmbedding(
            state_dim=state_dim,
            action_dim=action_dim,
            embedding_dim=rkhs_embedding_dim,
            kernel_type=kernel_type,
            use_neural_features=True,
        )
        
        # 凸正则化损失
        # 从config获取use_occupancy_measure，默认False以节省内存
        use_occupancy_measure = config.get('use_occupancy_measure', False)
        self.convex_loss = ConvexRegularizationLoss(
            tau=tau,
            alpha=0.1,
            use_occupancy_measure=use_occupancy_measure,
        )
        
        # 参考策略网络（行为克隆）
        self.reference_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # 价值网络：支持三种Critic类型
        if use_transformer_critic:
            # 使用Transformer Critic
            self.critic = TransformerCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                hidden_dim=hidden_dim,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                max_seq_len=config.get('max_seq_len', 100) if config else 100,
                use_action=config.get('critic_use_action', True) if config else True,
            )
        elif use_mfvi:
            # 使用MFVI Critic
            self.critic = MFVICritic(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                kernel_type=kernel_type,
                gamma=gamma,
            )
        else:
            # 使用标准Critic（默认）
            self.critic = StandardCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_layers=3,
                gamma=gamma,
            )
        
        # 目标网络（用于稳定训练）
        self.target_transformer = copy.deepcopy(self.transformer)
        self.target_critic = copy.deepcopy(self.critic)
        
        # 初始化参数
        self._init_weights()
        
        # 移动到设备
        self.to(device)
        
    def _init_weights(self):
        """初始化网络权重"""
        # Transformer权重初始化
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # 参考策略网络初始化
        for layer in self.reference_policy:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        # 价值网络初始化
        for p in self.critic.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode_trajectory(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        编码轨迹序列
        
        参数:
            states: [B, T, S] 状态序列
            actions: [B, T, A] 动作序列（可选）
            agent_ids: [B, T] 智能体ID（可选）
            rewards: [B, T, 1] 奖励序列（可选）
            
        返回:
            dict包含编码结果
        """
        return self.transformer(
            states=states,
            actions=actions,
            agent_ids=agent_ids,
            rewards=rewards,
            return_all=True,
        )
    
    def predict_action(
        self,
        states: torch.Tensor,
        actions_history: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        预测动作
        
        参数:
            states: [B, S] 或 [B, T, S] 状态
            actions_history: [B, T-1, A] 历史动作（可选）
            agent_ids: [B] 或 [B, T] 智能体ID（可选）
            mask: [B, A] 动作掩码（可选）
            deterministic: 是否使用确定性策略
            
        返回:
            actions: 动作
            log_probs: 对数概率
            info: 附加信息
        """
        # 确保 states 是 float32 类型
        if states.dtype != torch.float32:
            states = states.float()
        
        # 确保状态维度正确
        if states.dim() == 2:
            states = states.unsqueeze(1)  # [B, 1, S]
        
        # 编码轨迹
        outputs = self.transformer(
            states=states,
            actions=actions_history,
            agent_ids=agent_ids,
        )
        
        logits = outputs["logits"]
        values = outputs["values"]
        
        # 应用动作掩码
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e10)
        
        # 修复过度自信：添加logit裁剪（防止logit值过大导致概率接近1.0）
        # 更严格的裁剪：从-10到10改为-5到5，进一步限制过度自信
        logits = torch.clamp(logits, min=-5.0, max=5.0)
        
        # 修复过度自信：添加温度缩放（通过config可配置，默认1.0平衡探索和利用）
        # 温度>1会增加探索，温度<1会减少探索
        # 从1.5降低到1.0，防止过度随机化
        temperature = self.config.get('action_temperature', 1.0) if self.config else 1.0
        if temperature > 0:
            logits = logits / temperature
        
        # 采样动作
        # 检查logits是否包含NaN或Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # 如果logits无效，使用零logits
            logits = torch.zeros_like(logits)
        
        if deterministic:
            # 确定性策略：选择最大logit的动作
            actions = torch.argmax(logits, dim=-1)
        else:
            # 随机策略：从分布中采样
            # Categorical需要2维logits [batch*seq, num_classes]
            logits_shape = logits.shape
            logits_flat = logits.view(-1, logits_shape[-1])  # [B*T, A]
            # 再次检查flat logits
            if torch.isnan(logits_flat).any() or torch.isinf(logits_flat).any():
                logits_flat = torch.zeros_like(logits_flat)
            action_dist = torch.distributions.Categorical(logits=logits_flat)
            actions_flat = action_dist.sample()  # [B*T]
            actions = actions_flat.view(logits_shape[:-1])  # [B, T]
        
        # 计算对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)
        
        info = {
            "logits": logits,
            "values": values,
            "action_dist": torch.softmax(logits, dim=-1),
        }
        
        return actions, log_probs, info
    
    def predict_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: Optional[torch.Tensor] = None,
        empirical_dist: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        使用RKHS预测状态转移
        
        参数:
            states: [..., S] 当前状态
            actions: [..., A] 动作
            next_states: [..., S] 下一个状态（可选）
            empirical_dist: [..., K, S] 经验分布（可选）
            
        返回:
            dict包含转移预测结果
        """
        return self.rkhs_model(
            states=states,
            actions=actions,
            next_states=next_states,
            empirical_dist=empirical_dist,
            return_embeddings=True,
        )
    
    def compute_value(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        empirical_dist: Optional[torch.Tensor] = None,
        use_target: bool = False,
    ) -> torch.Tensor:
        """
        计算状态价值
        
        参数:
            states: [..., S] 状态
            actions: [..., A] 动作（可选）
            empirical_dist: [..., K, S] 经验分布（可选）
            use_target: 是否使用目标网络
            
        返回:
            values: [..., 1] 价值
        """
        if use_target:
            critic = self.target_critic
        else:
            critic = self.critic
        
        if isinstance(critic, MFVICritic):
            # MFVI critic需要动作和经验分布
            if actions is None:
                # 如果没有提供动作，使用当前策略预测
                actions, _, _ = self.predict_action(states)
            
            # 如果actions是离散索引[B, T]，需要转换为one-hot编码[B, T, A]
            if actions.dim() == 2 and actions.dtype == torch.long:
                # actions是离散索引，转换为one-hot
                batch_size, seq_len = actions.shape
                action_dim = self.action_dim
                actions_onehot = torch.zeros(batch_size, seq_len, action_dim, 
                                            device=actions.device, dtype=torch.float32)
                actions_onehot.scatter_(-1, actions.unsqueeze(-1), 1.0)
                actions = actions_onehot
            
            values = critic(
                states=states,
                actions=actions,
                empirical_dist=empirical_dist,
            )
        elif isinstance(critic, TransformerCritic):
            # Transformer Critic可以接受动作（如果use_action=True）
            # 如果没有提供动作且需要动作，使用当前策略预测
            if critic.use_action and actions is None:
                actions, _, _ = self.predict_action(states)
            
            values = critic(
                states=states,
                actions=actions,
                empirical_dist=empirical_dist,
            )
        else:
            # 标准Critic（只需要状态）
            # StandardCritic可以处理序列输入，不需要特殊处理
            values = critic(
                states=states,
                actions=actions,  # 为了接口兼容性，但不使用
                empirical_dist=empirical_dist,  # 为了接口兼容性，但不使用
            )
        
        return values
    
    def compute_convex_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算凸正则化损失
        
        参数:
            states: [B, T, S] 状态
            actions: [B, T] 或 [B, T, A] 动作
            rewards: [B, T, 1] 奖励
            next_states: [B, T, S] 下一个状态
            agent_ids: [B, T] 智能体ID（可选）
            action_mask: [B, T, A] 动作掩码（可选）
            
        返回:
            dict包含损失项
        """
        # 编码轨迹获取策略logits
        outputs = self.transformer(
            states=states,
            actions=actions,
            agent_ids=agent_ids,
        )
        policy_logits = outputs["logits"]
        
        # 获取参考策略logits
        with torch.no_grad():
            ref_outputs = self.target_transformer(
                states=states,
                actions=actions,
                agent_ids=agent_ids,
            )
            ref_logits = ref_outputs["logits"]
        
        # 计算凸损失
        loss_dict = self.convex_loss(
            policy_logits=policy_logits,
            ref_policy_logits=ref_logits,
            rewards=rewards,
            states=states,
            actions=actions if actions.dim() == 2 else torch.argmax(actions, dim=-1),
            action_mask=action_mask,
            gamma=self.gamma,
        )
        
        return loss_dict
    
    def update_reference_policy(
        self,
        offline_data: Dict[str, torch.Tensor],
        num_epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
    ):
        """
        更新参考策略网络（行为克隆）
        
        参数:
            offline_data: 离线数据集
            num_epochs: 训练轮数
            batch_size: 批量大小
            lr: 学习率
        """
        optimizer = torch.optim.Adam(self.reference_policy.parameters(), lr=lr)
        
        states = offline_data["states"]
        actions = offline_data["actions"]
        
        if actions.dim() == 3 and actions.shape[-1] > 1:
            # one-hot动作
            action_labels = torch.argmax(actions, dim=-1)
        else:
            # 离散动作索引
            action_labels = actions.long()
        
        dataset_size = states.shape[0]
        num_batches = (dataset_size + batch_size - 1) // batch_size
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            # 随机打乱数据
            indices = torch.randperm(dataset_size)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = action_labels[batch_indices]
                
                # 前向传播
                logits = self.reference_policy(batch_states)
                
                # 计算损失
                loss = F.cross_entropy(
                    logits.view(-1, self.action_dim),
                    batch_actions.view(-1),
                )
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            print(f"Reference Policy Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def update_target_networks(self, tau: float = 0.005):
        """
        软更新目标网络
        
        参数:
            tau: 软更新系数
        """
        # 更新Transformer目标网络
        for target_param, param in zip(self.target_transformer.parameters(), 
                                      self.transformer.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # 更新Critic目标网络
        for target_param, param in zip(self.target_critic.parameters(), 
                                      self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save_checkpoint(self, path: str, config: Optional[Dict[str, Any]] = None):
        """保存检查点"""
        # 使用传入的 config 或模型保存的 config
        save_config = config if config is not None else self.config
        
        # 从模型获取参数
        checkpoint_config = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'n_agents': self.n_agents,
            'hidden_dim': self.hidden_dim,
            'transformer_layers': self.transformer_layers,
            'transformer_heads': self.transformer_heads,
            'rkhs_embedding_dim': self.rkhs_embedding_dim,
            'kernel_type': self.kernel_type,
            'tau': self.tau,
            'gamma': self.gamma,
        }
        
        # 从 transformer 获取 max_seq_len
        if hasattr(self.transformer, 'max_seq_len'):
            checkpoint_config['max_seq_len'] = self.transformer.max_seq_len
        elif save_config and 'max_seq_len' in save_config:
            checkpoint_config['max_seq_len'] = save_config['max_seq_len']
        elif hasattr(self.transformer, 'pos_encoding') and hasattr(self.transformer.pos_encoding, 'pe'):
            # 从位置编码推断 max_seq_len
            pe_shape = self.transformer.pos_encoding.pe.shape
            if len(pe_shape) >= 2:
                checkpoint_config['max_seq_len'] = pe_shape[1]
        
        # 从 config 获取额外信息
        if save_config:
            if 'critic_use_action' in save_config:
                checkpoint_config['critic_use_action'] = save_config['critic_use_action']
            if 'use_transformer_critic' in save_config:
                checkpoint_config['use_transformer_critic'] = save_config['use_transformer_critic']
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'transformer_state_dict': self.transformer.state_dict(),
            'rkhs_model_state_dict': self.rkhs_model.state_dict(),
            'critic_state_dict': self.critic.state_dict() if hasattr(self.critic, 'state_dict') else None,
            'reference_policy_state_dict': self.reference_policy.state_dict(),
            'config': checkpoint_config
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if 'transformer_state_dict' in checkpoint:
            self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        
        if 'rkhs_model_state_dict' in checkpoint:
            self.rkhs_model.load_state_dict(checkpoint['rkhs_model_state_dict'])
        
        if 'critic_state_dict' in checkpoint and checkpoint['critic_state_dict'] is not None:
            if hasattr(self.critic, 'load_state_dict'):
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        if 'reference_policy_state_dict' in checkpoint:
            self.reference_policy.load_state_dict(checkpoint['reference_policy_state_dict'])
        
        print(f"Checkpoint loaded from {path}")
    
    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        compute_loss: bool = False,
    ) -> Dict[str, Any]:
        """
        前向传播
        
        参数:
            states: [B, T, S] 状态
            actions: [B, T, A] 动作（可选）
            agent_ids: [B, T] 智能体ID（可选）
            rewards: [B, T, 1] 奖励（可选）
            next_states: [B, T, S] 下一个状态（可选）
            action_mask: [B, T, A] 动作掩码（可选）
            compute_loss: 是否计算损失
            
        返回:
            dict包含所有输出
        """
        # 编码轨迹
        transformer_outputs = self.transformer(
            states=states,
            actions=actions,
            agent_ids=agent_ids,
            rewards=rewards,
        )
        
        # 预测转移（如果提供了动作）
        transition_outputs = None
        if actions is not None:
            # 确保 actions 格式正确：rkhs_model 期望 [B, T, action_dim] 或 [B, action_dim] 格式
            # 检查 actions 的形状是否与 states 兼容
            states_batch_dim = states.shape[0] if states.dim() >= 2 else 1
            actions_batch_dim = actions.shape[0] if actions.dim() >= 1 else 1
            
            # 如果批次维度不匹配，跳过
            if states_batch_dim != actions_batch_dim:
                transition_outputs = None
            else:
                # 检查 actions 的最后一个维度
                if actions.dim() == 2:
                    # [B, ...] 格式
                    if actions.shape[-1] == self.action_dim:
                        # 已经是 one-hot 格式 [B, action_dim]
                        actions_for_rkhs = actions
                        # 如果 states 是 [B, T, S]，需要扩展 actions
                        if states.dim() == 3:
                            actions_for_rkhs = actions_for_rkhs.unsqueeze(1).expand(-1, states.shape[1], -1)
                    elif actions.shape[-1] == 1 and actions.dtype == torch.long:
                        # [B, 1] 索引格式，无法确定如何转换为 one-hot，跳过
                        transition_outputs = None
                    else:
                        # 其他格式，尝试使用
                        actions_for_rkhs = actions
                elif actions.dim() == 3:
                    # [B, T, ...] 格式
                    if actions.shape[-1] == self.action_dim:
                        # 已经是 one-hot 格式 [B, T, action_dim]
                        actions_for_rkhs = actions
                    else:
                        # 形状不匹配，跳过
                        transition_outputs = None
                else:
                    # 维度不匹配，跳过
                    transition_outputs = None
                
                # 如果 actions_for_rkhs 已设置，调用 rkhs_model
                if transition_outputs is None and 'actions_for_rkhs' in locals():
                    try:
                        transition_outputs = self.rkhs_model(
                            states=states,
                            actions=actions_for_rkhs,
                            next_states=next_states,
                        )
                    except Exception as e:
                        # 调试信息：只在第一次出错时打印
                        if not hasattr(self, '_rkhs_error_printed'):
                            print(f"ERROR: RKHS model forward failed: {e}")
                            print(f"  actions shape: {actions.shape if actions is not None else None}")
                            print(f"  states shape: {states.shape if states is not None else None}")
                            import traceback
                            traceback.print_exc()
                            self._rkhs_error_printed = True
                        transition_outputs = None
        
        # 计算价值
        value_outputs = self.compute_value(states)
        
        # 组装结果
        results = {
            "logits": transformer_outputs["logits"],
            "values": transformer_outputs.get("values", value_outputs),
            "encoded": transformer_outputs.get("encoded"),
        }
        
        if transition_outputs is not None:
            next_state_pred = transition_outputs.get("next_state_pred")
            # 调试：只在第一次检查时打印
            if not hasattr(self, '_rkhs_debug_printed'):
                if next_state_pred is None:
                    print(f"WARNING: next_state_pred is None in transition_outputs!")
                    print(f"  transition_outputs keys: {transition_outputs.keys()}")
                elif "next_state_pred" not in transition_outputs:
                    print(f"WARNING: next_state_pred not in transition_outputs!")
                    print(f"  transition_outputs keys: {transition_outputs.keys()}")
                else:
                    print(f"INFO: next_state_pred found! shape: {next_state_pred.shape}")
                self._rkhs_debug_printed = True
            results.update({
                "next_state_pred": next_state_pred,
                "transition_prob": transition_outputs.get("transition_prob"),
                "rkhs_embeddings": transition_outputs.get("embeddings"),
            })
        
        # 计算损失（如果需要）
        if compute_loss and rewards is not None and actions is not None:
            convex_loss = self.compute_convex_loss(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                agent_ids=agent_ids,
                action_mask=action_mask,
            )
            results["loss_dict"] = convex_loss
        
        return results


