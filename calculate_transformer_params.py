#!/usr/bin/env python3
"""
计算MultiAgentTransformer的参数量
"""

import torch
import torch.nn as nn
from models.transformer_models import MultiAgentTransformer

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def count_parameters_by_module(model):
    """按模块统计参数量"""
    params_by_module = {}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        if module_name not in params_by_module:
            params_by_module[module_name] = 0
        params_by_module[module_name] += param.numel()
    return params_by_module

# 默认配置（根据你的模型）
state_dim = 1257  # Hanabi环境的状态维度
action_dim = 40   # Hanabi环境的动作维度
n_agents = 4      # 智能体数量
hidden_dim = 256  # 隐藏层维度（根据你的配置）
num_layers = 3    # Transformer层数（根据你的配置）
num_heads = 8     # 注意力头数（根据你的配置）
max_seq_len = 150 # 最大序列长度（根据你的配置）

print("=" * 80)
print("MultiAgentTransformer 参数量计算")
print("=" * 80)
print(f"\n配置:")
print(f"  state_dim: {state_dim}")
print(f"  action_dim: {action_dim}")
print(f"  n_agents: {n_agents}")
print(f"  hidden_dim: {hidden_dim}")
print(f"  num_layers: {num_layers}")
print(f"  num_heads: {num_heads}")
print(f"  max_seq_len: {max_seq_len}")

# 创建模型
model = MultiAgentTransformer(
    state_dim=state_dim,
    action_dim=action_dim,
    n_agents=n_agents,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    max_seq_len=max_seq_len,
)

# 计算总参数量
total_params, trainable_params = count_parameters(model)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")

# 按模块统计
params_by_module = count_parameters_by_module(model)
print(f"\n按模块统计:")
for module_name, param_count in sorted(params_by_module.items(), key=lambda x: x[1], reverse=True):
    percentage = param_count / total_params * 100
    print(f"  {module_name:30s}: {param_count:10,} ({percentage:5.2f}%)")

# 详细计算每个组件的参数量
print(f"\n详细计算:")

# 1. State Projection
state_proj_params = state_dim * hidden_dim + hidden_dim  # weight + bias
print(f"  1. state_proj: {state_proj_params:,} = {state_dim} × {hidden_dim} + {hidden_dim}")

# 2. Positional Encoding (无参数，只有buffer)
print(f"  2. pos_encoding: 0 (只有buffer，无参数)")

# 3. Agent Embedding
agent_embedding_params = n_agents * hidden_dim
print(f"  3. agent_embedding: {agent_embedding_params:,} = {n_agents} × {hidden_dim}")

# 4. Transformer Blocks
# 每个TransformerBlock包含:
#   - MultiHeadAttention: 4 * (hidden_dim * head_dim) + hidden_dim * hidden_dim
#     head_dim = hidden_dim // num_heads
#   - FeedForward: 2 * (hidden_dim * ffn_dim) + ffn_dim + hidden_dim
#     通常 ffn_dim = 4 * hidden_dim
head_dim = hidden_dim // num_heads
ffn_dim = 4 * hidden_dim  # 通常FFN维度是hidden_dim的4倍

# MultiHeadAttention参数
# Q, K, V投影: 3 * (hidden_dim * hidden_dim + hidden_dim)
# 输出投影: hidden_dim * hidden_dim + hidden_dim
attention_params_per_layer = 3 * (hidden_dim * hidden_dim + hidden_dim) + (hidden_dim * hidden_dim + hidden_dim)

# FeedForward参数
# 第一个线性层: hidden_dim * ffn_dim + ffn_dim
# 第二个线性层: ffn_dim * hidden_dim + hidden_dim
ffn_params_per_layer = (hidden_dim * ffn_dim + ffn_dim) + (ffn_dim * hidden_dim + hidden_dim)

# LayerNorm参数 (每个LayerNorm有2个参数: weight和bias)
# 每个TransformerBlock有2个LayerNorm: attention后和FFN后
layernorm_params_per_layer = 2 * (hidden_dim + hidden_dim)  # 2个LayerNorm，每个有weight和bias

transformer_block_params = attention_params_per_layer + ffn_params_per_layer + layernorm_params_per_layer
total_transformer_blocks_params = transformer_block_params * num_layers

print(f"  4. transformer_blocks ({num_layers}层):")
print(f"     每层参数量: {transformer_block_params:,}")
print(f"     - MultiHeadAttention: {attention_params_per_layer:,}")
print(f"       * Q/K/V投影: 3 × ({hidden_dim} × {hidden_dim} + {hidden_dim}) = {3 * (hidden_dim * hidden_dim + hidden_dim):,}")
print(f"       * 输出投影: {hidden_dim} × {hidden_dim} + {hidden_dim} = {hidden_dim * hidden_dim + hidden_dim:,}")
print(f"     - FeedForward: {ffn_params_per_layer:,}")
print(f"       * 第一层: {hidden_dim} × {ffn_dim} + {ffn_dim} = {hidden_dim * ffn_dim + ffn_dim:,}")
print(f"       * 第二层: {ffn_dim} × {hidden_dim} + {hidden_dim} = {ffn_dim * hidden_dim + hidden_dim:,}")
print(f"     - LayerNorm (2个): {layernorm_params_per_layer:,}")
print(f"     总参数量: {total_transformer_blocks_params:,}")

# 5. Policy Head
# Sequential(
#   Linear(hidden_dim, hidden_dim) + LayerNorm + ReLU + Dropout
#   Linear(hidden_dim, action_dim)
# )
policy_head_params = (hidden_dim * hidden_dim + hidden_dim) + (hidden_dim + hidden_dim) + (hidden_dim * action_dim + action_dim)
print(f"  5. policy_head: {policy_head_params:,}")
print(f"     - Linear1: {hidden_dim} × {hidden_dim} + {hidden_dim} = {hidden_dim * hidden_dim + hidden_dim:,}")
print(f"     - LayerNorm: {hidden_dim} + {hidden_dim} = {hidden_dim + hidden_dim:,}")
print(f"     - Linear2: {hidden_dim} × {action_dim} + {action_dim} = {hidden_dim * action_dim + action_dim:,}")

# 6. Value Head
# Sequential(
#   Linear(hidden_dim, hidden_dim) + LayerNorm + ReLU + Dropout
#   Linear(hidden_dim, 1)
# )
value_head_params = (hidden_dim * hidden_dim + hidden_dim) + (hidden_dim + hidden_dim) + (hidden_dim * 1 + 1)
print(f"  6. value_head: {value_head_params:,}")
print(f"     - Linear1: {hidden_dim} × {hidden_dim} + {hidden_dim} = {hidden_dim * hidden_dim + hidden_dim:,}")
print(f"     - LayerNorm: {hidden_dim} + {hidden_dim} = {hidden_dim + hidden_dim:,}")
print(f"     - Linear2: {hidden_dim} × 1 + 1 = {hidden_dim * 1 + 1:,}")

# 手动计算总和
manual_total = (state_proj_params + 
                agent_embedding_params + 
                total_transformer_blocks_params + 
                policy_head_params + 
                value_head_params)

print(f"\n手动计算总和: {manual_total:,}")
print(f"实际模型参数量: {total_params:,}")
print(f"差异: {abs(manual_total - total_params):,}")

print("\n" + "=" * 80)











