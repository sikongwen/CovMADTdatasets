#!/usr/bin/env python3
"""计算MADDPG模型的内存需求"""

# 已知参数
obs_dim = 164520  # 观察维度
action_dim = 3    # 动作维度
n_agents = 40     # 智能体数量
hidden_dim = 128  # 隐藏层维度

# 单个智能体的状态维度
single_agent_state_dim = obs_dim

print("=" * 70)
print("MADDPG内存需求详细计算")
print("=" * 70)
print(f"观察维度（单个智能体）: {single_agent_state_dim:,}")
print(f"动作维度: {action_dim}")
print(f"智能体数量: {n_agents}")
print(f"隐藏层维度: {hidden_dim}")
print()

# Actor网络参数
print("1. Actor网络（每个智能体2个：主网络 + target网络）")
actor_layer1 = single_agent_state_dim * hidden_dim + hidden_dim
actor_layer2 = hidden_dim * hidden_dim + hidden_dim
actor_layer3 = hidden_dim * (hidden_dim // 2) + (hidden_dim // 2)
actor_output = (hidden_dim // 2) * action_dim + action_dim
actor_params = actor_layer1 + actor_layer2 + actor_layer3 + actor_output
actor_memory_mb = (actor_params * 4) / (1024 ** 2)
print(f"   单个Actor参数: {actor_params:,}")
print(f"   单个Actor内存: {actor_memory_mb:.2f} MB")
print(f"   所有Actor（{n_agents}个智能体 × 2个网络）: {actor_memory_mb * 2 * n_agents:.2f} MB ({actor_memory_mb * 2 * n_agents / 1024:.2f} GB)")

# Critic网络参数
print("\n2. Critic网络（每个智能体2个：主网络 + target网络）")
critic_input_dim = single_agent_state_dim * n_agents + action_dim * n_agents
critic_layer1 = critic_input_dim * hidden_dim + hidden_dim
critic_layer2 = hidden_dim * hidden_dim + hidden_dim
critic_layer3 = hidden_dim * (hidden_dim // 2) + (hidden_dim // 2)
critic_output = (hidden_dim // 2) * 1 + 1
critic_params = critic_layer1 + critic_layer2 + critic_layer3 + critic_output
critic_memory_mb = (critic_params * 4) / (1024 ** 2)
print(f"   Critic输入维度: {critic_input_dim:,}")
print(f"   单个Critic参数: {critic_params:,}")
print(f"   单个Critic内存: {critic_memory_mb:.2f} MB")
print(f"   所有Critic（{n_agents}个智能体 × 2个网络）: {critic_memory_mb * 2 * n_agents:.2f} MB ({critic_memory_mb * 2 * n_agents / 1024:.2f} GB)")

# 总模型内存
total_model = actor_memory_mb * 2 * n_agents + critic_memory_mb * 2 * n_agents
print(f"\n3. 模型参数总内存: {total_model:.2f} MB ({total_model/1024:.2f} GB)")

# 训练时额外内存
optimizer_memory = total_model * 2.5
gradient_memory = total_model
batch_memory = 64 * (single_agent_state_dim * n_agents * 4) / (1024 ** 2)

total_training = total_model + optimizer_memory + gradient_memory + batch_memory
print(f"\n4. 训练时总内存需求:")
print(f"   模型参数: {total_model:.2f} MB ({total_model/1024:.2f} GB)")
print(f"   优化器状态: {optimizer_memory:.2f} MB ({optimizer_memory/1024:.2f} GB)")
print(f"   梯度: {gradient_memory:.2f} MB ({gradient_memory/1024:.2f} GB)")
print(f"   批次数据: {batch_memory:.2f} MB")
print(f"   ========================================")
print(f"   总计: {total_training:.2f} MB ({total_training/1024:.2f} GB)")

print("\n" + "=" * 70)
print("不同配置对比")
print("=" * 70)

configs = [
    ("40智能体, hidden=128", 40, 128),
    ("20智能体, hidden=128", 20, 128),
    ("40智能体, hidden=64", 40, 64),
    ("20智能体, hidden=64", 20, 64),
]

for name, n, h in configs:
    s = single_agent_state_dim
    a_params = (s * h + h + h * h + h + h * (h//2) + h//2 + (h//2) * action_dim + action_dim) * 4 / (1024**2)
    c_input = s * n + action_dim * n
    c_params = (c_input * h + h + h * h + h + h * (h//2) + h//2 + (h//2) * 1 + 1) * 4 / (1024**2)
    model_mem = (a_params * 2 * n) + (c_params * 2 * n)
    train_mem = model_mem * 4.5
    print(f"{name:30s}: {train_mem/1024:6.2f} GB")

print("=" * 70)
















