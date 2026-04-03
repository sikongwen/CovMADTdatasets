"""
模型加载示例脚本
演示如何加载训练好的模型权重进行推理或继续训练
"""
import torch
import numpy as np
from pettingzoo.classic import hanabi_v5
from main import R2D2Net, select_action, DEVICE

def load_and_test_model(checkpoint_path, num_episodes=5):
    """
    加载模型并测试性能
    
    参数:
        checkpoint_path: 检查点文件路径
        num_episodes: 测试的episode数量
    """
    # 初始化环境
    env = hanabi_v5.env(players=4, max_life_tokens=6)  # 4个智能体，max_life=6
    env.reset(seed=0)
    
    obs, _, _, _, _ = env.last()
    obs_dim = len(obs["observation"])
    act_dim = env.action_space(env.agents[0]).n
    
    # 创建网络
    net = R2D2Net(obs_dim, act_dim).to(DEVICE)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.eval()  # 设置为评估模式
    
    print(f"✓ 模型加载成功")
    print(f"  - 训练episode: {checkpoint.get('episode', 'N/A')}")
    print(f"  - Epsilon: {checkpoint.get('eps', 'N/A'):.4f}")
    print(f"\n开始测试 {num_episodes} 个episodes...\n")
    
    # 测试模型
    total_rewards = []
    
    for ep in range(num_episodes):
        episode_reward = 0
        hidden_states = {}
        env.reset()
        
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, _ = env.last()
            
            if terminated or truncated:
                env.step(None)
                continue
            
            if agent not in hidden_states:
                hidden_states[agent] = net.init_hidden(1)
            
            obs_vec = torch.FloatTensor(
                obs["observation"]
            ).view(1, 1, -1).to(DEVICE)
            
            with torch.no_grad():
                q, hidden = net(obs_vec, hidden_states[agent])
                hidden_states[agent] = hidden
            
            # 使用贪婪策略（epsilon=0）
            action = select_action(
                q.squeeze(0).squeeze(0).cpu().numpy(),
                obs["action_mask"],
                eps=0.0  # 不使用探索
            )
            
            episode_reward += reward
            env.step(action)
        
        total_rewards.append(episode_reward)
        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}")
    
    avg_reward = np.mean(total_rewards)
    print(f"\n平均奖励: {avg_reward:.2f}")
    print(f"最佳奖励: {max(total_rewards):.2f}")
    print(f"最差奖励: {min(total_rewards):.2f}")
    
    env.close()
    return avg_reward

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python load_model_example.py <checkpoint_path> [num_episodes]")
        print("示例: python load_model_example.py ./checkpoints/best_model.pt 10")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    load_and_test_model(checkpoint_path, num_episodes)



