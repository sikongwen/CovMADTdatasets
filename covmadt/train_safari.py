"""
SAFARI算法训练脚本

使用离线数据训练SAFARI模型
"""
import argparse
import os
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from algorithms.safari import SAFARI
from algorithms.safari_trainer import SAFARITrainer


def load_offline_data(data_path: str, seq_len: int = 100):
    """加载离线数据"""
    print(f"加载离线数据: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        states = np.array(f['states'])
        actions = np.array(f['actions'])
        rewards = np.array(f['rewards'])
        next_states = np.array(f['next_states'])
        dones = np.array(f['dones'])
        
        # 获取元数据
        obs_dim = f.attrs.get('obs_dim', states.shape[-1])
        act_dim = f.attrs.get('act_dim', actions.max() + 1 if len(actions.shape) == 1 else actions.shape[-1])
        num_agents = f.attrs.get('num_agents', 1)
    
    # 处理数据维度
    if states.ndim == 2:
        # [N, S] -> [N, 1, S]
        states = states[:, np.newaxis, :]
        next_states = next_states[:, np.newaxis, :]
    
    if actions.ndim == 1:
        # [N] -> [N, 1]
        actions = actions[:, np.newaxis]
    
    if rewards.ndim == 1:
        # [N] -> [N, 1]
        rewards = rewards[:, np.newaxis, np.newaxis]
    elif rewards.ndim == 2:
        # [N, 1] -> [N, 1, 1]
        rewards = rewards[:, :, np.newaxis]
    
    if dones.ndim == 1:
        # [N] -> [N, 1, 1]
        dones = dones[:, np.newaxis, np.newaxis]
    elif dones.ndim == 2:
        # [N, 1] -> [N, 1, 1]
        dones = dones[:, :, np.newaxis]
    
    # 转换为torch tensor
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'dones': dones,
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'num_agents': num_agents,
    }


def create_dataset(data_dict, seq_len=100):
    """创建序列数据集"""
    states = data_dict['states']
    actions = data_dict['actions']
    rewards = data_dict['rewards']
    next_states = data_dict['next_states']
    dones = data_dict['dones']
    
    # 如果序列长度大于1，需要创建滑动窗口
    if seq_len > 1:
        # 这里简化处理，直接使用单个时间步
        # 实际应用中可能需要创建序列
        pass
    
    return TensorDataset(states, actions, rewards, next_states, dones)


def main():
    parser = argparse.ArgumentParser(description='使用离线数据训练SAFARI模型')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True,
                       help='离线数据文件路径（H5格式）')
    parser.add_argument('--seq_len', type=int, default=1,
                       help='序列长度')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='学习率')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='RKHS嵌入维度')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='网络层数')
    parser.add_argument('--kernel_type', type=str, default='rbf',
                       help='核函数类型（rbf/linear/poly）')
    parser.add_argument('--kernel_bandwidth', type=float, default=1.0,
                       help='RBF核带宽')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='悲观惩罚系数')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='保守性系数')
    
    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/safari',
                       help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs/safari',
                       help='日志保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("SAFARI 离线训练")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"设备: {device}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"悲观惩罚系数: {args.beta}")
    print(f"保守性系数: {args.alpha}")
    print("=" * 60)
    
    # 加载数据
    data_dict = load_offline_data(args.data_path, args.seq_len)
    
    obs_dim = data_dict['obs_dim']
    act_dim = data_dict['act_dim']
    num_agents = data_dict['num_agents']
    
    print(f"\n观察维度: {obs_dim}")
    print(f"动作维度: {act_dim}")
    print(f"智能体数量: {num_agents}")
    
    # 创建数据集
    print("\n创建数据集...")
    dataset = create_dataset(data_dict, args.seq_len)
    
    # 划分训练集和验证集
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    # 创建模型
    print("\n创建SAFARI模型...")
    model = SAFARI(
        state_dim=obs_dim,
        action_dim=act_dim,
        n_agents=num_agents,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        kernel_type=args.kernel_type,
        kernel_bandwidth=args.kernel_bandwidth,
        gamma=args.gamma,
        beta=args.beta,
        alpha=args.alpha,
        device=device,
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    print("\n创建训练器...")
    trainer = SAFARITrainer(
        model=model,
        learning_rate=args.learning_rate,
        device=device,
        log_dir=args.log_dir,
    )
    
    # 创建保存目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 开始训练
    print("\n开始训练...")
    metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=args.checkpoint_dir,
        save_freq=10,
    )
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最佳验证损失: {metrics['best_val_loss']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
















