"""
ComaDICE算法训练脚本

使用离线数据训练ComaDICE模型
"""
import argparse
import os
import torch
import numpy as np

from algorithms.comadice import ComaDICE
from algorithms.comadice_trainer import ComaDICETrainer
from data.dataset import OfflineDataset
from utils.logger import Logger


def main():
    parser = argparse.ArgumentParser(description='使用离线数据训练ComaDICE模型')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True,
                       help='离线数据文件路径（H5格式）')
    parser.add_argument('--env_name', type=str, default='hanabi_v5',
                       help='环境名称（用于获取环境信息）')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--seq_len', type=int, default=100,
                       help='序列长度')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='网络层数')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='保守性系数')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='重要性权重正则化系数')
    
    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志保存目录')
    parser.add_argument('--experiment_name', type=str, default='comadice_training',
                       help='实验名称')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    parser.add_argument('--num_agents', type=int, default=None,
                       help='智能体数量（如果不指定，从环境获取）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("ComaDICE 离线训练")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"设备: {device}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"保守性系数: {args.alpha}")
    print(f"权重正则化系数: {args.beta}")
    print("=" * 60)
    
    # 初始化环境以获取维度信息
    print("\n初始化环境以获取维度信息...")
    
    # 支持不同的环境
    if args.env_name == "hanabi_v5":
        from pettingzoo.classic import hanabi_v5
        env = hanabi_v5.env(players=4, max_life_tokens=6)  # 4个智能体，max_life=6
        env.reset(seed=0)
        obs, _, _, _, _ = env.last()
        obs_dim = len(obs["observation"])
        act_dim = env.action_space(env.agents[0]).n
        num_agents = args.num_agents if args.num_agents is not None else len(env.agents)
    else:
        raise ValueError(f"不支持的环境: {args.env_name}")
    
    print(f"观察维度: {obs_dim}")
    print(f"动作维度: {act_dim}")
    print(f"智能体数量: {num_agents}")
    
    env.close()
    
    # 加载数据
    print(f"\n加载离线数据: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"数据文件不存在: {args.data_path}")
    
    # 创建训练和验证数据集
    print("创建数据集...")
    train_dataset = OfflineDataset(
        data_path=args.data_path,
        split="train",
        train_ratio=args.train_ratio,
        seq_len=args.seq_len,
        single_agent_action_dim=act_dim,
        num_agents=num_agents,
    )
    
    val_dataset = OfflineDataset(
        data_path=args.data_path,
        split="val",
        train_ratio=args.train_ratio,
        seq_len=args.seq_len,
        single_agent_action_dim=act_dim,
        num_agents=num_agents,
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    print("\n创建ComaDICE模型...")
    model = ComaDICE(
        state_dim=obs_dim,
        action_dim=act_dim,
        n_agents=num_agents,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        device=device,
    ).to(device)
    
    print(f"模型参数量: {model.get_num_params():,}")
    
    # 创建日志记录器
    log_dir = os.path.join(args.log_dir, args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir=log_dir, experiment_name=args.experiment_name)
    
    # 创建训练配置
    config = {
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "checkpoint_dir": args.checkpoint_dir,
    }
    
    # 创建训练器
    print("\n创建训练器...")
    trainer = ComaDICETrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        logger=logger,
        device=device,
    )
    
    # 开始训练
    print("\n开始训练...")
    print("=" * 60)
    metrics = trainer.train()
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"模型保存在: {args.checkpoint_dir}")
    print(f"日志保存在: {log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()




