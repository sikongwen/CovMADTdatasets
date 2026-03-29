#!/usr/bin/env python3
"""
MFVI 微调诊断工具

用于分析微调过程中的损失、奖励和梯度，帮助找出问题所在。
"""
import argparse
import json
import numpy as np
import os
import sys
from pathlib import Path

def analyze_rewards(rewards_file):
    """分析奖励趋势"""
    print("\n" + "="*60)
    print("📊 奖励分析")
    print("="*60)
    
    if not os.path.exists(rewards_file):
        print(f"❌ 文件不存在: {rewards_file}")
        return
    
    with open(rewards_file, 'r') as f:
        data = json.load(f)
    
    rewards = data.get('episode_rewards', [])
    if len(rewards) == 0:
        print("❌ 没有奖励数据")
        return
    
    rewards = np.array(rewards)
    
    # 分段统计
    n = len(rewards)
    segments = [
        ("前25%", rewards[:n//4]),
        ("25%-50%", rewards[n//4:n//2]),
        ("50%-75%", rewards[n//2:3*n//4]),
        ("后25%", rewards[3*n//4:]),
    ]
    
    print(f"\n总episode数: {n}")
    print(f"\n分段统计:")
    for name, seg_rewards in segments:
        if len(seg_rewards) > 0:
            mean = np.mean(seg_rewards)
            std = np.std(seg_rewards)
            min_val = np.min(seg_rewards)
            max_val = np.max(seg_rewards)
            print(f"  {name:10s}: 平均={mean:8.2f}, 标准差={std:8.2f}, 范围=[{min_val:8.2f}, {max_val:8.2f}]")
    
    # 趋势分析
    first_100 = rewards[:100] if len(rewards) >= 100 else rewards
    last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
    
    if len(first_100) > 0 and len(last_100) > 0:
        improvement = np.mean(last_100) - np.mean(first_100)
        improvement_pct = (improvement / (abs(np.mean(first_100)) + 1e-8)) * 100)
        
        print(f"\n趋势分析:")
        print(f"  前100个episode平均: {np.mean(first_100):.2f}")
        print(f"  后100个episode平均: {np.mean(last_100):.2f}")
        print(f"  提升: {improvement:.2f} ({improvement_pct:+.1f}%)")
        
        if improvement > 0:
            print(f"  ✅ 奖励提升，微调有效果")
        elif improvement < -10:
            print(f"  ❌ 奖励下降超过10，可能存在问题")
            print(f"     建议: 降低学习率、降低训练频率、降低RKHS损失权重")
        else:
            print(f"  ⚠️  奖励基本不变，可能需要调整参数")
    
    # 稳定性分析
    if len(rewards) > 100:
        recent_std = np.std(rewards[-100:])
        early_std = np.std(rewards[:100])
        
        print(f"\n稳定性分析:")
        print(f"  前100个episode标准差: {early_std:.2f}")
        print(f"  后100个episode标准差: {recent_std:.2f}")
        
        if recent_std < early_std * 0.8:
            print(f"  ✅ 稳定性提升")
        elif recent_std > early_std * 1.2:
            print(f"  ⚠️  稳定性下降，可能训练不稳定")
            print(f"     建议: 降低学习率、增加梯度裁剪、降低训练频率")


def analyze_losses(log_file):
    """分析损失值（如果日志文件存在）"""
    print("\n" + "="*60)
    print("📉 损失分析")
    print("="*60)
    
    if not os.path.exists(log_file):
        print(f"⚠️  日志文件不存在: {log_file}")
        print(f"   无法分析损失，请检查训练日志")
        return
    
    # 尝试从日志中提取损失信息
    print("提示: 请手动检查训练日志中的损失值")
    print("\n理想情况:")
    print("  policy_loss: 1.0 - 2.0")
    print("  value_loss: 1.0 - 3.0")
    print("  rkhs_loss: < 10.0  (如果 > 50.0，需要降低权重)")
    print("  convex_loss: 0.5 - 2.0")
    print("  total_loss: 各损失项之和")
    print("\n如果 RKHS 损失过大:")
    print("  --lambda_rkhs_finetune 0.001  # 降低权重")
    print("\n如果价值损失不收敛:")
    print("  --lambda_value_finetune 2.5  # 增加权重")


def generate_recommendations(rewards_file):
    """根据分析结果生成建议"""
    print("\n" + "="*60)
    print("💡 优化建议")
    print("="*60)
    
    if not os.path.exists(rewards_file):
        return
    
    with open(rewards_file, 'r') as f:
        data = json.load(f)
    
    rewards = np.array(data.get('episode_rewards', []))
    if len(rewards) < 100:
        print("⚠️  数据不足，无法给出准确建议")
        return
    
    first_100 = rewards[:100]
    last_100 = rewards[-100:]
    improvement = np.mean(last_100) - np.mean(first_100)
    
    print("\n根据当前情况，推荐使用以下配置:\n")
    
    if improvement < -10:
        print("# 配置: 超保守微调（奖励下降）")
        print("python evaluate_covmadt.py \\")
        print("    --checkpoint <your_checkpoint> \\")
        print("    --env_name hanabi_v5 \\")
        print("    --use_mfvi \\")
        print("    --enable_finetune \\")
        print("    --num_episodes 20000 \\")
        print("    --learning_rate 1e-5 \\")
        print("    --train_freq 10 \\")
        print("    --epsilon 0.005 \\")
        print("    --lambda_rkhs_finetune 0.005 \\")
        print("    --lambda_convex_finetune 0.02 \\")
        print("    --lambda_value_finetune 2.5 \\")
        print("    --grad_clip 0.15 \\")
        print("    --warmup_episodes 200 \\")
        print("    --min_buffer_size 2000 \\")
        print("    --use_ppo \\")
        print("    --normalize_advantages \\")
        print("    --recalibrate_value_head")
    elif improvement < 0:
        print("# 配置: 保守微调（奖励略有下降）")
        print("python evaluate_covmadt.py \\")
        print("    --checkpoint <your_checkpoint> \\")
        print("    --env_name hanabi_v5 \\")
        print("    --use_mfvi \\")
        print("    --enable_finetune \\")
        print("    --num_episodes 15000 \\")
        print("    --learning_rate 3e-5 \\")
        print("    --train_freq 5 \\")
        print("    --epsilon 0.01 \\")
        print("    --lambda_rkhs_finetune 0.01 \\")
        print("    --lambda_convex_finetune 0.05 \\")
        print("    --lambda_value_finetune 2.0 \\")
        print("    --grad_clip 0.2 \\")
        print("    --warmup_episodes 100 \\")
        print("    --use_ppo \\")
        print("    --normalize_advantages")
    elif improvement < 5:
        print("# 配置: 标准微调（奖励略有提升）")
        print("python evaluate_covmadt.py \\")
        print("    --checkpoint <your_checkpoint> \\")
        print("    --env_name hanabi_v5 \\")
        print("    --use_mfvi \\")
        print("    --enable_finetune \\")
        print("    --num_episodes 10000 \\")
        print("    --learning_rate 5e-5 \\")
        print("    --train_freq 3 \\")
        print("    --epsilon 0.01 \\")
        print("    --lambda_rkhs_finetune 0.01 \\")
        print("    --lambda_convex_finetune 0.05 \\")
        print("    --lambda_value_finetune 2.0 \\")
        print("    --use_ppo \\")
        print("    --normalize_advantages")
    else:
        print("# 配置: 当前配置效果良好，可以继续使用")
        print("# 或者尝试稍微增加学习率以加快收敛:")
        print("python evaluate_covmadt.py \\")
        print("    --checkpoint <your_checkpoint> \\")
        print("    --env_name hanabi_v5 \\")
        print("    --use_mfvi \\")
        print("    --enable_finetune \\")
        print("    --num_episodes 10000 \\")
        print("    --learning_rate 8e-5 \\")
        print("    --train_freq 2 \\")
        print("    --epsilon 0.01 \\")
        print("    --lambda_rkhs_finetune 0.01 \\")
        print("    --lambda_convex_finetune 0.05 \\")
        print("    --lambda_value_finetune 2.0 \\")
        print("    --use_ppo \\")
        print("    --normalize_advantages")


def main():
    parser = argparse.ArgumentParser(description='MFVI 微调诊断工具')
    parser.add_argument('--rewards_file', type=str, required=True,
                       help='奖励JSON文件路径 (episode_rewards.json)')
    parser.add_argument('--log_file', type=str, default=None,
                       help='训练日志文件路径（可选）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔍 MFVI 微调诊断工具")
    print("="*60)
    
    # 分析奖励
    analyze_rewards(args.rewards_file)
    
    # 分析损失（如果提供日志文件）
    if args.log_file:
        analyze_losses(args.log_file)
    
    # 生成建议
    generate_recommendations(args.rewards_file)
    
    print("\n" + "="*60)
    print("📚 诊断完成")
    print("="*60)


if __name__ == '__main__':
    main()



