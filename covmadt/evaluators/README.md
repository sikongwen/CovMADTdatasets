# 评估器模块使用指南

## 概述

`evaluators` 模块提供了模块化的模型评估功能，可以方便地在代码中导入和使用。

## R2D2 评估器

### 基本使用

```python
from evaluators.r2d2_evaluator import R2D2Evaluator, load_r2d2_model
from pettingzoo.classic import hanabi_v5
import torch

# 1. 初始化环境
env = hanabi_v5.env()
env.reset(seed=0)
obs, _, _, _, _ = env.last()
obs_dim = len(obs["observation"])
act_dim = env.action_space(env.agents[0]).n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载模型
model = load_r2d2_model(
    checkpoint_path="./checkpoints/best_model.pt",
    obs_dim=obs_dim,
    act_dim=act_dim,
    device=device
)

# 3. 创建评估器
evaluator = R2D2Evaluator(
    model=model,
    env=env,
    device=device,
    epsilon=0.0,  # 完全贪婪策略
)

# 4. 执行评估
all_episode_stats, stats_summary = evaluator.evaluate(
    num_episodes=100,
    verbose=True
)

# 5. 打印统计摘要
evaluator.print_summary(stats_summary)

# 6. 保存结果
saved_files = evaluator.save_results(
    all_episode_stats,
    stats_summary,
    output_dir="./evaluation_results/r2d2",
    save_detailed=True,
)

# 7. 生成图表
evaluator.plot_results(all_episode_stats, "./evaluation_results/r2d2")
```

### 单独评估一个 Episode

```python
# 评估单个episode
episode_stats = evaluator.evaluate_episode()
print(f"Final Score: {episode_stats['final_score']}")
print(f"Information Efficiency: {episode_stats['information_efficiency']:.3f}")
```

### 自定义评估流程

```python
# 手动控制评估流程
all_episode_stats = []
for i in range(10):
    episode_stats = evaluator.evaluate_episode()
    all_episode_stats.append(episode_stats)
    
    # 实时查看结果
    if i % 5 == 0:
        recent_scores = [s['final_score'] for s in all_episode_stats[-5:]]
        print(f"最近5个episode平均得分: {np.mean(recent_scores):.2f}")

# 计算统计摘要
stats_summary = evaluator.compute_statistics(all_episode_stats)
evaluator.print_summary(stats_summary)
```

## API 参考

### R2D2Evaluator

#### 初始化参数

- `model`: R2D2Net 模型实例
- `env`: 环境实例
- `device`: torch.device
- `epsilon`: 探索率（默认: 0.0，完全贪婪）

#### 主要方法

- `evaluate_episode()`: 评估单个episode，返回统计信息字典
- `evaluate(num_episodes, verbose)`: 评估多个episodes，返回所有统计信息和摘要
- `compute_statistics(all_episode_stats)`: 计算统计摘要
- `save_results(...)`: 保存评估结果到文件
- `plot_results(...)`: 生成可视化图表
- `print_summary(stats_summary)`: 打印统计摘要

### load_r2d2_model

加载R2D2模型的辅助函数。

**参数:**
- `checkpoint_path`: 检查点文件路径
- `obs_dim`: 观察维度
- `act_dim`: 动作维度
- `device`: 设备

**返回:**
- `R2D2Net`: 加载的模型实例

## 评估指标

评估器会记录以下指标：

1. **Final Score（最终得分）**: 游戏最终得分（0-25）
2. **Perfect Game Rate（满分率）**: 达到25分的游戏比例
3. **Information Efficiency（信息效率）**: 得分与信息令牌使用的比率
4. **Life Loss Rate（生命损失率）**: 生命令牌损失率
5. **Risk Control Score（风险控制得分）**: 1 - 生命损失率
6. **Episode Reward（Episode奖励）**: 累计奖励
7. **Episode Steps（Episode步数）**: 总步数

## 输出文件

评估器会生成以下文件：

1. `r2d2_evaluation_summary.json`: 统计摘要（JSON格式）
2. `r2d2_evaluation_results.csv`: 每个episode的详细数据（CSV格式）
3. `r2d2_evaluation_detailed.json`: 完整详细信息（可选，JSON格式）
4. `r2d2_evaluation_plots.png`: 综合评估图表（如果matplotlib可用）

## 完整示例

```python
"""
完整的R2D2评估示例
"""
from evaluators.r2d2_evaluator import R2D2Evaluator, load_r2d2_model
from pettingzoo.classic import hanabi_v5
import torch
import os

def main():
    # 配置
    checkpoint_path = "./checkpoints/best_model.pt"
    output_dir = "./evaluation_results/r2d2"
    num_episodes = 100
    epsilon = 0.0
    
    # 初始化环境
    env = hanabi_v5.env()
    env.reset(seed=0)
    obs, _, _, _, _ = env.last()
    obs_dim = len(obs["observation"])
    act_dim = env.action_space(env.agents[0]).n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"观察维度: {obs_dim}, 动作维度: {act_dim}")
    
    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    model = load_r2d2_model(checkpoint_path, obs_dim, act_dim, device)
    
    # 创建评估器
    evaluator = R2D2Evaluator(
        model=model,
        env=env,
        device=device,
        epsilon=epsilon,
    )
    
    # 执行评估
    print(f"开始评估 {num_episodes} 个episodes...")
    all_episode_stats, stats_summary = evaluator.evaluate(
        num_episodes=num_episodes,
        verbose=True
    )
    
    # 打印结果
    evaluator.print_summary(stats_summary)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    saved_files = evaluator.save_results(
        all_episode_stats,
        stats_summary,
        output_dir,
        save_detailed=True,
    )
    
    print(f"\n结果已保存到: {output_dir}")
    for file_type, file_path in saved_files.items():
        print(f"  - {file_type}: {file_path}")
    
    # 生成图表
    evaluator.plot_results(all_episode_stats, output_dir)
    
    env.close()

if __name__ == "__main__":
    main()
```

## 与命令行脚本的对比

使用模块化评估器的优势：

1. **可编程性**: 可以在代码中灵活使用，不限于命令行
2. **可扩展性**: 易于继承和扩展功能
3. **可测试性**: 便于单元测试
4. **可集成性**: 可以轻松集成到其他工作流中

命令行脚本 `evaluate_r2d2.py` 内部也使用了这个模块化评估器，提供了便捷的命令行接口。





