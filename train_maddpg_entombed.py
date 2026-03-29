"""
MADDPG训练脚本 - Entombed Cooperative (PettingZoo Atari, parallel)
"""
import argparse
import os
import json
from datetime import datetime
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from pettingzoo.atari import entombed_cooperative_v3
from supersuit import color_reduction_v0, resize_v1, frame_stack_v1

from algorithms.maddpg import MADDPG


def create_entombed_env(preprocess=True, seed=None):
    """创建Entombed Cooperative环境（并可选预处理）"""
    env = entombed_cooperative_v3.parallel_env(render_mode=None)
    if preprocess:
        env = color_reduction_v0(env)
        env = resize_v1(env, 84, 84)
        env = frame_stack_v1(env, stack_size=4)
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    return env


def get_env_info(env):
    """获取环境信息"""
    first_agent = env.agents[0]
    obs_space = env.observation_space(first_agent)
    action_space = env.action_space(first_agent)

    if hasattr(obs_space, "shape"):
        obs_dim = int(np.prod(obs_space.shape))
    elif hasattr(obs_space, "spaces"):
        obs_dim = int(np.sum([np.prod(space.shape) for space in obs_space.spaces.values()]))
    else:
        obs_dim = 128

    action_dim = action_space.n if hasattr(action_space, "n") else 4
    num_agents = len(env.agents)
    return obs_dim, action_dim, num_agents


def process_observation(obs):
    """处理观察：转换为向量格式"""
    if isinstance(obs, dict) and "observation" in obs:
        obs = obs["observation"]
    if isinstance(obs, np.ndarray):
        return obs.flatten()
    return obs


def setup_chinese_font():
    """设置中文字体，如果找不到则返回False"""
    import matplotlib.font_manager as fm

    chinese_fonts = [
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "STHeiti",
        "STSong",
    ]

    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams["font.sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def plot_training_curve(eval_results, save_path):
    """绘制训练曲线"""
    iterations = [r["episode"] for r in eval_results]
    mean_rewards = [r["mean_reward"] for r in eval_results]
    std_rewards = [r["std_reward"] for r in eval_results]

    has_chinese_font = setup_chinese_font()
    if has_chinese_font:
        mean_label = "平均奖励"
        std_label = "标准差"
        ylabel = "平均奖励 (episodes)"
        title = "MADDPG 训练曲线 (Entombed)"
    else:
        mean_label = "Mean Reward"
        std_label = "Std Dev"
        ylabel = "Mean Reward (episodes)"
        title = "MADDPG Training Curve (Entombed)"

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, mean_rewards, "b-", label=mean_label, linewidth=2)
    plt.fill_between(
        iterations,
        np.array(mean_rewards) - np.array(std_rewards),
        np.array(mean_rewards) + np.array(std_rewards),
        alpha=0.3,
        color="blue",
        label=std_label,
    )
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


def evaluate_model(model, preprocess_atari=True, num_episodes=20, device="cuda", seed=None):
    """评估模型，返回平均奖励"""
    episode_rewards = []

    for episode in range(num_episodes):
        env = create_entombed_env(preprocess=preprocess_atari, seed=seed + episode if seed is not None else None)
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _ = reset_out
        else:
            obs = reset_out

        agent_ids = list(env.possible_agents)
        sample_obs = process_observation(obs[agent_ids[0]])
        zero_obs = np.zeros_like(sample_obs)
        obs_dict = {aid: process_observation(obs.get(aid, zero_obs)) for aid in agent_ids}

        done = False
        episode_reward = 0.0
        while not done:
            state = np.concatenate([obs_dict[aid] for aid in agent_ids])
            actions_arr = model.select_actions(state, epsilon=0.0)
            actions = {aid: int(actions_arr[idx]) for idx, aid in enumerate(agent_ids) if aid in env.agents}

            obs, rewards, terminations, truncations, infos = env.step(actions)
            rewards_sum = sum(float(rewards.get(aid, 0.0)) for aid in agent_ids)
            episode_reward += rewards_sum

            for aid in agent_ids:
                if aid in obs:
                    obs_dict[aid] = process_observation(obs[aid])
                else:
                    obs_dict[aid] = zero_obs.copy()

            done = len(env.agents) == 0

        episode_rewards.append(episode_reward)
        env.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
    }


def main():
    parser = argparse.ArgumentParser(description="MADDPG训练脚本 - Entombed Cooperative")

    # 训练参数
    parser.add_argument("--num_episodes", type=int, default=1000, help="训练episode数")
    parser.add_argument("--max_steps", type=int, default=1000, help="每个episode最大步数")
    parser.add_argument("--batch_size", type=int, default=64, help="批量大小")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--lr_actor", type=float, default=1e-3, help="Actor学习率")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="Critic学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--tau", type=float, default=0.01, help="软更新系数")
    parser.add_argument("--epsilon_start", type=float, default=0.1, help="初始探索率")
    parser.add_argument("--epsilon_end", type=float, default=0.02, help="最低探索率")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="探索率衰减")
    parser.add_argument("--train_freq", type=int, default=1, help="每N步更新一次")

    # 评估参数
    parser.add_argument("--eval_freq", type=int, default=50, help="每N个episode评估一次")
    parser.add_argument("--eval_episodes", type=int, default=20, help="每次评估的episode数")

    # 其他参数
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志保存目录")
    parser.add_argument("--experiment_name", type=str, default="maddpg_entombed", help="实验名称")
    parser.add_argument("--device", type=str, default="cuda", help="设备（cuda/cpu）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_preprocess", action="store_true", help="禁用Atari图像预处理")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.log_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    print("=" * 60)
    print("MADDPG 训练 - Entombed Cooperative")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"实验目录: {exp_dir}")
    print("=" * 60)

    # 初始化环境以获取维度信息
    env = create_entombed_env(preprocess=not args.no_preprocess, seed=args.seed)
    obs_dim, action_dim, num_agents = get_env_info(env)
    env.close()

    print(f"观察维度: {obs_dim}")
    print(f"动作维度: {action_dim}")
    print(f"智能体数量: {num_agents}")

    # 创建MADDPG模型
    maddpg = MADDPG(
        state_dim=obs_dim * num_agents,
        action_dim=action_dim,
        n_agents=num_agents,
        single_agent_action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        tau=args.tau,
        discrete=True,
        device=device,
    )

    eval_results = []
    episode_rewards = deque(maxlen=100)

    for episode in tqdm(range(1, args.num_episodes + 1), desc="训练进度"):
        env = create_entombed_env(preprocess=not args.no_preprocess, seed=args.seed + episode)
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _ = reset_out
        else:
            obs = reset_out

        agent_ids = list(env.possible_agents)
        sample_obs = process_observation(obs[agent_ids[0]])
        zero_obs = np.zeros_like(sample_obs)
        obs_dict = {aid: process_observation(obs.get(aid, zero_obs)) for aid in agent_ids}

        episode_reward = 0.0
        done = False
        step_count = 0
        epsilon = max(args.epsilon_end, args.epsilon_start * (args.epsilon_decay ** episode))

        while not done and step_count < args.max_steps:
            state = np.concatenate([obs_dict[aid] for aid in agent_ids])
            actions_arr = maddpg.select_actions(state, epsilon=epsilon)
            actions = {aid: int(actions_arr[idx]) for idx, aid in enumerate(agent_ids) if aid in env.agents}

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            rewards_sum = sum(float(rewards.get(aid, 0.0)) for aid in agent_ids)
            episode_reward += rewards_sum

            next_obs_dict = {}
            for aid in agent_ids:
                if aid in next_obs:
                    next_obs_dict[aid] = process_observation(next_obs[aid])
                else:
                    next_obs_dict[aid] = zero_obs.copy()

            next_state = np.concatenate([next_obs_dict[aid] for aid in agent_ids])
            done = len(env.agents) == 0

            maddpg.push_transition(
                state=state,
                actions=actions_arr,
                reward=rewards_sum,
                next_state=next_state,
                done=done,
            )

            if len(maddpg.replay_buffer) >= args.batch_size and step_count % args.train_freq == 0:
                maddpg.update(args.batch_size)

            obs_dict = next_obs_dict
            step_count += 1

        episode_rewards.append(episode_reward)
        env.close()

        if episode % args.eval_freq == 0:
            eval_stats = evaluate_model(
                maddpg,
                preprocess_atari=not args.no_preprocess,
                num_episodes=args.eval_episodes,
                device=device,
                seed=args.seed + episode * 10000,
            )
            eval_results.append({
                "episode": episode,
                "mean_reward": eval_stats["mean_reward"],
                "std_reward": eval_stats["std_reward"],
                "min_reward": eval_stats["min_reward"],
                "max_reward": eval_stats["max_reward"],
            })

            print(f"\nEpisode {episode}: 评估结果")
            print(f"  平均奖励: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
            print(f"  范围: [{eval_stats['min_reward']:.4f}, {eval_stats['max_reward']:.4f}]")

            eval_path = os.path.join(exp_dir, "eval_results.json")
            with open(eval_path, "w") as f:
                json.dump(eval_results, f, indent=2)

            best_path = os.path.join(args.checkpoint_dir, "maddpg_entombed_best.pt")
            maddpg.save(best_path)

    final_path = os.path.join(args.checkpoint_dir, "maddpg_entombed_final.pt")
    maddpg.save(final_path)

    if len(eval_results) > 0:
        plot_path = os.path.join(exp_dir, "training_curve.png")
        plot_training_curve(eval_results, plot_path)

    print("\n训练完成！")
    print(f"模型保存在: {args.checkpoint_dir}")
    print(f"日志保存在: {exp_dir}")


if __name__ == "__main__":
    main()










