import os
import json
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from agent.dqn_agent import BasicQAgent
from agent.emdqn_agent import EMDQNAgent
from agent.ddqn.ddqn_agent import DDQNAgent
from train_utils import train_dqn_agent
import argparse
import warnings
from env_utils import create_env

num_frames = 8

warnings.filterwarnings("ignore", category=DeprecationWarning)

import gym

class FrameSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDQN agents on Super Mario Bros")
    parser.add_argument("--use_custom_reward", action="store_true")
    parser.add_argument(
        "--num_train_steps", type=int, default=250_000, help="Number of training steps"
    )
    parser.add_argument("--max_train_episode_steps", type=int, default=10_000)

    parser.add_argument("--save_every", type=int, default=10000, help="Save checkpoint frequency")
    parser.add_argument("--eval_every", type=int, default=10_000, help="Evaluation frequency")
    parser.add_argument(
        "--num_eval_episodes", type=int, default=1, help="Number of evaluation episodes"
    ) 
    parser.add_argument(
        "--max_eval_steps", type=int, default=10000, help="Maximum evaluation steps per episode"
    )
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--frame_stack_size", type=int, default=8, help="Number of frames to stack")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--alpha_mem", type=float, default=0.01, help="Memory alpha parameter for EMDQN"
    )
    parser.add_argument(
        "--ep_init", type=float, default=0.78, help="Exploration probability (initial)"
    )
    parser.add_argument(
        "--ep_final", type=float, default=0.1, help="Exploration probability (final)"
    )
    parser.add_argument("--num_steps_before_decay", type=int, default=0)
    parser.add_argument("--num_decay_steps", type=int, default=250000)

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--memory_update_freq", type=int, default=1_000, help="Memory update frequency for EMDQN"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="ddqn_check", help="Directory to save checkpoints"
    )

    parser.add_argument("--num_steps_between_target_updates", type=int, default=500)

    parser.add_argument(
        "--agent_type",
        type=str,
        default="ddqn",
        choices=["dqn", "emdqn", "ddqn", "both"],
        help="Agent type to train",
    )
    parser.add_argument("--do_eval", action="store_true", help="Whether to do evals")
    parser.add_argument(
        "--backbone_input_shape",
        type=int,
        nargs=2,
        default=[num_frames, 84, 84],
        help="Input shape for backbone network",
    )
    parser.add_argument("--backbone_channels", type=int, default=3, help="Channels per image")
    parser.add_argument("--device", type=str, default='cuda', help="Device used for training")

    parser.add_argument(
        "--backbone_conv_channels",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="Conv channels for backbone",
    )
    parser.add_argument(
        "--backbone_output_dim", type=int, default=256, help="Output dimension of backbone"
    )
    parser.add_argument("--action_emb_dim", type=int, default=16, help="Action embedding dimension")
    parser.add_argument(
        "--reward_predictor_dims",
        type=int,
        nargs="+",
        default=[256, 128, 64, 32],
        help="Reward predictor hidden layers",
    )
    parser.add_argument(
        "--target_update_freq", type=int, default=100, help="Target network update frequency for DDQN"
    )
    parser.add_argument("--replay_buffer_batch_size", type=int, default=512)
    parser.add_argument("--agent_update_frequency", type=int, default=4)
    parser.add_argument("--replay_buffer_sample_size", type=int, default=2048)
    parser.add_argument("--replay_buffer_max_size", type=int, default=50_000)
    return parser.parse_args()


def save_args_to_json(args: argparse.Namespace, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    args_dict: dict = vars(args)
    for key, value in args_dict.items():
        if isinstance(value, (list, tuple)):
            args_dict[key] = list(value)
    with open(os.path.join(directory, "params.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_args_to_json(args, args.checkpoint_dir)

    env = create_env(stack_size=num_frames)
    num_actions = env.action_space.n

    # obs = env.reset()
    # print(obs.shape)
    # exit()

    q_params = {
        "input_shape": tuple(args.backbone_input_shape),
        "num_actions": num_actions
    }

    train_params = {
        "num_train_steps": args.num_train_steps,
        "save_every": args.save_every,
        "do_eval": args.do_eval,
        "eval_every": args.eval_every,
        "num_eval_episodes": args.num_eval_episodes,
        "max_eval_steps_per_episode": args.max_eval_steps,
        "render": args.render,
        "frame_stack_size": args.frame_stack_size,
        "use_custom_reward": args.use_custom_reward,
        "max_train_episode_steps": args.max_train_episode_steps,
        "replay_buffer_batch_size": args.replay_buffer_batch_size,
        "replay_buffer_sample_size": args.replay_buffer_sample_size,
        "replay_buffer_max_size": args.replay_buffer_max_size,
        "agent_update_frequency": args.agent_update_frequency,
    }

    ep_sched_params = {
        "ep_init": args.ep_init,
        "ep_final": args.ep_final,
        "num_steps_before_decay": args.num_steps_before_decay,
        "num_decay_steps": args.num_decay_steps,
        "warmup_ep": 0.9,
        "warmup_steps": 111000
    }

    if args.agent_type in ["emdqn", "both"]:
        emdqn_agent = EMDQNAgent(
            num_actions=num_actions,
            q_params=q_params,
            ep_sched_params=ep_sched_params,
            lr=args.lr,
            ep=args.ep,
            gamma=args.gamma,
            alpha_mem=args.alpha_mem,
            memory_update_freq=args.memory_update_freq,
            stack_size=args.frame_stack_size,
        )
        train_dqn_agent(
            emdqn_agent,
            env,
            args.checkpoint_dir,
            train_params,
        )

    if args.agent_type in ["dqn", "both"]:
        dqn_agent = BasicQAgent(
            num_actions=num_actions,
            ep_sched_params=ep_sched_params,
            q_params=q_params,
            lr=args.lr,
            ep=args.ep,
            gamma=args.gamma,
        )
        train_dqn_agent(
            dqn_agent,
            env,
            args.checkpoint_dir,
            train_params,
        )

    if args.agent_type in ["ddqn", "both"]:
        ddqn_agent = DDQNAgent(
            num_actions=num_actions,
            q_params=q_params,
            ep_sched_params=ep_sched_params,
            lr=args.lr,
            gamma=args.gamma,
            target_update_freq=args.target_update_freq,
            num_steps_between_target_updates=args.num_steps_between_target_updates
        )
        ddqn_agent.load("ddqn_check/", step = 110000)
        print("Model restored. Runnin start eval")
        train_dqn_agent(
            ddqn_agent,
            env,
            args.checkpoint_dir,
            train_params,
        )
