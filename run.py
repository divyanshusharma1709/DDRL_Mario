import os
import json
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from agent.dqn_agent import BasicQAgent
from agent.emdqn_agent import EMDQNAgent
from train_utils import train_dqn_agent
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN agents on Super Mario Bros")
    parser.add_argument("--use_custom_reward", action="store_true")
    parser.add_argument(
        "--num_train_steps", type=int, default=100_000, help="Number of training steps"
    )
    parser.add_argument("--save_every", type=int, default=10_000, help="Save checkpoint frequency")
    parser.add_argument("--eval_every", type=int, default=10_000, help="Evaluation frequency")
    parser.add_argument(
        "--num_eval_episodes", type=int, default=1, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max_eval_steps", type=int, default=5_000, help="Maximum evaluation steps per episode"
    )
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--frame_stack_size", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--alpha_mem", type=float, default=0.05, help="Memory alpha parameter for EMDQN"
    )
    parser.add_argument("--ep", type=float, default=0.025, help="Exploration probability")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--memory_update_freq", type=int, default=1_000, help="Memory update frequency for EMDQN"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="both",
        choices=["dqn", "emdqn", "both"],
        help="Agent type to train",
    )
    parser.add_argument("--do_eval", action="store_true", help="Whether to do evals")

    # Q-network parameters
    parser.add_argument(
        "--backbone_input_shape",
        type=int,
        nargs=2,
        default=[240, 256],
        help="Input shape for backbone network",
    )
    parser.add_argument("--backbone_channels", type=int, default=3, help="Channels per image")
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

    return parser.parse_args()


def save_args_to_json(args: argparse.Namespace, directory: str) -> None:
    """Save arguments to a JSON file in the specified directory."""
    os.makedirs(directory, exist_ok=True)

    # Convert args to a dictionary
    args_dict: dict = vars(args)

    # Convert any non-serializable types to strings or appropriate formats
    for key, value in args_dict.items():
        if isinstance(value, (list, tuple)):
            args_dict[key] = list(value)

    # Write to JSON file
    with open(os.path.join(directory, "params.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4)


if __name__ == "__main__":
    args = parse_args()

    # Create the checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Save arguments to JSON file in the checkpoint directory
    save_args_to_json(args, args.checkpoint_dir)

    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    q_params = dict(
        backbone_input_shape=tuple(args.backbone_input_shape),
        backbone_channels_per_image=args.backbone_channels,
        backbone_frame_stack_size=args.frame_stack_size,
        backbone_conv_channels=args.backbone_conv_channels,
        backbone_output_dim=args.backbone_output_dim,
        action_emb_table_size=env.action_space.n,
        action_emb_dim=args.action_emb_dim,
        reward_predictor_hidden_layer_dims=args.reward_predictor_dims,
    )

    train_params = dict(
        num_train_steps=args.num_train_steps,
        save_every=args.save_every,
        do_eval=True,
        eval_every=args.eval_every,
        num_eval_episodes=args.num_eval_episodes,
        max_eval_steps_per_episode=args.max_eval_steps,
        render=args.render,
        frame_stack_size=args.frame_stack_size,
        use_custom_reward=args.use_custom_reward,
    )

    if args.agent_type in ["emdqn", "both"]:
        emdqn_agent = EMDQNAgent(
            num_actions=env.action_space.n,
            q_params=q_params,
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
            num_actions=env.action_space.n,
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
