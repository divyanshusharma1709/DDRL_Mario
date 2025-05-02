from agent.dqn_agent import BasicQAgent
from agent.emdqn_agent import EMDQNAgent
from env_utils import create_env
from eval_utils import eval_agent
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run evaluation for Mario agent")
    parser.add_argument("--agent_type", type=str, default="dqn", help="Type of agent")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="dqn_final_v0", help="Directory for checkpoints"
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=[150000], help="Step numbers to load"
    )
    parser.add_argument("--stack_size", type=int, default=8, help="Number of frames to stack")

    args = parser.parse_args()

    agent_type = args.agent_type
    checkpoint_dir = args.checkpoint_dir
    stack_size = args.stack_size

    env = create_env(stack_size=stack_size, env_version="v0")
    num_actions = env.action_space.n
    q_params = {
        "backbone_input_shape": [84, 84],
        "backbone_channels_per_image": stack_size,
        "backbone_conv_channels": [32, 64, 128, 256],
        "backbone_output_dim": 512,
        "action_emb_table_size": num_actions,
        "action_emb_dim": 16,
        "reward_predictor_hidden_layer_dims": [256],
    }
    ep_sched_params = {
        "ep_values": [],
        "ep_decay_step_counts": [],
    }
    agent = None
    for step in args.steps:
        if agent_type == "dqn":
            agent = BasicQAgent(
                num_actions=num_actions,
                q_params=q_params,
                ep_sched_params=ep_sched_params,
                lr=0.0,
                gamma=0.0,
                num_steps_between_target_updates=0,
            )
        elif agent_type == "emdqn":
            agent = EMDQNAgent(
                num_actions=num_actions,
                q_params=q_params,
                ep_sched_params=ep_sched_params,
                lr=0.0,
                gamma=0.0,
                alpha_mem=0.0,
                memory_update_freq=0,
                stack_size=args.stack_size,
                num_steps_between_target_updates=0,
            )
        assert agent is not None
        agent.load(checkpoint_dir=checkpoint_dir, step=step)
        eval_agent(
            agent=agent,
            curr_train_step=step,
            num_episodes=1,
            max_eval_steps_per_episode=100000,
            render=False,
            frame_stack_size=stack_size,
            use_custom_reward=True,
            checkpoint_dir=checkpoint_dir,
            env_version="v0",
        )
