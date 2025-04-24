from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from agent.dqn_agent import BasicQAgent
from agent.emdqn_agent import EMDQNAgent
from train_utils import train_dqn_agent

if __name__ == "__main__":
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    q_params = dict(
        backbone_input_shape=(240, 256),
        backbone_channels_per_image=3,
        backbone_frame_stack_size=4,
        backbone_conv_channels=[64, 128, 256, 512],
        backbone_output_dim=256,
        action_emb_table_size=env.action_space.n,
        action_emb_dim=16,
        reward_predictor_hidden_layer_dims=[256, 128, 64, 32],
    )

    checkpoint_dir = "checkpoints"

    # train_params = dict(
    #     num_train_steps=1_000_000,
    #     save_every=100_000,
    #     do_eval=True,
    #     eval_every=50_000,
    #     num_eval_episodes=3,
    #     max_eval_steps_per_episode=7500,
    #     render=False,
    # )

    train_params = dict(
        num_train_steps=100_000,
        save_every=10_000,
        do_eval=True,
        eval_every=10_000,
        num_eval_episodes=1,
        max_eval_steps_per_episode=5_000,
        render=False,
        frame_stack_size=q_params["backbone_frame_stack_size"],
    )

    lr = 0.0001
    alpha_mem = 0.05
    ep = 0.025
    gamma = 0.99
    emdqn_agent = EMDQNAgent(
        num_actions=env.action_space.n,
        q_params=q_params,
        lr=lr,
        ep=ep,
        gamma=gamma,
        alpha_mem=alpha_mem,
        memory_update_freq=1_000,
        stack_size=train_params["frame_stack_size"],
    )
    dqn_agent = BasicQAgent(
        num_actions=env.action_space.n,
        q_params=q_params,
        lr=lr,
        ep=ep,
        gamma=gamma,
    )
    train_dqn_agent(
        emdqn_agent,
        env,
        checkpoint_dir + f"_emdqn_{alpha_mem=}_{lr=}",
        train_params,
    )
    train_dqn_agent(
        dqn_agent,
        env,
        checkpoint_dir + f"_dqn_{lr=}",
        train_params,
    )
