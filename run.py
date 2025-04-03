from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from agent.q_agent import BasicQAgent
from train_utils import train_dagger

if __name__ == "__main__":
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    agent = BasicQAgent(
        num_actions=env.action_space.n,
        q_params=dict(
            backbone_input_shape=(240, 256, 3),
            backbone_conv_channels=[64, 256, 1024],
            backbone_output_dim=256,
            action_emb_table_size=env.action_space.n,
            action_emb_dim=32,
            reward_predictor_hidden_layer_dims=[128, 64, 32],
        ),
        lr=5e-5,
        ep=0.05,
    )

    checkpoint_dir = "checkpoints"

    train_params = dict(
        num_train_steps=100000,
        train_set_size=5000,
        train_batch_size=64,
        train_every=2500,
        save_every=10000,
        do_eval=True,
        eval_every=10000,
        num_eval_steps=10000,
        render=False,
    )

    train_dagger(agent, env, checkpoint_dir, train_params)
