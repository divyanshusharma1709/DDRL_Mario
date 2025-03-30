from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from baseline_q_agent import BasicQAgent
from train_utils import train_dagger

if __name__ == "__main__":
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    agent = BasicQAgent(
        num_actions=env.action_space.n,
        q_params=dict(
            backbone_input_shape=(240, 256, 3),
            backbone_conv_channels=[64, 128, 256, 512, 1024, 2048],
            backbone_output_dim=256,
            action_emb_table_size=env.action_space.n,
            action_emb_dim=32,
            reward_predictor_hidden_layer_dims=[128, 64, 32],
        ),
        lr=5e-5,
    )
    results = train_dagger(
        agent,
        env,
        num_train_steps=100_000,
        train_set_size=10_000,
        train_batch_size=256,
        train_every=2_500,
        eval_every=2_500,
        num_eval_steps=10_000,
        render=True,
        ep=0.05,
    )
    print(results)
