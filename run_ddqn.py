from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from agent.ddqn.ddqn_agent import DDQNAgent  # Use DDQN agent
from train_utils import train_agent

if __name__ == "__main__":
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    agent = DDQNAgent(
        num_actions=env.action_space.n,
        q_params=dict(
            backbone_input_shape=(240, 256, 3),
            backbone_conv_channels=[64, 128, 256, 512, 1024],
            backbone_output_dim=256,
            action_emb_table_size=env.action_space.n,
            action_emb_dim=32,
            reward_predictor_hidden_layer_dims=[128, 64, 32],
        ),
        lr=5e-5,
        gamma=0.95,
        ep=0.05,
        target_update_freq=100,  # Adjust as needed
    )

    checkpoint_dir = "checkpoints"

    train_params = dict(
        num_train_steps=10000,  # Small for testing
        train_set_size=5000,
        train_every=1,
        save_every=10,
        do_eval=True,
        eval_every=50,
        num_eval_episodes=1,
        max_eval_steps_per_episode=5000,
        render=False,
    )

    train_agent(agent, env, checkpoint_dir, train_params)
