import tqdm
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym import Env
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from base_agent import BaseAgent
from baseline_q_agent import BasicQAgent
from eval_utils import eval_agent


def train_dagger(
    agent: BaseAgent,
    env: Env,
    num_train_steps: int = 50_000,
    train_set_size: int = 1_000,
    train_batch_size: int = 256,
    train_every: int = 2_500,
    eval_every: int = 5_000,
    num_eval_steps: int = 1_000,
    ep: float = 0.05,
    render: bool = False,
):
    done = True
    total_reward = 0.0
    state_buffer, action_buffer, reward_buffer = [], [], []
    mean_eval_rewards = []
    eval_steps = []
    for step in tqdm.tqdm(range(num_train_steps), desc="Training"):
        agent.eval()
        if done:
            state = env.reset()

        action = agent.act(state, ep=ep)
        next_state, reward, done, _ = env.step(action)

        total_reward += reward

        if step > 0 and step % train_every == 0:
            agent.update(
                state_buffer,
                action_buffer,
                reward_buffer,
                train_set_size,
                train_batch_size,
            )
        else:
            state_buffer.append(state)
            action_buffer.append(action)
            reward_buffer.append(reward)

        if step % eval_every == 0 and eval_every > 0:
            mean_eval_rewards.append(eval_agent(agent, env, num_eval_steps))
            eval_steps.append(step)

        state = next_state

        if render:
            env.render()

    return {"mean_eval_rewards": mean_eval_rewards, "eval_steps": eval_steps}


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
    )

    results = train_dagger(
        agent,
        env,
        num_train_steps=1000,
        train_set_size=50,
        train_batch_size=16,
        train_every=100,
        eval_every=200,
        num_eval_steps=100,
    )
    print(results)
