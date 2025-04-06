import typing as T
import numpy as np
import tqdm

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from agent.base_agent import BaseAgent


def eval_agent(
    agent: BaseAgent,
    num_episodes: int,
    max_eval_steps_per_episode: int,
    render: bool,
) -> T.Dict[str, T.Any]:
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    agent.eval()

    episode_lengths = []
    episode_rewards = []

    progress_bar = tqdm.tqdm(range(num_episodes), desc="Eval")

    for _ in progress_bar:
        episode_reward = 0.0
        episode_length = 0
        done = False
        state = env.reset()
        while not done and episode_length < max_eval_steps_per_episode:

            action = agent.act(state)
            state, reward, done, _ = env.step(action)

            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

            progress_bar.set_postfix(
                ep_len=episode_length,
                avg_r=np.mean(episode_rewards),
                avg_len=np.mean(episode_lengths),
            )

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()

    return {
        "average_episode_reward": np.mean(episode_rewards),
        "average_episode_length": np.mean(episode_lengths),
    }
