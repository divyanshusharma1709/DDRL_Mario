import typing as T
import numpy as np
import tqdm

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from agent.base_agent import BaseAgent
from rl_utils import compute_reward


def eval_agent(
    agent: BaseAgent,
    num_episodes: int,
    max_eval_steps_per_episode: int,
    frame_stack_size: int,
    render: bool,
) -> T.Dict[str, T.Any]:
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    agent.eval()

    episode_lengths = []
    episode_rewards = []

    progress_bar = tqdm.tqdm(range(num_episodes), desc="Eval")
    prev_info = None
    for _ in progress_bar:
        episode_reward = 0.0
        episode_length = 0
        done = False
        state = env.reset()
        state_stack = [np.zeros_like(state) for _ in range(frame_stack_size)]
        while not done and episode_length < max_eval_steps_per_episode:
            state_stack = state_stack[1:]
            state_stack.append(state)
            action = agent.act(state_stack)
            state, reward, done, info = env.step(action)

            episode_reward += compute_reward(reward, info, prev_info)
            episode_length += 1

            if render:
                env.render()
            if len(episode_rewards) > 0:
                progress_bar.set_postfix(
                    ep_len=episode_length,
                    avg_r=np.mean(episode_rewards),
                    avg_len=np.mean(episode_lengths),
                )
            else:
                progress_bar.set_postfix(
                    ep_len=episode_length,
                    avg_r=episode_reward,
                    avg_len=episode_length,
                )

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()

    reward_array = np.array(episode_rewards)
    length_array = np.array(episode_lengths)
    return {
        "average_episode_reward": np.mean(reward_array),
        "average_episode_length": np.mean(length_array),
        "average_per_step_reward": np.mean(reward_array / length_array),
    }
