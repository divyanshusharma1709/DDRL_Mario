import numpy as np
import tqdm

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from agent.base_agent import BaseAgent


def eval_agent(
    agent: BaseAgent,
    num_steps: int,
) -> float:
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    agent.eval()
    total_reward, max_reward = 0.0, -np.inf
    done = True
    progress_bar = tqdm.tqdm(range(num_steps), desc="Eval")
    for _ in progress_bar:
        if done:
            max_reward = max(max_reward, total_reward)
            total_reward = 0.0
            state = env.reset()
        action = agent.act(state, ep=0.00)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        progress_bar.set_postfix(max_reward=f"{max_reward:.4f}")
    return max_reward
