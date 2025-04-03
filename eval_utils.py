import numpy as np
import tqdm

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from agent.base_agent import BaseAgent


def eval_agent(agent: BaseAgent, num_steps: int, render: bool) -> float:
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    agent.eval()

    total_reward = 0.0
    done = True
    progress_bar = tqdm.tqdm(range(num_steps), desc="Eval")
    for step in progress_bar:
        if done:
            state = env.reset()
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        progress_bar.set_postfix(
            average_reward=f"{total_reward / (step + 1):.4f}"
        )
        if render:
            env.render()

    env.close()

    return total_reward / num_steps
