from gym import Env
import numpy as np
import numpy.typing as np_typing
import tqdm
import torch

from base_agent import BaseAgent


def eval_agent(
    agent: BaseAgent,
    env: Env,
    num_steps: int,
) -> float:
    agent.eval()
    total_reward = 0.0
    done = True
    for _ in tqdm.tqdm(range(num_steps), desc="Eval"):
        if done:
            state = env.reset()
        action = agent.act(state, ep=0.00)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward / num_steps
