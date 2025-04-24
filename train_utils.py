import json
import os
import numpy as np
import torch
import typing as T
import tqdm
from gym import Env
from agent.base_agent import BaseAgent
from eval_utils import eval_agent
from rl_utils import compute_reward


DEFAULT_PARAM_DICT = dict(
    num_train_steps=100000,
    train_set_size=5000,
    train_batch_size=64,
    train_every=2500,
    save_every=10000,
    do_eval=True,
    eval_every=5000,
    num_eval_episodes=10,
    max_eval_steps_per_episode=5000,
    render=False,
    frame_stack_size=4,
)


def train_dqn_agent(
    agent: BaseAgent,
    env: Env,
    checkpoint_dir: str = "checkpoints",
    params: T.Optional[T.Dict[str, T.Any]] = None,
):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if params is None:
        params = {}

    with open(os.path.join(checkpoint_dir, "params.json"), "w", encoding="utf-8") as params_file:
        json.dump(params, params_file)

    # Get parameters
    save_every = params.get("save_every", DEFAULT_PARAM_DICT["save_every"])
    num_train_steps = params.get("num_train_steps", DEFAULT_PARAM_DICT["num_train_steps"])
    do_eval = params.get("do_eval", DEFAULT_PARAM_DICT["do_eval"])
    eval_every = params.get("eval_every", DEFAULT_PARAM_DICT["eval_every"])
    num_eval_episodes = params.get("num_eval_episodes", DEFAULT_PARAM_DICT["num_eval_episodes"])
    max_eval_steps_per_episode = params.get(
        "max_eval_steps_per_episode",
        DEFAULT_PARAM_DICT["max_eval_steps_per_episode"],
    )
    render = params.get("render", DEFAULT_PARAM_DICT["render"])
    frame_stack_size = params.get("frame_stack_size", DEFAULT_PARAM_DICT["frame_stack_size"])

    done = True
    metrics = {
        "average_eval_episode_reward": [],
        "average_eval_episode_length": [],
        "average_eval_per_step_reward": [],
    }

    episode_reward = 0.0
    prev_info = None
    state_stack = None
    pbar = tqdm.tqdm(range(num_train_steps), desc="Gathering")
    for step in pbar:
        agent.eval()
        if done:
            episode_reward = 0.0
            state = env.reset()
            state_stack = [np.zeros_like(state) for _ in range(frame_stack_size)]

        state_stack = state_stack[1:]
        state_stack.append(state)

        action = agent.act(state_stack)
        next_state, reward, done, info = env.step(action)

        state_stack.append(next_state)

        custom_reward = compute_reward(reward, info, prev_info)
        episode_reward += custom_reward

        agent.learn_one_step(state_stack[:-1], action, reward, state_stack[1:], done)
        state_stack = state_stack[1:]

        if do_eval and step % eval_every == 0 and eval_every > 0:
            eval_results_dict = eval_agent(
                agent=agent,
                num_episodes=num_eval_episodes,
                max_eval_steps_per_episode=max_eval_steps_per_episode,
                render=render,
                frame_stack_size=frame_stack_size,
            )
            metrics["average_eval_episode_reward"].append(
                eval_results_dict["average_episode_reward"]
            )
            metrics["average_eval_episode_length"].append(
                eval_results_dict["average_episode_length"]
            )
            metrics["average_eval_per_step_reward"].append(
                eval_results_dict["average_per_step_reward"]
            )

        if step > 0 and step % save_every == 0:
            agent.save(checkpoint_dir, step, metrics)

        state = next_state.copy()
        prev_info = info

        pbar.set_postfix(ep_reward=episode_reward, time_left=info["time"])

        if render:
            env.render()

    agent.save(checkpoint_dir, num_train_steps, metrics)
