import json
import os
import numpy as np
import torch
import typing as T
import tqdm
from gym import Env
from agent.base_agent import BaseAgent
from eval_utils import eval_agent


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
)


def train_agent(
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
    train_set_size = params.get("train_set_size", DEFAULT_PARAM_DICT["train_set_size"])
    train_batch_size = params.get("train_batch_size", DEFAULT_PARAM_DICT["train_batch_size"])
    train_every = params.get("train_every", DEFAULT_PARAM_DICT["train_every"])
    do_eval = params.get("do_eval", DEFAULT_PARAM_DICT["do_eval"])
    eval_every = params.get("eval_every", DEFAULT_PARAM_DICT["eval_every"])
    num_eval_episodes = params.get("num_eval_episodes", DEFAULT_PARAM_DICT["num_eval_episodes"])
    max_eval_steps_per_episode = params.get(
        "max_eval_steps_per_episode",
        DEFAULT_PARAM_DICT["max_eval_steps_per_episode"],
    )
    render = params.get("render", DEFAULT_PARAM_DICT["render"])

    done = True
    next_state_buffer, state_buffer, action_buffer, reward_buffer = (
        [],
        [],
        [],
        [],
    )

    metrics = {
        "average_eval_episode_reward": [],
        "average_eval_episode_length": [],
    }

    episode_reward = 0.0
    pbar = tqdm.tqdm(range(num_train_steps), desc="Gathering")
    for step in pbar:
        agent.eval()
        if done:
            episode_reward = 0.0
            state = env.reset()

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        episode_reward += reward

        if train_every == 1:
            state_tensor = torch.Tensor(state[np.newaxis, ...].copy())
            next_state_tensor = torch.Tensor(next_state[np.newaxis, ...].copy())
            action_tensor = torch.Tensor([[action]]).int()
            reward_tensor = torch.Tensor([[reward]])
            agent.learn_one_step(
                state_tensor.to(agent.device),
                action_tensor.to(agent.device),
                reward_tensor.to(agent.device),
                next_state_tensor.to(agent.device),
            )
        else:
            if step > 0 and step % train_every == 0:
                agent.batch_update(
                    state_buffer,
                    action_buffer,
                    reward_buffer,
                    next_state_buffer,
                    train_set_size,
                    train_batch_size,
                )
            else:
                next_state_buffer.append(next_state)
                state_buffer.append(state)
                action_buffer.append(action)
                reward_buffer.append(reward)

        if do_eval and step % eval_every == 0 and eval_every > 0:
            eval_results_dict = eval_agent(
                agent, num_eval_episodes, max_eval_steps_per_episode, render
            )
            metrics["average_eval_episode_reward"].append(
                eval_results_dict["average_episode_reward"]
            )
            metrics["average_eval_episode_length"].append(
                eval_results_dict["average_episode_length"]
            )

        if step > 0 and step % save_every == 0:
            agent.save(checkpoint_dir, step, metrics)

        state = next_state

        pbar.set_postfix(ep_reward=episode_reward)

        if render:
            env.render()

    agent.save(checkpoint_dir, num_train_steps, metrics)
