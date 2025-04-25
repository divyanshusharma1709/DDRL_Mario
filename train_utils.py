import collections
import json
import os
import numpy as np
import pandas as pd
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
    max_train_episode_steps=5000,
)


def add_eval_metrics(eval_metrics: T.Dict[str, T.Any], eval_results_dict: T.Dict[str, T.Any]):
    eval_metrics["average_eval_episode_reward"].append(eval_results_dict["average_episode_reward"])
    eval_metrics["average_eval_episode_length"].append(eval_results_dict["average_episode_length"])
    eval_metrics["average_eval_per_step_reward"].append(
        eval_results_dict["average_per_step_reward"]
    )


def add_train_metrics(train_metrics: T.Dict[str, T.Any], step_metrics: T.Dict[str, T.Any]):
    for key, value in step_metrics.items():
        train_metrics[key].append(value)


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
    use_custom_reward = params.get("use_custom_reward", False)
    max_episode_steps = params.get(
        "max_train_episode_steps", DEFAULT_PARAM_DICT["max_train_episode_steps"]
    )

    eval_metrics = collections.defaultdict(list)
    train_metrics = collections.defaultdict(list)

    episode_train_loss = 0.0
    episode_train_reward = 0.0
    episode_idx = 0
    episode_steps = 0
    episode_reward = 0.0

    prev_info = None
    state_stack = None
    done = True
    pbar = tqdm.tqdm(range(num_train_steps), desc="Gathering")
    for step in pbar:
        agent.eval()
        if done or episode_steps > max_episode_steps:
            if episode_steps > 0:
                add_train_metrics(
                    train_metrics,
                    {
                        "ep_idx": episode_idx,
                        "ep_steps": episode_steps,
                        "ep_train_loss_per_step": episode_train_loss / (episode_steps + 1),
                        "ep_train_reward_per_step": episode_train_reward / (episode_steps + 1),
                    },
                )

            episode_idx += 1
            episode_reward = 0.0
            episode_steps = 0
            episode_train_loss = 0.0
            episode_train_reward = 0.0

            state = env.reset()
            state_stack = [np.zeros_like(state) for _ in range(frame_stack_size)]
            prev_info = None

        state_stack = state_stack[1:]
        state_stack.append(state)

        action = agent.act(state_stack)
        next_state, reward, done, info = env.step(action)

        state_stack.append(next_state)

        if use_custom_reward:
            new_reward = compute_reward(reward, info, prev_info)
        else:
            new_reward = reward
        episode_reward += new_reward

        step_loss = agent.learn_one_step(state_stack[:-1], action, reward, state_stack[1:], done)
        state_stack = state_stack[1:]

        if do_eval and step % eval_every == 0 and step != 0 and eval_every > 0:
            eval_results_dict = eval_agent(
                agent=agent,
                num_episodes=num_eval_episodes,
                max_eval_steps_per_episode=max_eval_steps_per_episode,
                render=render,
                frame_stack_size=frame_stack_size,
                curr_train_step=step,
            )
            add_eval_metrics(eval_metrics, eval_results_dict)

        if step > 0 and step % save_every == 0:
            agent.save(checkpoint_dir, step, eval_metrics)

        pbar.set_postfix(ep_reward=episode_reward, time=info["time"])

        if render:
            env.render()

        episode_train_reward += new_reward
        episode_train_loss += step_loss
        episode_steps += 1
        state = next_state.copy()
        prev_info = info

    agent.save(checkpoint_dir, num_train_steps, eval_metrics)

    pd.DataFrame(train_metrics).to_csv(f"{checkpoint_dir}/train_metrics.csv")
