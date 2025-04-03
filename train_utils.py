import json
import os
import typing as T
import tqdm
from gym import Env
from agent.base_agent import BaseAgent
from eval_utils import eval_agent


def extract_params(
    checkpoint_dir: str, params: T.Optional[T.Dict[str, T.Any]] = None
):
    if len(os.listdir(checkpoint_dir)) > 0:
        print("reading params from existing checkpoint")
        # read the parameters
        with open(
            os.path.join(checkpoint_dir, "params.json"), "r", encoding="utf-8"
        ) as params_file:
            return json.load(params_file)
    else:
        print("creating new checkpoint")
        default_params = {
            "save_every": 2_500,
            "num_train_steps": 50_000,
            "train_set_size": 1_000,
            "train_batch_size": 256,
            "train_every": 2_500,
            "do_eval": False,
            "eval_every": 5_000,
            "num_eval_steps": 1_000,
            "ep": 0.05,
            "render": False,
        }
        if params is not None:
            for key, default_value in default_params.items():
                default_params[key] = params.get(key, default_value)

        with open(
            os.path.join(checkpoint_dir, "params.json"), "w", encoding="utf-8"
        ) as params_file:
            json.dump(default_params, params_file)

        return default_params


def extract_metrics(checkpoint_dir: str) -> T.Tuple[int, T.Dict[str, T.Any]]:
    if len(os.listdir(checkpoint_dir)) == 0:
        return 0, {"mean_eval_rewards": [], "eval_steps": []}
    checkpoints = [
        ckpt for ckpt in os.listdir(checkpoint_dir) if os.path.isdir(ckpt)
    ]
    if len(checkpoints) == 0:
        return 0, {"mean_eval_rewards": [], "eval_steps": []}
    latest = sorted(checkpoints)[-1]
    start_step = int(latest.split("=")[-1])
    with open(
        os.path.join(checkpoint_dir, latest, "metrics.json"),
        "r",
        encoding="utf-8",
    ) as metrics_file:
        return start_step, json.load(metrics_file)


def train_dagger(
    agent: BaseAgent,
    env: Env,
    checkpoint_dir: str = "checkpoints",
    params: T.Optional[T.Dict[str, T.Any]] = None,
):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Get parameters
    params_dict = extract_params(checkpoint_dir, params)
    save_every = params_dict["save_every"]
    num_train_steps = params_dict["num_train_steps"]
    train_set_size = params_dict["train_set_size"]
    train_batch_size = params_dict["train_batch_size"]
    train_every = params_dict["train_every"]
    do_eval = params_dict["do_eval"]
    eval_every = params_dict["eval_every"]
    num_eval_steps = params_dict["num_eval_steps"]
    ep = params_dict["ep"]
    render = params_dict["render"]

    done = True
    next_state_buffer, state_buffer, action_buffer, reward_buffer = (
        [],
        [],
        [],
        [],
    )

    start_step, metrics = extract_metrics(checkpoint_dir)

    episode_reward = 0.0
    pbar = tqdm.tqdm(
        range(num_train_steps), desc="Gathering", initial=start_step
    )
    for step in pbar:
        agent.eval()
        if done:
            episode_reward = 0.0
            state = env.reset()

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        episode_reward += reward

        if step > 0 and step % train_every == 0:
            agent.update(
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
            metrics["mean_eval_rewards"].append(
                eval_agent(agent, num_eval_steps, render)
            )
            metrics["eval_steps"].append(step)

        if step > 0 and step % save_every == 0:
            agent.save(checkpoint_dir, step, metrics)

        state = next_state

        pbar.set_postfix(ep_reward=episode_reward)

        if render:
            env.render()

    agent.save(checkpoint_dir, num_train_steps, metrics)
