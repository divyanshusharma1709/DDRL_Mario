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
from IPython import get_ipython
from tqdm.notebook import tqdm as tqdm_notebook
from frame_processor import FrameProcessor
from agent.ddqn.replay import ReplayBuffer


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
    if params is None:
        params = {}

    # --- Parameters ---
    save_every = params.get("save_every", DEFAULT_PARAM_DICT["save_every"])
    num_train_steps = params.get("num_train_steps", DEFAULT_PARAM_DICT["num_train_steps"])
    do_eval = params.get("do_eval", DEFAULT_PARAM_DICT["do_eval"])
    eval_every = params.get("eval_every", DEFAULT_PARAM_DICT["eval_every"])
    num_eval_episodes = params.get("num_eval_episodes", DEFAULT_PARAM_DICT["num_eval_episodes"])
    max_eval_steps_per_episode = params.get("max_eval_steps_per_episode", DEFAULT_PARAM_DICT["max_eval_steps_per_episode"])
    render = params.get("render", DEFAULT_PARAM_DICT["render"])
    frame_stack_size = params.get("frame_stack_size", DEFAULT_PARAM_DICT["frame_stack_size"])
    use_custom_reward = params.get("use_custom_reward", False)
    max_episode_steps = params.get("max_train_episode_steps", DEFAULT_PARAM_DICT["max_train_episode_steps"])

    # --- Frame processor created once ---
    frame_processor = FrameProcessor(
        frame_height=240,
        frame_width=256,
        stack_size=frame_stack_size,
        grayscale=True,
        resize=False
    )

    eval_metrics = collections.defaultdict(list)
    train_metrics = collections.defaultdict(list)

    episode_train_loss = 0.0
    episode_train_reward = 0.0
    episode_idx = 0
    episode_steps = 0
    episode_reward = 0.0
    episode_max_x_pos = 0.0
    prev_info = None
    done = True

    # --- Stuck detection vars ---
    prev_x_pos = 0
    stuck_counter = 0
    stuck_threshold = 50

    # --- Progress bar ---
    pbar = tqdm.tqdm(range(num_train_steps), desc="Gathering")

    for step in pbar:
        if step == 50000:
            print("Resetting replay buffer at 50k steps")
            agent.replay_buffer = ReplayBuffer(capacity=100_000)
        agent.eval()

        if done or episode_steps >= max_episode_steps:
            if episode_steps > 0:
                add_train_metrics(
                    train_metrics,
                    {
                        "ep_idx": episode_idx,
                        "ep_steps": episode_steps,
                        "ep_train_loss_per_step": episode_train_loss / (episode_steps + 1),
                        "ep_total_loss": episode_train_loss,
                        "ep_train_reward_per_step": episode_train_reward / (episode_steps + 1),
                        "ep_total_reward": episode_train_reward,
                        "ep_max_x_pos": max(prev_info["x_pos"], episode_max_x_pos),
                    },
                )
                pd.DataFrame(train_metrics).to_csv(f"{checkpoint_dir}/train_metrics.csv")

            episode_idx += 1
            episode_reward = 0.0
            episode_steps = 0
            episode_train_loss = 0.0
            episode_train_reward = 0.0
            episode_max_x_pos = 0.0

            state = env.reset()
            state_stack = frame_processor.reset(state)
            prev_info = None

            prev_x_pos = 0
            stuck_counter = 0

        # --- Act ---
        start_epsilon = 1.0
        final_epsilon = 0.05
        decay_steps = 100000
        
        agent.ep = max(final_epsilon, start_epsilon - (start_epsilon - final_epsilon) * (step / decay_steps))
        if step > 50000 and step % 10000 == 0:
            agent.ep = max(agent.ep, 0.3)
        action = agent.act(state_stack)

        # --- Step env ---
        JUMP_ACTIONS = {2, 4, 5}

        hold_jump_frames = 3  # hold jump for 3 frames
        total_reward = 0.0
        final_info = None

        if action in JUMP_ACTIONS:
            for _ in range(hold_jump_frames):
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                final_info = info
                if done:
                    break
        else:
            next_state, reward, done, info = env.step(action)
            total_reward = reward
            final_info = info
        
        # --- Strong Reward Shaping ---
        if prev_info is not None and final_info is not None:
            x_diff = final_info["x_pos"] - prev_info["x_pos"]
            time_diff = final_info["time"] - prev_info["time"]

            # Bonus for good forward jump
            if action in {2, 4, 5}:  # Jump actions
                if x_diff > 5:
                    total_reward += 10.0  # Small forward leap
                if x_diff > 15:
                    total_reward += 20.0  # Bigger forward jump
                if x_diff > 30:
                    total_reward += 30.0  # Huge success jump

            # Bonus for surviving after jump
            if time_diff > 10:
                total_reward += 5.0  # Lived longer

            # Bonus for crossing key distance markers
            if prev_info["x_pos"] <= 200 < final_info["x_pos"]:
                total_reward += 20.0
            if prev_info["x_pos"] <= 400 < final_info["x_pos"]:
                total_reward += 40.0
            if prev_info["x_pos"] <= 600 < final_info["x_pos"]:
                total_reward += 60.0



        # --- Stuck detection ---
        curr_x_pos = final_info.get('x_pos', 0)
        if curr_x_pos <= prev_x_pos:
            stuck_counter += 1
        else:
            stuck_counter = 0
        prev_x_pos = curr_x_pos
        if stuck_counter >= stuck_threshold:
            done = True

        # --- Preprocess next frame and update stack ---
        next_state_stack = frame_processor.step(next_state)

        # --- Compute reward ---
        if use_custom_reward:
            new_reward = compute_reward(total_reward, final_info, prev_info)
        else:
            new_reward = total_reward
        episode_reward += new_reward

        # --- Train step ---
        step_loss = agent.learn_one_step(state_stack, action, new_reward, next_state_stack, done)

        if step_loss is not None:
            episode_train_loss += step_loss
        episode_train_reward += new_reward
        episode_steps += 1
        episode_max_x_pos = max(episode_max_x_pos, final_info["x_pos"])

        # --- Bookkeeping ---
        state_stack = next_state_stack
        prev_info = final_info

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

    # --- Final save ---
    agent.save(checkpoint_dir, num_train_steps, eval_metrics)
    pd.DataFrame(train_metrics).to_csv(f"{checkpoint_dir}/train_metrics.csv")
    if do_eval:
        pd.DataFrame(eval_metrics).to_csv(f"{checkpoint_dir}/eval_metrics.csv")
