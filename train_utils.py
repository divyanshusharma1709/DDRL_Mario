import random
import collections
import pandas as pd
import typing as T
import tqdm
from gym import Env
from agent.base_agent import BaseAgent
from eval_utils import eval_agent
from rl_utils import compute_reward, ReplayBuffer


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
    replay_buffer_batch_size=64,
    agent_update_frequency=5000,
    replay_buffer_sample_size=50000,
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
    batch_size = params.get(
        "replay_buffer_batch_size", DEFAULT_PARAM_DICT["replay_buffer_batch_size"]
    )
    agent_update_frequency = params.get(
        "agent_update_frequency", DEFAULT_PARAM_DICT["agent_update_frequency"]
    )
    replay_buffer_sample_size = params.get(
        "replay_buffer_sample_size", DEFAULT_PARAM_DICT["replay_buffer_sample_size"]
    )

    eval_metrics = collections.defaultdict(list)
    train_metrics = collections.defaultdict(list)

    replay_buffer = ReplayBuffer(max_size=100000, batch_size=batch_size)

    episode_train_loss = 0.0
    episode_train_reward = 0.0
    episode_idx = 0
    episode_steps = 0
    episode_reward = 0.0
    episode_max_x_pos = 0.0

    stuck_counter = 0
    max_stuck_iters = 250

    prev_info = None
    done = True
    pbar = tqdm.tqdm(range(num_train_steps), desc="Gathering")
    for step in pbar:
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
            prev_info = None

        state = state.__array__()
        action = agent.act(step, state)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.__array__()
        if use_custom_reward:
            new_reward = compute_reward(reward, info, prev_info)
        else:
            new_reward = reward
        episode_reward += new_reward

        step_loss = agent.compute_loss(
            step, *agent.tensorize(state, action, new_reward, next_state)
        ).item()

        replay_buffer.store(state, action, new_reward, next_state)

        if step > 0 and step % agent_update_frequency == 0:
            sample_size = min(replay_buffer_sample_size, len(replay_buffer))
            sample = replay_buffer.sample(n=sample_size)
            agent.learn_batch(step, sample)

        agent.update_state(step, state, action, reward, done)

        if do_eval and step % eval_every == 0 and step != 0 and eval_every > 0:
            eval_results_dict = eval_agent(
                agent=agent,
                num_episodes=num_eval_episodes,
                max_eval_steps_per_episode=max_eval_steps_per_episode,
                render=render,
                curr_train_step=step,
                use_custom_reward=use_custom_reward,
                frame_stack_size=frame_stack_size,
            )
            add_eval_metrics(eval_metrics, eval_results_dict)

        if step > 0 and step % save_every == 0:
            agent.save(checkpoint_dir, step, eval_metrics)

        if render:
            env.render()

        # check if stuck
        if prev_info and info["x_pos"] <= prev_info["x_pos"]:
            stuck_counter += 1
        else:
            stuck_counter = 0
        if stuck_counter >= max_stuck_iters:
            done = True

        pbar.set_postfix(
            ep_idx=episode_idx,
            ep_rew=episode_reward,
            ep_max_x=episode_max_x_pos,
            time=info["time"],
            ep=agent.ep_sched.get_epsilon(step),
            stuck_cnt=f"{stuck_counter}/{max_stuck_iters}",
        )

        episode_train_reward += new_reward
        episode_train_loss += step_loss
        episode_steps += 1
        episode_max_x_pos = max(episode_max_x_pos, info["x_pos"])
        state = next_state.copy()
        prev_info = info

    agent.save(checkpoint_dir, num_train_steps, eval_metrics)
    pd.DataFrame(train_metrics).to_csv(f"{checkpoint_dir}/train_metrics.csv")
    if do_eval:
        pd.DataFrame(eval_metrics).to_csv(f"{checkpoint_dir}/eval_metrics.csv")
