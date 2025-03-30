import tqdm
from gym import Env
from agent.base_agent import BaseAgent
from eval_utils import eval_agent


def train_dagger(
    agent: BaseAgent,
    env: Env,
    num_train_steps: int = 50_000,
    train_set_size: int = 1_000,
    train_batch_size: int = 256,
    train_every: int = 2_500,
    eval_every: int = 5_000,
    num_eval_steps: int = 1_000,
    ep: float = 0.05,
    render: bool = False,
):
    done = True
    state_buffer, action_buffer, reward_buffer = [], [], []
    mean_eval_rewards = []
    eval_steps = []
    episode_reward = 0.0
    pbar = tqdm.tqdm(range(num_train_steps), desc="Gathering")
    for step in pbar:
        agent.eval()
        if done:
            episode_reward = 0.0
            state = env.reset()

        action = agent.act(state, ep=ep)
        next_state, reward, done, _ = env.step(action)

        episode_reward += reward

        if step > 0 and step % train_every == 0:
            agent.update(
                state_buffer,
                action_buffer,
                reward_buffer,
                train_set_size,
                train_batch_size,
            )
        else:
            state_buffer.append(state)
            action_buffer.append(action)
            reward_buffer.append(reward)

        if step % eval_every == 0 and eval_every > 0:
            mean_eval_rewards.append(eval_agent(agent, env, num_eval_steps))
            eval_steps.append(step)

        state = next_state

        pbar.set_postfix(ep_reward=episode_reward)

        if render:
            env.render()

    return {"mean_eval_rewards": mean_eval_rewards, "eval_steps": eval_steps}
