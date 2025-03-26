import random
import torch
import tqdm
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from baseline_q_agent import BasicQAgent


class ExperienceBuffer(Dataset):
    def __init__(self, states, actions, rewards):
        self.states = states
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
        )


def eval_agent(
    agent: BasicQAgent, num_episodes: int, max_episode_steps: int = 500
) -> float:
    agent.eval()
    rewards = []
    for _ in tqdm.tqdm(range(num_episodes)):
        done = False
        state = env.reset()
        total_reward = 0
        steps = 0
        while not done and steps < max_episode_steps:
            state_tensor = torch.Tensor(state.copy())[None, :]
            action = agent.act(state_tensor, ep=0.00)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
    return sum(rewards) / num_episodes


env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

q_params = dict(
    backbone_input_shape=(240, 256, 3),
    backbone_conv_channels=[64, 128, 256, 512, 1024, 2048],
    backbone_output_dim=256,
    action_emb_table_size=env.action_space.n,
    action_emb_dim=32,
    reward_predictor_hidden_layer_dims=[128, 64, 32],
)

train_every = 1_000
train_set_size = 1_000
train_batch_size = 64
total_train_steps = 50_000
num_eval_episodes = 10
max_eval_episode_steps = 500
states, actions, rewards = [], [], []

agent = BasicQAgent(
    num_actions=env.action_space.n,
    q_params=q_params,
)

avg_eval_reward_init = eval_agent(
    agent,
    num_eval_episodes,
    max_eval_episode_steps,
)

done = True
total_reward = 0.0
total_rewards = []
for step in tqdm.tqdm(range(total_train_steps)):
    agent.eval()
    if done:
        if total_reward != 0:
            total_rewards.append(total_reward)
        total_reward = 0.0
        state = env.reset()
    state_tensor = torch.Tensor(state.copy())[None, :]
    action = agent.act(state_tensor, ep=0.05)
    next_state, reward, done, info = env.step(action)

    if step > 0 and step % train_every == 0:
        agent.train()
        all_states = np.vstack([state[np.newaxis, :] for state in states])
        states_t = torch.Tensor(all_states)
        actions_t = torch.Tensor(actions).int()
        rewards_t = torch.Tensor(rewards)
        perm = torch.randperm(len(states))[:train_set_size]
        train_dataset = ExperienceBuffer(
            states_t[perm],
            actions_t[perm],
            rewards_t[perm],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=False,
        )
        agent.learn(train_loader)
    else:
        actions.append(action)
        states.append(state)
        rewards.append(reward)

    total_reward += reward
    state = next_state

print(total_rewards)

avg_eval_reward_final = eval_agent(
    agent,
    num_eval_episodes,
    max_eval_episode_steps,
)

print(f"{avg_eval_reward_init = :.2f}")
print(f"{avg_eval_reward_final = :.2f}")

env.close()
