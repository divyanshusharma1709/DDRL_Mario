import torch
from torch.utils.data.dataset import Dataset


class ExperienceBuffer(Dataset):
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
        )
