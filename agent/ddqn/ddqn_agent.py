import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.base_agent import BaseAgent
from agent.q_agent import Q

class DDQNAgent(BaseAgent):
    def __init__(self, num_actions, q_params, lr=1e-4, gamma=0.95, ep=0.05, target_update_freq=100):
        """
        Initializes the DDQN agent.
        
        Parameters:
          num_actions (int): Number of available actions.
          q_params (dict): Parameters for the Q network (backbone input shape, conv channels, etc.).
          lr (float): Learning rate.
          gamma (float): Discount factor.
          ep (float): Epsilon for the epsilon-greedy policy.
          target_update_freq (int): Frequency (in steps) to update the target network.
        """
        super().__init__(ep)
        self.gamma = gamma
        self.num_actions = num_actions
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        # Create the online Q-network and the target Q-network.
        self.q = Q(**q_params)
        self.target_q = Q(**q_params)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optim = torch.optim.AdamW(self.q.parameters(), lr=lr)

    def a2t(self, action, batch_size):
        """
        Helper function that converts an action (int) into a tensor of shape (batch_size, 1).
        """
        return (
            torch.Tensor([action])[None, :]
            .int()
            .repeat(batch_size, 1)
            .to(self.device)
        )

    def learn_one_step(self, state, action, reward, next_state):
        """
        Performs one update step using the DDQN update rule.
        """
        self.train()  # Set the network to training mode

        # Predict Q-values for the current state-action pair using the online network.
        predicted_q = self.q(state, action)
        batch_size = state.shape[0]

        # Compute Q-values for all actions at the next state using the online network.
        q_values_next_online = []
        for a in range(self.num_actions):
            q_val = self.q(next_state, self.a2t(a, batch_size))
            q_values_next_online.append(q_val)
        q_values_next_online = torch.cat(q_values_next_online, dim=-1)
        # Select best actions for next state based on the online network.
        best_actions = torch.argmax(q_values_next_online, dim=-1, keepdim=True)

        # Compute Q-values for all actions at the next state using the target network.
        q_values_next_target = []
        for a in range(self.num_actions):
            q_val = self.target_q(next_state, self.a2t(a, batch_size))
            q_values_next_target.append(q_val)
        q_values_next_target = torch.cat(q_values_next_target, dim=-1)
        # Gather the target Q-value for the best action.
        target_q_selected = q_values_next_target.gather(1, best_actions).detach()

        # Compute the target using the DDQN rule.
        target = reward.view(-1, 1) + self.gamma * target_q_selected

        # Calculate loss and perform a gradient update.
        loss = F.mse_loss(predicted_q, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


        # Periodically update the target network.
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())
        return loss.item()

    def act(self, state, greedy=False, return_tensor=False):
        """
        Chooses an action using an epsilon-greedy policy.
        """
        # Convert state to tensor if not already.
        if isinstance(state, torch.Tensor):
            s = state
        else:
            s = torch.Tensor(state.copy())[None, :].to(self.device)
        batch_size = s.shape[0]
        # If greedy, epsilon is 0.
        ep = 0.0 if greedy else self.ep
        if random.random() < 1 - ep:
            q_values = []
            for a in range(self.num_actions):
                q_val = self.q(s, self.a2t(a, batch_size)).detach()
                q_values.append(q_val)
            q_values = torch.cat(q_values, dim=-1)
            result = torch.argmax(q_values, dim=-1)
        else:
            result = (
                torch.Tensor(
                    [random.randrange(self.num_actions) for _ in range(batch_size)]
                )
                .int()
                .to(self.device)
            )
        if batch_size == 1:
            return result if return_tensor else int(result[0])
        return result if return_tensor else result.cpu().numpy()

    def _get_model_path(self, checkpoint_dir, step):
        return f"{checkpoint_dir}/step={step}"

    def save(self, checkpoint_dir, step, metrics):
        """
        Saves the model parameters and training metrics.
        """
        path = self._get_model_path(checkpoint_dir, step)
        os.makedirs(path, exist_ok=True)
        torch.save(self.q.state_dict(), os.path.join(path, "model.pt"))
        with open(os.path.join(path, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f)

    def load(self, checkpoint_dir, step):
        """
        Loads the model parameters and returns the training metrics.
        """
        path = self._get_model_path(checkpoint_dir, step)
        state = torch.load(os.path.join(path, "model.pt"), map_location=self.device)
        self.q.load_state_dict(state)
        self.target_q.load_state_dict(state)
        with open(os.path.join(path, "metrics.json"), "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return metrics

    def train(self):
        self.q.train()

    def eval(self):
        self.q.eval()
