import typing as T
from torchvision import transforms as Tr
import gym
from gym.wrappers import RecordEpisodeStatistics, RecordVideo, FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.spaces import Box
import numpy as np
import torch

# credit: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = Tr.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = Tr.Compose([Tr.Resize(self.shape, antialias=True), Tr.Normalize(0, 255)])
        observation = transforms(observation).squeeze(0)
        return observation


def create_env(stack_size: int, video_dir: T.Optional[str] = None, env_version: str = "v0"):
    env = gym_super_mario_bros.make(f"SuperMarioBros-{env_version}")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=stack_size)
    if video_dir is not None:
        env = RecordEpisodeStatistics(env)
        env = RecordVideo(
            env, video_folder=video_dir, episode_trigger=lambda episode_id: True, name_prefix="eval"
        )
    return env
