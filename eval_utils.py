import typing as T
import numpy as np
import tqdm
import os

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import RecordEpisodeStatistics, RecordVideo
from agent.base_agent import BaseAgent

from moviepy import VideoFileClip, concatenate_videoclips

def stitch_videos(video_dir, eval_episode):
    """
    Stitch together evaluation videos for one evaluation call.
    
    Parameters:
        video_dir (str): Directory containing eval videos.
        output_filename (str): The filename for the stitched output video.
    """
    # Gather video filenames
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if (f.endswith('.mp4') and "eval-episode" in f)]
    # Load video clips
    clips = [VideoFileClip(f) for f in video_files]
    # Concatenate videos
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(video_dir + "/" + str(eval_episode) + "_stitched_op.mp4", codec="libx264")
    for f in video_files:
        os.remove(f)


def eval_agent(
    agent: BaseAgent,
    num_episodes: int,
    max_eval_steps_per_episode: int,
    render: bool,
    curr_train_step: int,
    video_dir: str = "eval_vids/",
    stuck_threshold: int = 200, 
    progress_threshold: int = 5  
) -> T.Dict[str, T.Any]:
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda episode_id: True, name_prefix="eval")

    agent.eval()

    episode_lengths = []
    episode_rewards = []

    progress_bar = tqdm.tqdm(range(num_episodes), desc="Eval")

    for _ in progress_bar:
        episode_reward = 0.0
        episode_length = 0
        done = False
        # Stuck check
        last_x_pos = None
        stuck_counter = 0

        state = env.reset()
        while not done and episode_length < max_eval_steps_per_episode:

            action = agent.act(state)
            state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            
            # Stuck check
            x_pos = info.get("x_pos", None)
            # print("Current Position: ", x_pos)
            if x_pos is not None:
                if last_x_pos is None:
                    last_x_pos = x_pos
                else:
                    # Count stuck episode.
                    if abs(x_pos - last_x_pos) < progress_threshold:
                        stuck_counter += 1
                        # print(stuck_counter)
                    else:
                        stuck_counter = 0
                        last_x_pos = x_pos
                    # Break if above threshold
                    if stuck_counter >= stuck_threshold:
                        print(f"\nTerminating episode due to being stuck after {episode_length} steps.")
                        break

            if render:
                env.render()

            progress_bar.set_postfix(
                ep_len=episode_length,
                avg_r=np.mean(episode_rewards) if len(episode_rewards) > 0 else 0,
                avg_len=np.mean(episode_lengths) if len(episode_lengths) > 0 else 0,
            )

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()
    stitch_videos(video_dir, curr_train_step)

    return {
        "average_episode_reward": np.mean(episode_rewards),
        "average_episode_length": np.mean(episode_lengths),
    }
