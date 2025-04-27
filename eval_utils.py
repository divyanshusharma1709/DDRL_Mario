import typing as T
import numpy as np
import tqdm
import os

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import RecordEpisodeStatistics, RecordVideo
from agent.base_agent import BaseAgent
from rl_utils import compute_reward

from moviepy import VideoFileClip, concatenate_videoclips

from frame_processor import FrameProcessor

frame_processor = FrameProcessor(
    frame_height=240,
    frame_width=256,
    stack_size=4,
    grayscale=True,
    resize=False
)

def stitch_videos(video_dir, eval_episode):
    """
    Stitch together evaluation videos for one evaluation call.

    Parameters:
        video_dir (str): Directory containing eval videos.
        output_filename (str): The filename for the stitched output video.
    """
    # Gather video filenames
    video_files = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if (f.endswith(".mp4") and "eval-episode" in f)
    ]
    # Load video clips
    clips = [VideoFileClip(f) for f in video_files]
    # Concatenate videos
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(
        video_dir + "/" + str(eval_episode) + "_stitched_op.mp4", codec="libx264"
    )
    for f in video_files:
        os.remove(f)


def eval_agent(
    agent: BaseAgent,
    num_episodes: int,
    max_eval_steps_per_episode: int,
    frame_stack_size: int,
    render: bool,
    curr_train_step: int,
    video_dir: str = "eval_vids/",
    stuck_threshold: int = 1000,
    progress_threshold: int = 3,
) -> T.Dict[str, T.Any]:
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(
        env, video_folder=video_dir, episode_trigger=lambda episode_id: True, name_prefix="eval"
    )

    agent.eval()

    episode_lengths = []
    episode_rewards = []

    progress_bar = tqdm.tqdm(range(num_episodes), desc="Eval")
    prev_info = None
    for _ in progress_bar:
        episode_reward = 0.0
        episode_length = 0
        done = False
        # Stuck check
        last_x_pos = None
        stuck_counter = 0

        state = env.reset()
        state_stack = frame_processor.reset(state)

        while not done and episode_length < max_eval_steps_per_episode:
            action = agent.act(state_stack)

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



            next_state_stack = frame_processor.step(next_state)

            episode_reward += compute_reward(total_reward, final_info, prev_info)
            episode_length += 1

            # Bookkeeping
            state_stack = next_state_stack
            prev_info = final_info

            # Stuck check
            x_pos = final_info.get("x_pos", None)
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
                        print(
                            f"\nTerminating episode due to being stuck after {episode_length} steps."
                        )
                        break

            if render:
                env.render()

            if len(episode_rewards) > 0:
                progress_bar.set_postfix(
                    ep_len=episode_length,
                    avg_r=np.mean(episode_rewards),
                    avg_len=np.mean(episode_lengths),
                )
            else:
                progress_bar.set_postfix(
                    ep_len=episode_length,
                    avg_r=episode_reward,
                    avg_len=episode_length,
                )

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()
    stitch_videos(video_dir, curr_train_step)

    reward_array = np.array(episode_rewards)
    length_array = np.array(episode_lengths)

    return {
        "average_episode_reward": np.mean(reward_array),
        "average_episode_length": np.mean(length_array),
        "average_per_step_reward": np.mean(reward_array / length_array),
    }
