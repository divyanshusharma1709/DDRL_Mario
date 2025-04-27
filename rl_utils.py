def compute_reward(base_reward, info, prev_info):
    if prev_info is None:
        return base_reward

    lives_reward = -100 if info["life"] < prev_info["life"] else 0
    score_reward = (info["score"] - prev_info["score"]) / 100.0
    delta_x = info["x_pos"] - prev_info["x_pos"]

    # Main x_pos reward
    x_pos_reward = delta_x * 0.5  # Stronger than before

    # Big leap bonus
    big_leap_bonus = 0.0
    if delta_x > 10:
        big_leap_bonus = 10.0
    elif delta_x > 5:
        big_leap_bonus = 5.0

    # Status reward
    if info["status"] != prev_info["status"]:
        status_reward = 100 if info["status"] != "small" else 0
    else:
        status_reward = 5 if info["status"] != "small" else 0

    # Living bonus
    living_bonus = 0.05

    total_reward = (
        base_reward + lives_reward + score_reward + x_pos_reward + big_leap_bonus + status_reward + living_bonus
    )
    return total_reward
