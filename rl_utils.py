import typing as T


def compute_reward(
    base_reward: float,
    info: T.Dict[str, T.Any],
    prev_info: T.Optional[T.Dict[str, T.Any]],
) -> float:
    if prev_info is None:
        return base_reward
    if base_reward == 0:
        base_reward = -2
    lives_reward = -100 if info["life"] < prev_info["life"] else 0
    score_reward = (info["score"] - prev_info["score"]) / 100
    if info["status"] != prev_info["status"]:
        status_reward = 100 if info["status"] != "small" else 0
    else:
        status_reward = 5 if info["status"] != "small" else 0
    total_reward = base_reward + lives_reward + score_reward + status_reward
    return total_reward
