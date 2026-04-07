
from __future__ import annotations

import numpy as np

from iov_power_channel.envs.mobile_network_env import OBS_IDX


class RandomPolicy:
    name = "Random"

    def __init__(self, env):
        self.env = env

    def act(self, obs):
        return self.env.action_space.sample()


class MaxPowerMaxChannelPolicy:
    name = "MaxPowerMaxChannel"

    def __init__(self, env):
        self.env = env

    def act(self, obs):
        n = obs.shape[0]
        action = np.ones((n, 2), dtype=np.float32)
        action[:, 0] = 1.0
        action[:, 1] = 1.0
        return action


class LoadAwareHeuristicPolicy:
    """
    Non-DRL baseline using the observation only.
    """
    name = "LoadAwareHeuristic"

    def __init__(self, env):
        self.env = env

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        n = obs.shape[0]
        action = np.zeros((n, 2), dtype=np.float32)

        cov = obs[:, OBS_IDX.COV_UTIL]
        reqp = obs[:, OBS_IDX.REQ_P_NORM]
        inter = obs[:, OBS_IDX.INTER_NORM]
        demand = obs[:, OBS_IDX.AVG_DEMAND_NORM]
        nearby = obs[:, OBS_IDX.NEARBY_POT]
        load = obs[:, OBS_IDX.LOAD_RATIO_NORM]
        bs_type = obs[:, OBS_IDX.BS_TYPE]

        power = 0.12 + 0.50 * cov + 0.28 * reqp + 0.12 * inter
        channel = 0.10 + 0.45 * nearby + 0.30 * demand + 0.20 * load

        # small tier-specific adjustment
        micro_mask = bs_type > 0.75
        power[micro_mask] += 0.08
        channel[micro_mask] -= 0.05

        action[:, 0] = np.clip(power, 0.05, 1.0)
        action[:, 1] = np.clip(channel, 0.05, 1.0)
        return action
