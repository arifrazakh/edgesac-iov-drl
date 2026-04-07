
from __future__ import annotations

import warnings
import random
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.patches import Circle

try:
    from IPython.display import display
    _HAVE_IPY = True
except Exception:
    _HAVE_IPY = False


# =========================
# PHY HELPERS
# =========================
def rx_sensitivity_dBm(bw_hz: float, nf_db: float = 7.0, snr_req_db: float = -5.0) -> float:
    return -174.0 + 10.0 * np.log10(max(bw_hz, 1.0)) + nf_db + snr_req_db


MOD_ORDER = 256
BITS_PER_CODEWORD = int(np.log2(MOD_ORDER))
NR_OVERHEAD = 0.85
SNR_GAP_DB = 1.5
MAX_SE = float(BITS_PER_CODEWORD)

MIMO_MAX_RANK = {"Ma": 4, "Mi": 8}

BEAM_GAIN_DB = {
    "Ma": {"tx_main": 8.0, "tx_side": -3.0, "rx": 0.0},
    "Mi": {"tx_main": 15.0, "tx_side": -5.0, "rx": 7.0},
}

UE_INTERF_RX_DB = 0.0
MACRO_REUSE_ONE = False

OBS_IDX = SimpleNamespace(
    BS_TYPE=0, TX_NORM=1, CH_UTIL=2, COV_UTIL=3, LOAD_RATIO_NORM=4, NEARBY_POT=5, AVG_SPEED=6,
    REQ_P_NORM=7, AVG_RADIAL_V=8, SP_VAR=9, NEIGHBOR_TX_NORM=10, AVG_DEMAND_NORM=11, INTER_NORM=12
)


def spectral_efficiency_from_sinr(SINR_dB: float, gap_db: float = SNR_GAP_DB, max_se: float = MAX_SE) -> float:
    gamma = 10.0 ** (SINR_dB / 10.0) / (10.0 ** (gap_db / 10.0))
    se = np.log2(1.0 + gamma)
    return float(np.clip(se, 0.0, max_se))


def mimo_rank_and_total_se(
    SINR_dB: float,
    max_layers: int,
    gap_db: float = SNR_GAP_DB,
    max_se: float = MAX_SE
) -> Tuple[int, float]:
    sinr_lin = 10.0 ** (SINR_dB / 10.0)
    best_L = 1
    best_sum_se = spectral_efficiency_from_sinr(SINR_dB, gap_db, max_se)
    for L in range(2, max_layers + 1):
        per_layer_snr_dB = 10.0 * np.log10(max(sinr_lin / max(L, 1), 1e-12))
        se_layer = spectral_efficiency_from_sinr(per_layer_snr_dB, gap_db, max_se)
        corr_eff = max(0.5, 1.0 - 0.07 * (L - 1))
        sum_se = L * se_layer * corr_eff
        if sum_se > best_sum_se:
            best_sum_se, best_L = sum_se, L
    return best_L, float(best_sum_se)


def default_bs_locations() -> List[Tuple[float, float]]:
    return [
        (1250, 1400), (3750, 1400), (1250, 3550), (3750, 3550),
        (400, 400), (400, 1200), (400, 2100), (1200, 400), (1200, 2200),
        (2100, 400), (2100, 1200), (2100, 2100), (2900, 400), (2900, 1200),
        (3700, 400), (3700, 2200), (4500, 400), (4500, 1200), (2900, 2100),
        (4500, 2100), (400, 2800), (400, 3600), (1200, 2800), (2100, 2800),
        (2100, 3600), (400, 4500), (1200, 4500), (2100, 4500), (2900, 2800),
        (3800, 2800), (4500, 2800), (2900, 3600), (2900, 4500), (4500, 3600),
        (3800, 4500), (4500, 4500),
    ]


class Channel:
    def __init__(self, id: int, frequency: float, bandwidth: float, noise_figure_db: float = 7.0):
        self.id = id
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.users: List["User"] = []
        self.base_station: Optional["BaseStation"] = None
        self.temperature = 293.15
        self.noise_figure_db = noise_figure_db

    def calculate_noise_power(self) -> float:
        k = 1.38e-23
        N = k * self.temperature * self.bandwidth
        NF = 10.0 ** (self.noise_figure_db / 10.0)
        return N * NF


class BaseStation:
    def __init__(self, id: int, transmit_power: float, height: float, location, type_bs: str):
        self.id = id
        self.transmit_power = transmit_power
        self.height = height
        self.type_bs = type_bs
        self.location = np.array(location, dtype=float)
        self.users: List["User"] = []
        self.assigned_channels: List[Channel] = []
        self.coverage_area = 0.0
        self.per_channel_power: Dict[int, float] = {}

        self.MAX_COVERAGE = {"Ma": 2000.0, "Mi": 350.0}
        self.COV_MARGIN_DB = {"Ma": 3.0, "Mi": 20.0}
        self.SNR_REQ_DB = {"Ma": -5.0, "Mi": 5.0}

    def _pl_uma_los(self, d3d: float, f_ghz: float, h_ut: float = 1.5) -> float:
        c = 3e8
        f_hz = f_ghz * 1e9
        h_bs = float(self.height)
        d2d = max(np.sqrt(max(d3d**2 - (h_bs - h_ut)**2, 1e-9)), 1.0)
        h_bs_eff = h_bs - 1.0
        h_ut_eff = h_ut - 1.0
        d_bp = 4.0 * h_bs_eff * h_ut_eff * f_hz / c
        pl1 = 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(f_ghz)
        if d2d <= d_bp:
            return pl1
        pl2 = 28.0 + 40.0 * np.log10(d3d) + 20.0 * np.log10(f_ghz) - 9.0 * np.log10(d_bp**2 + (h_bs - h_ut)**2)
        return pl2

    def _pl_uma_nlos(self, d3d: float, f_ghz: float, h_ut: float = 1.5) -> float:
        pl_los = self._pl_uma_los(d3d, f_ghz, h_ut=h_ut)
        pl_nlos = 13.54 + 39.08 * np.log10(d3d) + 20.0 * np.log10(f_ghz) - 0.6 * (h_ut - 1.5)
        return max(pl_nlos, pl_los)

    def _pl_umi_los(self, d3d: float, f_ghz: float) -> float:
        return 32.4 + 21.0 * np.log10(f_ghz) + 20.0 * np.log10(d3d)

    def _pl_umi_nlos(self, d3d: float, f_ghz: float, h_ut: float = 1.5) -> float:
        pl_los = self._pl_umi_los(d3d, f_ghz)
        pl_nlos = 22.4 + 35.3 * np.log10(d3d) + 21.3 * np.log10(f_ghz) - 0.3 * (h_ut - 1.5)
        return max(pl_nlos, pl_los)

    def _eirp_dbm(self, transmit_power_mW: float) -> float:
        g = BEAM_GAIN_DB[self.type_bs]
        tx_dbm = 10.0 * np.log10(max(transmit_power_mW, 1e-15))
        return tx_dbm + g["tx_main"] + g["rx"]

    def calculate_path_loss(self, distance: float, frequency: float, user_height: float = 1.5) -> float:
        d2d = max(float(distance), 0.1)
        f_ghz = frequency / 1e9
        d3d = np.sqrt(d2d**2 + (self.height - user_height) ** 2)

        if self.type_bs == "Ma":
            p_los = min(18.0 / d2d, 1.0) * (1.0 - np.exp(-d2d / 63.0)) + np.exp(-d2d / 63.0)
            pl_los = self._pl_uma_los(d3d, f_ghz, h_ut=user_height)
            pl_nlos = self._pl_uma_nlos(d3d, f_ghz, h_ut=user_height)
        else:
            p_los = min(18.0 / d2d, 1.0) * (1.0 - np.exp(-d2d / 36.0)) + np.exp(-d2d / 36.0)
            pl_los = self._pl_umi_los(d3d, f_ghz)
            pl_nlos = self._pl_umi_nlos(d3d, f_ghz, h_ut=user_height)

        return float(p_los * pl_los + (1.0 - p_los) * pl_nlos)

    def update_coverage_area_from_power(self, transmit_power_mW: float, frequency_hz: float) -> float:
        if self.per_channel_power:
            p_list = list(self.per_channel_power.values())
            p_ch = float(np.percentile(p_list, 80))
        else:
            p_ch = transmit_power_mW / max(len(self.assigned_channels), 1)

        if self.assigned_channels:
            bw_hz = self.assigned_channels[0].bandwidth
            nf_db = self.assigned_channels[0].noise_figure_db
        else:
            bw_hz = 100e6 if self.type_bs == "Ma" else 200e6
            nf_db = 7.0

        sens_dBm = rx_sensitivity_dBm(
            bw_hz, nf_db=nf_db, snr_req_db=self.SNR_REQ_DB[self.type_bs]
        )
        eirp_plus_grx_dbm = self._eirp_dbm(p_ch)
        path_loss_budget_dB = eirp_plus_grx_dbm - sens_dBm - self.COV_MARGIN_DB[self.type_bs]

        self.coverage_area = self.find_distance_for_path_loss(path_loss_budget_dB, frequency_hz)
        cap = self.MAX_COVERAGE.get(self.type_bs)
        if cap is not None:
            self.coverage_area = min(self.coverage_area, float(cap))
        return self.coverage_area

    def find_distance_for_path_loss(self, path_loss_dB: float, frequency: float) -> float:
        d_min, d_max = 1.0, 10000.0
        tol = 0.1
        for _ in range(30):
            d_mid = 0.5 * (d_min + d_max)
            PL_mid = self.calculate_path_loss(d_mid, frequency)
            if abs(PL_mid - path_loss_dB) < tol:
                return d_mid
            if PL_mid < path_loss_dB:
                d_min = d_mid
            else:
                d_max = d_mid
        return d_mid

    def assign_channels(self, num_channels: int, available_channels: List[Channel]) -> None:
        num_channels = min(num_channels, len(available_channels))
        self.assigned_channels = available_channels[:num_channels]
        for ch in self.assigned_channels:
            ch.base_station = self

    def clear_assigned_channels(self) -> None:
        self.assigned_channels = []
        self.per_channel_power = {}

    def find_available_channel(self) -> Optional[Channel]:
        for ch in self.assigned_channels:
            if len(ch.users) == 0:
                return ch
        return None


class User:
    def __init__(self, id: int, location, velocity, demand: float):
        self.id = id
        self.location = np.array(location, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.height = 1.5
        self.channel: List[Channel] = []
        self.channel_SINR: List[float] = []
        self.SINR = 0.0
        self.SINR_ma = 0.0
        self.SINR_mi = 0.0
        self.SINR_ma_list: List[float] = []
        self.SINR_mi_list: List[float] = []
        self.data_rate = 0.0
        self.data_rate_ma = 0.0
        self.data_rate_mi = 0.0
        self.demand = float(demand)
        self.speed = float(np.linalg.norm(velocity))
        self.waypoint = None
        self.pause_time = 0
        self.dir_axis = None
        self.dir_sign = 1
        self.next_intersection = None
        self.mimo_layers: List[int] = []

    def clear_channel(self) -> None:
        self.channel = []
        self.channel_SINR = []
        self.SINR_ma_list = []
        self.SINR_mi_list = []
        self.mimo_layers = []

    def calculate_demand_from_rng(self, rng) -> None:
        app_types = {"video": 50, "gaming": 100, "browsing": 20}
        app = rng.choice(list(app_types.keys()))
        self.demand = float(app_types[app] * (1 + rng.uniform(0, 0.5)))

    def calculate_data_rate(self) -> None:
        self.data_rate_ma = 0.0
        self.data_rate_mi = 0.0
        self.mimo_layers = []
        for i, ch in enumerate(self.channel):
            bw_eff = ch.bandwidth * NR_OVERHEAD
            SINR_dB = self.channel_SINR[i]
            max_rank = MIMO_MAX_RANK[ch.base_station.type_bs]
            L, total_se = mimo_rank_and_total_se(SINR_dB, max_rank, gap_db=SNR_GAP_DB, max_se=MAX_SE)
            dr_Mbps = (bw_eff * total_se) / 1e6
            self.mimo_layers.append(L)
            if ch.base_station.type_bs == "Ma":
                self.data_rate_ma += dr_Mbps
            else:
                self.data_rate_mi += dr_Mbps
        self.data_rate = self.data_rate_ma + self.data_rate_mi

    def calculate_latency(self) -> float:
        if not self.channel:
            return 100.0
        avg_d = np.mean([np.linalg.norm(self.location - ch.base_station.location) for ch in self.channel])
        prop_delay = (avg_d / 3e8) * 1e3
        proc_delay = 0.5
        sched_delay = 0.5
        num_users_on_channel = sum(len(ch.users) for ch in self.channel) / len(self.channel)
        queue_delay = 0.5 + num_users_on_channel / (self.data_rate + 1e-6)
        return float(np.clip(prop_delay + proc_delay + sched_delay + queue_delay, 0.0, 100.0))


class MobileNetwork(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        num_base_stations: int = 36,
        num_users: int = 100,
        num_channels: int = 265,
        area_size: float = 5000.0,
        bs_loc: Optional[List[Tuple[float, float]]] = None,
        num_steps: int = 0,
        max_steps: int = 200,
        render_mode: str = "human",
        mobility_model: str = "manhattan",
        road_spacing: float = 250.0,
        v_mean: float = 12.0,
        v_std: float = 3.0,
        stop_prob: float = 0.05,
        max_stop_steps: int = 3,
        seed: int = 42,
    ):
        super().__init__()
        self.render_mode = render_mode if render_mode in self.metadata["render_modes"] else None
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.num_steps = num_steps
        self.max_steps = max_steps
        self.num_base_stations = num_base_stations
        self.num_users = num_users
        self.num_channels = num_channels
        self.area_size = [area_size, area_size]

        self.mobility_model = mobility_model
        self.road_spacing = road_spacing
        self.v_mean = v_mean
        self.v_std = v_std
        self.stop_prob = stop_prob
        self.max_stop_steps = max_stop_steps
        self._build_road_grid()

        self.ma_transmission_power = 40000.0
        self.mi_transmission_power = 10000.0
        self.max_ma_channels = 100
        self.max_mi_channels = 10

        if bs_loc is None:
            bs_loc = default_bs_locations()
        if len(bs_loc) < num_base_stations:
            raise ValueError("Not enough BS locations were provided.")

        self.bs_loc = list(bs_loc[:num_base_stations])
        self.type_bs = ["Ma"] * min(4, num_base_stations) + ["Mi"] * max(0, num_base_stations - 4)
        self.transmit_power = [self.ma_transmission_power] * min(4, num_base_stations) + [self.mi_transmission_power] * max(0, num_base_stations - 4)
        self.height = [30] * min(4, num_base_stations) + [10] * max(0, num_base_stations - 4)

        self.base_stations = [
            BaseStation(i, self.transmit_power[i], self.height[i], self.bs_loc[i], self.type_bs[i])
            for i in range(num_base_stations)
        ]

        self.users: List[User] = []
        for i in range(num_users):
            if self.mobility_model == "manhattan":
                loc, vel = self._spawn_on_grid()
            else:
                loc = self.np_random.uniform(0, area_size, 2)
                vel = self.np_random.integers(-10, 10, 2)
            demand = self.np_random.integers(100, 280)
            self.users.append(User(i, loc, vel, demand))
        for u in self.users:
            if self.mobility_model == "manhattan":
                self._init_user_manhattan(u)
            else:
                u.speed = float(np.linalg.norm(u.velocity))
                u.waypoint = self.np_random.uniform(0, area_size, 2)

        self.macro_carrier_frequencies = [3.5e9] if MACRO_REUSE_ONE else [3.4e9, 3.5e9, 3.6e9, 3.7e9, 3.8e9]
        self.micro_carrier_frequencies = [24.5e9, 25.5e9, 26.5e9, 27.5e9, 28.5e9]

        num_channels_per_carrier_ma = 135
        num_channels_per_carrier_mi = 130
        macro_bw_hz = 3.6e6
        micro_bw_hz = 14.4e6

        self.macro_channels: List[Channel] = []
        self.micro_channels: List[Channel] = []

        ch_id = 0
        for f in self.macro_carrier_frequencies:
            for _ in range(num_channels_per_carrier_ma):
                self.macro_channels.append(Channel(ch_id, f, macro_bw_hz, noise_figure_db=7.0))
                ch_id += 1
        for f in self.micro_carrier_frequencies:
            for _ in range(num_channels_per_carrier_mi):
                self.micro_channels.append(Channel(ch_id, f, micro_bw_hz, noise_figure_db=9.0))
                ch_id += 1

        self.cap_per_freq = defaultdict(int)
        for ch in (self.macro_channels + self.micro_channels):
            self.cap_per_freq[ch.frequency] += 1

        low = np.array([[0.001, 0.001]] * num_base_stations, dtype=np.float32)
        high = np.array([[1.0, 1.0]] * num_base_stations, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(num_base_stations, 13), dtype=np.float32)

        self.seed(seed)
        self._assign_frequency_subsets()
        self.prev_assoc_by_user = {u.id: {"Ma": None, "Mi": None} for u in self.users}
        self.current_episode_reward = 0.0
        self.last_info = {}

        self.fig = self.ax = self.disp = None
        if self.render_mode == "human":
            try:
                self.fig, self.ax = plt.subplots(figsize=(7, 7))
                self.disp = display(self.fig, display_id="sim_fig") if _HAVE_IPY else None
            except Exception as e:
                warnings.warn(f"Render init failed: {e}")
                self.fig = self.ax = self.disp = None
                self.render_mode = None

    # ---------- mobility ----------
    def _build_road_grid(self):
        L = self.area_size[0]
        spacing = max(50.0, float(self.road_spacing))
        self.grid_xs = np.arange(0.0, L + 1e-6, spacing)
        self.grid_ys = np.arange(0.0, L + 1e-6, spacing)

    def _snap_to_grid(self, p):
        x = self.grid_xs[np.argmin(np.abs(self.grid_xs - p[0]))]
        y = self.grid_ys[np.argmin(np.abs(self.grid_ys - p[1]))]
        return np.array([x, y], dtype=float)

    def _spawn_on_grid(self):
        L = self.area_size[0]
        if self.np_random.random() < 0.5:
            y = float(self.np_random.choice(self.grid_ys))
            x = float(self.np_random.uniform(0, L))
            dir_axis, dir_sign = "x", 1 if self.np_random.random() < 0.5 else -1
        else:
            x = float(self.np_random.choice(self.grid_xs))
            y = float(self.np_random.uniform(0, L))
            dir_axis, dir_sign = "y", 1 if self.np_random.random() < 0.5 else -1
        loc = np.array([x, y], dtype=float)
        speed = float(max(1.0, self.np_random.normal(self.v_mean, self.v_std)))
        if dir_axis == "x":
            vel = np.array([dir_sign * speed, 0.0], dtype=float)
        else:
            vel = np.array([0.0, dir_sign * speed], dtype=float)
        return loc, vel

    def _init_user_manhattan(self, u: User):
        L = self.area_size[0]
        u.location = self._snap_to_grid(u.location)
        if abs(u.velocity[0]) >= abs(u.velocity[1]):
            u.dir_axis = "x"
            u.dir_sign = 1 if u.velocity[0] >= 0 else -1
        else:
            u.dir_axis = "y"
            u.dir_sign = 1 if u.velocity[1] >= 0 else -1
        u.speed = float(max(1.0, np.linalg.norm(u.velocity)))
        eps = 1e-9
        if u.dir_axis == "x":
            if u.dir_sign > 0:
                candidates = self.grid_xs[self.grid_xs > (u.location[0] + eps)]
                x_next = candidates.min() if candidates.size > 0 else L
            else:
                candidates = self.grid_xs[self.grid_xs < (u.location[0] - eps)]
                x_next = candidates.max() if candidates.size > 0 else 0.0
            u.next_intersection = np.array([x_next, u.location[1]], dtype=float)
        else:
            if u.dir_sign > 0:
                candidates = self.grid_ys[self.grid_ys > (u.location[1] + eps)]
                y_next = candidates.min() if candidates.size > 0 else L
            else:
                candidates = self.grid_ys[self.grid_ys < (u.location[1] - eps)]
                y_next = candidates.max() if candidates.size > 0 else 0.0
            u.next_intersection = np.array([u.location[0], y_next], dtype=float)
        u.pause_time = 0

    def _advance_user_manhattan(self, u: User):
        L = self.area_size[0]
        if u.pause_time > 0:
            u.pause_time -= 1
            return
        delta = u.next_intersection - u.location
        dist = float(np.linalg.norm(delta))
        step = min(dist, u.speed)
        if dist > 1e-9:
            u.location += (delta / dist) * step
        if np.linalg.norm(u.location - u.next_intersection) <= 1e-6 or step == dist:
            if self.np_random.random() < self.stop_prob:
                u.pause_time = int(self.np_random.integers(1, self.max_stop_steps + 1))
            r = self.np_random.random()
            if r < 0.2:
                if u.dir_axis == "x":
                    u.dir_axis = "y"; u.dir_sign = 1 if u.dir_sign > 0 else -1
                else:
                    u.dir_axis = "x"; u.dir_sign = -1 if u.dir_sign > 0 else 1
            elif r < 0.4:
                if u.dir_axis == "x":
                    u.dir_axis = "y"; u.dir_sign = -1 if u.dir_sign > 0 else 1
                else:
                    u.dir_axis = "x"; u.dir_sign = 1 if u.dir_sign > 0 else -1
            u.speed = float(max(1.0, self.np_random.normal(self.v_mean, self.v_std)))
            if u.dir_axis == "x":
                candidates = self.grid_xs[self.grid_xs > u.location[0]] if u.dir_sign > 0 else self.grid_xs[self.grid_xs < u.location[0]]
                x_next = (candidates.min() if u.dir_sign > 0 else candidates.max()) if candidates.size > 0 else (L if u.dir_sign > 0 else 0.0)
                u.next_intersection = np.array([x_next, self._snap_to_grid(u.location)[1]], dtype=float)
                u.velocity = np.array([u.dir_sign * u.speed, 0.0], dtype=float)
            else:
                candidates = self.grid_ys[self.grid_ys > u.location[1]] if u.dir_sign > 0 else self.grid_ys[self.grid_ys < u.location[1]]
                y_next = (candidates.min() if u.dir_sign > 0 else candidates.max()) if candidates.size > 0 else (L if u.dir_sign > 0 else 0.0)
                u.next_intersection = np.array([self._snap_to_grid(u.location)[0], y_next], dtype=float)
                u.velocity = np.array([0.0, u.dir_sign * u.speed], dtype=float)

    def update_user_location(self):
        if self.mobility_model == "manhattan":
            for u in self.users:
                self._advance_user_manhattan(u)
                u.location = np.clip(u.location, 0.0, self.area_size[0])

    def seed(self, seed=None):
        if seed is None:
            seed = random.randrange(0, 2**32 - 1)
        seed32 = int(seed) & 0xFFFFFFFF
        self.np_random, _ = gym.utils.seeding.np_random(seed32)
        random.seed(seed32)
        np.random.seed(seed32)
        return [seed32]

    def _assign_frequency_subsets(self):
        self.bs_carrier_frequency = {}
        ma_idx, mi_idx = 0, 0
        for bs in self.base_stations:
            if bs.type_bs == "Ma":
                f = self.macro_carrier_frequencies[0] if MACRO_REUSE_ONE else self.macro_carrier_frequencies[ma_idx % len(self.macro_carrier_frequencies)]
                ma_idx += 1
            else:
                f = self.micro_carrier_frequencies[mi_idx % len(self.micro_carrier_frequencies)]
                mi_idx += 1
            self.bs_carrier_frequency[bs.id] = f

    # ---------- helper methods ----------
    def _noise_mW_for(self, bs: BaseStation) -> float:
        if bs.assigned_channels:
            return bs.assigned_channels[0].calculate_noise_power() * 1e3
        bw_hz = 100e6 if bs.type_bs == "Ma" else 200e6
        k = 1.38e-23
        T = 293.15
        NF = 10 ** (7 / 10)
        return k * T * bw_hz * NF * 1e3

    def _est_rate_one_channel_Mbps(self, u: User, bs: BaseStation) -> float:
        if not bs.assigned_channels:
            return 0.0
        f = self.bs_carrier_frequency[bs.id]
        d = float(np.linalg.norm(u.location - bs.location))
        PL_dB = bs.calculate_path_loss(d, f)
        n = max(len(bs.assigned_channels), 1)
        p_ch = bs.transmit_power / n
        g = BEAM_GAIN_DB[bs.type_bs]
        g_tx = 10 ** (g["tx_main"] / 10)
        g_rx = 10 ** (g["rx"] / 10)
        sig_mW = p_ch * g_tx * g_rx / (10.0 ** (PL_dB / 10.0))
        noise_mW = self._noise_mW_for(bs)
        SINR = sig_mW / max(noise_mW, 1e-12)
        SINR_dB = 10 * np.log10(max(SINR, 1e-12))
        bw = bs.assigned_channels[0].bandwidth * NR_OVERHEAD
        _, total_se = mimo_rank_and_total_se(SINR_dB, MIMO_MAX_RANK[bs.type_bs], gap_db=SNR_GAP_DB, max_se=MAX_SE)
        return (bw * total_se) / 1e6

    def _best_bs_in_cov(self, u: User, type_filter: Optional[str] = None):
        best, best_score = None, 0.0
        for bs in self.base_stations:
            if type_filter is not None and bs.type_bs != type_filter:
                continue
            if not any(len(c.users) == 0 for c in bs.assigned_channels):
                continue
            d = np.linalg.norm(u.location - bs.location)
            if d > bs.coverage_area:
                continue
            r = self._est_rate_one_channel_Mbps(u, bs)
            load = sum(1 for c in bs.assigned_channels if len(c.users) > 0) / max(1, len(bs.assigned_channels))
            score = r * (1.0 - 0.5 * load)
            if score > best_score:
                best_score = score
                best = bs
        return best, best_score

    def _estimate_interference_mW(self, u: User, f: float, exclude_bs: BaseStation) -> float:
        co = [b for b in self.base_stations if b is not exclude_bs and self.bs_carrier_frequency[b.id] == f]
        I = 0.0
        g_rx_int = 10.0 ** (UE_INTERF_RX_DB / 10.0)
        for bi in co:
            p_i = self._per_channel_tx_power_mW(bi)
            d_i = np.linalg.norm(u.location - bi.location)
            PL_i = bi.calculate_path_loss(d_i, f)
            gi = BEAM_GAIN_DB[bi.type_bs]
            g_tx_i = 10.0 ** (gi["tx_side"] / 10.0)
            users_in_cov_bi = [uu for uu in self.users if np.linalg.norm(uu.location - bi.location) <= bi.coverage_area]
            util = min(1.0, len(users_in_cov_bi) / max(1, len(bi.assigned_channels)))
            I += util * p_i * g_tx_i * g_rx_int / (10.0 ** (PL_i / 10.0))
        return I

    @staticmethod
    def _waterfill(P_total, h_list, n_list, p_floor_list=None, tol=1e-6, max_it=60):
        K = len(h_list)
        if K == 0:
            return []
        h = np.asarray(h_list, dtype=float)
        n = np.asarray(n_list, dtype=float)
        p_floor = np.zeros(K, dtype=float) if p_floor_list is None else np.asarray(p_floor_list, dtype=float)
        lo, hi = 0.0, 1e12

        def alloc(lmbd):
            base = 1.0 / max(lmbd, 1e-18)
            out = np.maximum(p_floor, base - n / np.maximum(h, 1e-18))
            return np.maximum(out, 0.0)

        p_floor_sum = float(np.sum(p_floor))
        if p_floor_sum > P_total:
            if p_floor_sum < 1e-12:
                return [0.0] * K
            scaled = P_total * (p_floor / p_floor_sum)
            return list(np.maximum(scaled, 0.0))

        for _ in range(max_it):
            mid = 0.5 * (lo + hi)
            p = alloc(mid)
            s = float(np.sum(p))
            if abs(s - P_total) <= tol:
                return list(np.maximum(p, 0.0))
            if s > P_total:
                lo = mid
            else:
                hi = mid
        return list(np.maximum(alloc(hi), 0.0))

    def calculate_required_power_for_distance(self, distance: float, base_station: BaseStation) -> float:
        f = self.bs_carrier_frequency[base_station.id]
        pl_dB = base_station.calculate_path_loss(distance, f)
        if base_station.assigned_channels:
            bw_hz = base_station.assigned_channels[0].bandwidth
            nf_db = base_station.assigned_channels[0].noise_figure_db
        else:
            bw_hz = 100e6 if base_station.type_bs == "Ma" else 200e6
            nf_db = 7.0
        sens_dBm = rx_sensitivity_dBm(bw_hz, nf_db=nf_db, snr_req_db=-5.0)
        g = BEAM_GAIN_DB[base_station.type_bs]
        req_tx_dBm = sens_dBm + pl_dB - (g["tx_main"] + g["rx"])
        return 10.0 ** (req_tx_dBm / 10.0)

    def assign_channels_on_demand(self, max_macro_per_user: int = 1, max_micro_per_user: int = 1) -> None:
        for u in self.users:
            u.clear_channel()

        for u in self.users:
            bs_ma, _ = self._best_bs_in_cov(u, type_filter="Ma")
            if bs_ma:
                ch = bs_ma.find_available_channel()
                if ch:
                    u.channel.append(ch)
                    ch.users.append(u)

        for u in self.users:
            if any(c.base_station.type_bs == "Mi" for c in u.channel):
                continue
            bs_mi, _ = self._best_bs_in_cov(u, type_filter="Mi")
            if bs_mi:
                ch = bs_mi.find_available_channel()
                if ch:
                    u.channel.append(ch)
                    ch.users.append(u)

    def calculate_SINR(self, user: User) -> None:
        user.channel_SINR = []
        user.SINR_ma_list = []
        user.SINR_mi_list = []

        for ch in user.channel:
            bs = ch.base_station
            d = np.linalg.norm(user.location - bs.location)
            PL_dB = bs.calculate_path_loss(d, ch.frequency)

            p_tx_ch = bs.per_channel_power.get(ch.id, None)
            if p_tx_ch is None:
                n = max(len(bs.assigned_channels), 1)
                p_tx_ch = bs.transmit_power / n

            g = BEAM_GAIN_DB[bs.type_bs]
            g_tx_main = 10.0 ** (g["tx_main"] / 10.0)
            g_rx_sig = 10.0 ** (g["rx"] / 10.0)
            signal_mW = p_tx_ch * g_tx_main * g_rx_sig / (10.0 ** (PL_dB / 10.0))

            if bs.type_bs == "Ma" and not MACRO_REUSE_ONE:
                interference_mW = 0.0
            else:
                co = {b for (b, _c) in self.frequency_to_channels[ch.frequency] if b != bs}
                interference_mW = 0.0
                g_rx_int = 10.0 ** (UE_INTERF_RX_DB / 10.0)
                for bi in co:
                    p_i = self._per_channel_tx_power_mW(bi)
                    d_i = np.linalg.norm(user.location - bi.location)
                    PL_i = bi.calculate_path_loss(d_i, ch.frequency)
                    gi = BEAM_GAIN_DB[bi.type_bs]
                    g_tx_i = 10.0 ** (gi["tx_side"] / 10.0)
                    users_in_cov_bi = [uu for uu in self.users if np.linalg.norm(uu.location - bi.location) <= bi.coverage_area]
                    util = min(1.0, len(users_in_cov_bi) / max(1, len(bi.assigned_channels)))
                    interference_mW += util * p_i * g_tx_i * g_rx_int / (10.0 ** (PL_i / 10.0))

            noise_W = ch.calculate_noise_power()
            noise_mW = noise_W * 1e3
            denom = max(interference_mW + noise_mW, 1e-15)
            SINR_lin = signal_mW / denom if signal_mW > 0 else 0.0
            SINR_dB = float(np.clip(10.0 * np.log10(SINR_lin) if SINR_lin > 0 else -100.0, -100.0, 30.0))
            user.channel_SINR.append(SINR_dB)
            if bs.type_bs == "Ma":
                user.SINR_ma_list.append(SINR_dB)
            else:
                user.SINR_mi_list.append(SINR_dB)

        user.SINR_ma = 10.0 * np.log10(np.mean([10.0 ** (x / 10.0) for x in user.SINR_ma_list])) if user.SINR_ma_list else -100.0
        user.SINR_mi = 10.0 * np.log10(np.mean([10.0 ** (x / 10.0) for x in user.SINR_mi_list])) if user.SINR_mi_list else -100.0

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.current_episode_reward = 0.0
        self.num_steps = 0

        L = self.area_size[0]
        for u in self.users:
            if self.mobility_model == "manhattan":
                loc, vel = self._spawn_on_grid()
                u.location = loc
                u.velocity = vel
                self._init_user_manhattan(u)
            else:
                u.location = self.np_random.uniform(0, L, 2)
                u.waypoint = self.np_random.uniform(0, L, 2)
            u.data_rate = 0.0
            u.SINR = 0.0
            u.clear_channel()

        self.assigned_channels = {}
        for ch in self.macro_channels + self.micro_channels:
            ch.users = []
            ch.base_station = None

        for bs in self.base_stations:
            bs.users = []
            bs.clear_assigned_channels()
            bs.transmit_power = self.ma_transmission_power if bs.type_bs == "Ma" else self.mi_transmission_power
            f = self.bs_carrier_frequency[bs.id]
            if bs.type_bs == "Ma":
                avail = [c for c in self.macro_channels if c.frequency == f and c not in self.assigned_channels]
            else:
                avail = [c for c in self.micro_channels if c.frequency == f and c not in self.assigned_channels]
            bs.assign_channels(1, avail[:1])
            for ch in bs.assigned_channels:
                self.assigned_channels[ch] = None
            bs.update_coverage_area_from_power(bs.transmit_power, f)

        self.update_user_location()
        self.prev_assoc_by_user = {u.id: {"Ma": None, "Mi": None} for u in self.users}
        self.last_info = {}
        return self.get_observation(), {}

    def step(self, action):
        self.num_steps += 1

        for u in self.users:
            if self.np_random.random() < 0.1:
                u.calculate_demand_from_rng(self.np_random)

        power_consumption_penalty = 0.0
        total_data_rate = 0.0
        total_mi_data_rate = 0.0
        total_ma_data_rate = 0.0
        user_latencies = []

        action = np.clip(action, self.action_space.low, self.action_space.high)
        power_fraction = action[:, 0]
        channel_fraction = action[:, 1]

        for u in self.users:
            u.clear_channel()
        for bs in self.base_stations:
            bs.clear_assigned_channels()
        for ch in self.macro_channels + self.micro_channels:
            ch.users.clear()
            ch.base_station = None

        self.assigned_channels = {}
        self.frequency_to_channels = defaultdict(list)

        user_locations = np.array([u.location for u in self.users])
        total_ma_channels = 0
        total_mi_channels = 0
        requested_ma_channels_per_freq = defaultdict(int)
        requested_mi_channels_per_freq = defaultdict(int)

        for i, bs in enumerate(self.base_stations):
            f = self.bs_carrier_frequency[bs.id]
            bs.transmit_power = float(power_fraction[i]) * (self.ma_transmission_power if bs.type_bs == "Ma" else self.mi_transmission_power)
            bs.update_coverage_area_from_power(bs.transmit_power, f)

            if bs.type_bs == "Ma":
                avail_pool = [c for c in self.macro_channels if c.frequency == f and c not in self.assigned_channels]
                max_cap = int(self.max_ma_channels)
            else:
                avail_pool = [c for c in self.micro_channels if c.frequency == f and c not in self.assigned_channels]
                max_cap = int(self.max_mi_channels)

            req_ch = int(np.rint(float(channel_fraction[i]) * max_cap))
            req_ch = int(np.clip(req_ch, 0, len(avail_pool)))
            if req_ch == 0 and len(avail_pool) > 0 and bs.transmit_power > 1e-6:
                req_ch = 1

            if req_ch > 0:
                bs.assign_channels(req_ch, avail_pool)
                for ch in bs.assigned_channels:
                    self.assigned_channels[ch] = None
                    self.frequency_to_channels[ch.frequency].append((bs, ch))
                    if bs.type_bs == "Ma":
                        total_ma_channels += 1
                    else:
                        total_mi_channels += 1

            if bs.type_bs == "Ma":
                requested_ma_channels_per_freq[f] += req_ch
            else:
                requested_mi_channels_per_freq[f] += req_ch

            bs.update_coverage_area_from_power(bs.transmit_power, f)

            dists = np.linalg.norm(user_locations - bs.location, axis=1)
            idx = np.where(dists <= bs.coverage_area)[0]
            if idx.size > 0:
                p90 = float(np.percentile(dists[idx], 90))
                req_tx_mW_per_ch = self.calculate_required_power_for_distance(p90, bs)
                req_tx_mW_total = req_tx_mW_per_ch * max(1, len(bs.assigned_channels))
                denom = self.ma_transmission_power if bs.type_bs == "Ma" else self.mi_transmission_power
                under = max(0.0, (req_tx_mW_total - bs.transmit_power) / max(denom, 1e-9))
                over = max(0.0, (bs.transmit_power - req_tx_mW_total) / max(denom, 1e-9))
                power_consumption_penalty += 1.2 * under + 0.3 * np.tanh(over)
                min_total = 0.85 * req_tx_mW_total
                if bs.transmit_power < min_total:
                    bs.transmit_power = min_total
                    bs.update_coverage_area_from_power(bs.transmit_power, f)

        self.assign_channels_on_demand(max_macro_per_user=1, max_micro_per_user=1)

        for bs in self.base_stations:
            bs.per_channel_power = {}
            ch_act, H, N, floors = [], [], [], []
            for ch in bs.assigned_channels:
                if not ch.users:
                    continue
                u = ch.users[0]
                f = self.bs_carrier_frequency[bs.id]
                d = float(np.linalg.norm(u.location - bs.location))
                PL_dB = bs.calculate_path_loss(d, f)
                g = BEAM_GAIN_DB[bs.type_bs]
                h = (10 ** (g["tx_main"] / 10) * 10 ** (g["rx"] / 10)) / (10 ** (PL_dB / 10.0))
                noise_mW = ch.calculate_noise_power() * 1e3
                I_mW = self._estimate_interference_mW(u, f, bs)
                prio = float(np.clip(0.5 + (u.demand / 150.0), 0.5, 1.5))
                N_eff = (noise_mW + I_mW) / prio
                ch_act.append(ch)
                H.append(h)
                N.append(N_eff)
                floors.append(0.0)

            if not ch_act:
                continue

            p_list = self._waterfill(float(bs.transmit_power), H, N, p_floor_list=floors)
            for ch, p in zip(ch_act, p_list):
                bs.per_channel_power[ch.id] = float(max(p, 0.0))

        for u in self.users:
            self.calculate_SINR(u)
            u.calculate_data_rate()
            user_latencies.append(u.calculate_latency())
            total_data_rate += u.data_rate
            total_mi_data_rate += u.data_rate_mi
            total_ma_data_rate += u.data_rate_ma

        ma_power = sum(bs.transmit_power for bs in self.base_stations if bs.type_bs == "Ma")
        mi_power = sum(bs.transmit_power for bs in self.base_stations if bs.type_bs == "Mi")
        total_power = ma_power + mi_power

        served = sum(min(u.data_rate, u.demand) for u in self.users)
        total_demand = sum(u.demand for u in self.users) + 1e-9
        util = served / total_demand
        n_ma = sum(1 for bs in self.base_stations if bs.type_bs == "Ma")
        n_mi = self.num_base_stations - n_ma
        budget_mW = n_ma * self.ma_transmission_power + n_mi * self.mi_transmission_power
        power_norm = total_power / max(budget_mW, 1e-9)
        reward = float(np.clip(10.0 * util - 2.0 * power_norm, -10.0, 10.0))

        rates = [u.data_rate for u in self.users]
        best_sinrs = [max(u.channel_SINR) if u.channel_SINR else -100.0 for u in self.users]
        avg_rate = float(np.mean(rates)) if rates else 0.0
        avg_latency = float(np.mean(user_latencies)) if user_latencies else 100.0
        lat_p95 = float(np.percentile(user_latencies, 95)) if user_latencies else 100.0

        over_num, cap_sum = 0, 0
        for f in self.macro_carrier_frequencies:
            over_num += max(0, requested_ma_channels_per_freq[f] - self.cap_per_freq[f])
            cap_sum += self.cap_per_freq[f]
        for f in self.micro_carrier_frequencies:
            over_num += max(0, requested_mi_channels_per_freq[f] - self.cap_per_freq[f])
            cap_sum += self.cap_per_freq[f]
        over_frac = float(over_num) / max(cap_sum, 1)

        info = {
            "reward": reward,
            "util": util,
            "power_norm": power_norm,
            "total_data_rate": total_data_rate,
            "ma_data_rate": total_ma_data_rate,
            "mi_data_rate": total_mi_data_rate,
            "total_power": total_power,
            "avg_rate": avg_rate,
            "avg_latency": avg_latency,
            "lat_p95": lat_p95,
            "sinr_p50": float(np.percentile(best_sinrs, 50)) if best_sinrs else -100.0,
            "jain_fairness": self._jain([min(u.data_rate / max(u.demand, 1e-9), 1.0) for u in self.users]),
            "energy_eff_Mb_per_J": total_data_rate / max(total_power * 1e-3, 1e-12),
            "total_channels": total_ma_channels + total_mi_channels,
            "total_ma_channels": total_ma_channels,
            "total_mi_channels": total_mi_channels,
            "assoc_unique": sum(1 for u in self.users if u.channel) / max(self.num_users, 1),
            "over_request_frac": over_frac,
            "power_pen_norm": float(np.tanh(power_consumption_penalty / max(self.num_base_stations, 1))),
        }

        self.current_episode_reward += reward
        self.last_info = info
        obs = self.get_observation()
        self.update_user_location()
        terminated = False
        truncated = self.num_steps >= self.max_steps
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _jain(values: List[float]) -> float:
        s = np.array(values, dtype=float)
        if len(s) == 0 or np.sum(s**2) == 0:
            return 0.0
        return float((s.sum() ** 2) / (len(s) * np.sum(s ** 2) + 1e-9))

    def get_observation(self):
        obs = []
        max_speed = 25.0
        for bs in self.base_stations:
            users_in_cov = [u for u in self.users if np.linalg.norm(u.location - bs.location) <= bs.coverage_area]
            max_power = self.ma_transmission_power if bs.type_bs == "Ma" else self.mi_transmission_power
            max_ch = self.max_ma_channels if bs.type_bs == "Ma" else self.max_mi_channels
            bs_type = 0.5 if bs.type_bs == "Ma" else 1.0
            tx_norm = np.clip(bs.transmit_power / max_power, 0, 1)
            used = sum(1 for c in bs.assigned_channels if len(c.users) > 0)
            ch_util = np.clip(used / max_ch, 0, 1)
            cov_util = len(users_in_cov) / self.num_users if self.num_users > 0 else 0.0
            load_ratio = len(users_in_cov) / max(1, len(bs.assigned_channels))
            load_ratio_norm = float(np.clip(load_ratio, 0.0, 2.0) / 2.0)
            nearby_pot = len([u for u in self.users if np.linalg.norm(u.location - bs.location) <= bs.coverage_area * 1.5]) / self.num_users if self.num_users > 0 else 0.0
            avg_speed = np.clip((sum(u.speed for u in users_in_cov) / len(users_in_cov) if users_in_cov else 0.0) / max_speed, 0, 1)

            max_dist = max([np.linalg.norm(u.location - bs.location) for u in users_in_cov]) if users_in_cov else 0.0
            if users_in_cov:
                req_p_per_ch = self.calculate_required_power_for_distance(max_dist, bs)
                n_ch = max(1, len(bs.assigned_channels))
                req_total = req_p_per_ch * n_ch
            else:
                req_total = 0.0
            req_p_norm = np.clip(req_total / max_power if max_power > 0 else 0.0, 0, 1)

            avg_radial_v = 0.5
            sp_var = np.clip(np.var([u.speed for u in users_in_cov]) / ((max_speed**2) / 4) if len(users_in_cov) > 1 else 0.0, 0, 1)

            f = self.bs_carrier_frequency[bs.id]
            co_bs = [o for o in self.base_stations if o is not bs and self.bs_carrier_frequency[o.id] == f]
            K = 10
            if users_in_cov:
                sampled_users = users_in_cov if len(users_in_cov) <= K else list(self.np_random.choice(users_in_cov, K, replace=False))
            else:
                sampled_users = []

            I_b = 0.0
            g_rx = 10.0 ** (UE_INTERF_RX_DB / 10.0)
            for b2 in co_bs:
                p_i = self._per_channel_tx_power_mW(b2)
                gi = BEAM_GAIN_DB[b2.type_bs]
                g_tx_i = 10.0 ** (gi["tx_side"] / 10.0)
                users_in_cov_b2 = [uu for uu in self.users if np.linalg.norm(uu.location - b2.location) <= b2.coverage_area]
                util = min(1.0, len(users_in_cov_b2) / max(1, len(b2.assigned_channels)))
                if sampled_users:
                    acc = 0.0
                    for uu in sampled_users:
                        d_i = np.linalg.norm(uu.location - b2.location)
                        PL_i = b2.calculate_path_loss(d_i, f)
                        acc += p_i * g_tx_i * g_rx / (10.0 ** (PL_i / 10.0))
                    I_b += util * (acc / len(sampled_users))
            I_dBm = 10.0 * np.log10(max(I_b, 1e-12))
            inter_norm = np.clip((I_dBm + 120.0) / 120.0, 0.0, 1.0)

            avg_demand_norm = float(np.clip(np.mean([u.demand for u in users_in_cov]) / 300.0, 0, 1)) if users_in_cov else 0.0
            neigh = [
                o.transmit_power / (self.ma_transmission_power if o.type_bs == "Ma" else self.mi_transmission_power)
                for o in self.base_stations
                if o is not bs and np.linalg.norm(bs.location - o.location) < 1000.0
            ]
            neighbor_tx_norm = float(np.mean(neigh)) if len(neigh) > 0 else 0.0

            obs.append([
                bs_type, tx_norm, ch_util, cov_util, load_ratio_norm, nearby_pot, avg_speed,
                req_p_norm, avg_radial_v, sp_var, neighbor_tx_norm, avg_demand_norm, inter_norm
            ])
        return np.array(obs, dtype=np.float32)

    # ---------- wrappers ----------
    def render(self, mode="human", action=None):
        if self.render_mode != "human" or self.ax is None:
            return
        self.ax.clear()
        if self.mobility_model == "manhattan":
            for x in self.grid_xs:
                self.ax.plot([x, x], [0, self.area_size[0]], linestyle=":", linewidth=0.5, alpha=0.25, color="gray")
            for y in self.grid_ys:
                self.ax.plot([0, self.area_size[0]], [y, y], linestyle=":", linewidth=0.5, alpha=0.25, color="gray")
        for bs in self.base_stations:
            color = "blue" if bs.type_bs == "Ma" else "green"
            circle = Circle(bs.location, bs.coverage_area, color=color, alpha=0.18)
            self.ax.add_patch(circle)
            self.ax.plot(bs.location[0], bs.location[1], "^" if bs.type_bs == "Ma" else "s", color=color, markersize=8)
        for u in self.users:
            self.ax.plot(u.location[0], u.location[1], "ro", markersize=3)
            for ch in u.channel:
                bs = ch.base_station
                self.ax.plot([u.location[0], bs.location[0]], [u.location[1], bs.location[1]], "k-", lw=0.5, alpha=0.4)
        self.ax.set_xlim(0, self.area_size[0])
        self.ax.set_ylim(0, self.area_size[1])
        self.ax.set_title(f"IoV map | step={self.num_steps}")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True, alpha=0.2)
        self.fig.canvas.draw()
        if self.disp is not None:
            self.disp.update(self.fig)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


class FlattenActionObservationWrapper(gym.Wrapper):
    """
    SB3-friendly wrapper:
    - observation: flatten (N, F) -> (N*F,)
    - action: flatten (N, 2) -> (N*2,)
    """
    def __init__(self, env: MobileNetwork):
        super().__init__(env)
        n, f = env.observation_space.shape
        self.n_bs = n
        self.obs_dim = f
        self.action_space = spaces.Box(low=env.action_space.low.flatten(), high=env.action_space.high.flatten(), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n * f,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs.flatten(), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.n_bs, 2)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.flatten(), reward, terminated, truncated, info
