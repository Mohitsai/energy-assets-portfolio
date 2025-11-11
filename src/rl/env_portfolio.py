import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, assets: list[str], window: int = 5, tc_bps: float = 1.0, risk_lambda: float = 0.0):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.assets = assets
        self.n = len(assets)
        self.window = window
        self.tc = tc_bps / 10000.0
        self.risk_lambda = risk_lambda

        # build observation features: last window of returns for each asset plus simple indicators
        self.ret_cols = [f"ret_1_{a}" for a in assets]
        self.obs_cols = []
        for a in assets:
            self.obs_cols += [f"ret_1_{a}", f"sma_5_{a}", f"sma_20_{a}", f"vol_10_{a}", f"zprice_{a}"]

        self.idx = window  # start after window days
        self.prev_w = np.ones(self.n) / self.n

        low = -np.inf * np.ones(len(self.obs_cols) * window)
        high = np.inf * np.ones(len(self.obs_cols) * window)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # action is logits R^n, we softmax to simplex weights
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n,), dtype=np.float32)

    def _obs(self):
        frames = []
        for w in range(self.idx - self.window, self.idx):
            frames.append(self.df.loc[w, self.obs_cols].values.astype(np.float32))
        return np.concatenate(frames, axis=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = self.window
        self.prev_w = np.ones(self.n) / self.n
        return self._obs(), {}

    def step(self, action):
        # convert logits to weights on simplex
        logits = np.clip(action, -10, 10)
        e = np.exp(logits - logits.max())
        w = e / e.sum()

        # portfolio return next day
        next_ret_vec = self.df.loc[self.idx, self.ret_cols].values.astype(np.float32)
        port_ret = float((w * next_ret_vec).sum())

        # transaction cost on turnover
        turnover = np.abs(w - self.prev_w).sum()
        cost = self.tc * turnover

        # simple risk penalty on variance proxy (squared return)
        reward = port_ret - cost - self.risk_lambda * (port_ret ** 2)

        self.prev_w = w
        self.idx += 1
        terminated = self.idx >= len(self.df) - 1
        obs = self._obs() if not terminated else self._obs()
        info = {"port_ret": port_ret, "turnover": turnover, "weights": w}
        return obs, reward, terminated, False, info
