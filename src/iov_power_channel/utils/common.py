
from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path: str | os.PathLike, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize_infos(infos: List[Dict]) -> Dict[str, float]:
    if not infos:
        return {}
    keys = sorted({k for d in infos for k in d.keys()})
    out = {}
    for k in keys:
        vals = [float(d[k]) for d in infos if k in d]
        if vals:
            out[k] = float(np.mean(vals))
    return out
