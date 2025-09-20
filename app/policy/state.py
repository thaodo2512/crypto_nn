from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class PolicyState:
    """Holds minimal state for EV policy.

    - last_entry_ts: timestamp of last entry (starts cooldown window)
    - has_position: whether a position is currently open
    - side: 'LONG' | 'SHORT' | None for the open position
    - entry_close: close price at entry
    - tp_px, sl_px: absolute barrier prices
    - bars_since_entry: integer bars elapsed since entry
    """

    last_entry_ts: Optional[pd.Timestamp] = None
    has_position: bool = False
    side: Optional[str] = None
    entry_close: float = float("nan")
    tp_px: float = float("nan")
    sl_px: float = float("nan")
    bars_since_entry: int = 0

    def reset_position(self) -> None:
        self.has_position = False
        self.side = None
        self.entry_close = float("nan")
        self.tp_px = float("nan")
        self.sl_px = float("nan")
        self.bars_since_entry = 0

