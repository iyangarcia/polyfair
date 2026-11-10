"""SQLite persistence: the fill log is the source of truth for positions."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .models import Fill, Signal

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY, ts REAL, mode TEXT, market TEXT, token_id TEXT,
    outcome TEXT, side TEXT, shares REAL, price REAL, fee REAL, order_id TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY, ts REAL, mode TEXT, market TEXT, token_id TEXT,
    outcome TEXT, side TEXT, fair REAL, touch REAL, limit_price REAL, shares REAL,
    edge REAL, status TEXT, detail TEXT, decision_ms REAL
);
CREATE TABLE IF NOT EXISTS equity (
    ts REAL, mode TEXT, equity REAL, cash REAL, exposure REAL, realized REAL, fees REAL
);
"""


class Store:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(_SCHEMA)

    def record_fill(self, f: Fill) -> None:
        self.db.execute(
            "INSERT INTO fills (ts, mode, market, token_id, outcome, side, shares, price, fee, order_id)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f.ts, f.mode, f.market, f.token_id, f.outcome, f.side, f.shares, f.price, f.fee, f.order_id),
        )
        self.db.commit()

    def record_order(self, mode: str, s: Signal, status: str, detail: str, decision_ms: float) -> None:
        self.db.execute(
            "INSERT INTO orders (ts, mode, market, token_id, outcome, side, fair, touch, limit_price,"
            " shares, edge, status, detail, decision_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), mode, s.market, s.token_id, s.outcome, s.side, s.fair, s.touch, s.limit,
             s.shares, s.edge, status, detail, decision_ms),
        )
        self.db.commit()

    def record_equity(self, mode: str, equity: float, cash: float, exposure: float,
                      realized: float, fees: float) -> None:
        self.db.execute("INSERT INTO equity VALUES (?,?,?,?,?,?,?)",
                        (time.time(), mode, equity, cash, exposure, realized, fees))
        self.db.commit()

    def fills(self, mode: str) -> list[Fill]:
        rows = self.db.execute(
            "SELECT ts, mode, market, token_id, outcome, side, shares, price, fee, order_id"
            " FROM fills WHERE mode = ? ORDER BY id", (mode,),
        ).fetchall()
        return [Fill(*r) for r in rows]

    def order_stats(self, mode: str) -> list[tuple]:
        return self.db.execute(
            "SELECT status, COUNT(*), AVG(decision_ms), AVG(edge) FROM orders WHERE mode = ? GROUP BY status",
            (mode,),
        ).fetchall()

    def close(self) -> None:
        self.db.close()
