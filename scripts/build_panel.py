"""Build the hourly panel from the verified archives (data/raw -> data/panel)."""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fasa.panel import build, save

cfg = yaml.safe_load((ROOT / "configs" / "alpha_protocol.yaml").read_text())
pool = json.loads((ROOT / "data" / "pool.json").read_text())["pool"]
p = build(ROOT / "data" / "raw", pool, cfg["calendar"]["warmup_from"], cfg["calendar"]["final"][1])
save(p, ROOT / "data" / "panel")
o = p["perp"]["open"]
print(o.shape, "perp hours x symbols;", int(o.notna().sum().sum()), "bars;", len(p["funding"]), "funding settlements")
print("last bar per symbol (delisted before the end):")
last = o.apply(lambda c: c.last_valid_index())
print(last[last < o.index[-1]].dt.strftime("%Y-%m-%d").to_string())
print("funding intervals seen:", sorted(p["funding"]["interval_h"].unique()))
