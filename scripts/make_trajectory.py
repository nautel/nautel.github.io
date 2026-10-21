"""Render images/ego_trajectory.png — ego trajectory in the city frame.

Equal scale (1 m x = 1 m y), gradient red->green over time, t1/t2 marked.
Run from anywhere:  python scripts/make_trajectory.py
Requires the Argoverse log under <project>/data/... (see scripts/README.md).
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

LOG = "273c1883-673a-36bf-b124-88311b1a80be"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                       # mysite/nautel.github.io
PROJ = os.path.dirname(os.path.dirname(REPO))      # VO_tutorial
LOGDIR = os.path.join(PROJ, "data/tracking_train1_v1.1/argoverse-tracking/train1", LOG)
POSES = os.path.join(LOGDIR, "poses")
OUT = os.path.join(REPO, "images")

ts1, ts2 = 315975640448534784, 315975643412234000

rows = []
for fp in glob.glob(f"{POSES}/city_SE3_egovehicle_*.json"):
    ts = int(os.path.basename(fp).split("_")[-1].split(".")[0])
    t = json.load(open(fp))["translation"]
    rows.append((ts, t[0], t[1]))
rows.sort()
ts_arr = np.array([r[0] for r in rows])
xs = np.array([r[1] for r in rows])
ys = np.array([r[2] for r in rows])

i1 = int(np.argmin(np.abs(ts_arr - ts1)))
i2 = int(np.argmin(np.abs(ts_arr - ts2)))

# window: a few poses before t1, forward to the rightmost point after t2 (the turn segment)
start = max(0, i1 - 6)
end = i2 + int(np.argmax(xs[i2:i2 + 2000]))
sx, sy = xs[start:end+1], ys[start:end+1]

# gradient line red -> green along time
pts = np.array([sx, sy]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap="RdYlGn", linewidth=5)
lc.set_array(np.linspace(0, 1, len(segs)))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.add_collection(lc)
ax.plot(xs[i1], ys[i1], "o", color="magenta", ms=15, label="t1", zorder=5)
ax.plot(xs[i2], ys[i2], "o", color="cyan", ms=15, label="t2", zorder=5)
ax.set_xlim(sx.min() - 3, sx.max() + 3)
ax.set_ylim(sy.min() - 3, sy.max() + 3)
ax.set_aspect("equal")          # 1 m in x == 1 m in y (true scale)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Ego trajectory (city frame)")
ax.legend(loc="lower right")
ax.grid(True, ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(f"{OUT}/ego_trajectory.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote", f"{OUT}/ego_trajectory.png", "| poses", start, "->", end)
