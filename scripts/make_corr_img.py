"""Render images/VO/correspondences.png — hand-annotated 2D-2D correspondences.

Reads argoverse_2_E_1.pkl.pkl and links each matched point across the two images.
Run from anywhere:  python scripts/make_corr_img.py
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "273c1883-673a-36bf-b124-88311b1a80be"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROJ = os.path.dirname(os.path.dirname(REPO))
IMG = os.path.join(PROJ, "data/tracking_train1_v1.1/argoverse-tracking/train1", LOG, "ring_front_center")
PKL = os.path.join(PROJ, "argoverse_2_E_1.pkl.pkl")
OUT = os.path.join(REPO, "images", "VO")
os.makedirs(OUT, exist_ok=True)
ts1, ts2 = 315975640448534784, 315975643412234000

def load(ts):
    try:
        return plt.imread(f"{IMG}/ring_front_center_{ts}.jpg")
    except Exception:
        from PIL import Image
        return np.asarray(Image.open(f"{IMG}/ring_front_center_{ts}.jpg"))

img1, img2 = load(ts1), load(ts2)

d = pickle.load(open(PKL, "rb"))
X1, Y1 = np.array(d["x1"]), np.array(d["y1"])
X2, Y2 = np.array(d["x2"]), np.array(d["y2"])
n = len(X1)

h = max(img1.shape[0], img2.shape[0])
w1 = img1.shape[1]
canvas = np.zeros((h, w1 + img2.shape[1], 3), dtype=img1.dtype)
canvas[:img1.shape[0], :w1] = img1
canvas[:img2.shape[0], w1:] = img2

colors = plt.cm.hsv(np.linspace(0, 1, n, endpoint=False))
C1, C2 = "#ff00ff", "#00d0e0"

W = canvas.shape[1]
fig, ax = plt.subplots(figsize=(12.8, 12.8 * h / W + 0.5))
ax.imshow(canvas)
for i in range(n):
    c = colors[i]
    p1 = (X1[i], Y1[i])
    p2 = (X2[i] + w1, Y2[i])
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color=c, lw=1.4, alpha=0.9)
    for p in (p1, p2):
        ax.plot(p[0], p[1], "o", color=c, ms=9, markeredgecolor="black",
                markeredgewidth=0.8, clip_on=False)
ax.axvline(w1, color="white", lw=2)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-0.5, W - 0.5)
ax.set_ylim(h - 0.5, -0.5)
ax.text(22, 30, "t1", color="white", fontsize=15, fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc=C1, ec="none"))
ax.text(w1 + 22, 30, "t2", color="white", fontsize=15, fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc=C2, ec="none"))
fig.subplots_adjust(left=0.008, right=0.992, top=0.99, bottom=0.02)
fig.savefig(f"{OUT}/correspondences.png", dpi=130)
plt.close(fig)
print("wrote", f"{OUT}/correspondences.png", "| correspondences:", n)
