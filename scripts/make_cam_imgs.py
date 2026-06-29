"""Render images/VO/cam_images_t1_t2.png — the two front-center camera frames.

t1 (magenta border) and t2 (cyan border), matching the trajectory dot colours.
Run from anywhere:  python scripts/make_cam_imgs.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "273c1883-673a-36bf-b124-88311b1a80be"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROJ = os.path.dirname(os.path.dirname(REPO))
IMG = os.path.join(PROJ, "data/tracking_train1_v1.1/argoverse-tracking/train1", LOG, "ring_front_center")
OUT = os.path.join(REPO, "images", "VO")
os.makedirs(OUT, exist_ok=True)

ts1, ts2 = 315975640448534784, 315975643412234000

def load(ts):
    try:
        return plt.imread(f"{IMG}/ring_front_center_{ts}.jpg")
    except Exception:
        from PIL import Image
        import numpy as np
        return np.asarray(Image.open(f"{IMG}/ring_front_center_{ts}.jpg"))

img1, img2 = load(ts1), load(ts2)
C1, C2 = "#ff00ff", "#00d0e0"   # t1 / t2 colours

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for a, im, color, lab in ((ax[0], img1, C1, "t1"), (ax[1], img2, C2, "t2")):
    a.imshow(im)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(4)
    a.text(0.015, 0.96, lab, transform=a.transAxes, color="white",
           fontsize=15, fontweight="bold", ha="left", va="top",
           bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none"))

fig.subplots_adjust(left=0.008, right=0.992, top=0.99, bottom=0.01, wspace=0.03)
fig.savefig(f"{OUT}/cam_images_t1_t2.png", dpi=130)
plt.close(fig)
print("wrote", f"{OUT}/cam_images_t1_t2.png")
