"""Render images/VO/epilines.png — epipolar lines on both images.

Each annotated point in one image maps to a 1D epipolar line in the other; the
lines converge at the epipole. We fit E (RANSAC), get F = K^-T E K^-1, and draw
the lines + points for the RANSAC inliers (so points sit on their lines).
Run from anywhere:  python scripts/make_epilines.py
"""
import os, json, pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cv2.setRNGSeed(0)
LOG = "273c1883-673a-36bf-b124-88311b1a80be"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROJ = os.path.dirname(os.path.dirname(REPO))
LOGDIR = os.path.join(PROJ, "data/tracking_train1_v1.1/argoverse-tracking/train1", LOG)
IMG = os.path.join(LOGDIR, "ring_front_center")
CALIB = os.path.join(LOGDIR, "vehicle_calibration_info.json")
PKL = os.path.join(PROJ, "argoverse_2_E_1.pkl.pkl")
OUT = os.path.join(REPO, "images", "VO")
os.makedirs(OUT, exist_ok=True)
ts1, ts2 = 315975640448534784, 315975643412234000

def load(ts):
    try:
        return plt.imread(f"{IMG}/ring_front_center_{ts}.jpg").copy()
    except Exception:
        from PIL import Image
        return np.asarray(Image.open(f"{IMG}/ring_front_center_{ts}.jpg")).copy()

img1, img2 = load(ts1), load(ts2)

# intrinsics K from calibration
fc = [x for x in json.load(open(CALIB))["camera_data_"]
      if x["key"].endswith("ring_front_center")][0]["value"]
K = np.array([[fc["focal_length_x_px_"], 0, fc["focal_center_x_px_"]],
              [0, fc["focal_length_y_px_"], fc["focal_center_y_px_"]],
              [0, 0, 1]])

d = pickle.load(open(PKL, "rb"))
kpts1 = np.hstack([np.array(d["x1"]).reshape(-1,1), np.array(d["y1"]).reshape(-1,1)]).astype(np.int32)
kpts2 = np.hstack([np.array(d["x2"]).reshape(-1,1), np.array(d["y2"]).reshape(-1,1)]).astype(np.int32)

E, mask = cv2.findEssentialMat(kpts1, kpts2, K, method=cv2.RANSAC, threshold=0.1)
F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)          # cam2_F_cam1

inl = mask.ravel() == 1
p1, p2 = kpts1[inl], kpts2[inl]
n = len(p1)
colors = (plt.cm.hsv(np.linspace(0, 1, n, endpoint=False))[:, :3] * 255).astype(int)

def draw(img, lines, pts):
    h, w = img.shape[:2]
    for line, pt, col in zip(lines, pts, colors):
        c = tuple(int(x) for x in col)
        x0, y0 = 0, int(-line[2] / line[1])
        x1, y1 = w, int(-(line[2] + line[0] * w) / line[1])
        cv2.line(img, (x0, y0), (x1, y1), c, 3)
        cv2.circle(img, (int(pt[0]), int(pt[1])), 13, c, -1)
    return img

lines2 = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2).astype(float), 1, F).reshape(-1, 3)
img2 = draw(img2, lines2, p2)
lines1 = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2).astype(float), 2, F).reshape(-1, 3)
img1 = draw(img1, lines1, p1)

C1, C2 = "#ff00ff", "#00d0e0"
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for a, im, color, lab in ((ax[0], img1, C1, "t1"), (ax[1], img2, C2, "t2")):
    a.imshow(im)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(4)
    a.text(0.015, 0.96, lab, transform=a.transAxes, color="white", fontsize=15,
           fontweight="bold", ha="left", va="top",
           bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none"))
fig.subplots_adjust(left=0.008, right=0.992, top=0.99, bottom=0.01, wspace=0.03)
fig.savefig(f"{OUT}/epilines.png", dpi=130)
plt.close(fig)
print("wrote", f"{OUT}/epilines.png", "| inliers:", n)
