"""Reproduce the VO pipeline numbers shown in the blog (prints only, no images).

Reads poses + calibration JSON directly (no argoverse package needed) and the
hand-annotated pkl, then prints ground-truth and recovered (R, t) for comparison.
Run from anywhere:  python scripts/run_pipeline.py
"""
import os, json, pickle
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

np.set_printoptions(suppress=True)
cv2.setRNGSeed(0)

LOG = "273c1883-673a-36bf-b124-88311b1a80be"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROJ = os.path.dirname(os.path.dirname(REPO))
LOGDIR = os.path.join(PROJ, "data/tracking_train1_v1.1/argoverse-tracking/train1", LOG)
POSES = os.path.join(LOGDIR, "poses")
CALIB = os.path.join(LOGDIR, "vehicle_calibration_info.json")
PKL = os.path.join(PROJ, "argoverse_2_E_1.pkl.pkl")
ts1, ts2 = 315975640448534784, 315975643412234000

class SE3:
    def __init__(self, R, t): self.R = np.asarray(R, float); self.t = np.asarray(t, float)
    def inverse(self): return SE3(self.R.T, -self.R.T @ self.t)
    def compose(self, o): return SE3(self.R @ o.R, self.R @ o.t + self.t)

def quat_wxyz_to_R(q):
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()

def city_SE3_ego(ts):
    d = json.load(open(f"{POSES}/city_SE3_egovehicle_{ts}.json"))
    return SE3(quat_wxyz_to_R(d["rotation"]), d["translation"])

city_SE3_egot1 = city_SE3_ego(ts1)
city_SE3_egot2 = city_SE3_ego(ts2)
print("# [1] ego position (city frame)")
print("t1 =", np.round(city_SE3_egot1.t, 2))
print("t2 =", np.round(city_SE3_egot2.t, 2))

egot1_SE3_egot2 = city_SE3_egot1.inverse().compose(city_SE3_egot2)
print("\n# [2] relative pose (egovehicle frame)")
print("rotation zyx (deg) =", np.round(Rotation.from_matrix(egot1_SE3_egot2.R).as_euler("zyx", degrees=True), 2))
print("translation (m)    =", np.round(egot1_SE3_egot2.t, 2))

c = json.load(open(CALIB))
fc = [x for x in c["camera_data_"] if x["key"].endswith("ring_front_center")][0]["value"]
egovehicle_SE3_camera = SE3(quat_wxyz_to_R(fc["vehicle_SE3_camera_"]["rotation"]["coefficients"]),
                            fc["vehicle_SE3_camera_"]["translation"])
K = np.array([[fc["focal_length_x_px_"], 0, fc["focal_center_x_px_"]],
              [0, fc["focal_length_y_px_"], fc["focal_center_y_px_"]],
              [0, 0, 1]])

city_SE3_cam1 = city_SE3_egot1.compose(egovehicle_SE3_camera)
city_SE3_cam2 = city_SE3_egot2.compose(egovehicle_SE3_camera)
cam1_SE3_cam2 = city_SE3_cam1.inverse().compose(city_SE3_cam2)
gt_rot = Rotation.from_matrix(cam1_SE3_cam2.R).as_euler("zyx", degrees=True)
gt_t_unit = cam1_SE3_cam2.t / np.linalg.norm(cam1_SE3_cam2.t)
print("\n# [3] GROUND TRUTH (camera frame)")
print("rotation zyx (deg) =", np.round(gt_rot, 3))
print("translation (m)    =", np.round(cam1_SE3_cam2.t, 2))

d = pickle.load(open(PKL, "rb"))
img1_kpts = np.hstack([np.array(d["x1"]).reshape(-1,1), np.array(d["y1"]).reshape(-1,1)]).astype(np.int32)
img2_kpts = np.hstack([np.array(d["x2"]).reshape(-1,1), np.array(d["y2"]).reshape(-1,1)]).astype(np.int32)
print("\n# [4] correspondences:", len(img1_kpts))

cam2_E_cam1, inlier_mask = cv2.findEssentialMat(img1_kpts, img2_kpts, K, method=cv2.RANSAC, threshold=0.1)
print("Num inliers:", int(inlier_mask.sum()), "/", len(img1_kpts))

_, cam2_R_cam1, cam2_t_cam1, _ = cv2.recoverPose(cam2_E_cam1, img1_kpts, img2_kpts, K, mask=inlier_mask.copy())
print("\n# [5] recoverPose -> cam2_T_cam1 (inverse direction)")
print("rotation zyx (deg) =", np.round(Rotation.from_matrix(cam2_R_cam1).as_euler("zyx", degrees=True), 2))
print("translation        =", np.round(cam2_t_cam1.squeeze(), 2))

cam1_SE3_cam2_est = SE3(cam2_R_cam1, cam2_t_cam1.squeeze()).inverse()
est_rot = Rotation.from_matrix(cam1_SE3_cam2_est.R).as_euler("zyx", degrees=True)
est_t_unit = cam1_SE3_cam2_est.t / np.linalg.norm(cam1_SE3_cam2_est.t)
err = np.rad2deg(np.arccos(np.clip(gt_t_unit @ est_t_unit, -1, 1)))
print("\n# [6] after inverting -> compare to ground truth")
print("Estimated rotation:", np.round(est_rot, 2))
print("Ground truth      :", np.round(gt_rot, 2))
print("Estimated t (unit):", np.round(est_t_unit, 3))
print("Ground truth t    :", np.round(gt_t_unit, 3))
print("Translation angular error (deg):", round(float(err), 2))
