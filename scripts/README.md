# Figure / result scripts

Helpers that regenerate the blog figures and the numbers quoted in the posts.

## Prerequisites

```bash
pip install numpy matplotlib opencv-python scipy pillow
```

Plus the Argoverse 1 log (not in this repo), expected at:

```
<project>/data/tracking_train1_v1.1/argoverse-tracking/train1/273c1883-673a-36bf-b124-88311b1a80be/
<project>/argoverse_2_E_1.pkl.pkl          # hand-annotated correspondences
```

where `<project>` is the parent two levels above this repo
(`.../VO_tutorial/`, i.e. `../../` from the repo root). Paths are resolved
relative to each script, so they run from anywhere.

## Scripts

| Script | Output |
|---|---|
| `make_trajectory.py` | `images/ego_trajectory.png` — ego trajectory, equal scale, t1/t2 |
| `make_cam_imgs.py`   | `images/VO/cam_images_t1_t2.png` — camera frames at t1/t2 |
| `make_corr_img.py`   | `images/VO/correspondences.png` — annotated 2D-2D matches |
| `make_epilines.py`   | `images/VO/epilines.png` — epipolar lines on both images |
| `run_pipeline.py`    | prints ground-truth vs recovered (R, t) — no image |

```bash
python scripts/make_trajectory.py
python scripts/run_pipeline.py
```
