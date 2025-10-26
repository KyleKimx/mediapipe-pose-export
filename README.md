# MediaPipe Pose Export: 33 Landmarks to Video + CSV/MAT/NPZ

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![OpenCV](https://img.shields.io/badge/opencv-4.x-brightgreen)]()
[![MediaPipe](https://img.shields.io/badge/mediapipe-tasks-orange)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)]()

Run **MediaPipe Pose (Tasks API)** on a video, overlay **all 33 landmarks** with indices, and export per-frame pose data to **CSV**, **MAT**, and **NPZ**, plus a `_meta.json` summary.  
Also writes an annotated **AVI** (or MP4) video.

> Repo: https://github.com/KyleKimx/mediapipe-pose-export

---

## Demo

<img src="docs/demo.gif" width="720" alt="demo gif">

> Add a 3–5s sample or GIF showing the indexed 33 landmarks. Avoid any private footage.

---

## Project Structure

mediapipe-pose-export/
<br />
├─ assets/models/pose_landmarker_full.task # model (see Setup)
<br />
├─ data/videos/sample.mp4 # small demo video
<br />
├─ outputs/ # results (generated)
<br />
├─ src/extract_pose.py # main script (CLI)
<br />
├─ notebooks/quick_viz.ipynb # optional
<br />
├─ requirements.txt
<br />
├─ .gitignore
<br />
├─ LICENSE
<br />
└─ README.md

---

## Setup

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Model

Download ```pose_landmarker_full.task``` to ```assets/models/```.

+ Official: MediaPipe Pose Landmarker (Tasks API).

+ You can store via Git LFS or provide a small download script.

### 3) Data

Place a short test video at ```data/videos/sample.mp4``` (or pass your own via ```--video```).

***
### Usage
```bash
python src/extract_pose.py \
  --model assets/models/pose_landmarker_full.task \
  --video data/videos/sample.mp4 \
  --outdir outputs \
  --format avi            # or mp4
  --force_landscape       # optional
```

### Outputs (same stem as input video)

+ ```outputs/<name>.avi``` – annotated video with 33 landmarks (indexed)
+ ```outputs/<name>.npz``` – NumPy arrays
+ ```outputs/<name>.mat``` – MATLAB file
+ ```outputs/<name>.csv``` – long-format table (frame × landmark)
+ ```outputs/<name>_meta.json``` – summary (fps, duration, detected frames, etc.)

---
### Output Schema

#### CSV columns

+ ```frame``` – frame index (0-based)
+ ```t_ms``` – timestamp in milliseconds
+ ```lm_id``` – landmark ID (0–32)
+ ```x_px```, ```y_px``` – pixel coordinates
+ ```z_norm``` – normalized depth (relative, often negative closer to camera)
+ ```visibility```, ```presence``` – model confidence (0–1)
+ ```wx_m```, ```wy_m```, ```wz_m``` – world coordinates in meters (hip-centered)

#### NPZ/MAT keys

+ ```timestamps_ms``` ```[N]```
+ ```pix_xy``` ```[N, 33, 2]```
+ ```pix_z``` ```[N, 33]```
+ ```visibility``` ```[N, 33]```
+ ```presence``` ```[N, 33]```
+ ```world_xyz``` ```[N, 33, 3]```

---
### Notes

+ Effective pose sampling rate is reported as ```pose_sampling_hz``` in ```_meta.json```.
+ For MP4 output, use ```--format mp4``` or change codec to ```mp4v``` in the script.
+ Avoid pushing sensitive footage; use a public demo or synthetic data.

---
### Roadmap

+ Optional JSON dump per frame (all points)
+ Smoothing & interpolation utilities
+ Velocity/speed computation examples
+ Wide-format export for ML pipelines

---
### License

MIT © Byounguk Kim

---
### Citation
```bash
If you use this code, please cite this repository and MediaPipe:
@software{kim_mediapipe_pose_export_2025,
  title        = {MediaPipe Pose Export: 33 Landmarks to Video + CSV/MAT/NPZ},
  author       = {Byounguk Kim},
  year         = {2025},
  url          = {https://github.com/KyleKimx/mediapipe-pose-export}
}

---

# 2) LICENSE (MIT)

```text
MIT License

Copyright (c) 2025 Byounguk Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

```
