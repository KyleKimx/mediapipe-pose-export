from pathlib import Path
import json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from scipy.io import savemat

# =========================
# Pose extraction pipeline:
# (Read FPS -> Inference -> Draw -> Save video + arrays)
# Outputs: .avi, .npz, .mat, .csv, _meta.json
# =========================

# ---------- Paths ----------
BASE = Path(__file__).resolve().parent
model_path = BASE / "assets" / "models" / "pose_landmarker_full.task"
video_path = BASE / "data" / "videos" / "CASE1-WEEK1-2025.03.15-PRE.mp4"

assert model_path.exists(), f"Model not found: {model_path}"
assert video_path.exists(), f"Video not found: {video_path}"

# Output video path (keep same stem under /outputs, use .avi as requested)
out_dir = BASE / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{video_path.stem}.avi"

# ---------- MediaPipe setup ----------
BaseOptions = python.BaseOptions
RunningMode = vision.RunningMode

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    # output_segmentation_masks=False,
)

landmarker = vision.PoseLandmarker.create_from_options(options)

# ---------- Video IO ----------
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Can't open the video: {video_path}")

# Input metadata
in_fps = cap.get(cv2.CAP_PROP_FPS)
in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
in_nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Guard against broken FPS metadata
if in_fps <= 0 or in_fps > 120:
    in_fps = 30.0

print(f"[INFO] opened: {in_w}x{in_h} @ {in_fps:.2f}fps, frames={in_nf}")

# Optional rotation to force landscape display
force_landscape = False
rotate_code = cv2.ROTATE_90_CLOCKWISE  # change if needed

# Determine output frame size (after optional rotation)
if force_landscape and in_h > in_w:
    out_size = (in_h, in_w)  # swapped due to rotation
else:
    out_size = (in_w, in_h)

# Output video writer (AVI/XVID as requested)
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # use "mp4v" if you want .mp4 instead
writer = cv2.VideoWriter(str(out_path), fourcc, in_fps, out_size)
if not writer.isOpened():
    raise RuntimeError("VideoWriter failed to open.")

# ---------- Accumulators for export ----------
detected_frames = 0                       # number of frames where a pose was detected
ts_list = []                              # [N] timestamps (ms)
pix_xy = []                               # [N, 33, 2] pixel coords
pix_z = []                                # [N, 33] normalized z
vis = []                                  # [N, 33] visibility
pres = []                                 # [N, 33] presence
world_xyz = []                            # [N, 33, 3] world coords (meters)

ts_ms = 0.0
frame_idx = 0

# Precompute id->name map (not strictly needed for drawing indices, but handy)
try:
    PL = mp.solutions.pose.PoseLandmark
    id2name = {pl.value: pl.name for pl in PL}
except Exception:
    id2name = {i: f"LM_{i}" for i in range(33)}

try:
    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        # Optional rotation
        if force_landscape and in_h > in_w:
            bgr = cv2.rotate(bgr, rotate_code)

        # Inference
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_img, int(ts_ms))

        # Current frame size (after any rotation)
        h, w = bgr.shape[:2]

        # 1Hz console summary
        if frame_idx % max(1, int(in_fps)) == 0:
            if res.pose_landmarks:
                lm0 = res.pose_landmarks[0]
                nose = lm0[0]
                Ls = lm0[11]
                Rs = lm0[12]
                getv = lambda L: getattr(L, "visibility", float("nan"))
                getp = lambda L: getattr(L, "presence", float("nan"))
                print(
                    f"[{frame_idx:05d}] poses=1 "
                    f"nose=({nose.x*w:.1f},{nose.y*h:.1f}) vis={getv(nose):.2f} pres={getp(nose):.2f} | "
                    f"L_sh=({Ls.x*w:.1f},{Ls.y*h:.1f}) vis={getv(Ls):.2f} | "
                    f"R_sh=({Rs.x*w:.1f},{Rs.y*h:.1f}) vis={getv(Rs):.2f}"
                )
            else:
                print(f"[{frame_idx:05d}] poses=0")

        # Default NaNs for this frame
        L = 33
        px = np.full((L, 2), np.nan, dtype=np.float32)
        pz = np.full((L,), np.nan, dtype=np.float32)
        pv = np.full((L,), np.nan, dtype=np.float32)
        pp = np.full((L,), np.nan, dtype=np.float32)
        wx = np.full((L, 3), np.nan, dtype=np.float32)

        # Extract results -> accumulators
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]  # first/only person
            for i, l in enumerate(lm):
                # image coords are normalized [0..1] -> convert to pixels
                px[i, 0] = float(l.x) * w
                px[i, 1] = float(l.y) * h
                pz[i] = float(l.z)
                # confidence fields
                pv[i] = float(l.visibility) if hasattr(l, "visibility") else np.nan
                pp[i] = float(l.presence) if hasattr(l, "presence") else np.nan

            # world coords (meters, hip-centered origin)
            if res.pose_world_landmarks:
                wlm = res.pose_world_landmarks[0]
                for i, l in enumerate(wlm):
                    wx[i, 0] = float(l.x)
                    wx[i, 1] = float(l.y)
                    wx[i, 2] = float(l.z)

            # Draw overlay: connections + points + index labels
            detected_frames += 1
            lm_list = landmark_pb2.NormalizedLandmarkList()
            lm_list.landmark.extend(
                [landmark_pb2.NormalizedLandmark(x=lm2.x, y=lm2.y, z=lm2.z)
                 for lm2 in res.pose_landmarks[0]]
            )
            mp.solutions.drawing_utils.draw_landmarks(
                bgr,
                lm_list,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

            for i, l in enumerate(res.pose_landmarks[0]):
                cx, cy = int(l.x * w), int(l.y * h)
                cv2.circle(bgr, (cx, cy), 3, (255, 255, 255), -1)
                cv2.putText(
                    bgr, str(i), (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                )

        # Write frame & collect arrays
        writer.write(bgr)
        ts_list.append(int(ts_ms))
        pix_xy.append(px)
        pix_z.append(pz)
        vis.append(pv)
        pres.append(pp)
        world_xyz.append(wx)

        # Next frame
        frame_idx += 1
        ts_ms += 1000.0 / in_fps  # timestamp in ms for detect_for_video

finally:
    # Always release resources
    cap.release()
    writer.release()
    landmarker.close()
    cv2.destroyAllWindows()

# ---------- Verify output video length ----------
chk = cv2.VideoCapture(str(out_path))
out_fps = chk.get(cv2.CAP_PROP_FPS) or in_fps
out_nf = int(chk.get(cv2.CAP_PROP_FRAME_COUNT))
chk.release()

in_dur = (in_nf / in_fps) if in_nf > 0 else None
out_dur = (out_nf / out_fps) if out_fps > 0 else None
print(f"[VERIFY] in_frames={in_nf}, out_frames={out_nf}, in_dur≈{in_dur:.2f}s, out_dur≈{out_dur:.2f}s")
print(f"[DONE] saved -> {out_path}")

# ---------- Summary stats + save arrays ----------
N = len(ts_list)
if N >= 2:
    dur_s = (ts_list[-1] - ts_list[0]) / 1000.0
else:
    dur_s = (in_nf / in_fps) if in_nf > 0 else 0.0

pose_sampling_hz = (detected_frames / dur_s) if dur_s > 0 else float("nan")
input_fps_reported = in_fps
output_fps_reported = out_fps

print("[STATS] frames_total=", N)
print("[STATS] frames_detected=", detected_frames)
print(f"[STATS] duration≈{dur_s:.3f}s")
print(f"[STATS] pose_sampling_hz≈{pose_sampling_hz:.3f} Hz")
print(f"[STATS] input_video_fps≈{input_fps_reported:.3f}, output_video_fps≈{output_fps_reported:.3f}")

# Save arrays / meta
npz_path = out_path.with_suffix(".npz")
mat_path = out_path.with_suffix(".mat")
csv_path = out_path.with_suffix(".csv")
meta_path = out_path.with_name(out_path.stem + "_meta.json")

data_np = {
    "timestamps_ms": np.asarray(ts_list, dtype=np.int64),   # [N]
    "pix_xy": np.asarray(pix_xy, dtype=np.float32),         # [N, 33, 2]
    "pix_z": np.asarray(pix_z, dtype=np.float32),           # [N, 33]
    "visibility": np.asarray(vis, dtype=np.float32),        # [N, 33]
    "presence": np.asarray(pres, dtype=np.float32),         # [N, 33]
    "world_xyz": np.asarray(world_xyz, dtype=np.float32),   # [N, 33, 3]
}
np.savez_compressed(npz_path, **data_np)
savemat(str(mat_path), data_np)

meta = {
    "input_video_path": str(video_path),
    "output_video_path": str(out_path),
    "input_video_fps_reported": float(input_fps_reported),
    "output_video_fps_reported": float(output_fps_reported),
    "frames_total": int(N),
    "frames_detected": int(detected_frames),
    "duration_sec": float(dur_s),
    "pose_sampling_hz": float(pose_sampling_hz),
    "landmarks_per_person": 33,
    "num_poses": 1,
    "model": "pose_landmarker_full.task",
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# Long-format CSV (frame x landmark rows)
try:
    import csv as _csv
    arr_ts = data_np["timestamps_ms"]
    arr_px = data_np["pix_xy"]
    arr_pz = data_np["pix_z"]
    arr_v = data_np["visibility"]
    arr_p = data_np["presence"]
    arr_w = data_np["world_xyz"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow([
            "frame", "t_ms", "lm_id",
            "x_px", "y_px", "z_norm",
            "visibility", "presence",
            "wx_m", "wy_m", "wz_m"
        ])
        for fi in range(N):
            t = int(arr_ts[fi])
            for li in range(33):
                x_px, y_px = arr_px[fi, li, 0], arr_px[fi, li, 1]
                z_norm = arr_pz[fi, li]
                visib = arr_v[fi, li]
                presc = arr_p[fi, li]
                wx, wy, wz = arr_w[fi, li, 0], arr_w[fi, li, 1], arr_w[fi, li, 2]
                writer.writerow([fi, t, li, x_px, y_px, z_norm, visib, presc, wx, wy, wz])
    print(f"[DONE] saved arrays: {npz_path}, {mat_path}, {csv_path}, {meta_path}")
except Exception as e:
    print("[WARN] CSV export skipped:", e)
