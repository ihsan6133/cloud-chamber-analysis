
import argparse
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------------
def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open {path}. Your OpenCV build may lack WebM/VP9 support.\n"
            f"Convert first:  ffmpeg -i {path} -c:v libx264 -crf 12 video.mp4"
        )
    return cap


def to_gray(frame, crop):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if crop:
        x, y, w, h = crop
        g = g[y:y + h, x:x + w]
    return g


def estimate_background(path, crop, stride=10, max_samples=150):
    """Median over sparsely sampled frames. Tracks are transient, so the
    median keeps only what is always there (static clutter / lighting)."""
    cap = open_video(path)
    samples, i = [], 0
    while len(samples) < max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            samples.append(to_gray(frame, crop))
        i += 1
    cap.release()
    if not samples:
        raise SystemExit("No frames could be read.")
    return np.median(np.stack(samples), axis=0).astype(np.float32)


# ----------------------------------------------------------------------------
def robust_sigma(img):
    """Noise level estimate from the median absolute deviation."""
    med = np.median(img)
    return 1.4826 * np.median(np.abs(img - med)) + 1e-6


def clean_frame(gray, bg, a):
    g = gray.astype(np.float32)

    # 2. background subtraction (tracks are brighter than background)
    diff = np.clip(g - bg, 0, None)

    # 3. noise suppression
    if a.denoise == "nlm":  # slower, better edge preservation
        d8 = np.clip(diff, 0, 255).astype(np.uint8)
        diff = cv2.fastNlMeansDenoising(d8, None, h=a.nlm_h).astype(np.float32)
    else:
        diff = cv2.GaussianBlur(diff, (0, 0), a.blur)

    # 4. remove broad haze: structures wider than the kernel vanish,
    #    thin tracks survive
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.tophat, a.tophat))
    diff = cv2.morphologyEx(diff, cv2.MORPH_TOPHAT, k)

    # 5. hysteresis threshold, thresholds relative to measured noise
    sigma = robust_sigma(diff)
    base = np.median(diff)
    low = (diff > base + a.low_k * sigma).astype(np.uint8)
    high = (diff > base + a.high_k * sigma).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    strong = np.unique(labels[high > 0])  # components touching a strong pixel
    for lab in strong:
        if lab == 0:
            continue
        x, y, w, h, area = stats[lab]
        if area >= a.min_area and max(w, h) >= a.min_len:
            keep[lab] = True
    mask = keep[labels].astype(np.uint8) * 255

    # 6. cleaned image: keep original intensity only near kept tracks
    near = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    out = np.where(near > 0, diff, 0)
    out = out / max(np.percentile(out[near > 0], 99.5), 1.0) if (near > 0).any() else out
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return out, mask


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out", default="out")
    p.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                   help="crop to the chamber region, e.g. --crop 100 50 800 600")
    p.add_argument("--bg-stride", type=int, default=10)
    p.add_argument("--blur", type=float, default=1.2, help="gaussian sigma (px)")
    p.add_argument("--denoise", choices=["gauss", "nlm"], default="gauss")
    p.add_argument("--nlm-h", type=float, default=6.0)
    p.add_argument("--tophat", type=int, default=31, help="odd, > track width")
    p.add_argument("--low-k", type=float, default=2.5)
    p.add_argument("--high-k", type=float, default=5.0)
    p.add_argument("--min-area", type=int, default=25)
    p.add_argument("--min-len", type=int, default=15, help="min track extent (px)")
    p.add_argument("--save-frames", action="store_true")
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.save_frames:
        (out_dir / "frames").mkdir(exist_ok=True)

    print("Estimating static background...")
    bg = estimate_background(a.video, a.crop, a.bg_stride)
    cv2.imwrite(str(out_dir / "background.png"), bg.astype(np.uint8))

    cap = open_video(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h, w = bg.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_clean = cv2.VideoWriter(str(out_dir / "cleaned.mp4"), fourcc, fps, (w, h), False)
    w_mask = cv2.VideoWriter(str(out_dir / "mask.mp4"), fourcc, fps, (w, h), False)

    idx, preview_done = 0, False
    print("Processing frames...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = to_gray(frame, a.crop)
        cleaned, mask = clean_frame(gray, bg, a)
        w_clean.write(cleaned)
        w_mask.write(mask)
        if a.save_frames:
            cv2.imwrite(str(out_dir / "frames" / f"{idx:06d}.png"), cleaned)
        if not preview_done and mask.sum() > 0 and idx > 30:
            side = np.hstack([gray, cleaned, mask])
            cv2.imwrite(str(out_dir / "preview.png"), side)
            preview_done = True
        idx += 1
        if idx % 100 == 0:
            print(f"  {idx} frames")

    cap.release()
    w_clean.release()
    w_mask.release()
    print(f"Done: {idx} frames -> {out_dir}/")


if __name__ == "__main__":
    main()