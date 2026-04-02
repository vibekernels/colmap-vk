#!/usr/bin/env python3
"""Fuse monocular depth maps into a colored PLY point cloud.

For videos with limited translational parallax (e.g. head-mounted POV footage),
COLMAP's stereo depth maps are noise. This script uses Depth Anything V2 for
per-image depth estimation, aligns predictions to COLMAP's metric scale, and
keeps only points where multiple frames agree on the 3D position.

Usage: python3 fuse_depth.py <workspace_dir>

The workspace must contain:
  dense/sparse/   - undistorted sparse model (from image_undistorter)
  dense/images/   - undistorted images

Output: <workspace_dir>/dense_cloud.ply
"""
import numpy as np
import pycolmap
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import least_squares


def load_depth_model():
    """Load Depth Anything V2 model (raw float output)."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = 'depth-anything/Depth-Anything-V2-Small-hf'
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForDepthEstimation.from_pretrained(name).to('cuda').eval()
    return processor, model


@torch.no_grad()
def predict_depth(processor, model, image_path, target_h, target_w):
    """Get raw float depth prediction, resized to target dimensions."""
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors='pt').to('cuda')
    pred = model(**inputs).predicted_depth
    pred = F.interpolate(pred.unsqueeze(0), size=(target_h, target_w),
                         mode='bilinear', align_corners=False)
    return pred.squeeze().cpu().numpy()


def align_depth_per_frame(mono, sparse_2d, sparse_metric, h, w):
    """Align mono depth to metric using shifted inverse: metric = a / (mono + b).

    Falls back to simpler models if optimization fails.
    Returns aligned metric depth map, or None on failure.
    """
    pairs = []
    for (px, py), md in zip(sparse_2d, sparse_metric):
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < w and 0 <= iy < h:
            mv = mono[iy, ix]
            if mv > 1e-6 and md > 0:
                pairs.append((mv, md))

    if len(pairs) < 3:
        return None

    mono_v = np.array([p[0] for p in pairs])
    met_v = np.array([p[1] for p in pairs])

    # Fit: metric = a / (mono + b) using robust least squares
    def residual(params, m, t):
        a, b = params
        pred = a / (m + b)
        return (pred - t) / t

    try:
        r = least_squares(residual, [np.median(met_v) * np.median(mono_v), 1.0],
                          args=(mono_v, met_v), method='lm', max_nfev=200)
        a, b = r.x
        pred = a / (mono_v + b)
        rel_err = np.median(np.abs(pred - met_v) / met_v)
    except Exception:
        rel_err = 1.0

    if rel_err < 0.3 and a > 0:
        metric = a / (mono + b)
        metric[mono + b <= 0] = 0
        metric[metric <= 0] = 0
        return metric

    # Fallback: linear fit with RANSAC
    best_inliers = 0
    best_a, best_b = None, None
    rng = np.random.RandomState(42)
    n = len(mono_v)

    for _ in range(100):
        idx = rng.choice(n, 2, replace=False)
        A = np.stack([mono_v[idx], np.ones(2)], axis=1)
        try:
            params = np.linalg.solve(A, met_v[idx])
        except np.linalg.LinAlgError:
            continue
        pred = params[0] * mono_v + params[1]
        rel_err = np.abs(pred - met_v) / met_v
        inliers = np.sum(rel_err < 0.25)
        if inliers > best_inliers:
            best_inliers = inliers
            best_a, best_b = params

    if best_a is None:
        return None

    metric = best_a * mono + best_b
    metric[metric <= 0] = 0
    return metric


def main():
    ws = sys.argv[1]
    rec = pycolmap.Reconstruction(f'{ws}/dense/sparse')
    cam = list(rec.cameras.values())[0]
    fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]

    # Build sparse point visibility per image
    image_sparse = {}
    for pt3d_id, pt3d in rec.points3D.items():
        for elem in pt3d.track.elements:
            img_id = elem.image_id
            if img_id not in rec.images:
                continue
            img = rec.images[img_id]
            if not img.has_pose:
                continue
            M = np.array(img.cam_from_world().matrix())
            R, t = M[:3, :3], M[:3, 3]
            pc = R @ pt3d.xyz + t
            if pc[2] <= 0:
                continue
            px = fx * pc[0] / pc[2] + cx
            py = fy * pc[1] / pc[2] + cy
            if img_id not in image_sparse:
                image_sparse[img_id] = []
            image_sparse[img_id].append((px, py, pc[2]))

    print("  Loading depth model...")
    processor, model = load_depth_model()

    img_items = sorted(rec.images.items())
    sample_img = None
    for img_id, img in img_items:
        if img.has_pose:
            ip = f'{ws}/dense/images/{img.name}'
            if os.path.exists(ip):
                sample_img = np.array(Image.open(ip))
                break
    h_img, w_img = sample_img.shape[:2]
    sx, sy = w_img / cam.width, h_img / cam.height
    fxs, fys, cxs, cys = fx * sx, fy * sy, cx * sx, cy * sy

    # Phase 1: Generate per-frame point clouds with per-frame alignment
    print("  Phase 1: Predicting and aligning depth maps...")
    frame_clouds = []  # list of (points_world, colors) per frame

    for idx, (img_id, img) in enumerate(img_items):
        if not img.has_pose:
            continue
        ip = f'{ws}/dense/images/{img.name}'
        if not os.path.exists(ip):
            continue

        mono = predict_depth(processor, model, ip, h_img, w_img)

        sparse_data = image_sparse.get(img_id, [])
        pts_2d = [(p[0] * sx, p[1] * sy) for p in sparse_data]
        metric_depths = [p[2] for p in sparse_data]

        metric_depth = align_depth_per_frame(mono, pts_2d, metric_depths, h_img, w_img)
        if metric_depth is None:
            continue

        rgb = np.array(Image.open(ip))
        if rgb.shape[0] != h_img or rgb.shape[1] != w_img:
            rgb = np.array(Image.open(ip).resize((w_img, h_img)))

        M = np.array(img.cam_from_world().matrix())
        R, t = M[:3, :3], M[:3, 3]

        # Subsample every 2nd pixel for multi-view checking
        ys, xs = np.where(metric_depth > 0)
        m = (ys % 2 == 0) & (xs % 2 == 0)
        ys, xs = ys[m], xs[m]
        ds = metric_depth[ys, xs]

        med = np.median(ds)
        m = (ds > med * 0.1) & (ds < med * 3)
        ys, xs, ds = ys[m], xs[m], ds[m]

        if len(ds) == 0:
            continue

        pc = np.stack([(xs - cxs) / fxs * ds, (ys - cys) / fys * ds, ds], -1)
        pw = (pc - t) @ R
        frame_clouds.append((pw.astype(np.float32), rgb[ys, xs].astype(np.uint8)))

        if (idx + 1) % 20 == 0:
            print(f"    [{idx+1}/{len(img_items)}] {len(ds):,} pts")

    print(f"  {len(frame_clouds)} frames processed")

    # Phase 2: Multi-view consistency via voxel voting
    # Points that appear in multiple frames at the same location are likely real surfaces
    print("  Phase 2: Multi-view consistency filtering...")

    # Determine voxel size from scene scale
    all_pts_sample = np.concatenate([fc[0][::10] for fc in frame_clouds])
    c = np.median(all_pts_sample, 0)
    scene_radius = np.percentile(np.linalg.norm(all_pts_sample - c, axis=1), 90)
    voxel_size = scene_radius / 150  # ~150 voxels across scene
    print(f"    Scene radius: {scene_radius:.2f}, voxel size: {voxel_size:.4f}")

    # Count how many frames contribute to each voxel
    voxel_counts = {}
    voxel_points = {}  # voxel_key -> (accumulated_xyz, accumulated_rgb, count)

    for fi, (pts, cols) in enumerate(frame_clouds):
        vkeys = np.floor(pts / voxel_size).astype(np.int32)
        for i in range(0, len(pts), 3):  # further subsample for voting speed
            k = (vkeys[i, 0], vkeys[i, 1], vkeys[i, 2])
            if k not in voxel_counts:
                voxel_counts[k] = set()
                voxel_points[k] = [np.zeros(3, dtype=np.float64),
                                   np.zeros(3, dtype=np.float64), 0]
            voxel_counts[k].add(fi)
            voxel_points[k][0] += pts[i]
            voxel_points[k][1] += cols[i].astype(np.float64)
            voxel_points[k][2] += 1

    # Keep voxels seen by >= min_views frames
    min_views = 3
    good_voxels = {k for k, v in voxel_counts.items() if len(v) >= min_views}
    print(f"    Voxels: {len(voxel_counts)} total, {len(good_voxels)} with >= {min_views} views")

    # Build final point cloud from good voxels (use averaged position)
    pts_final = np.empty((len(good_voxels), 3), dtype=np.float32)
    cols_final = np.empty((len(good_voxels), 3), dtype=np.uint8)
    for i, k in enumerate(good_voxels):
        info = voxel_points[k]
        pts_final[i] = (info[0] / info[2]).astype(np.float32)
        cols_final[i] = np.clip(info[1] / info[2], 0, 255).astype(np.uint8)

    print(f"  Multi-view filtered: {len(pts_final):,} points")

    # Remove spatial outliers
    c = np.median(pts_final, 0)
    dists = np.linalg.norm(pts_final - c, axis=1)
    m = dists < np.percentile(dists, 95)
    pts_final, cols_final = pts_final[m], cols_final[m]

    out = f'{ws}/dense_cloud.ply'
    with open(out, 'wb') as f:
        f.write((
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(pts_final)}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        ).encode())
        packed = np.empty(len(pts_final), dtype=[('xyz', '<f4', 3), ('rgb', 'u1', 3)])
        packed['xyz'] = pts_final
        packed['rgb'] = cols_final
        f.write(packed.tobytes())

    print(f"  {len(pts_final):,} points, {os.path.getsize(out) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
