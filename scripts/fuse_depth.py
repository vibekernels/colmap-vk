#!/usr/bin/env python3
"""Fuse COLMAP unfiltered photometric depth maps into a colored PLY point cloud.

COLMAP's built-in stereo_fusion requires filtered depth maps, which can be empty
for videos with limited translational parallax (e.g. head-mounted POV footage).
This script works directly with unfiltered photometric depth maps instead.

Usage: python3 fuse_depth.py <workspace_dir>

The workspace must contain:
  dense/sparse/              - undistorted sparse model (from image_undistorter)
  dense/images/              - undistorted images
  dense/stereo/depth_maps/   - photometric depth maps (from patch_match_stereo)

Output: <workspace_dir>/dense_cloud.ply
"""
import numpy as np
import pycolmap
import os
import sys
from PIL import Image


def read_colmap_depth(path):
    """Read a COLMAP binary depth map.

    Format: ASCII header "width&height&channels&" followed by float32 data.
    """
    with open(path, 'rb') as f:
        raw = f.read()
    hend = 0
    for i in range(min(30, len(raw))):
        if raw[i] not in range(32, 127):
            hend = i
            break
    parts = raw[:hend].decode('ascii').rstrip('&').split('&')
    w, h = int(parts[0]), int(parts[1])
    use = min(len(raw) - hend, w * h * 4)
    use = (use // 4) * 4
    d = np.frombuffer(raw[hend:hend + use], dtype=np.float32)
    if len(d) < w * h:
        d = np.pad(d, (0, w * h - len(d)))
    return d.reshape(h, w)


def main():
    ws = sys.argv[1]
    model = pycolmap.Reconstruction(f'{ws}/dense/sparse')
    cam = list(model.cameras.values())[0]
    fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]

    pts_list, col_list = [], []
    for _, img in sorted(model.images.items()):
        if not img.has_pose:
            continue
        dp = f'{ws}/dense/stereo/depth_maps/{img.name}.photometric.bin'
        if not os.path.exists(dp) or os.path.getsize(dp) == 0:
            continue

        depth = read_colmap_depth(dp)
        h, w = depth.shape
        sx, sy = w / cam.width, h / cam.height
        fxs, fys, cxs, cys = fx * sx, fy * sy, cx * sx, cy * sy

        rgb = np.array(Image.open(f'{ws}/dense/images/{img.name}').resize((w, h)))

        M = np.array(img.cam_from_world().matrix())
        R, t = M[:3, :3], M[:3, 3]

        ys, xs = np.where(depth > 0)
        ds = depth[depth > 0]

        # Filter outlier depths and subsample pixels
        med = np.median(ds)
        m = (ds < med * 3) & (ys % 3 == 0) & (xs % 3 == 0)
        ys, xs, ds = ys[m], xs[m], ds[m]

        # Backproject to camera coords, then transform to world coords
        pc = np.stack([(xs - cxs) / fxs * ds, (ys - cys) / fys * ds, ds], -1)
        pw = (pc - t) @ R
        pts_list.append(pw)
        col_list.append(rgb[ys, xs])

    if not pts_list:
        print("  0 points, 0.0 MB")
        sys.exit(0)

    pts = np.concatenate(pts_list).astype(np.float32)
    cols = np.concatenate(col_list).astype(np.uint8)

    # Remove spatial outliers
    c = np.median(pts, 0)
    dists = np.linalg.norm(pts - c, axis=1)
    m = dists < np.percentile(dists, 95)
    pts, cols = pts[m], cols[m]

    out = f'{ws}/dense_cloud.ply'
    with open(out, 'wb') as f:
        f.write((
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(pts)}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        ).encode())
        for i in range(len(pts)):
            f.write(pts[i].tobytes())
            f.write(cols[i].tobytes())

    print(f"  {len(pts):,} points, {os.path.getsize(out) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
