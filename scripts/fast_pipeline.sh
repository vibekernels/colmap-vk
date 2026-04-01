#!/bin/bash
# Fast video-to-3D pipeline using COLMAP with GLOMAP (global SfM).
# Produces a dense colored point cloud from a video in ~60 seconds on an RTX 5090.
#
# Usage: ./scripts/fast_pipeline.sh <video_path> [workspace_dir]
#
# Requirements: ffmpeg, COLMAP (built with CUDA), python3 with numpy/pycolmap/Pillow
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLMAP="${COLMAP:-colmap}"
INPUT_VIDEO="$1"
WS="${2:-./colmap_workspace}"

if [ -z "$INPUT_VIDEO" ]; then
  echo "Usage: $0 <video_path> [workspace_dir]"
  exit 1
fi

# Tunable parameters
FPS=2         # Frame extraction rate (lower = faster, fewer frames)
MAX_IMG=500   # Max image dimension in pixels
MAX_FEAT=2048 # Max SIFT features per image
OVERLAP=5     # Sequential matching overlap window

rm -rf "$WS"
mkdir -p "$WS/images"

TOTAL_START=$(date +%s%N)

# 1. Extract frames
echo "=== Extracting frames at ${FPS}fps ==="
S=$(date +%s%N)
ffmpeg -i "$INPUT_VIDEO" -vf "fps=$FPS,scale=$MAX_IMG:-2" -q:v 4 "$WS/images/frame_%05d.jpg" -loglevel warning 2>&1
E=$(date +%s%N); echo "  Frames: $(ls $WS/images/*.jpg | wc -l), Time: $(( (E-S)/1000000 ))ms"

# 2. Feature extraction (GPU SIFT)
echo "=== Feature extraction ==="
S=$(date +%s%N)
$COLMAP feature_extractor \
  --database_path "$WS/database.db" \
  --image_path "$WS/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --FeatureExtraction.use_gpu 1 \
  --FeatureExtraction.max_image_size $MAX_IMG \
  --SiftExtraction.max_num_features $MAX_FEAT \
  2>&1 | tail -1
E=$(date +%s%N); echo "  Time: $(( (E-S)/1000000 ))ms"

# 3. Sequential matching (GPU)
echo "=== Sequential matching ==="
S=$(date +%s%N)
$COLMAP sequential_matcher \
  --database_path "$WS/database.db" \
  --FeatureMatching.use_gpu 1 \
  --SequentialMatching.overlap $OVERLAP \
  --SequentialMatching.loop_detection 0 \
  2>&1 | tail -1
E=$(date +%s%N); echo "  Time: $(( (E-S)/1000000 ))ms"

# 4. Global mapper (GLOMAP) — much faster than incremental for video
echo "=== Global mapper ==="
S=$(date +%s%N)
mkdir -p "$WS/sparse"
$COLMAP global_mapper \
  --database_path "$WS/database.db" \
  --image_path "$WS/images" \
  --output_path "$WS/sparse" \
  2>&1 | tail -1
E=$(date +%s%N); echo "  Time: $(( (E-S)/1000000 ))ms"

# Fallback to incremental mapper if global fails
if [ ! -f "$WS/sparse/0/images.bin" ]; then
  echo "Global mapper failed, trying incremental..."
  $COLMAP mapper \
    --database_path "$WS/database.db" \
    --image_path "$WS/images" \
    --output_path "$WS/sparse" \
    --Mapper.ba_global_max_num_iterations 10 \
    --Mapper.ba_global_max_refinements 1 \
    2>&1 | tail -3
fi

# 5. Undistort images
echo "=== Undistortion ==="
S=$(date +%s%N)
$COLMAP image_undistorter \
  --image_path "$WS/images" \
  --input_path "$WS/sparse/0" \
  --output_path "$WS/dense" \
  --output_type COLMAP \
  --max_image_size $MAX_IMG \
  2>&1 | tail -1
E=$(date +%s%N); echo "  Time: $(( (E-S)/1000000 ))ms"

# 6. PatchMatch stereo (fast settings, GPU)
echo "=== PatchMatch stereo ==="
S=$(date +%s%N)
$COLMAP patch_match_stereo \
  --workspace_path "$WS/dense" \
  --workspace_format COLMAP \
  --PatchMatchStereo.max_image_size $MAX_IMG \
  --PatchMatchStereo.window_radius 3 \
  --PatchMatchStereo.num_samples 7 \
  --PatchMatchStereo.num_iterations 1 \
  --PatchMatchStereo.geom_consistency false \
  --PatchMatchStereo.filter false \
  2>&1 | tail -1
E=$(date +%s%N); echo "  Time: $(( (E-S)/1000000 ))ms"

# 7. Depth map fusion (custom, handles unfiltered depth maps)
echo "=== Depth fusion ==="
S=$(date +%s%N)
python3 "$SCRIPT_DIR/fuse_depth.py" "$WS"
E=$(date +%s%N); echo "  Time: $(( (E-S)/1000000 ))ms"

TOTAL_END=$(date +%s%N)
echo ""
echo "=== TOTAL: $(( (TOTAL_END-TOTAL_START)/1000000 ))ms ==="
echo "Output: $WS/dense_cloud.ply"
