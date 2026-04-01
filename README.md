COLMAP
======

About
-----

COLMAP is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo
(MVS) pipeline with a graphical and command-line interface. It offers a wide
range of features for reconstruction of ordered and unordered image collections.
The software is licensed under the new BSD license.

The latest source code is available at https://github.com/colmap/colmap. COLMAP
builds on top of existing works and when using specific algorithms within
COLMAP, please also cite the original authors, as specified in the source code,
and consider citing relevant third-party dependencies (most notably
ceres-solver, poselib, sift-gpu, vlfeat).

Download
--------

* Binaries for **Windows** and other resources can be downloaded
  from https://github.com/colmap/colmap/releases.
* Binaries for **Linux/Unix/BSD** are available at
  https://repology.org/metapackage/colmap/versions.
* Pre-built **Docker** images are available at
  https://hub.docker.com/r/colmap/colmap.
* Conda packages are available at https://anaconda.org/conda-forge/colmap and
  can be installed with `conda install colmap`
* **Python bindings** are available at https://pypi.org/project/pycolmap.
  CUDA-enabled wheels are available at https://pypi.org/project/pycolmap-cuda12.
* To **build from source**, please see https://colmap.github.io/install.html.

Getting Started
---------------

1. Download pre-built binaries or build from source.
2. Download one of the provided [sample datasets](https://demuc.de/colmap/datasets/)
   or use your own images.
3. Use the **automatic reconstruction** to easily build models
   with a single click or command.

Fast Video-to-3D Pipeline
-------------------------

This fork includes a fast pipeline script that reconstructs a dense colored
point cloud from a video file in about 60 seconds on an NVIDIA RTX 5090.

### Quick Start

```bash
# Build COLMAP with CUDA
mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DGUI_ENABLED=OFF \
  -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=native
ninja

# Run the pipeline
export COLMAP=./build/src/colmap/exe/colmap
bash scripts/fast_pipeline.sh input_video.mov ./output
# Output: ./output/dense_cloud.ply
```

### Pipeline Steps

The script runs 7 steps end-to-end:

| Step | What it does | GPU? |
|------|-------------|------|
| 1. Frame extraction | Extracts frames from video at 2fps, scaled to 1000px | No (ffmpeg) |
| 2. Feature extraction | SIFT features at 500px (2048 per image) | Yes |
| 3. Sequential matching | Matches adjacent frames (overlap=5) | Yes |
| 4. Global mapper | GLOMAP global SfM (solves all poses at once) | No |
| 5. Undistortion | Removes lens distortion for dense stereo | No |
| 6. PatchMatch stereo | Estimates per-pixel depth maps | Yes |
| 7. Depth fusion | Backprojects depth maps to 3D point cloud | No (Python) |

### Benchmark (61s iPhone video, 1920x1080, RTX 5090)

| Step | Time |
|------|------|
| Frame extraction | 3s |
| Feature extraction | 8s |
| Sequential matching | 11s |
| Global mapper (GLOMAP) | 7s |
| Undistortion | 3s |
| PatchMatch stereo | 26s |
| Depth fusion | 5s |
| **Total** | **~64s** |

Result: 122/122 frames registered, ~3.8M point dense cloud at 1000px with color.

### Tuning

Edit the variables at the top of `scripts/fast_pipeline.sh`:

- **`FPS`** (default: 2) — Frame extraction rate. Higher = more frames = better
  quality but slower. 1-3 is good for handheld video.
- **`MAX_IMG`** (default: 1000) — Max image dimension for dense reconstruction.
  Directly affects PatchMatch speed and depth map resolution.
- **`FEAT_IMG`** (default: 500) — Max image dimension for feature extraction.
  Kept lower than `MAX_IMG` since SfM doesn't need full resolution.
- **`MAX_FEAT`** (default: 2048) — Max SIFT features per image. More features =
  better matching but slower extraction/matching.
- **`OVERLAP`** (default: 5) — Sequential matching window. How many adjacent
  frames to match against. Higher = more robust but slower matching.

The PatchMatch parameters in the script are set aggressively for speed
(`num_iterations=1`, `window_radius=3`). Increase these for better depth quality
at the cost of speed.

### Why Custom Depth Fusion?

COLMAP's built-in `stereo_fusion` requires geometrically consistent (filtered)
depth maps. For videos with limited translational parallax (e.g. head-mounted
POV footage, mostly rotational camera motion), COLMAP's depth filtering removes
nearly all depth estimates, producing empty results.

The included `scripts/fuse_depth.py` works with raw unfiltered photometric depth
maps instead. It backprojects each depth map to 3D using camera poses, applies
basic outlier removal (3x median depth filter + 95th percentile spatial filter),
and writes a binary PLY file.

### Why GLOMAP Instead of Incremental Mapper?

The incremental mapper (`colmap mapper`) adds images one at a time and runs
bundle adjustment after each batch. For 1800+ video frames, this takes over an
hour, with bundle adjustment dominating the runtime.

GLOMAP (`colmap global_mapper`) solves all camera rotations and translations
globally in a single pass, then runs bundle adjustment once. This is
dramatically faster for video input (13s vs 67+ minutes on 122 frames).

### Requirements

- COLMAP built with CUDA support
- ffmpeg (for frame extraction)
- Python 3 with `numpy`, `pycolmap`, `Pillow` (for depth fusion)

Documentation
-------------

The documentation is available [here](https://colmap.github.io/).

To build and update the documentation at the documentation website,
follow [these steps](https://colmap.github.io/install.html#documentation).

Support
-------

Please, use [GitHub Discussions](https://github.com/colmap/colmap/discussions)
for questions and the [GitHub issue tracker](https://github.com/colmap/colmap)
for bug reports, feature requests/additions, etc.

Acknowledgments
---------------

COLMAP was originally written by [Johannes Schönberger](https://demuc.de/) with
funding provided by his PhD advisors Jan-Michael Frahm and Marc Pollefeys.
The team of core project maintainers currently includes
[Johannes Schönberger](https://github.com/ahojnnes),
[Paul-Edouard Sarlin](https://github.com/sarlinpe),
[Shaohui Liu](https://github.com/B1ueber2y), and
[Linfei Pan](https://lpanaf.github.io/).

The Python bindings in PyCOLMAP were originally added by
[Mihai Dusmanu](https://github.com/mihaidusmanu),
[Philipp Lindenberger](https://github.com/Phil26AT), and
[Paul-Edouard Sarlin](https://github.com/sarlinpe).

The project has also benefitted from countless community contributions, including
bug fixes, improvements, new features, third-party tooling, and community
support (special credits to [Torsten Sattler](https://tsattler.github.io)).

Citation
--------

If you use this project for your research, please cite:

    @inproceedings{schoenberger2016sfm,
        author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
        title={Structure-from-Motion Revisited},
        booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
    }

    @inproceedings{schoenberger2016mvs,
        author={Sch\"{o}nberger, Johannes Lutz and Zheng, Enliang and Pollefeys, Marc and Frahm, Jan-Michael},
        title={Pixelwise View Selection for Unstructured Multi-View Stereo},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2016},
    }

If you use the global SfM pipeline (GLOMAP), please cite:

    @inproceedings{pan2024glomap,
        author={Pan, Linfei and Barath, Daniel and Pollefeys, Marc and Sch\"{o}nberger, Johannes Lutz},
        title={{Global Structure-from-Motion Revisited}},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2024},
    }

If you use the image retrieval / vocabulary tree engine, please cite:

    @inproceedings{schoenberger2016vote,
        author={Sch\"{o}nberger, Johannes Lutz and Price, True and Sattler, Torsten and Frahm, Jan-Michael and Pollefeys, Marc},
        title={A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval},
        booktitle={Asian Conference on Computer Vision (ACCV)},
        year={2016},
    }

Contribution
------------

Contributions (bug reports, bug fixes, improvements, etc.) are very welcome and
should be submitted in the form of new issues and/or pull requests on GitHub.

License
-------

The COLMAP library is licensed under the new BSD license. Note that this text
refers only to the license for COLMAP itself, independent of its thirdparty
dependencies, which are separately licensed. Building COLMAP with these
dependencies may affect the resulting COLMAP license.

    Copyright (c), ETH Zurich and UNC Chapel Hill.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.

        * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
          its contributors may be used to endorse or promote products derived
          from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
