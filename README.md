# COL780 Computer Vision - AR Tag Assignment

Source bundle: https://drive.google.com/drive/folders/1HDRoBewJfYyAX2xlf77KgNo26_CotpuB

Beginner-friendly code walkthrough (Windows): `docs/BEGINNER_GUIDE_WINDOWS.md`

## Solution Overview
This implementation solves the AR assignment using a custom image-processing pipeline, with OpenCV used only for I/O, display, and matrix utilities.

Implemented pipeline:
- Frame preprocessing: custom grayscale, Gaussian blur, Sobel, threshold.
- Tag detection: custom contour tracing, polygon simplification, quad filtering, corner ordering.
- Tag decoding: custom homography + inverse warping to canonical view, orientation marker detection, 4-bit ID extraction.
- Task 2 AR overlay: template image projected onto detected tag using custom homography/warp.
- Task 3 AR rendering: pose from homography + intrinsics, centered cube rendering, optional OBJ wireframe rendering, temporal pose smoothing.

Core files:
- `main.cpp`: runtime pipeline, overlay logic, 3D projection.
- `task1.cpp`, `image_processing.h`: custom CV/geometry functions.
- `scripts/calibrate_camera.py`: camera intrinsics generator.
- `scripts/run_demo.sh`: one-command build and run helper.
- `scripts/prepare_demo_assets.py`: generates local placeholder template/OBJ assets.

## Setup (Linux/WSL)
1. Verify environment and missing packages:
```bash
./scripts/setup_linux_deps.sh
```
2. Auto-install missing dependencies (Ubuntu/Debian):
```bash
./scripts/setup_linux_deps.sh --install
```
3. Build:
```bash
make clean && make
```
4. (Optional) Generate calibration file:
```bash
python3 scripts/calibrate_camera.py --source 0 --pattern 9x6 --min-frames 20 --output camera_intrinsics.yml
```

## Setup (Windows Native)
This project can run on Windows using MSYS2 (UCRT64 toolchain).

1. Verify dependencies:
```powershell
.\scripts\setup_windows_deps.ps1
```
2. Install missing dependencies:
```powershell
.\scripts\setup_windows_deps.ps1 -Install
```
3. Build:
```powershell
.\scripts\build_windows.ps1
```
4. Intrinsics file:
- `camera_intrinsics.yml` is now included in repo as an approximate default (generated from video resolution).
- For better Task 3 pose quality, generate a real calibration file with a chessboard:
```bash
python3 scripts/calibrate_camera.py --source 0 --pattern 9x6 --min-frames 20 --output camera_intrinsics.yml
```

## How To Test
Run provided videos:
```bash
./ar_tag_detector Tag0.mp4
./ar_tag_detector multipleTags.mp4
```

Run with template + calibrated intrinsics + OBJ:
```bash
./ar_tag_detector Tag0.mp4 iitd_logo_template.jpg camera_intrinsics.yml wolf.obj
```

Use helper runner:
```bash
./scripts/run_demo.sh
./scripts/run_demo.sh multipleTags.mp4 iitd_logo_template.jpg camera_intrinsics.yml wolf.obj
```

Webcam test:
```bash
./ar_tag_detector 0
```

Windows test equivalents:
```powershell
.\scripts\test_windows.ps1                      # default mode=task1 (tags + IDs only)
.\scripts\test_windows.ps1 -Source multipleTags.mp4 -Mode task1
```

Debug mode (recommended when tags are not visible):
```powershell
.\scripts\test_windows.ps1 -Source Tag0.mp4 -Mode task1 -DebugDir debug_out
```

This writes per-frame diagnostics to `debug_out`:
- `frame_*_gray.png`
- `frame_*_blurred.png`
- `frame_*_sobel.png`
- `frame_*_binary.png`
- `frame_*_contours.png`
- `frame_*_warped_tag_*.png`
- `frame_*_warped_binary_*.png`
- `frame_*_final.png`

How to use debug outputs:
- If `binary` is mostly black/noisy white, thresholding is the issue.
- If `contours` has no clear tag shapes, contour extraction/filtering is failing.
- If `warped_binary` is unclear, homography/quad corner quality is weak.

Optional full test (only if these files exist locally):
```powershell
.\scripts\test_windows.ps1 -Source Tag0.mp4 -Template iitd_logo_template.jpg -Intrinsics camera_intrinsics.yml -Obj wolf.obj
```

Built-in placeholder assets (already generated in this repo):
- `assets/iitd_logo_template.jpg`
- `assets/wolf.obj`

Run with placeholders:
```powershell
.\scripts\test_windows.ps1 -Source Tag0.mp4 -Template assets\iitd_logo_template.jpg -Intrinsics camera_intrinsics.yml -Obj assets\wolf.obj -Mode all
```

To regenerate placeholders:
```bash
python3 scripts/prepare_demo_assets.py
```

Expected checks:
- Tag boundary and corner markers are stable.
- Displayed ID/orientation is consistent as tag rotates.
- Template is correctly aligned with tag orientation.
- 3D cube/OBJ remains anchored with minimal flicker.
