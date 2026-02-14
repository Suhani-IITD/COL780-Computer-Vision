# Repository Guidelines

## Project Structure & Module Organization
This repository contains a compact C++ Computer Vision assignment implementation.
- `main.cpp`: executable entry point, video loop, and AR-tag pipeline orchestration.
- `task1.cpp`: core image-processing and geometry routines (blur, threshold, contour handling, homography, warping).
- `image_processing.h`: shared declarations for custom processing utilities.
- `A1/README.md`: assignment-specific notes.
- `HW1_Geometry_Assignment.pdf`: problem statement/reference document.

Keep new algorithm code in `task1.cpp` and expose only reusable interfaces in `image_processing.h`.

## Build, Test, and Development Commands
- `make all`: builds `ar_tag_detector.exe`.
- `make run`: runs `clean`, rebuilds, then executes with `Tag0.mp4`.
- `make clean`: removes objects and executable.

Example local build command (useful when adapting to Linux/macOS OpenCV installs):
```bash
g++ -std=c++11 -Wall main.cpp task1.cpp -o ar_tag_detector $(pkg-config --cflags --libs opencv4)
```

## Coding Style & Naming Conventions
- Language target: C++11.
- Indentation: 4 spaces; keep brace style consistent with existing files.
- Naming: functions use `snake_case` (e.g., `custom_blur_separable`); type aliases use `PascalCase` (e.g., `Contour`).
- Prefer `const` references for large inputs and explicit `cv::` namespace usage.
- Keep debug prints temporary; remove noisy per-frame logs before merging.

## Assignment Constraint (Important)
- Do **not** use OpenCV image-processing primitives for core logic.
- Implement these operations manually in project code: grayscale conversion, Gaussian blur, thresholding, contour detection, homography estimation, perspective warping, and similar pipeline steps.
- OpenCV is allowed only for utility tasks such as media I/O (`imread`, `VideoCapture`, `imshow`, `imwrite`), matrix containers, and drawing/visualization for debugging.
- When adding new functionality, prefer extending `task1.cpp` + `image_processing.h` with custom implementations instead of calling high-level `cv::` processing APIs.

## Testing Guidelines
There is no formal automated test suite yet. Validate changes by:
- Building cleanly with `-Wall`.
- Running on sample videos and checking contour overlays, corner ordering, and warp output windows.
- Comparing custom outputs against OpenCV-based paths where both exist.

If you add test helpers (for example `test_functions.cpp/.h`), ensure `Makefile` sources stay in sync.

## Commit & Pull Request Guidelines
Current history favors short, imperative commit messages (e.g., `Add sources link to README`).
- Use: `Verb + scope` style (`Fix corner sorting for convex quads`).
- Keep commits focused and logically atomic.

PRs should include:
- What changed and why.
- How it was validated (commands + input video used).
- Visual evidence for CV changes (screenshots/frame captures when relevant).

## Configuration Notes
`Makefile` OpenCV paths are currently Windows/MSYS-style (`C:/msys64/mingw64`). Update `OPENCV_DIR`, include paths, and libraries to match your local environment before building.
