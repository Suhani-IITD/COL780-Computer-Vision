# AR Assignment Implementation Plan (Status)

## Goal
Complete HW1 AR tasks using custom image-processing and geometry code (OpenCV only for I/O, display, and basic utilities).

## Phase 1: Build Stabilization and Constraint Enforcement
Status: Completed
1. Removed missing `test_functions` dependencies from `main.cpp` and `Makefile`.
2. Kept a single custom pipeline in runtime flow.
3. Replaced OpenCV contour filters with custom geometry checks used in runtime.
4. Verified clean build with current source set.

## Phase 2: Task 1 (Detection + ID)
Status: Completed
1. Implemented custom candidate extraction:
   - grayscale -> blur -> Sobel -> threshold -> custom contour extraction.
   - contour simplification + quadrilateral filtering + corner ordering.
2. Implemented custom tag normalization:
   - custom homography to canonical square.
   - custom inverse warp to orthographic view.
3. Implemented tag decoding:
   - orientation marker detection.
   - canonical rotation.
   - 4-bit ID decode from central 2x2.
4. Added overlays for corners, ID, and orientation label.

## Phase 3: Task 2 (2D Augmented Reality)
Status: Completed (manual validation pending)
1. Load template image once.
2. Compute template-to-tag homography for each detected tag.
3. Warp template onto frame with custom inverse mapping and mask compositing.
4. Manual validation remains for webcam and stress cases.

## Phase 4: Task 3 (3D Augmented Reality)
Status: Implemented (calibrated-intrinsics file still required for best results)
1. Added intrinsics loading from YAML (`K` or `camera_matrix`) with default fallback.
2. Implemented pose estimation from homography and intrinsics (R, t).
3. Implemented centered 3D cube projection on tag.
4. Added optional OBJ wireframe rendering and temporal pose smoothing (EMA).

## Phase 5: Validation and Deliverables
Status: In progress
1. Build + non-GUI smoke runs completed on provided videos in WSL.
2. Remaining: full GUI/manual validation on webcam and report capture for rubric evidence.
3. Remaining: calibrated intrinsics capture (`camera_intrinsics.yml`) for best Task 3 alignment.
