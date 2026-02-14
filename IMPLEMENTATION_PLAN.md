# AR Assignment Implementation Plan

## Goal
Complete HW1 AR tasks using custom image-processing and geometry code (OpenCV only for I/O, display, and basic utilities).

## Phase 1: Build Stabilization and Constraint Enforcement
1. Remove missing `test_functions` dependencies from `main.cpp` and `Makefile`.
2. Keep a single custom pipeline in runtime flow.
3. Replace OpenCV contour filters (`isContourConvex`, `contourArea`, OpenCV contour drawing calls) with custom equivalents where needed.
4. Ensure project compiles cleanly with current source set.

## Phase 2: Task 1 (Detection + ID)
1. Robust candidate extraction:
   - grayscale -> blur -> Sobel/edge -> threshold -> custom contour extraction.
   - contour simplification + quadrilateral filtering + corner ordering.
2. Tag normalization:
   - custom homography to canonical square.
   - custom warp to orthographic top view.
3. Tag decoding:
   - detect orientation marker.
   - rotate to canonical orientation.
   - decode 4-bit ID from inner grid.
4. Draw result overlays (corners + decoded ID on frame).

## Phase 3: Task 2 (2D Augmented Reality)
1. Load template image once.
2. Compute template-to-tag homography for each detected tag.
3. Warp template onto frame with masking and proper orientation.
4. Validate multi-tag and varying-angle behavior.

## Phase 4: Task 3 (3D Augmented Reality)
1. Add camera calibration workflow and store intrinsic matrix.
2. Estimate pose from homography and intrinsics (R, t).
3. Project 3D cube onto tag.
4. Optional bonus: OBJ rendering pipeline and temporal smoothing.

## Phase 5: Validation and Deliverables
1. Run on all provided videos and collect outputs for each task.
2. Document assumptions, parameter settings, and failure cases.
3. Prepare concise demo/report checklist aligned with rubric.
