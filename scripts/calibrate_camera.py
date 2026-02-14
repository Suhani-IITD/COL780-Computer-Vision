#!/usr/bin/env python3
import argparse
import os
import sys

try:
    import numpy as np
except ImportError:
    print("Error: numpy is not installed.")
    print("Install in WSL with: sudo apt-get install -y python3-numpy")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Error: python3-opencv is not installed.")
    print("Install in WSL with: sudo apt-get install -y python3-opencv")
    sys.exit(1)


def parse_pattern(value: str):
    parts = value.lower().split('x')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('Pattern must be like 9x6')
    try:
        cols = int(parts[0])
        rows = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Pattern values must be integers') from exc
    if cols <= 1 or rows <= 1:
        raise argparse.ArgumentTypeError('Pattern dimensions must be > 1')
    return cols, rows


def open_capture(source_text: str):
    if source_text.isdigit():
        return cv2.VideoCapture(int(source_text))
    return cv2.VideoCapture(source_text)


def main():
    parser = argparse.ArgumentParser(description='Interactive camera calibration utility for AR assignment')
    parser.add_argument('--source', default='0', help='Webcam index or video path (default: 0)')
    parser.add_argument('--pattern', type=parse_pattern, default='9x6', help='Chessboard inner-corner pattern COLSxROWS (default: 9x6)')
    parser.add_argument('--square-size', type=float, default=1.0, help='Physical square size (default: 1.0 arbitrary units)')
    parser.add_argument('--min-frames', type=int, default=20, help='Minimum captured boards required (default: 20)')
    parser.add_argument('--output', default='camera_intrinsics.yml', help='Output YAML path (default: camera_intrinsics.yml)')
    args = parser.parse_args()

    cols, rows = args.pattern
    pattern_size = (cols, rows)

    cap = open_capture(str(args.source))
    if not cap.isOpened():
        print(f'Error: could not open source: {args.source}')
        return 1

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = []
    imgpoints = []
    image_size = None

    print('Calibration capture started.')
    print('Keys: c = capture current chessboard, q = finish and calibrate')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ok, frame = cap.read()
        if not ok:
            print('Error: failed to read frame from source.')
            break

        image_size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        display = frame.copy()
        if found:
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, pattern_size, corners_sub, found)
        else:
            corners_sub = None

        cv2.putText(display, f'Captured: {len(objpoints)} / {args.min_frames}', (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, 'Press c to capture, q to finish', (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Calibration Capture', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if found and corners_sub is not None:
                objpoints.append(objp.copy())
                imgpoints.append(corners_sub)
                print(f'Captured frame {len(objpoints)}')
            else:
                print('No chessboard detected in current frame; not captured.')
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if image_size is None:
        print('Error: no frames processed.')
        return 1

    if len(objpoints) < args.min_frames:
        print(f'Error: only {len(objpoints)} valid captures. Need at least {args.min_frames}.')
        return 1

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    if not ret:
        print('Error: calibration failed.')
        return 1

    total_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], reprojected, cv2.NORM_L2)
        total_error += err * err
        total_points += len(objpoints[i])

    rms_reprojection = np.sqrt(total_error / max(total_points, 1))

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fs = cv2.FileStorage(args.output, cv2.FILE_STORAGE_WRITE)
    fs.write('K', K)
    fs.write('camera_matrix', K)
    fs.write('dist_coeffs', dist)
    fs.write('rms_reprojection_error', float(rms_reprojection))
    fs.write('image_width', int(image_size[0]))
    fs.write('image_height', int(image_size[1]))
    fs.release()

    print(f'Calibration saved to: {args.output}')
    print('K =')
    print(K)
    print(f'RMS reprojection error: {rms_reprojection:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
