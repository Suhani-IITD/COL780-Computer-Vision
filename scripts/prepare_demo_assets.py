#!/usr/bin/env python3
import os
import cv2
import numpy as np

ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "iitd_logo_template.jpg")
OBJ_PATH = os.path.join(ASSETS_DIR, "wolf.obj")


def make_logo(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.full((800, 800, 3), 245, dtype=np.uint8)

    maroon = (60, 30, 140)  # BGR
    gold = (40, 180, 220)
    black = (30, 30, 30)

    cv2.circle(img, (400, 320), 230, maroon, -1)
    cv2.circle(img, (400, 320), 200, (245, 245, 245), -1)
    cv2.circle(img, (400, 320), 170, maroon, 6)

    # Simple geometric motif for center emblem-like shape
    pts = np.array([[400, 210], [510, 350], [400, 520], [290, 350]], np.int32)
    cv2.fillConvexPoly(img, pts, gold)
    cv2.circle(img, (400, 350), 55, maroon, -1)
    cv2.circle(img, (400, 350), 32, gold, -1)

    cv2.putText(img, "IITD", (300, 710), cv2.FONT_HERSHEY_SIMPLEX, 2.1, maroon, 5, cv2.LINE_AA)
    cv2.putText(img, "Indian Institute of Technology Delhi", (65, 765),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, black, 2, cv2.LINE_AA)

    cv2.imwrite(path, img)


def make_wolf_obj(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Lightweight stylized low-poly wolf head (placeholder demo mesh)
    vertices = [
        (-0.30, 0.00, 0.45),  # 1 nose L
        (0.30, 0.00, 0.45),   # 2 nose R
        (0.00, 0.12, 0.58),   # 3 nose top
        (-0.42, 0.25, 0.20),  # 4 cheek L
        (0.42, 0.25, 0.20),   # 5 cheek R
        (-0.36, 0.55, -0.02), # 6 eye brow L
        (0.36, 0.55, -0.02),  # 7 eye brow R
        (-0.20, 0.88, -0.16), # 8 ear base L
        (0.20, 0.88, -0.16),  # 9 ear base R
        (-0.35, 1.12, -0.20), # 10 ear tip L
        (0.35, 1.12, -0.20),  # 11 ear tip R
        (0.00, 0.72, -0.28),  # 12 head top
        (-0.24, 0.25, -0.32), # 13 jaw back L
        (0.24, 0.25, -0.32),  # 14 jaw back R
        (0.00, -0.08, 0.08),  # 15 chin
    ]

    faces = [
        (1, 2, 3),
        (1, 3, 4), (2, 5, 3),
        (4, 3, 6), (5, 7, 3),
        (6, 7, 12),
        (6, 8, 12), (7, 12, 9),
        (8, 10, 12), (9, 12, 11),
        (4, 6, 13), (5, 14, 7),
        (4, 13, 15), (5, 15, 14),
        (1, 4, 15), (2, 15, 5),
        (13, 12, 14),
        (6, 12, 13), (7, 14, 12),
        (8, 9, 12),
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Placeholder low-poly wolf head OBJ for AR demo\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")


def main():
    make_logo(LOGO_PATH)
    make_wolf_obj(OBJ_PATH)
    print(f"Generated: {LOGO_PATH}")
    print(f"Generated: {OBJ_PATH}")


if __name__ == "__main__":
    main()
