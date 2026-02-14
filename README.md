# COL780-Computer-Vision

Sources: https://drive.google.com/drive/folders/1HDRoBewJfYyAX2xlf77KgNo26_CotpuB

## Build
```bash
make clean && make
```

## Run Detector
```bash
# video
./ar_tag_detector Tag0.mp4

# webcam
./ar_tag_detector 0

# with template + intrinsics + obj
./ar_tag_detector Tag0.mp4 iitd_logo_template.jpg camera_intrinsics.yml wolf.obj
```

You can also use the helper runner:
```bash
# defaults: source=Tag0.mp4, intrinsics=camera_intrinsics.yml if it exists
./scripts/run_demo.sh

# custom source and extras
./scripts/run_demo.sh multipleTags.mp4 iitd_logo_template.jpg camera_intrinsics.yml wolf.obj
```

If you need to pass intrinsics/obj without template, the app supports `-` placeholder:
```bash
./ar_tag_detector Tag0.mp4 - camera_intrinsics.yml wolf.obj
```

## Camera Calibration Helper
Generate `camera_intrinsics.yml` with a chessboard:
```bash
python3 scripts/calibrate_camera.py --source 0 --pattern 9x6 --min-frames 20 --output camera_intrinsics.yml
```

Controls while running calibration:
- `c`: capture current detected chessboard
- `q`: finish and calibrate

If OpenCV Python bindings are missing in WSL:
```bash
sudo apt-get install -y python3-opencv python3-numpy
```
