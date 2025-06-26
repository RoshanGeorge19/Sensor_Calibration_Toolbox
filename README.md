# GMIND Sensor Calibration Toolbox

This repository provides a comprehensive set of tools for calibrating cameras and LiDAR sensors, including both intrinsic and extrinsic calibration workflows. It is designed for use with the GMIND SDK and supports robust, user-friendly calibration via both command-line scripts and a PyQt GUI.

## Folder Overview

- **camera_intrinsics/**
  - Tools for calibrating camera intrinsic parameters (focal length, principal point, distortion) using checkerboard images and OpenCV.
  - Includes a `CameraModel` utility class for 2D/3D mapping and projection.
- **lidar_camera_extrinsics/**
  - PyQt-based GUI for interactive LiDAR-camera extrinsic calibration (rotation and translation between sensors).
  - Supports robust calibration, outlier analysis, visualization, and point cloud projection.

## Quick Start

### 1. Camera Intrinsic Calibration

1. Place checkerboard images in a folder.
2. Edit `camera_intrinsics/opencv_checkerboard_calibrate.py` to set your image directory, checkerboard size, and square size.
3. Run:
   ```bash
   python camera_intrinsics/opencv_checkerboard_calibrate.py
   ```
4. Results are saved to `GMIND_Calibration.json`.

### 2. LiDAR-Camera Extrinsic Calibration

1. Run the GUI:
   ```bash
   python lidar_camera_extrinsics/main.py
   ```
2. Load an image, collect 2D-3D point pairs, set intrinsics, and run calibration.
3. Use advanced tools for robust fitting, diagnostics, and visualization.

## Requirements

- Python 3.7+
- PyQt5 (for GUI)
- OpenCV (`cv2`)
- NumPy
- SciPy

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## File Highlights

- `camera_intrinsics/opencv_checkerboard_calibrate.py` — Intrinsic calibration script
- `camera_intrinsics/camera.py` — CameraModel utility class
- `lidar_camera_extrinsics/main.py` — Main GUI for extrinsic calibration
- `lidar_camera_extrinsics/extrinsic_calibration.py` — Core calibration routines

## Documentation

See the `README.md` in each subfolder for detailed instructions and workflow examples.

## Notes

- For best results, use high-quality, well-distributed calibration images and accurate point correspondences.
- All calibration results are compatible with the GMIND SDK format.

---

For questions or issues, please contact the GMIND SDK development team.
