# Camera Intrinsics Calibration Toolbox

This folder provides tools for calibrating camera intrinsic parameters using checkerboard images and OpenCV. It includes scripts for robust calibration, result export, and a camera model utility class.

## Features

- **Checkerboard-based camera calibration** using OpenCV
- **RANSAC-style robust calibration** to select best image subsets
- **Automatic detection** of checkerboard corners in batches of images
- **Exports calibration results** to a JSON file in GMIND format
- **CameraModel class** for easy 2D/3D mapping and projection

## How to Use

### 1. Prepare Checkerboard Images
- Collect multiple images of a checkerboard pattern from your camera.
- Place all images in a directory (JPG, PNG, BMP, etc. supported).

### 2. Configure Calibration Script
- Edit `opencv_checkerboard_calibrate.py`:
  - Set `IMAGES_DIR` to your image folder path.
  - Set `CHECKERBOARD` to your checkerboard's inner corner grid size (columns, rows).
  - Set `SQUARE_SIZE` to the physical size of one square (in meters or mm).

### 3. Run Calibration

```bash
python opencv_checkerboard_calibrate.py
```

- The script will:
  - Detect checkerboard corners in all images
  - Run multiple RANSAC-style calibrations to find the best subset
  - Print the best camera matrix and distortion coefficients
  - Save results to `GMIND_Calibration.json`

### 4. Output
- `GMIND_Calibration.json` contains the camera intrinsics in a format compatible with the GMIND SDK.

## CameraModel Utility
- `camera.py` provides a `CameraModel` class for loading intrinsics, projecting 3D points, and unprojecting pixels.
- Example usage:

```python
from camera import CameraModel
model = CameraModel(camera_matrix, dist_coeffs)
image_points = model.project_point(points_3d)
```

## Requirements
- Python 3.7+
- OpenCV (`cv2`)
- NumPy

Install dependencies with:

```bash
pip install -r ../requirements.txt
```

## File Overview
- `opencv_checkerboard_calibrate.py` — Main calibration script
- `camera.py` — CameraModel class for intrinsics and projection utilities

---

For questions or issues, please contact the GMIND SDK development team.
