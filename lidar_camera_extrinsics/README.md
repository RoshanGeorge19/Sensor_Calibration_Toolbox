# LiDAR-Camera Extrinsic Calibration Toolbox

This folder contains tools for calibrating the extrinsic parameters (rotation and translation) between a LiDAR sensor and a camera, using 2D-3D point correspondences. The main application is a PyQt-based GUI for interactive calibration, visualization, and diagnostics.

## Features

- **Interactive GUI** for collecting 2D-3D correspondences and running calibration
- **Robust calibration** (RANSAC, Huber loss, bundle adjustment)
- **Outlier and point contribution analysis**
- **Visualization** of projected 3D points onto the image
- **Import/export** of point correspondences
- **Support for custom camera intrinsics**

## How to Use

### 1. Launch the Toolbox

Run the main application:

```bash
python main.py
```

### 2. Workflow

1. **Load an Image**: Click 'Load Image' and select your camera image.
2. **Collect Point Pairs**:
   - Click on the image to select a 2D point.
   - Enter the corresponding 3D LiDAR coordinates (X, Y, Z) and click 'Add Point Pair'.
   - Repeat for at least 4 pairs (more is better).
3. **Manage Points**:
   - Remove, clear, import, or export point pairs as needed.
   - Undo/redo actions are available via the menu or shortcuts.
4. **Set Camera Intrinsics**:
   - Use the 'Intrinsics' tab to view or edit camera parameters.
   - Load/save intrinsics from/to a file, or apply changes to the toolbox.
5. **Run Calibration**:
   - Click 'Run Calibration' to compute the extrinsic parameters.
   - Results and errors are shown in the log.
6. **Advanced Tools**:
   - 'Robust Calibration (RANSAC)': Run a robust fit to reject outliers.
   - 'Bundle Adjustment': Refine extrinsics using multiple views.
   - 'Analyze Outliers': Identify and report outlier correspondences.
   - 'Point Contribution': See how each point affects calibration error.
   - 'Visualize Projections': Overlay projected 3D points on the image.
   - 'Load PCD' and 'Project PCD onto Image': Validate calibration with dense point clouds.

### 3. Keyboard Shortcuts

- **Ctrl+N**: Add point pair
- **Delete**: Remove selected point
- **Ctrl+Shift+C**: Clear all points
- **Ctrl+R**: Run calibration
- **Ctrl+E**: Export points
- **Ctrl+I**: Import points
- **Ctrl+Z**: Undo
- **Ctrl+Y**: Redo

## File Overview

- `main.py` — Main PyQt GUI application
- `extrinsic_calibration.py` — Core calibration and optimization routines
- `lidar_projection.py` — Utilities for projecting LiDAR points to image

## Requirements

- Python 3.7+
- PyQt5
- OpenCV (`cv2`)
- NumPy
- SciPy

Install dependencies with:

```bash
pip install -r ../requirements.txt
```

## Example

1. Load an image and add at least 4 point pairs.
2. Set camera intrinsics if needed.
3. Click 'Run Calibration'.
4. Use advanced tools for diagnostics or robust fitting.

## Notes

- For best results, use well-distributed, accurate point correspondences.
- The toolbox supports standard OpenCV camera models.
- All calibration results and logs are shown in the GUI.

---

For questions or issues, please contact the GMIND SDK development team.
