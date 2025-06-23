import cv2
import numpy as np
import glob
import os
import random
import json

# --- User parameters ---
# Directory containing checkerboard images (change as needed)
IMAGES_DIR = 'H:/14-02-2022 - OfficeCalibAll/1644848398-2-withLIDAR/19305290/BMPOut/'  # e.g. './calib_images/'
# Checkerboard dimensions (number of inner corners per row and column)
CHECKERBOARD = (11, 6)  # (columns, rows) of inner corners
# Size of a square in your checkerboard (in meters or mm)
SQUARE_SIZE = 0.09445  

# --- Prepare object points ---
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# --- Find all images ---
image_paths = glob.glob(os.path.join(IMAGES_DIR, '*.jpg')) + \
              glob.glob(os.path.join(IMAGES_DIR, '*.png')) + \
              glob.glob(os.path.join(IMAGES_DIR, '*.jpeg')) + \
              glob.glob(os.path.join(IMAGES_DIR, '*.bmp'))

if not image_paths:
    print(f"No images found in {IMAGES_DIR}")
    exit(1)

# --- Detect points in all images first ---
all_objpoints = []
all_imgpoints = []
all_used_files = []
for idx, fname in enumerate(image_paths):
    print(f"[{idx+1}/{len(image_paths)}] Detecting: {os.path.basename(fname)}", end='')
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        all_objpoints.append(objp)
        all_imgpoints.append(corners2)
        all_used_files.append(fname)
        print("  [OK]")
    else:
        print("  [NO CHESSBOARD]")

print(f"\nDetected chessboard in {len(all_objpoints)} out of {len(image_paths)} images.")
if len(all_objpoints) < 20:
    print("Not enough valid images for calibration. Exiting.")
    exit(1)

# --- RANSAC-style calibration using detected points ---
RANSAC_ITER = 100
SAMPLE_SIZE = 60  # Number of images per trial
best_rms = float('inf')
best_mtx = None
best_dist = None
best_subset = None
best_objpoints = None
best_imgpoints = None

for i in range(RANSAC_ITER):
    print(f"\n--- RANSAC Iteration {i+1}/{RANSAC_ITER} ---")
    idxs = random.sample(range(len(all_objpoints)), min(SAMPLE_SIZE, len(all_objpoints)))
    objpoints = [all_objpoints[j] for j in idxs]
    imgpoints = [all_imgpoints[j] for j in idxs]
    used_files = [all_used_files[j] for j in idxs]
    print(f"  -> Using {len(objpoints)} images for calibration.")
    if len(objpoints) < 20:
        print(f"  Not enough valid images for calibration. Skipping iteration.")
        continue
    img_shape = cv2.imread(used_files[0]).shape
    gray_shape = (img_shape[1], img_shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    if ret < best_rms:
        best_rms = ret
        best_mtx = mtx
        best_dist = dist
        best_subset = used_files
        best_objpoints = objpoints
        best_imgpoints = imgpoints
    print(f"  Calibration RMS error: {ret:.4f} (used {len(objpoints)} images)")
    print(f"  Best RMS so far: {best_rms:.4f}")

print("\n=== Best Calibration Results ===")
print("Best RMS re-projection error:", best_rms)
print("Camera matrix (K):\n", best_mtx)
print("Distortion coefficients:\n", best_dist.ravel())
print(f"Used {len(best_subset)} images:")
for f in best_subset:
    print(f"  {f}")

# --- Write to JSON in GMIND format ---
fx = float(best_mtx[0, 0])
fy = float(best_mtx[1, 1])
cx = float(best_mtx[0, 2])
cy = float(best_mtx[1, 2])
dist_list = best_dist.ravel().tolist()

# You can change this sensor name as needed
SENSOR_NAME = 'flir8.9'
calib_data = {
    SENSOR_NAME: {
        'intrinsics': {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'dist': dist_list
        },
        'extrinsics': {
            'R': [[1,0,0],[0,1,0],[0,0,1]],
            't': [0,0,0]
        }
    }
}
with open('GMIND_Calibration.json', 'w') as f:
    json.dump(calib_data, f, indent=2)
print('Calibration JSON written to GMIND_Calibration.json')
