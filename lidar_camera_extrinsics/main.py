"""
LIDAR-Camera Calibration Tool
--------------------------------------------------
Interactive LiDAR-camera calibration tool for associating 2D image points with 3D LiDAR points, running robust extrinsic calibration, and visualising results.

Author: Roshan George, Dara Molloy
Date: 30-06-2025
Licence: MIT

Features:
 - Interactive point selection and pairing
 - Robust calibration (RANSAC, bundle adjustment)
 - Intrinsics/extrinsics import/export
 - Undo/redo, reset, and advanced diagnostics
 - PCD projection visualisation
--------------------------------------------------
"""
import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QListWidget, QLineEdit, QMessageBox, QTextEdit, QTabWidget, QFormLayout, QComboBox, QProgressDialog, QAction, QMenuBar, QMenu, QInputDialog, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent
from PyQt5.QtCore import Qt, QPoint
import extrinsic_calibration as calibration_runner
import json
import os

class CalibrationToolbox(QMainWindow):
    # Default camera intrinsics for FLIR 8.9mm (OpenCV standard fields, no skew/fisheye)
    CAMERA_INTRINSICS = {
        "fx": 1823.2131641887504,  # Focal length in x
        "fy": 1820.6052667772403,  # Focal length in y
        "cx": 2066.627144354761,   # Principal point x
        "cy": 1134.983254011596,   # Principal point y
        "model": "standard",
        "dist": [
            -0.11493534187041937,  # k1 (radial distortion)
            0.07868437224005516,   # k2 (radial distortion)
            0.0005806697561345654, # p1 (tangential distortion)
            0.0006835092446758971, # p2 (tangential distortion)
            -0.01980248809606338,  # k3 (radial distortion)
            0.0, 0.0, 0.0,         # k4, k5, k6 (higher-order radial)
            0.0, 0.0, 0.0, 0.0     # s1, s2, s3, s4 (thin prism)
        ]
    }

    def __init__(self):
        """
        Initialize the main calibration toolbox window and state.
        """
        super().__init__()
        self.setWindowTitle('LIDAR-Camera Calibration Tool')
        self.setGeometry(100, 100, 1400, 900)
        self.initUI()
        # State variables
        self.true_original_image = None  # Original loaded image (numpy array)
        self.collected_image_points = []  # List of 2D image points
        self.collected_lidar_points = []  # List of 3D lidar points
        self.temp_clicked_2d_point = None  # Temporary 2D point (before pairing)
        self.zoom = 1.0  # Current zoom factor
        self.pan_offset = QPoint(0, 0)  # Current pan offset
        self.last_pan_point = None  # Last pan anchor point
        # Undo/redo stacks for point management
        self.undo_stack = []
        self.redo_stack = []
        # Store last projected 3D points for persistent overlay
        self.last_projected_3d_points = None

    def push_undo_state(self):
        """
        Save a deep copy of the current state for undo functionality.
        """
        import copy
        state = (
            copy.deepcopy(self.collected_image_points),
            copy.deepcopy(self.collected_lidar_points),
            [self.point_list.item(i).text() for i in range(self.point_list.count())]
        )
        self.undo_stack.append(state)
        # Clear redo stack on new action
        self.redo_stack.clear()

    def pop_undo_state(self, to_redo=True):
        """
        Undo the last action, optionally saving the current state for redo.
        """
        if not self.undo_stack:
            return
        import copy
        state = (
            copy.deepcopy(self.collected_image_points),
            copy.deepcopy(self.collected_lidar_points),
            [self.point_list.item(i).text() for i in range(self.point_list.count())]
        )
        if to_redo:
            self.redo_stack.append(state)
        prev_img_pts, prev_lidar_pts, prev_list = self.undo_stack.pop()
        self.collected_image_points = prev_img_pts
        self.collected_lidar_points = prev_lidar_pts
        self.point_list.clear()
        for txt in prev_list:
            self.point_list.addItem(txt)
        self.status_label.setText('Undo performed.')
        self.log('Undo performed.')

    def pop_redo_state(self):
        """
        Redo the last undone action.
        """
        if not self.redo_stack:
            return
        import copy
        state = (
            copy.deepcopy(self.collected_image_points),
            copy.deepcopy(self.collected_lidar_points),
            [self.point_list.item(i).text() for i in range(self.point_list.count())]
        )
        self.undo_stack.append(state)
        next_img_pts, next_lidar_pts, next_list = self.redo_stack.pop()
        self.collected_image_points = next_img_pts
        self.collected_lidar_points = next_lidar_pts
        self.point_list.clear()
        for txt in next_list:
            self.point_list.addItem(txt)
        self.status_label.setText('Redo performed.')
        self.log('Redo performed.')

    def initUI(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Calibration tab (existing UI)
        calib_tab = QWidget()
        calib_layout = QHBoxLayout()
        calib_tab.setLayout(calib_layout)
        self.tabs.addTab(calib_tab, "Calibration")

        # Left: Image display and log
        left_layout = QVBoxLayout()
        calib_layout.addLayout(left_layout)

        self.image_label = QLabel('Load an image to begin')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet('background: #222; color: #fff;')
        self.image_label.setFixedSize(1280, 720)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.on_image_click
        self.image_label.wheelEvent = self.on_image_wheel
        self.image_label.mouseMoveEvent = self.on_image_drag
        self.image_label.mouseReleaseEvent = self.on_image_release
        left_layout.addWidget(self.image_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(120)
        left_layout.addWidget(self.log_text)

        # Right: Controls
        control_layout = QVBoxLayout()
        calib_layout.addLayout(control_layout)

        self.load_image_btn = QPushButton('Load Image')
        self.load_image_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_image_btn)

        self.point_list = QListWidget()
        control_layout.addWidget(self.point_list)

        coord_layout = QHBoxLayout()
        self.x_input = QLineEdit(); self.x_input.setPlaceholderText('X')
        self.y_input = QLineEdit(); self.y_input.setPlaceholderText('Y')
        self.z_input = QLineEdit(); self.z_input.setPlaceholderText('Z')
        coord_layout.addWidget(self.x_input)
        coord_layout.addWidget(self.y_input)
        coord_layout.addWidget(self.z_input)
        control_layout.addLayout(coord_layout)

        self.add_point_btn = QPushButton('Add Point Pair')
        self.add_point_btn.clicked.connect(self.add_point_pair)
        control_layout.addWidget(self.add_point_btn)

        self.remove_point_btn = QPushButton('Remove Selected Point')
        self.remove_point_btn.clicked.connect(self.remove_selected_point)
        control_layout.addWidget(self.remove_point_btn)


        self.status_label = QLabel('Status: Ready')
        control_layout.addWidget(self.status_label)

        control_layout.addStretch()


        self.run_calib_btn = QPushButton('Run Calibration')
        self.run_calib_btn.clicked.connect(self.run_calibration)
        # Highlight the button: larger, bold, colored
        self.run_calib_btn.setStyleSheet('background-color: #ffcc00; color: #222; font-weight: bold; font-size: 18px; border: 2px solid #bba500; padding: 8px;')
        control_layout.addWidget(self.run_calib_btn)

        self.reset_btn = QPushButton('Reset')
        self.reset_btn.clicked.connect(self.reset_toolbox)
        control_layout.addWidget(self.reset_btn)

        self.export_btn = QPushButton('Export Points')
        self.export_btn.clicked.connect(self.export_points)
        control_layout.addWidget(self.export_btn)

        self.import_btn = QPushButton('Import Points')
        self.import_btn.clicked.connect(self.import_points)
        control_layout.addWidget(self.import_btn)


        # Intrinsics tab
        intr_tab = QWidget()
        intr_layout = QVBoxLayout()
        intr_tab.setLayout(intr_layout)
        self.tabs.addTab(intr_tab, "Intrinsics")
        form = QFormLayout()
        self.fx_edit = QLineEdit(str(self.CAMERA_INTRINSICS["fx"]))
        self.fy_edit = QLineEdit(str(self.CAMERA_INTRINSICS["fy"]))
        self.cx_edit = QLineEdit(str(self.CAMERA_INTRINSICS["cx"]))
        self.cy_edit = QLineEdit(str(self.CAMERA_INTRINSICS["cy"]))
        form.addRow("fx", self.fx_edit)
        form.addRow("fy", self.fy_edit)
        form.addRow("cx", self.cx_edit)
        form.addRow("cy", self.cy_edit)
        # Standard model fields
        self.k1_edit = QLineEdit(str(self.CAMERA_INTRINSICS["dist"][0]))
        self.k2_edit = QLineEdit(str(self.CAMERA_INTRINSICS["dist"][1]))
        self.p1_edit = QLineEdit(str(self.CAMERA_INTRINSICS["dist"][2]))
        self.p2_edit = QLineEdit(str(self.CAMERA_INTRINSICS["dist"][3]))
        self.k3_edit = QLineEdit(str(self.CAMERA_INTRINSICS["dist"][4]))
        self.k4_edit = QLineEdit(str(self.CAMERA_INTRINSICS["dist"][5]))
        self.k5_edit = QLineEdit(str(self.CAMERA_INTRINSICS["dist"][6]))
        self.k6_edit = QLineEdit(str(self.CAMERA_INTRINSICS["dist"][7]))
        self.s1_edit = QLineEdit(str(self.CAMERA_INTRINSICS["dist"][8]))
        self.s2_edit = QLineEdit(str(self.CAMERA_INTRINSICS["dist"][9]))
        self.s3_edit = QLineEdit(str(self.CAMERA_INTRINSICS["dist"][10]))
        self.s4_edit = QLineEdit(str(self.CAMERA_INTRINSICS["dist"][11]))
        # Add all fields, but show/hide based on model
        self.std_fields = [
            ("k1", self.k1_edit), ("k2", self.k2_edit), ("p1", self.p1_edit), ("p2", self.p2_edit),
            ("k3", self.k3_edit), ("k4", self.k4_edit), ("k5", self.k5_edit), ("k6", self.k6_edit),
            ("s1", self.s1_edit), ("s2", self.s2_edit), ("s3", self.s3_edit), ("s4", self.s4_edit)
        ]
        for label, widget in self.std_fields:
            form.addRow(label, widget)
        intr_layout.addLayout(form)
        btn_layout = QHBoxLayout()
        self.load_intr_btn = QPushButton('Load Intrinsics')
        self.save_intr_btn = QPushButton('Save Intrinsics')
        self.apply_intr_btn = QPushButton('Apply to Toolbox')
        btn_layout.addWidget(self.load_intr_btn)
        btn_layout.addWidget(self.save_intr_btn)
        btn_layout.addWidget(self.apply_intr_btn)
        intr_layout.addLayout(btn_layout)
        self.apply_intr_btn.clicked.connect(self.apply_intrinsics_to_toolbox)
        self.load_intr_btn.clicked.connect(self.load_intrinsics_from_file)
        self.save_intr_btn.clicked.connect(self.save_intrinsics_to_file)

        # Add advanced calibration/diagnostic buttons
        self.outlier_btn = QPushButton('Analyse Outliers')
        self.outlier_btn.clicked.connect(self.analyse_outliers)
        control_layout.addWidget(self.outlier_btn)

        self.contribution_btn = QPushButton('Point Contribution')
        self.contribution_btn.clicked.connect(self.analyse_point_contributions)
        control_layout.addWidget(self.contribution_btn)

        # Removed Visualize Projections button

        # Menu bar
        menubar = self.menuBar()
        # Remove Help menu and About action
        # Keyboard shortcuts
        self.add_point_btn.setShortcut('Ctrl+N')
        self.remove_point_btn.setShortcut('Delete')
        self.reset_btn.setShortcut('Ctrl+Shift+C')
        self.run_calib_btn.setShortcut('Ctrl+R')
        self.export_btn.setShortcut('Ctrl+E')
        self.import_btn.setShortcut('Ctrl+I')
        # Undo/Redo actions
        undo_action = QAction('Undo', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.triggered.connect(self.undo_action)
        menubar.addAction(undo_action)
        redo_action = QAction('Redo', self)
        redo_action.setShortcut('Ctrl+Y')
        redo_action.triggered.connect(self.redo_action)
        menubar.addAction(redo_action)

        # Removed Robust Calibration and Bundle Adjustment buttons

        # --- Move PCD validation controls to Calibration tab ---
        # Combined Load & Project PCD button
        self.pcd_overlay_enabled = False
        self.pcd_btn = QPushButton('Load and Project PCD')
        self.pcd_btn.clicked.connect(self.load_and_project_pcd)
        control_layout.addWidget(self.pcd_btn)
        # Toggle overlay button
        self.toggle_pcd_btn = QPushButton('Toggle PCD Overlay')
        self.toggle_pcd_btn.setCheckable(True)
        self.toggle_pcd_btn.setChecked(False)
        self.toggle_pcd_btn.clicked.connect(self.toggle_pcd_overlay)
        control_layout.addWidget(self.toggle_pcd_btn)
        self.loaded_pcd_points = None
        self.loaded_pcd_intensities = None
        self.pcd_overlay_pixmap = None
        self.last_extrinsics = None
    def load_and_project_pcd(self):
        """
        Loads a PCD file and projects it onto the image, displaying the overlay if enabled.
        """
        fname, _ = QFileDialog.getOpenFileName(self, 'Load PCD', '', 'PCD files (*.pcd)')
        if not fname:
            return
        progress = self.show_progress("Loading and projecting PCD...")
        try:
            # Try Open3D if available, else fallback to simple parser
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(fname)
                points = np.asarray(pcd.points)
                if hasattr(pcd, 'intensities') and len(pcd.intensities) == len(points):
                    intensities = np.asarray(pcd.intensities)
                else:
                    intensities = None
            except ImportError:
                points, intensities = self.simple_pcd_loader(fname)
                if intensities is not None and np.all(intensities == 1.0):
                    intensities = None
            self.loaded_pcd_points = points
            self.loaded_pcd_intensities = intensities
            self.log(f'Loaded PCD: {fname} ({points.shape[0]} points)')
            # Project immediately if image and extrinsics are available
            if self.true_original_image is not None:
                self.project_pcd_onto_image(show_overlay=True)
        except Exception as e:
            self.log(f'Failed to load or project PCD: {e}')
        finally:
            progress.close()

    def toggle_pcd_overlay(self):
        """
        Toggle the display of the PCD overlay on the image.
        """
        self.pcd_overlay_enabled = not self.pcd_overlay_enabled
        self.display_image()
        # Remove Validation tab (do not add it)

    def log(self, msg):
        """
        Append a message to the log output.
        """
        self.log_text.append(msg)

    def fit_image_to_canvas(self):
        # Fit the image to the QLabel display area
        if self.true_original_image is None:
            return
        img_h, img_w, _ = self.true_original_image.shape
        disp_w, disp_h = self.image_label.width(), self.image_label.height()
        if img_w == 0 or img_h == 0 or disp_w == 0 or disp_h == 0:
            return
        scale_w = disp_w / img_w
        scale_h = disp_h / img_h
        fit_zoom = min(scale_w, scale_h)
        self.zoom = fit_zoom
        self.fit_zoom = fit_zoom  # Store for reference
        # Center the image if it's smaller than the label
        pan_x = max(0, (img_w * fit_zoom - disp_w) // 2)
        pan_y = max(0, (img_h * fit_zoom - disp_h) // 2)
        self.pan_offset = QPoint(0, 0)  # Always start at top-left for now

    def load_image(self):
        try:
            fname, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image files (*.jpg *.jpeg *.png *.bmp *.tiff)')
            if not fname:
                return
            img = cv2.imread(fname)
            if img is None:
                raise Exception('Could not load image.')
            self.true_original_image = img
            # --- Set label size to match image aspect ratio, max 1280x720 ---
            h, w = img.shape[:2]
            max_w, max_h = 1280, 720
            aspect = w / h
            if w > max_w or h > max_h:
                # Scale down to fit max size
                scale = min(max_w / w, max_h / h)
                disp_w = int(w * scale)
                disp_h = int(h * scale)
            else:
                disp_w, disp_h = w, h
            self.image_label.setFixedSize(disp_w, disp_h)
            self._zoom_initialized = False
            self.display_image(fit_to_canvas=True)
            self.status_label.setText('Image loaded. Click to select points.')
            self.log(f'Loaded image: {fname}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load image: {e}')
            self.log(f'Failed to load image: {e}')

    def resizeEvent(self, event):
        self._zoom_initialized = False
        self.display_image(fit_to_canvas=True)
        super().resizeEvent(event)

    def display_image(self, fit_to_canvas=False):
        if self.true_original_image is None:
            self.image_label.clear()
            return
        img = self.true_original_image.copy()
        h, w, _ = img.shape
        disp_w, disp_h = self.image_label.width(), self.image_label.height()
        # Fit image to canvas if requested or if zoom is not set
        if fit_to_canvas or not hasattr(self, '_zoom_initialized') or not self._zoom_initialized:
            scale_w = disp_w / w
            scale_h = disp_h / h
            self.zoom = min(scale_w, scale_h)
            self.fit_zoom = self.zoom  # Ensure fit_zoom is always set
            self.pan_offset = QPoint(0, 0)
            self._zoom_initialized = True
        zoomed_w, zoomed_h = int(w * self.zoom), int(h * self.zoom)
        img = cv2.resize(img, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LINEAR)
        # Pan offset
        x_off = self.pan_offset.x()
        y_off = self.pan_offset.y()
        x_off = max(0, min(x_off, max(0, zoomed_w - disp_w)))
        y_off = max(0, min(y_off, max(0, zoomed_h - disp_h)))
        # Center the image in the label if it's smaller than the label
        x_pad = max(0, (disp_w - zoomed_w) // 2)
        y_pad = max(0, (disp_h - zoomed_h) // 2)
        img_cropped = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        crop = img[y_off:y_off+min(zoomed_h, disp_h), x_off:x_off+min(zoomed_w, disp_w)]
        ch, cw = crop.shape[:2]
        img_cropped[y_pad:y_pad+ch, x_pad:x_pad+cw] = crop
        # Adjust all drawing coordinates by x_pad, y_pad below

        # --- Draw PCD overlay if enabled ---
        if self.pcd_overlay_enabled and self.loaded_pcd_points is not None and self.last_extrinsics is not None:
            try:
                # Use rotation matrix and tvec from extrinsics_result.json
                rot_matrix = np.array(self.last_extrinsics['rotation_matrix'], dtype=np.float32)
                tvec = np.array(self.last_extrinsics['tvec'], dtype=np.float32).reshape(3,1)
                fx = self.CAMERA_INTRINSICS["fx"]
                fy = self.CAMERA_INTRINSICS["fy"]
                cx = self.CAMERA_INTRINSICS["cx"]
                cy = self.CAMERA_INTRINSICS["cy"]
                dist = np.array(self.CAMERA_INTRINSICS["dist"][:5], dtype=np.float32)
                camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                dist_coeffs = dist.reshape(-1, 1)
                # Transform points using rotation matrix and tvec
                points = self.loaded_pcd_points
                # Apply rotation and translation: X_cam = R * X_lidar + t
                points_cam = (rot_matrix @ points.T).T + tvec.flatten()
                # Project to 2D
                proj_2d, _ = cv2.projectPoints(points_cam, np.zeros((3,1), dtype=np.float32), np.zeros((3,1), dtype=np.float32), camera_matrix, dist_coeffs)
                proj_2d = proj_2d.reshape(-1, 2)
                intensities = self.loaded_pcd_intensities
                zoom = self.zoom
                # Draw projected PCD points directly on img_cropped
                if intensities is not None:
                    int_min = intensities.min()
                    int_ptp = np.ptp(intensities) if len(intensities) > 0 else 1.0
                    norm_int = (intensities - int_min) / (int_ptp + 1e-6)
                    for pt, inten in zip(proj_2d, norm_int):
                        if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
                            continue
                        x_proj = int(pt[0] * zoom - x_off + x_pad)
                        y_proj = int(pt[1] * zoom - y_off + y_pad)
                        if 0 <= x_proj < disp_w and 0 <= y_proj < disp_h:
                            color = (0, int(255*inten), int(255*inten))  # BGR: black to yellow
                            cv2.circle(img_cropped, (x_proj, y_proj), 1, color, -1)
                else:
                    light_yellow = (0, 255, 255)  # BGR
                    for pt in proj_2d:
                        if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
                            continue
                        x_proj = int(pt[0] * zoom - x_off + x_pad)
                        y_proj = int(pt[1] * zoom - y_off + y_pad)
                        if 0 <= x_proj < disp_w and 0 <= y_proj < disp_h:
                            cv2.circle(img_cropped, (x_proj, y_proj), 1, light_yellow, -1)
            except Exception as e:
                self.log(f'Failed to draw PCD overlay: {e}')

        # --- Draw 2D points (orange) and projected 3D points (cyan), green if overlap ---
        circle_radius = 8
        circle_thickness = 2
        orange = (0, 165, 255)  # BGR for orange
        cyan = (255, 255, 0)    # BGR for cyan
        green = (0, 255, 0)     # BGR for green (overlap)
        # Compute display positions (add x_pad, y_pad)
        pts_2d_disp = [(int(pt[0] * self.zoom - x_off + x_pad), int(pt[1] * self.zoom - y_off + y_pad)) for pt in self.collected_image_points]
        pts_3d_disp = []
        if self.last_projected_3d_points is not None:
            pts_3d_disp = [(int(pt[0] * self.zoom - x_off + x_pad), int(pt[1] * self.zoom - y_off + y_pad)) for pt in self.last_projected_3d_points]
        # Draw green where overlap, else orange/cyan
        drawn_3d = set()
        for i, pt2d in enumerate(pts_2d_disp):
            overlap = False
            for j, pt3d in enumerate(pts_3d_disp):
                dist = np.hypot(pt2d[0] - pt3d[0], pt2d[1] - pt3d[1])
                if dist <= circle_radius:
                    cv2.circle(img_cropped, pt2d, circle_radius, green, circle_thickness)
                    drawn_3d.add(j)
                    overlap = True
                    break
            if not overlap:
                if 0 <= pt2d[0] < disp_w and 0 <= pt2d[1] < disp_h:
                    cv2.circle(img_cropped, pt2d, circle_radius, orange, circle_thickness)
        # Draw remaining 3D projected points (cyan)
        for j, pt3d in enumerate(pts_3d_disp):
            if j in drawn_3d:
                continue
            if 0 <= pt3d[0] < disp_w and 0 <= pt3d[1] < disp_h:
                cv2.circle(img_cropped, pt3d, circle_radius, cyan, circle_thickness)
        # Draw temp point (if not already in the list)
        if hasattr(self, 'temp_clicked_2d_point') and self.temp_clicked_2d_point is not None:
            if self.temp_clicked_2d_point not in self.collected_image_points:
                x_disp = int(self.temp_clicked_2d_point[0] * self.zoom - x_off + x_pad)
                y_disp = int(self.temp_clicked_2d_point[1] * self.zoom - y_off + y_pad)
                if 0 <= x_disp < disp_w and 0 <= y_disp < disp_h:
                    cv2.circle(img_cropped, (x_disp, y_disp), circle_radius, orange, circle_thickness)
        # --- Draw color legend in bottom-left corner (relative to image area) ---
        legend_items = [
            (orange, '2D Point'),
            (cyan,   '3D Projected'),
            (green,  'Overlap'),
        ]
        legend_x = x_pad + 15
        legend_y = y_pad + (zoomed_h if zoomed_h + y_pad < disp_h else disp_h) - 20 * len(legend_items) - 15
        box_w = 140
        box_h = 20 * len(legend_items) + 10
        # Draw background box (semi-transparent)
        overlay = img_cropped.copy()
        cv2.rectangle(overlay, (legend_x-5, legend_y-10), (legend_x+box_w, legend_y+box_h), (30,30,30), -1)
        alpha = 0.6
        img_cropped = cv2.addWeighted(overlay, alpha, img_cropped, 1-alpha, 0)
        # Draw legend items
        for i, (color, label) in enumerate(legend_items):
            cy = legend_y + 20*i + 10
            cv2.circle(img_cropped, (legend_x+10, cy), 7, color, -1)
            cv2.putText(img_cropped, label, (legend_x+25, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, disp_w, disp_h, disp_w*3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)

    def on_image_click(self, event):
        if self.true_original_image is None:
            return
        # Only handle left mouse button for point selection
        if event.button() != Qt.LeftButton:
            return
        disp_w, disp_h = self.image_label.width(), self.image_label.height()
        h, w, _ = self.true_original_image.shape
        x_disp = event.pos().x()
        y_disp = event.pos().y()
        # Map display coords to zoomed/panned image coords
        x_img = int((x_disp + self.pan_offset.x()) / self.zoom)
        y_img = int((y_disp + self.pan_offset.y()) / self.zoom)
        if x_img < 0 or y_img < 0 or x_img >= w or y_img >= h:
            return
        self.temp_clicked_2d_point = (x_img, y_img)
        self.status_label.setText(f'Clicked 2D: ({x_img}, {y_img})')
        self.log(f'Clicked 2D: ({x_img}, {y_img})')
        self.display_image()

    def on_image_wheel(self, event):
        # Zoom in/out centered at mouse position
        old_zoom = self.zoom
        mouse_pos = event.pos()
        disp_w, disp_h = self.image_label.width(), self.image_label.height()
        h, w, _ = self.true_original_image.shape if self.true_original_image is not None else (0, 0, 0)
        # Use fit-to-canvas zoom as minimum
        min_zoom = getattr(self, 'fit_zoom', 1.0)
        if event.angleDelta().y() > 0:
            new_zoom = min(10.0, self.zoom * 1.1)
        else:
            new_zoom = max(min_zoom, self.zoom / 1.1)
        if self.true_original_image is not None and new_zoom != self.zoom:
            # Calculate the image coordinate under the mouse before zoom
            x_img = (mouse_pos.x() + self.pan_offset.x()) / self.zoom
            y_img = (mouse_pos.y() + self.pan_offset.y()) / self.zoom
            # Update zoom
            self.zoom = new_zoom
            # Calculate new pan_offset so that the same image point stays under the mouse
            new_x_off = int(x_img * self.zoom - mouse_pos.x())
            new_y_off = int(y_img * self.zoom - mouse_pos.y())
            self.pan_offset = QPoint(max(0, new_x_off), max(0, new_y_off))
        self.display_image()
        self.log(f'Zoom: {self.zoom:.2f}')

    def on_image_drag(self, event):
        # Right mouse button drag for panning (anchored to mouse)
        if event.buttons() & Qt.RightButton:
            if not hasattr(self, '_pan_anchor') or self._pan_anchor is None:
                # Store anchor: (mouse_pos, pan_offset)
                self._pan_anchor = (event.pos(), QPoint(self.pan_offset))
            else:
                anchor_mouse, anchor_offset = self._pan_anchor
                delta = event.pos() - anchor_mouse
                new_offset = anchor_offset + delta
                # Clamp pan offset to image bounds
                if self.true_original_image is not None:
                    h, w, _ = self.true_original_image.shape
                    zoomed_w, zoomed_h = int(w * self.zoom), int(h * self.zoom)
                    disp_w, disp_h = self.image_label.width(), self.image_label.height()
                    max_x = max(0, zoomed_w - disp_w)
                    max_y = max(0, zoomed_h - disp_h)
                    new_offset.setX(max(0, min(new_offset.x(), max_x)))
                    new_offset.setY(max(0, min(new_offset.y(), max_y)))
                self.pan_offset = new_offset
                self.display_image()
        else:
            self._pan_anchor = None

    def on_image_release(self, event):
        self._pan_anchor = None

    def add_point_pair(self):
        if self.temp_clicked_2d_point is None:
            QMessageBox.warning(self, 'Error', 'Click a 2D point on the image first.')
            return
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.z_input.text())
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Invalid 3D coordinates.')
            return
        self.push_undo_state()
        self.collected_image_points.append(self.temp_clicked_2d_point)
        self.collected_lidar_points.append([x, y, z])
        self.point_list.addItem(f'2D: {self.temp_clicked_2d_point} <-> 3D: ({x:.2f}, {y:.2f}, {z:.2f})')
        self.temp_clicked_2d_point = None
        self.status_label.setText('Point pair added.')
        self.x_input.clear(); self.y_input.clear(); self.z_input.clear()
        self.log(f'Added point pair: 2D {self.collected_image_points[-1]}, 3D ({x}, {y}, {z})')
        self.display_image()

    def remove_selected_point(self):
        row = self.point_list.currentRow()
        if row >= 0:
            self.push_undo_state()
            self.point_list.takeItem(row)
            del self.collected_image_points[row]
            del self.collected_lidar_points[row]
        self.status_label.setText('Selected point removed.')
        self.log('Selected point removed.')
        return
    # If no point selected
        self.status_label.setText('No point selected.')


    def reset_toolbox(self):
        # Remove all points, image, overlays, undo/redo, and reset UI
        self.collected_image_points.clear()
        self.collected_lidar_points.clear()
        self.point_list.clear()
        self.temp_clicked_2d_point = None
        self.last_projected_3d_points = None
        self.true_original_image = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.loaded_pcd_points = None
        self.loaded_pcd_intensities = None
        self.last_extrinsics = None
        self.image_label.clear()
        self.status_label.setText('Toolbox reset.')
        self.log('Toolbox reset: all points, image, overlays, and state cleared.')
        # Optionally clear log
        self.log_text.clear()

    def show_progress(self, label_text="Working..."):
        dlg = QProgressDialog(label_text, None, 0, 0, self)
        dlg.setWindowTitle("Please Wait")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.show()
        QApplication.processEvents()
        return dlg

    def save_extrinsics_to_json(self, rvec, tvec, mean_error, filename="extrinsics_result.json"):
        # Save extrinsics and error to a JSON file in the current working directory
        try:
            rot_matrix, _ = cv2.Rodrigues(rvec)
            extrinsics = {
                "rotation_matrix": rot_matrix.tolist(),
                "tvec": tvec.flatten().tolist(),
                "mean_reprojection_error": float(mean_error)
            }
            with open(filename, "w") as f:
                json.dump(extrinsics, f, indent=2)
            self.log(f"Extrinsics saved to {os.path.abspath(filename)}")
        except Exception as e:
            self.log(f"Failed to save extrinsics: {e}")

    def run_calibration(self):
        # Prepare camera matrix and dist coeffs from GUI intrinsics
        fx = self.CAMERA_INTRINSICS["fx"]
        fy = self.CAMERA_INTRINSICS["fy"]
        cx = self.CAMERA_INTRINSICS["cx"]
        cy = self.CAMERA_INTRINSICS["cy"]
        dist = np.array(self.CAMERA_INTRINSICS["dist"][:5], dtype=np.float32)  # Use first 5 for OpenCV
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = dist.reshape(-1, 1)
        image_points = self.collected_image_points
        lidar_points = self.collected_lidar_points
        n_points = len(image_points)
        if n_points < 3 or len(lidar_points) < 3:
            QMessageBox.warning(self, 'Calibration', 'Need at least 3 point pairs.')
            return
        self.log('Running calibration (solvePnP, RANSAC, bundle adjustment)...')
        progress = self.show_progress("Running calibration...")
        try:
            # 1. Try direct solvePnP for 3+ points
            solvepnp_result, solvepnp_err = calibration_runner.run_calibration(image_points, lidar_points, camera_matrix, dist_coeffs)
            if solvepnp_result is None:
                self.status_label.setText('Calibration failed (solvePnP).')
                self.log(f'Calibration failed (solvePnP): {solvepnp_err}')
                QMessageBox.warning(self, 'Calibration', f'Calibration failed (solvePnP): {solvepnp_err}')
                self.last_projected_3d_points = None
                self.display_image()
                return
            self.log('solvePnP successful!')
            self.log(f'Rotation vector (rvec): {solvepnp_result["rvec"].flatten()}')
            self.log(f'Translation vector (tvec): {solvepnp_result["tvec"].flatten()}')
            self.log(f'Mean reprojection error: {solvepnp_result["mean_error"]:.2f} px')
            self.log(f'Per-point errors: {solvepnp_result["errors"]}')

            # 2. Only run RANSAC if at least 10 points
            if n_points >= 10:
                ransac_result, ransac_err = calibration_runner.run_robust_calibration(image_points, lidar_points, camera_matrix, dist_coeffs)
                if ransac_result is not None:
                    inliers = ransac_result['inliers']
                    res = ransac_result['result']
                    self.log('RANSAC robust calibration successful!')
                    self.log(f'Inliers: {np.sum(inliers)}/{len(inliers)}')
                    self.log(f'Rotation vector (rvec): {res["rvec"].flatten()}')
                    self.log(f'Translation vector (tvec): {res["tvec"].flatten()}')
                    self.log(f'Mean reprojection error: {res["mean_error"]:.2f} px')
                    self.log(f'Per-point errors: {res["errors"]}')
                    # Highlight inliers (green) and outliers (red) in the point list
                    for i in range(self.point_list.count()):
                        item = self.point_list.item(i)
                        if inliers[i]:
                            item.setBackground(Qt.green)
                            item.setForeground(Qt.black)
                        else:
                            item.setBackground(Qt.red)
                            item.setForeground(Qt.white)
                            self.log(f'Point {i}: OUTLIER')
                    # 3. Bundle adjustment on inliers for final refinement
                    inlier_img_pts = np.asarray(image_points)[inliers]
                    inlier_lidar_pts = np.asarray(lidar_points)[inliers]
                    ba_result, ba_err = calibration_runner.run_bundle_adjustment([inlier_img_pts], [inlier_lidar_pts], camera_matrix, dist_coeffs, rvec_init=res["rvec"], tvec_init=res["tvec"])
                    if ba_result is not None:
                        self.status_label.setText('Calibration successful (RANSAC + BA)!')
                        self.log('Bundle adjustment successful!')
                        self.log(f'Final Rotation vector (rvec): {ba_result["rvec"].flatten()}')
                        self.log(f'Final Translation vector (tvec): {ba_result["tvec"].flatten()}')
                        # Save extrinsics to JSON
                        self.save_extrinsics_to_json(ba_result["rvec"], ba_result["tvec"], res["mean_error"])
                        # Project 3D points for overlay
                        try:
                            proj_2d = calibration_runner.project_points(np.asarray(lidar_points, dtype=np.float32), ba_result["rvec"], ba_result["tvec"], camera_matrix, dist_coeffs)
                            self.last_projected_3d_points = proj_2d
                        except Exception as e:
                            self.last_projected_3d_points = None
                            self.log(f'Failed to compute projected 3D points: {e}')
                        self.display_image()
                        return
                    else:
                        self.status_label.setText('Calibration failed (Bundle Adjustment).')
                        self.log(f'Bundle adjustment failed: {ba_err}')
                        QMessageBox.warning(self, 'Calibration', f'Bundle adjustment failed: {ba_err}')
                        self.last_projected_3d_points = None
                        self.display_image()
                        return
                else:
                    self.log('RANSAC failed, using solvePnP result.')
            # For <10 points, or if RANSAC fails, use solvePnP result
            self.status_label.setText('Calibration successful (solvePnP only)!')
            self.save_extrinsics_to_json(solvepnp_result["rvec"], solvepnp_result["tvec"], solvepnp_result["mean_error"])
            try:
                proj_2d = calibration_runner.project_points(np.asarray(lidar_points, dtype=np.float32), solvepnp_result["rvec"], solvepnp_result["tvec"], camera_matrix, dist_coeffs)
                self.last_projected_3d_points = proj_2d
            except Exception as e:
                self.last_projected_3d_points = None
                self.log(f'Failed to compute projected 3D points: {e}')
            self.display_image()
        finally:
            try:
                progress.close()
            except Exception:
                pass

    def export_points(self):
        try:
            fname, _ = QFileDialog.getSaveFileName(self, 'Export Points', '', 'Numpy files (*.npz)')
            if not fname:
                return
            np.savez(fname, image_points=np.array(self.collected_image_points), lidar_points=np.array(self.collected_lidar_points))
            self.status_label.setText('Points exported.')
            self.log(f'Points exported to {fname}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to export points: {e}')
            self.log(f'Failed to export points: {e}')

    def import_points(self):
        try:
            fname, _ = QFileDialog.getOpenFileName(self, 'Import Points', '', 'Numpy files (*.npz)')
            if not fname:
                return
            data = np.load(fname)
            self.collected_image_points = data['image_points'].tolist()
            self.collected_lidar_points = data['lidar_points'].tolist()
            self.point_list.clear()
            for img_pt, lidar_pt in zip(self.collected_image_points, self.collected_lidar_points):
                self.point_list.addItem(f'2D: {tuple(img_pt)} <-> 3D: ({lidar_pt[0]:.2f}, {lidar_pt[1]:.2f}, {lidar_pt[2]:.2f})')
            self.status_label.setText('Points imported.')
            self.log(f'Points imported from {fname}')
            self.display_image()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to import points: {e}')
            self.log(f'Failed to import points: {e}')

    def apply_intrinsics_to_toolbox(self):
        try:
            self.CAMERA_INTRINSICS["fx"] = float(self.fx_edit.text())
            self.CAMERA_INTRINSICS["fy"] = float(self.fy_edit.text())
            self.CAMERA_INTRINSICS["cx"] = float(self.cx_edit.text())
            self.CAMERA_INTRINSICS["cy"] = float(self.cy_edit.text())
            self.CAMERA_INTRINSICS["dist"] = [
                float(self.k1_edit.text()), float(self.k2_edit.text()), float(self.p1_edit.text()), float(self.p2_edit.text()),
                float(self.k3_edit.text()), float(self.k4_edit.text()), float(self.k5_edit.text()), float(self.k6_edit.text()),
                float(self.s1_edit.text()), float(self.s2_edit.text()), float(self.s3_edit.text()), float(self.s4_edit.text())
            ]
            QMessageBox.information(self, 'Intrinsics', 'Intrinsics updated in toolbox.')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Invalid intrinsics: {e}')

    def load_intrinsics_from_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Intrinsics', '', 'JSON files (*.json);;NPZ files (*.npz)')
        if not fname:
            return
        try:
            if fname.endswith('.json'):
                import json
                with open(fname, 'r') as f:
                    data = json.load(f)
                intr = data.get('intrinsics', data)
            elif fname.endswith('.npz'):
                npz = np.load(fname)
                intr = {k: npz[k].item() if npz[k].shape == () else npz[k].tolist() for k in npz}
            else:
                raise Exception('Unsupported file type')
            self.fx_edit.setText(str(intr["fx"]))
            self.fy_edit.setText(str(intr["fy"]))
            self.cx_edit.setText(str(intr["cx"]))
            self.cy_edit.setText(str(intr["cy"]))
            dist = intr["dist"]
            self.k1_edit.setText(str(dist[0]))
            self.k2_edit.setText(str(dist[1]))
            self.p1_edit.setText(str(dist[2]))
            self.p2_edit.setText(str(dist[3]))
            self.k3_edit.setText(str(dist[4]))
            self.k4_edit.setText(str(dist[5]))
            self.k5_edit.setText(str(dist[6]))
            self.k6_edit.setText(str(dist[7]))
            self.s1_edit.setText(str(dist[8]))
            self.s2_edit.setText(str(dist[9]))
            self.s3_edit.setText(str(dist[10]))
            self.s4_edit.setText(str(dist[11]))
            QMessageBox.information(self, 'Intrinsics', 'Intrinsics loaded.')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to load intrinsics: {e}')

    def save_intrinsics_to_file(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save Intrinsics', '', 'JSON files (*.json);;NPZ files (*.npz)')
        if not fname:
            return
        try:
            intr = {
                "fx": float(self.fx_edit.text()),
                "fy": float(self.fy_edit.text()),
                "cx": float(self.cx_edit.text()),
                "cy": float(self.cy_edit.text()),
                "model": "standard",
                "dist": [
                    float(self.k1_edit.text()), float(self.k2_edit.text()), float(self.p1_edit.text()), float(self.p2_edit.text()),
                    float(self.k3_edit.text()), float(self.k4_edit.text()), float(self.k5_edit.text()), float(self.k6_edit.text()),
                    float(self.s1_edit.text()), float(self.s2_edit.text()), float(self.s3_edit.text()), float(self.s4_edit.text())
                ]
            }
            if fname.endswith('.json'):
                import json
                with open(fname, 'w') as f:
                    json.dump({"intrinsics": intr}, f, indent=2)
            elif fname.endswith('.npz'):
                np.savez(fname, **intr)
            else:
                raise Exception('Unsupported file type')
            QMessageBox.information(self, 'Intrinsics', 'Intrinsics saved.')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to save intrinsics: {e}')

    def analyse_outliers(self):
        # Outlier analysis using calibration_runner
        fx = self.CAMERA_INTRINSICS["fx"]
        fy = self.CAMERA_INTRINSICS["fy"]
        cx = self.CAMERA_INTRINSICS["cx"]
        cy = self.CAMERA_INTRINSICS["cy"]
        dist = np.array(self.CAMERA_INTRINSICS["dist"][:5], dtype=np.float32)
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = dist.reshape(-1, 1)
        image_points = np.asarray(self.collected_image_points, dtype=np.float32)
        lidar_points = np.asarray(self.collected_lidar_points, dtype=np.float32)
        progress = self.show_progress("Analysing outliers...")
        try:
            result, err = calibration_runner.analyze_outliers(image_points, lidar_points, camera_matrix, dist_coeffs)
        finally:
            progress.close()
        if result is None:
            self.log(err)
            return
        errors = result['errors']
        mean = result['mean']
        std = result['std']
        outlier_thresh = result['outlier_thresh']
        self.log('Outlier analysis:')
        for i, err_val in enumerate(errors):
            status = ''
            if err_val > outlier_thresh:
                status = 'OUTLIER!'
            elif err_val > 10:
                status = 'High error'
            self.log(f'Point {i}: error={err_val:.2f} px {status}')
        self.log(f'Mean error: {mean:.2f} px, Std: {std:.2f} px, Outlier threshold: {outlier_thresh:.2f} px')

    def analyse_point_contributions(self):
        # Leave-one-out analysis using calibration_runner
        fx = self.CAMERA_INTRINSICS["fx"]
        fy = self.CAMERA_INTRINSICS["fy"]
        cx = self.CAMERA_INTRINSICS["cx"]
        cy = self.CAMERA_INTRINSICS["cy"]
        dist = np.array(self.CAMERA_INTRINSICS["dist"][:5], dtype=np.float32)
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = dist.reshape(-1, 1)
        image_points = np.asarray(self.collected_image_points, dtype=np.float32)
        lidar_points = np.asarray(self.collected_lidar_points, dtype=np.float32)
        progress = self.show_progress("Analysing point contributions...")
        try:
            contributions, err = calibration_runner.analyze_point_contributions(image_points, lidar_points, camera_matrix, dist_coeffs)
        finally:
            progress.close()
        if contributions is None:
            self.log(err)
            return
        self.log('Point contribution analysis (leave-one-out):')
        for c in contributions:
            if c['success']:
                self.log(f"Point {c['index']}: mean error without = {c['mean_error']:.2f} px (delta {c['delta']:+.2f})")
            else:
                self.log(f"Point {c['index']}: calibration failed when removed. {c.get('error','')}")

    def visualise_projections(self):
        # Overlay projections on the image using current calibration
        if self.true_original_image is None:
            self.log('No image loaded.')
            return
        fx = self.CAMERA_INTRINSICS["fx"]
        fy = self.CAMERA_INTRINSICS["fy"]
        cx = self.CAMERA_INTRINSICS["cx"]
        cy = self.CAMERA_INTRINSICS["cy"]
        dist = np.array(self.CAMERA_INTRINSICS["dist"][:5], dtype=np.float32)
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = dist.reshape(-1, 1)
        image_points = np.asarray(self.collected_image_points, dtype=np.float32)
        lidar_points = np.asarray(self.collected_lidar_points, dtype=np.float32)
        if len(image_points) < 4:
            self.log('Need at least 4 points for projection visualisation.')
            return
        progress = None
        try:
            progress = self.show_progress("Visualising projections...")
            calib_result, err = calibration_runner.run_calibration(image_points, lidar_points, camera_matrix, dist_coeffs)
            if calib_result is None:
                self.log(f'Calibration required for visualisation: {err}')
                return
            proj_2d = calibration_runner.project_points(lidar_points, calib_result['rvec'], calib_result['tvec'], camera_matrix, dist_coeffs)
            # Draw on a copy of the image
            img = self.true_original_image.copy()
            for pt in proj_2d:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 2)
            for pt in image_points:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), 2)
            # Show in a new window using OpenCV
            cv2.imshow('Projection Visualization (Green=Projected, Red=Clicked)', img)
            cv2.waitKey(0)
            cv2.destroyWindow('Projection Visualization (Green=Projected, Red=Clicked)')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to visualize projections: {e}')
            self.log(f'Failed to visualise projections: {e}')
        finally:
            if progress is not None:
                progress.close()

    def undo_action(self):
        self.pop_undo_state()

    def redo_action(self):
        self.pop_redo_state()

    def load_pcd_file(self):
        progress = self.show_progress("Loading PCD file...")
        try:
            fname, _ = QFileDialog.getOpenFileName(self, 'Load PCD', '', 'PCD files (*.pcd)')
            if not fname:
                progress.close()
                return
            # Try Open3D if available, else fallback to simple parser
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(fname)
                points = np.asarray(pcd.points)
                # Try to get intensity from Open3D if present
                if hasattr(pcd, 'intensities') and len(pcd.intensities) == len(points):
                    intensities = np.asarray(pcd.intensities)
                else:
                    intensities = None  # No intensities present
            except ImportError:
                points, intensities = self.simple_pcd_loader(fname)
                if intensities is not None and np.all(intensities == 1.0):
                    intensities = None  # treat as no intensities
            self.loaded_pcd_points = points
            self.loaded_pcd_intensities = intensities
            self.log(f'Loaded PCD: {fname} ({points.shape[0]} points)')
        except Exception as e:
            self.log(f'Failed to load PCD: {e}')
        finally:
            progress.close()

    def simple_pcd_loader(self, fname):
        # Minimal PCD ASCII loader (XYZ + optional intensity)
        points = []
        intensities = []
        with open(fname, 'r') as f:
            header = True
            for line in f:
                if header:
                    if line.strip().startswith('DATA'):
                        header = False
                    continue
                vals = line.strip().split()
                if len(vals) >= 3:
                    try:
                        x, y, z = map(float, vals[:3])
                        points.append([x, y, z])
                        if len(vals) >= 4:
                            intensities.append(float(vals[3]))
                        else:
                            intensities.append(1.0)
                    except:
                        continue
        return np.array(points, dtype=np.float32), np.array(intensities, dtype=np.float32)

    def project_pcd_onto_image(self, show_overlay=False):
        """
        Projects loaded PCD onto the image using the last extrinsics and updates overlay state.
        If show_overlay is True, enables overlay display.
        """
        if self.true_original_image is None:
            self.log('No image loaded.')
            return
        if self.loaded_pcd_points is None:
            self.log('No PCD loaded.')
            return
        # Load extrinsics from last calibration JSON
        extr_file = 'extrinsics_result.json'
        try:
            with open(extr_file, 'r') as f:
                extr = json.load(f)
            self.last_extrinsics = extr
        except Exception as e:
            self.log(f'Failed to load extrinsics: {e}')
            return
        if show_overlay:
            self.pcd_overlay_enabled = True
            self.toggle_pcd_btn.setChecked(True)
        self.display_image()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CalibrationToolbox()
    win.show()
    sys.exit(app.exec_())
