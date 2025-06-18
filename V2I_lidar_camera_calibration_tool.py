import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import csv

from scipy.optimize import minimize, least_squares
from scipy.sparse import lil_matrix
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import os
try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# --- Global variables for GUI and data ---
true_original_image = None  # Stores the full resolution image
original_image = None  # Stores the image currently displayed in OpenCV (potentially resized)
true_original_image_dims = (0, 0)  # Stores (width, height) of true_original_image
collected_lidar_points = []
collected_image_points = []
temp_clicked_2d_point = None
last_calibration_results = None  # Store the last calibration results

WINDOW_NAME = "Select Points"
POINT_COLOR = (0, 0, 255)
TEMP_POINT_COLOR = (0, 255, 0)
POINT_RADIUS = 5
MAX_IMAGE_DISPLAY_WIDTH = 1280

# --- Zooming Globals ---
zoom_level = 1.0
view_rect = None  # (x, y, width, height) of the visible region IN TRUE ORIGINAL IMAGE COORDINATES
MIN_ZOOM_LEVEL = 1.0
MAX_ZOOM_LEVEL = 10.0
ZOOM_FACTOR_STEP = 1.1
opencv_window_shape = None  # (display_width, display_height) of the OpenCV window

# --- Camera Parameters ---
sensor_pixel_size = 2.1e-6  # meters
focal_length = 10.3e-3  # meters
image_width_pixels = 5472  # Original image width for intrinsic definition
image_height_pixels = 3648  # Original image height for intrinsic definition
fx = focal_length / sensor_pixel_size
fy = focal_length / sensor_pixel_size
cx = image_width_pixels / 2
cy = image_height_pixels / 2

camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((4, 1), dtype=np.float32)


editing_mode = False
selected_point_index = None

LAS_CLASSIFICATION_CODES = {
    0: "Created, never classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point (noise)",
    8: "Reserved",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    12: "Reserved",
    13: "Wire - Guard (Shield)",
    14: "Wire - Conductor (Phase)",
    15: "Transmission Tower",
    16: "Wire-structure Connector",
    17: "Bridge Deck",
    18: "High Noise"
}

def load_las_file(filepath):
    """Load LAS/LAZ file using laspy with intensity values"""
    if not HAS_LASPY:
        raise ImportError("laspy library not installed. Install with: pip install laspy")

    try:
        las_file = laspy.read(filepath)

        # Extract coordinates
        x = np.array(las_file.x)
        y = np.array(las_file.y)
        z = np.array(las_file.z)

        # Extract additional attributes
        intensity = np.array(las_file.intensity) if hasattr(las_file, 'intensity') else None
        classification = np.array(las_file.classification) if hasattr(las_file, 'classification') else None
        return_number = np.array(las_file.return_number) if hasattr(las_file, 'return_number') else None

        # Try to get RGB if available
        rgb = None
        if hasattr(las_file, 'red') and hasattr(las_file, 'green') and hasattr(las_file, 'blue'):
            red = np.array(las_file.red)
            green = np.array(las_file.green)
            blue = np.array(las_file.blue)
            rgb = np.column_stack((red, green, blue))

        # Stack coordinates into Nx3 array
        points = np.column_stack((x, y, z))

        # Create dictionary with all available data
        point_data = {
            'points': points,
            'intensity': intensity,
            'classification': classification,
            'return_number': return_number,
            'rgb': rgb
        }

        print(f"Loaded {len(points)} points from LAS file")
        if intensity is not None:
            print(f"Intensity range: [{np.min(intensity)}, {np.max(intensity)}]")
        if rgb is not None:
            print(f"RGB color data available")
        if classification is not None:
            unique_classes = np.unique(classification)
            print(f"Classifications present: {unique_classes}")

        return point_data

    except Exception as e:
        raise Exception(f"Failed to read LAS file: {str(e)}")

def load_point_cloud_file(filepath):
    """
    Load point cloud from various file formats (.las, .pcd, .csv, .txt, .ply)
    Returns: For LAS files - dictionary with points and attributes
             For other formats - numpy array of shape (N, 3) with X, Y, Z coordinates
    """
    file_ext = os.path.splitext(filepath)[1].lower()

    try:
        if file_ext == '.las' or file_ext == '.laz':
            return load_las_file(filepath)  # Returns dictionary
        elif file_ext == '.pcd':
            points = load_pcd_file(filepath)
            return {'points': points}  # Wrap in dictionary for consistency
        elif file_ext == '.ply':
            points = load_ply_file(filepath)
            return {'points': points}
        elif file_ext == '.csv':
            points = load_csv_file(filepath)
            return {'points': points}
        elif file_ext == '.txt':
            points = load_txt_file(filepath)
            return {'points': points}
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    except Exception as e:
        raise Exception(f"Error loading {file_ext} file: {str(e)}")

def load_pcd_file(filepath):
    """Load PCD file using Open3D"""
    if not HAS_OPEN3D:
        raise ImportError("open3d library not installed. Install with: pip install open3d")

    try:
        # Load PCD file
        pcd = o3d.io.read_point_cloud(filepath)

        if len(pcd.points) == 0:
            raise ValueError("PCD file contains no points")

        # Convert to numpy array
        points = np.asarray(pcd.points)

        print(f"Loaded {len(points)} points from PCD file")
        return points

    except Exception as e:
        raise Exception(f"Failed to read PCD file: {str(e)}")

def load_ply_file(filepath):
    """Load PLY file using Open3D"""
    if not HAS_OPEN3D:
        raise ImportError("open3d library not installed. Install with: pip install open3d")

    try:
        # Load PLY file
        mesh = o3d.io.read_triangle_mesh(filepath)
        if len(mesh.vertices) > 0:
            points = np.asarray(mesh.vertices)
        else:
            # Try as point cloud
            pcd = o3d.io.read_point_cloud(filepath)
            points = np.asarray(pcd.points)

        if len(points) == 0:
            raise ValueError("PLY file contains no points")

        print(f"Loaded {len(points)} points from PLY file")
        return points

    except Exception as e:
        raise Exception(f"Failed to read PLY file: {str(e)}")

def load_csv_file(filepath):
    """Load CSV file with various column formats"""
    if not HAS_PANDAS:
        # Fallback to basic CSV reading
        return load_csv_basic(filepath)

    try:
        df = pd.read_csv(filepath)

        # Try different column name conventions
        if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
            points = df[['x', 'y', 'z']].values
        elif 'X' in df.columns and 'Y' in df.columns and 'Z' in df.columns:
            points = df[['X', 'Y', 'Z']].values
        elif len(df.columns) >= 3:
            # Use first 3 columns
            points = df.iloc[:, :3].values
        else:
            raise ValueError("CSV must have at least 3 columns for X, Y, Z coordinates")

        # Remove any non-numeric rows
        points = points[~np.isnan(points).any(axis=1)]

        print(f"Loaded {len(points)} points from CSV file")
        return points

    except Exception as e:
        raise Exception(f"Failed to read CSV file: {str(e)}")

def load_csv_basic(filepath):
    """Basic CSV loading without pandas"""
    points = []
    try:
        with open(filepath, 'r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader, None)  # Skip header if present

            for row in csv_reader:
                if len(row) >= 3:
                    try:
                        x, y, z = float(row[0]), float(row[1]), float(row[2])
                        points.append([x, y, z])
                    except ValueError:
                        continue  # Skip invalid rows

        if len(points) == 0:
            raise ValueError("No valid points found in CSV file")

        points = np.array(points)
        print(f"Loaded {len(points)} points from CSV file")
        return points

    except Exception as e:
        raise Exception(f"Failed to read CSV file: {str(e)}")

def load_txt_file(filepath):
    """Load space/tab separated text file"""
    try:
        # Try to load as space-separated values
        points = np.loadtxt(filepath, usecols=(0, 1, 2))

        if points.ndim == 1:
            points = points.reshape(1, -1)

        if points.shape[1] < 3:
            raise ValueError("Text file must have at least 3 columns for X, Y, Z coordinates")

        # Take only first 3 columns
        points = points[:, :3]

        print(f"Loaded {len(points)} points from text file")
        return points

    except Exception as e:
        raise Exception(f"Failed to read text file: {str(e)}")

def filter_point_cloud_with_attributes(point_data, **kwargs):
    """
    Filter point cloud with attributes based on various criteria

    Parameters:
    - point_data: Dictionary containing 'points' and optional attributes
    - bbox: (min_x, min_y, min_z, max_x, max_y, max_z) bounding box
    - max_distance: maximum distance from origin
    - downsample: downsample factor (e.g., 10 means keep every 10th point)
    - z_range: (min_z, max_z) tuple to filter by height
    - intensity_range: (min_intensity, max_intensity) tuple to filter by intensity
    - classifications: list of classification codes to keep
    - return_numbers: list of return numbers to keep (e.g., [1] for first returns only)
    """
    points = point_data['points'].copy()

    # Initialize mask
    mask = np.ones(len(points), dtype=bool)

    # Bounding box filter
    if 'bbox' in kwargs:
        bbox = kwargs['bbox']
        if len(bbox) == 6:
            min_x, min_y, min_z, max_x, max_y, max_z = bbox
            bbox_mask = ((points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
                         (points[:, 1] >= min_y) & (points[:, 1] <= max_y) &
                         (points[:, 2] >= min_z) & (points[:, 2] <= max_z))
            mask &= bbox_mask
            print(f"Bounding box filter: {np.sum(mask)} points remaining")

    # Distance filter
    if 'max_distance' in kwargs:
        max_dist = kwargs['max_distance']
        distances = np.linalg.norm(points, axis=1)
        dist_mask = distances <= max_dist
        mask &= dist_mask
        print(f"Distance filter: {np.sum(mask)} points remaining")

    # Z-range filter
    if 'z_range' in kwargs:
        z_min, z_max = kwargs['z_range']
        z_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        mask &= z_mask
        print(f"Z-range filter: {np.sum(mask)} points remaining")

    # Intensity filter (LAS specific)
    if 'intensity_range' in kwargs and point_data.get('intensity') is not None:
        intensity_min, intensity_max = kwargs['intensity_range']
        intensity = point_data['intensity']
        intensity_mask = (intensity >= intensity_min) & (intensity <= intensity_max)
        mask &= intensity_mask
        print(f"Intensity filter: {np.sum(mask)} points remaining")

    # Classification filter (LAS specific)
    if 'classifications' in kwargs and point_data.get('classification') is not None:
        allowed_classes = kwargs['classifications']
        classification = point_data['classification']
        class_mask = np.isin(classification, allowed_classes)
        mask &= class_mask
        print(f"Classification filter: {np.sum(mask)} points remaining")

    # Return number filter (LAS specific)
    if 'return_numbers' in kwargs and point_data.get('return_number') is not None:
        allowed_returns = kwargs['return_numbers']
        return_number = point_data['return_number']
        return_mask = np.isin(return_number, allowed_returns)
        mask &= return_mask
        print(f"Return number filter: {np.sum(mask)} points remaining")

    # Apply mask to all data
    filtered_data = {'points': points[mask]}

    for key in ['intensity', 'classification', 'return_number', 'rgb']:
        if point_data.get(key) is not None:
            filtered_data[key] = point_data[key][mask]

    # Downsampling (applied last to maintain attribute alignment)
    if 'downsample' in kwargs:
        downsample_factor = kwargs['downsample']
        if downsample_factor > 1:
            ds_indices = np.arange(0, len(filtered_data['points']), downsample_factor)
            filtered_data['points'] = filtered_data['points'][ds_indices]

            for key in ['intensity', 'classification', 'return_number', 'rgb']:
                if filtered_data.get(key) is not None:
                    filtered_data[key] = filtered_data[key][ds_indices]

            print(f"Downsampling: {len(filtered_data['points'])} points remaining")

    return filtered_data

def get_point_cloud_info_with_attributes(point_data):
    """Get detailed statistics about the point cloud including attributes"""
    if isinstance(point_data, dict):
        points = point_data['points']
    else:
        points = point_data
        point_data = {'points': points}

    if len(points) == 0:
        return "Empty point cloud"

    info = []
    info.append(f"Number of points: {len(points):,}")
    info.append(f"X range: [{np.min(points[:, 0]):.2f}, {np.max(points[:, 0]):.2f}]")
    info.append(f"Y range: [{np.min(points[:, 1]):.2f}, {np.max(points[:, 1]):.2f}]")
    info.append(f"Z range: [{np.min(points[:, 2]):.2f}, {np.max(points[:, 2]):.2f}]")

    # Calculate centroid
    centroid = np.mean(points, axis=0)
    info.append(f"Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")

    # Calculate distances from origin
    distances = np.linalg.norm(points, axis=1)
    info.append(f"Distance range: [{np.min(distances):.2f}, {np.max(distances):.2f}] meters")

    # Intensity information
    if point_data.get('intensity') is not None:
        intensity = point_data['intensity']
        info.append(f"Intensity range: [{np.min(intensity)}, {np.max(intensity)}]")
        info.append(f"Intensity mean: {np.mean(intensity):.1f}")

    # Classification information
    if point_data.get('classification') is not None:
        classification = point_data['classification']
        unique_classes, counts = np.unique(classification, return_counts=True)
        info.append(f"Classifications: {dict(zip(unique_classes, counts))}")

    # Return number information
    if point_data.get('return_number') is not None:
        return_number = point_data['return_number']
        unique_returns, counts = np.unique(return_number, return_counts=True)
        info.append(f"Return numbers: {dict(zip(unique_returns, counts))}")

    # RGB information
    if point_data.get('rgb') is not None:
        info.append("RGB color data available")

    return "\n".join(info)

def filter_point_cloud(points, **kwargs):
    """
    Filter point cloud based on various criteria

    Parameters:
    - points: Nx3 numpy array
    - bbox: (min_x, min_y, min_z, max_x, max_y, max_z) bounding box
    - max_distance: maximum distance from origin
    - downsample: downsample factor (e.g., 10 means keep every 10th point)
    - z_range: (min_z, max_z) tuple to filter by height
    """
    filtered_points = points.copy()

    # Bounding box filter
    if 'bbox' in kwargs:
        bbox = kwargs['bbox']
        if len(bbox) == 6:
            min_x, min_y, min_z, max_x, max_y, max_z = bbox
            mask = ((filtered_points[:, 0] >= min_x) & (filtered_points[:, 0] <= max_x) &
                    (filtered_points[:, 1] >= min_y) & (filtered_points[:, 1] <= max_y) &
                    (filtered_points[:, 2] >= min_z) & (filtered_points[:, 2] <= max_z))
            filtered_points = filtered_points[mask]
            print(f"Bounding box filter: {len(filtered_points)} points remaining")

    # Distance filter
    if 'max_distance' in kwargs:
        max_dist = kwargs['max_distance']
        distances = np.linalg.norm(filtered_points, axis=1)
        mask = distances <= max_dist
        filtered_points = filtered_points[mask]
        print(f"Distance filter: {len(filtered_points)} points remaining")

    # Z-range filter
    if 'z_range' in kwargs:
        z_min, z_max = kwargs['z_range']
        mask = (filtered_points[:, 2] >= z_min) & (filtered_points[:, 2] <= z_max)
        filtered_points = filtered_points[mask]
        print(f"Z-range filter: {len(filtered_points)} points remaining")

    # Downsampling
    if 'downsample' in kwargs:
        downsample_factor = kwargs['downsample']
        if downsample_factor > 1:
            filtered_points = filtered_points[::downsample_factor]
            print(f"Downsampling: {len(filtered_points)} points remaining")

    return filtered_points

def get_point_cloud_info(points):
    """Get basic statistics about the point cloud"""
    if len(points) == 0:
        return "Empty point cloud"

    info = []
    info.append(f"Number of points: {len(points):,}")
    info.append(f"X range: [{np.min(points[:, 0]):.2f}, {np.max(points[:, 0]):.2f}]")
    info.append(f"Y range: [{np.min(points[:, 1]):.2f}, {np.max(points[:, 1]):.2f}]")
    info.append(f"Z range: [{np.min(points[:, 2]):.2f}, {np.max(points[:, 2]):.2f}]")

    # Calculate centroid
    centroid = np.mean(points, axis=0)
    info.append(f"Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")

    # Calculate distances from origin
    distances = np.linalg.norm(points, axis=1)
    info.append(f"Distance range: [{np.min(distances):.2f}, {np.max(distances):.2f}] meters")

    return "\n".join(info)

class EnhancedPointCloudFilterDialog:
    """Enhanced dialog for filtering point cloud options including LAS attributes"""

    def __init__(self, parent, point_data):
        self.result = None
        self.point_data = point_data

        # Determine if this is LAS data with attributes
        self.has_intensity = point_data.get('intensity') is not None
        self.has_classification = point_data.get('classification') is not None
        self.has_return_number = point_data.get('return_number') is not None
        self.has_rgb = point_data.get('rgb') is not None

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Point Cloud Filtering Options")
        dialog_height = 600 if (self.has_intensity or self.has_classification) else 500
        self.dialog.geometry(f"450x{dialog_height}")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (450 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (dialog_height // 2)
        self.dialog.geometry(f"450x{dialog_height}+{x}+{y}")

        self.create_widgets()

        # Wait for dialog to close
        self.dialog.wait_window()

    def create_widgets(self):
        # Create scrollable frame for all options
        canvas = tk.Canvas(self.dialog)
        scrollbar = tk.Scrollbar(self.dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Point cloud info
        info_frame = tk.Frame(scrollable_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(info_frame, text="Point Cloud Information:", font=("Arial", 10, "bold")).pack(anchor='w')
        info_text = tk.Text(info_frame, height=8, width=50)
        info_text.pack(fill=tk.X)
        info_text.insert(tk.END, get_point_cloud_info_with_attributes(self.point_data))
        info_text.config(state='disabled')

        # Basic filtering options
        filter_frame = tk.Frame(scrollable_frame)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(filter_frame, text="Basic Filtering Options:", font=("Arial", 10, "bold")).pack(anchor='w')

        # Distance filter
        dist_frame = tk.Frame(filter_frame)
        dist_frame.pack(fill=tk.X, pady=2)
        self.use_distance = tk.BooleanVar()
        tk.Checkbutton(dist_frame, text="Max distance from origin:", variable=self.use_distance).pack(side=tk.LEFT)
        self.max_distance = tk.Entry(dist_frame, width=10)
        self.max_distance.pack(side=tk.RIGHT)
        self.max_distance.insert(0, "100")

        # Z-range filter
        z_frame = tk.Frame(filter_frame)
        z_frame.pack(fill=tk.X, pady=2)
        self.use_z_range = tk.BooleanVar()
        tk.Checkbutton(z_frame, text="Z-range filter:", variable=self.use_z_range).pack(side=tk.LEFT)
        z_entry_frame = tk.Frame(z_frame)
        z_entry_frame.pack(side=tk.RIGHT)
        tk.Label(z_entry_frame, text="Min:").pack(side=tk.LEFT)
        self.z_min = tk.Entry(z_entry_frame, width=8)
        self.z_min.pack(side=tk.LEFT)
        tk.Label(z_entry_frame, text="Max:").pack(side=tk.LEFT)
        self.z_max = tk.Entry(z_entry_frame, width=8)
        self.z_max.pack(side=tk.LEFT)

        points = self.point_data['points']
        self.z_min.insert(0, f"{np.min(points[:, 2]):.1f}")
        self.z_max.insert(0, f"{np.max(points[:, 2]):.1f}")

        # Downsampling
        downsample_frame = tk.Frame(filter_frame)
        downsample_frame.pack(fill=tk.X, pady=2)
        self.use_downsample = tk.BooleanVar(value=True)
        tk.Checkbutton(downsample_frame, text="Downsample (keep every Nth point):",
                       variable=self.use_downsample).pack(side=tk.LEFT)
        self.downsample_factor = tk.Entry(downsample_frame, width=10)
        self.downsample_factor.pack(side=tk.RIGHT)

        # Set default downsample
        point_count = len(points)
        if point_count > 100000:
            default_downsample = 20
        elif point_count > 50000:
            default_downsample = 10
        elif point_count > 10000:
            default_downsample = 5
        else:
            default_downsample = 1
            self.use_downsample.set(False)

        self.downsample_factor.insert(0, str(default_downsample))

        # LAS-specific filtering options
        if self.has_intensity or self.has_classification or self.has_return_number:
            las_frame = tk.Frame(scrollable_frame)
            las_frame.pack(fill=tk.X, padx=10, pady=5)

            tk.Label(las_frame, text="LAS-Specific Filtering:", font=("Arial", 10, "bold")).pack(anchor='w')

            # Intensity filter
            if self.has_intensity:
                intensity_frame = tk.Frame(las_frame)
                intensity_frame.pack(fill=tk.X, pady=2)
                self.use_intensity = tk.BooleanVar()
                tk.Checkbutton(intensity_frame, text="Intensity range:", variable=self.use_intensity).pack(side=tk.LEFT)
                intensity_entry_frame = tk.Frame(intensity_frame)
                intensity_entry_frame.pack(side=tk.RIGHT)
                tk.Label(intensity_entry_frame, text="Min:").pack(side=tk.LEFT)
                self.intensity_min = tk.Entry(intensity_entry_frame, width=8)
                self.intensity_min.pack(side=tk.LEFT)
                tk.Label(intensity_entry_frame, text="Max:").pack(side=tk.LEFT)
                self.intensity_max = tk.Entry(intensity_entry_frame, width=8)
                self.intensity_max.pack(side=tk.LEFT)

                intensity = self.point_data['intensity']
                self.intensity_min.insert(0, f"{np.min(intensity)}")
                self.intensity_max.insert(0, f"{np.max(intensity)}")

            # Classification filter
            if self.has_classification:
                class_frame = tk.Frame(las_frame)
                class_frame.pack(fill=tk.X, pady=2)
                self.use_classification = tk.BooleanVar()
                tk.Checkbutton(class_frame, text="Classifications (comma-separated):",
                               variable=self.use_classification).pack(side=tk.LEFT)
                self.classifications = tk.Entry(class_frame, width=20)
                self.classifications.pack(side=tk.RIGHT)

                # Show available classifications
                unique_classes = np.unique(self.point_data['classification'])
                self.classifications.insert(0, ",".join(map(str, unique_classes)))

            # Return number filter
            if self.has_return_number:
                return_frame = tk.Frame(las_frame)
                return_frame.pack(fill=tk.X, pady=2)
                self.use_return_number = tk.BooleanVar()
                tk.Checkbutton(return_frame, text="Return numbers (comma-separated):",
                               variable=self.use_return_number).pack(side=tk.LEFT)
                self.return_numbers = tk.Entry(return_frame, width=15)
                self.return_numbers.pack(side=tk.RIGHT)

                unique_returns = np.unique(self.point_data['return_number'])
                self.return_numbers.insert(0, ",".join(map(str, unique_returns)))

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side="right", fill="y", pady=10, padx=(0, 10))

        # Buttons at bottom
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(button_frame, text="Apply Filters", command=self.apply_filters,
                  bg="lightgreen").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Use All Points", command=self.use_all_points,
                  bg="lightblue").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.RIGHT, padx=5)

    def apply_filters(self):
        try:
            filter_kwargs = {}

            # Basic filters
            if self.use_distance.get():
                max_dist = float(self.max_distance.get())
                filter_kwargs['max_distance'] = max_dist

            if self.use_z_range.get():
                z_min = float(self.z_min.get())
                z_max = float(self.z_max.get())
                filter_kwargs['z_range'] = (z_min, z_max)

            if self.use_downsample.get():
                downsample = int(self.downsample_factor.get())
                filter_kwargs['downsample'] = downsample

            # LAS-specific filters
            if self.has_intensity and self.use_intensity.get():
                intensity_min = int(self.intensity_min.get())
                intensity_max = int(self.intensity_max.get())
                filter_kwargs['intensity_range'] = (intensity_min, intensity_max)

            if self.has_classification and self.use_classification.get():
                class_str = self.classifications.get()
                class_list = [int(c.strip()) for c in class_str.split(',') if c.strip()]
                filter_kwargs['classifications'] = class_list

            if self.has_return_number and self.use_return_number.get():
                return_str = self.return_numbers.get()
                return_list = [int(r.strip()) for r in return_str.split(',') if r.strip()]
                filter_kwargs['return_numbers'] = return_list

            # Apply filters
            self.result = filter_point_cloud_with_attributes(self.point_data, **filter_kwargs)
            self.dialog.destroy()

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check your filter values: {e}")

    def use_all_points(self):
        self.result = self.point_data.copy()
        self.dialog.destroy()

    def cancel(self):
        self.result = None
        self.dialog.destroy()

class ColorOptionDialog:
    """Dialog for selecting color visualization option"""

    def __init__(self, parent, color_options):
        self.result = None
        self.color_options = color_options

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Color Visualization Options")
        self.dialog.geometry("350x250")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (350 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (250 // 2)
        self.dialog.geometry(f"350x250+{x}+{y}")

        self.create_widgets()

        # Wait for dialog to close
        self.dialog.wait_window()

    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.dialog)
        title_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(title_frame, text="Choose color visualization:",
                 font=("Arial", 12, "bold")).pack()

        # Color options
        option_frame = tk.Frame(self.dialog)
        option_frame.pack(fill=tk.X, padx=20, pady=10)

        self.color_var = tk.StringVar()

        for i, option in enumerate(self.color_options):
            rb = tk.Radiobutton(option_frame, text=option, variable=self.color_var,
                                value=option, font=("Arial", 10))
            rb.pack(anchor='w', pady=2)

            # Set first option as default
            if i == 0:
                self.color_var.set(option)

        # Description based on selection
        desc_frame = tk.Frame(self.dialog)
        desc_frame.pack(fill=tk.X, padx=20, pady=5)

        descriptions = {
            "Intensity": "LiDAR return intensity values\n(bright surfaces = high intensity)",
            "Distance from camera": "Distance in meters\n(blue = close, red = far)",
            "Height (Z coordinate)": "Elevation values\n(blue = low, red = high)",
            "RGB (original colors)": "Original colors from LiDAR\n(if available in the file)"
        }

        self.desc_label = tk.Label(desc_frame, text="", font=("Arial", 9),
                                   justify=tk.LEFT, fg="gray")
        self.desc_label.pack(anchor='w')

        # Update description when selection changes
        def update_description(*args):
            selected = self.color_var.get()
            desc = descriptions.get(selected, "")
            self.desc_label.config(text=desc)

        self.color_var.trace('w', update_description)
        update_description()  # Set initial description

        # Buttons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Button(button_frame, text="OK", command=self.ok_clicked,
                  bg="lightgreen").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.RIGHT, padx=5)

    def ok_clicked(self):
        self.result = self.color_var.get()
        self.dialog.destroy()

    def cancel_clicked(self):
        self.result = None
        self.dialog.destroy()

def get_classification_description(code):
    """Get human-readable description of LAS classification code"""
    return LAS_CLASSIFICATION_CODES.get(code, f"Unknown ({code})")

def show_classification_info():
    """Display LAS classification codes in a popup"""
    info_window = tk.Toplevel()
    info_window.title("LAS Classification Codes")
    info_window.geometry("400x500")

    # Create scrollable text
    text_frame = tk.Frame(info_window)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    text_widget = tk.Text(text_frame, wrap=tk.WORD)
    scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)

    # Add classification info
    text_widget.insert(tk.END, "LAS Classification Codes:\n\n")
    for code, description in LAS_CLASSIFICATION_CODES.items():
        text_widget.insert(tk.END, f"{code:2d}: {description}\n")

    text_widget.insert(tk.END, "\nCommon filtering examples:\n")
    text_widget.insert(tk.END, "• Ground only: 2\n")
    text_widget.insert(tk.END, "• Vegetation: 3,4,5\n")
    text_widget.insert(tk.END, "• Buildings: 6\n")
    text_widget.insert(tk.END, "• Ground + Buildings: 2,6\n")
    text_widget.insert(tk.END, "• Remove noise: exclude 7,18\n")

    text_widget.config(state=tk.DISABLED)

    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Close button
    tk.Button(info_window, text="Close", command=info_window.destroy).pack(pady=5)

def project_3d_to_2d(LIDAR_3D_PTS, camera_matrix_input, dist_coeffs_input, rvec, tvec):
    """
    Projects 3D points to 2D image points using a pinhole camera model.
    """
    rvec = np.asarray(rvec, dtype=np.float32)
    tvec = np.asarray(tvec, dtype=np.float32)
    points_2d, _ = cv2.projectPoints(LIDAR_3D_PTS, rvec, tvec, camera_matrix_input, dist_coeffs_input)
    points_2d = points_2d.reshape(-1, 2)
    return points_2d

def robust_objective_function(params, LIDAR_3D_PTS, IMAGE_2D_PTS, camera_matrix_input, dist_coeffs_input):
    """
    Enhanced objective function with better error handling and numerical stability.
    """
    try:
        rvec_opt = params[:3].reshape(3, 1)
        tvec_opt = params[3:].reshape(3, 1)

        # Check for reasonable parameter ranges
        if np.any(np.abs(rvec_opt) > 2 * np.pi):  # Rotation shouldn't exceed 2π
            return 1e10
        if np.any(np.abs(tvec_opt) > 1000):  # Translation shouldn't be extreme
            return 1e10

        projected_2d = project_3d_to_2d(LIDAR_3D_PTS, camera_matrix_input, dist_coeffs_input, rvec_opt, tvec_opt)

        # Check for invalid projections
        if np.any(np.isnan(projected_2d)) or np.any(np.isinf(projected_2d)):
            return 1e10

        # Check if points project to reasonable image coordinates
        if np.any(projected_2d < -1000) or np.any(projected_2d > 10000):
            return 1e10

        # Standard L2 error
        error = np.sum((projected_2d - IMAGE_2D_PTS) ** 2)

        return error

    except Exception as e:
        return 1e10

objective_function = robust_objective_function

def draw_points_on_image():
    global true_original_image, view_rect, opencv_window_shape, zoom_level
    global collected_image_points, temp_clicked_2d_point, true_original_image_dims

    if true_original_image is None or view_rect is None or opencv_window_shape is None:
        return

    # view_rect is in true_original_image coordinates
    roi_x, roi_y, roi_w, roi_h = map(int, view_rect)

    if roi_w <= 0 or roi_h <= 0:
        # Fallback in case of invalid ROI, though with proper view_rect management, this should be rare
        current_display_image = np.zeros((opencv_window_shape[1], opencv_window_shape[0], 3), dtype=np.uint8)
    else:
        # Extract the visible ROI from the *true_original_image*
        visible_roi_from_original = true_original_image[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
        # Resize this ROI to fit the OpenCV window
        current_display_image = cv2.resize(visible_roi_from_original, opencv_window_shape,
                                           interpolation=cv2.INTER_LINEAR)

    # Draw collected points: these are already in true original image coordinates
    for pt_orig in collected_image_points:
        ox, oy = map(int, pt_orig)
        # Check if the original point is within the current view_rect
        if ox >= roi_x and ox < roi_x + roi_w and oy >= roi_y and oy < roi_y + roi_h:
            # Calculate display coordinates relative to the ROI and then scale to window size
            disp_x = int(((ox - roi_x) / roi_w) * opencv_window_shape[0])
            disp_y = int(((oy - roi_y) / roi_h) * opencv_window_shape[1])
            cv2.circle(current_display_image, (disp_x, disp_y), POINT_RADIUS, POINT_COLOR, -1)

    # Draw the temporary clicked point: this is also in true original image coordinates
    if temp_clicked_2d_point:
        ox, oy = map(int, temp_clicked_2d_point)
        # Check if the original point is within the current view_rect
        if ox >= roi_x and ox < roi_x + roi_w and oy >= roi_y and oy < roi_y + roi_h:
            # Calculate display coordinates relative to the ROI and then scale to window size
            disp_x = int(((ox - roi_x) / roi_w) * opencv_window_shape[0])
            disp_y = int(((oy - roi_y) / roi_h) * opencv_window_shape[1])
            cv2.circle(current_display_image, (disp_x, disp_y), POINT_RADIUS, TEMP_POINT_COLOR, -1)

    cv2.imshow(WINDOW_NAME, current_display_image)
    cv2.waitKey(1)

def mouse_callback(event, x, y, flags, param):
    global temp_clicked_2d_point, label_2d_coord, true_original_image, view_rect, zoom_level, opencv_window_shape, true_original_image_dims

    if true_original_image is None or view_rect is None or opencv_window_shape is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        # x, y are the coordinates in the *displayed* OpenCV window
        clicked_display_x, clicked_display_y = x, y

        # Calculate proportional position within the displayed window
        prop_x_in_display = clicked_display_x / opencv_window_shape[0]
        prop_y_in_display = clicked_display_y / opencv_window_shape[1]

        # Calculate position within the current ROI (in true original image coordinates)
        # view_rect contains (x_start, y_start, width, height) of the ROI in true original image
        orig_x_in_roi = prop_x_in_display * view_rect[2]
        orig_y_in_roi = prop_y_in_display * view_rect[3]

        # Add the offset of the ROI to get the absolute position in the true original image
        orig_x = view_rect[0] + orig_x_in_roi
        orig_y = view_rect[1] + orig_y_in_roi

        temp_clicked_2d_point = (int(round(orig_x)), int(round(orig_y)))
        label_2d_coord.config(
            text=f"Clicked 2D (disp): ({clicked_display_x}, {clicked_display_y}) -> (orig): ({temp_clicked_2d_point[0]}, {temp_clicked_2d_point[1]})")
        draw_points_on_image()

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            # mouse_x_disp, mouse_y_disp are coordinates in the displayed window
            mouse_x_disp, mouse_y_disp = x, y

            # Calculate the corresponding mouse position in true original image coordinates
            # This is relative to the currently displayed ROI
            mouse_abs_x_in_original = view_rect[0] + (mouse_x_disp / opencv_window_shape[0]) * view_rect[2]
            mouse_abs_y_in_original = view_rect[1] + (mouse_y_disp / opencv_window_shape[1]) * view_rect[3]

            delta = 0
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except AttributeError:
                raw_delta = (flags >> 16) & 0xFFFF
                if raw_delta != 0:
                    if raw_delta > 32767:
                        delta = raw_delta - 65536
                    else:
                        delta = raw_delta
            if delta == 0:
                return

            old_zoom_level = zoom_level
            if delta > 0:  # Zoom in
                zoom_level *= ZOOM_FACTOR_STEP
            else:  # Zoom out
                zoom_level /= ZOOM_FACTOR_STEP
            zoom_level = max(MIN_ZOOM_LEVEL, min(zoom_level, MAX_ZOOM_LEVEL))

            # If zoom level hasn't effectively changed, or if it's already min zoom and view_rect covers whole image
            if abs(zoom_level - old_zoom_level) < 1e-3 and not \
                    (abs(zoom_level - MIN_ZOOM_LEVEL) < 1e-3 and \
                     tuple(map(int, view_rect)) == (0, 0, true_original_image_dims[0], true_original_image_dims[1])):
                return

            # Calculate new view_rect dimensions (in true original image coordinates)
            if abs(zoom_level - MIN_ZOOM_LEVEL) < 1e-3:
                # If zoomed out to min level, show the entire true original image
                new_vr_x, new_vr_y = 0, 0
                new_vr_w = true_original_image_dims[0]
                new_vr_h = true_original_image_dims[1]
            else:
                # Calculate new ROI width and height based on the new zoom level
                new_vr_w = true_original_image_dims[0] / zoom_level
                new_vr_h = true_original_image_dims[1] / zoom_level

                # Position the new ROI to keep the mouse point centered (as much as possible)
                new_vr_x = mouse_abs_x_in_original - (mouse_x_disp / opencv_window_shape[0]) * new_vr_w
                new_vr_y = mouse_abs_y_in_original - (mouse_y_disp / opencv_window_shape[1]) * new_vr_h

            # Ensure the new view_rect stays within the bounds of the true original image
            new_vr_x = max(0.0, new_vr_x)
            new_vr_y = max(0.0, new_vr_y)
            if new_vr_x + new_vr_w > true_original_image_dims[0]: new_vr_x = true_original_image_dims[0] - new_vr_w
            if new_vr_y + new_vr_h > true_original_image_dims[1]: new_vr_y = true_original_image_dims[1] - new_vr_h
            # Recalculate to ensure non-negative after boundary adjustments
            new_vr_x = max(0.0, new_vr_x)
            new_vr_y = max(0.0, new_vr_y)
            new_vr_w = min(new_vr_w, true_original_image_dims[0] - new_vr_x)
            new_vr_h = min(new_vr_h, true_original_image_dims[1] - new_vr_y)

            view_rect = (new_vr_x, new_vr_y, new_vr_w, new_vr_h)
            draw_points_on_image()

def load_image_action():
    global true_original_image, original_image, temp_clicked_2d_point, true_original_image_dims
    global zoom_level, view_rect, opencv_window_shape

    filepath = filedialog.askopenfilename(title="Select Image File",
                                          filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                                                     ("All files", "*.*")))
    if not filepath: return

    loaded_img = cv2.imread(filepath)
    if loaded_img is None:
        messagebox.showerror("Error", "Could not load image.")
        return

    true_original_image = loaded_img  # Store the full resolution image
    true_original_image_dims = (true_original_image.shape[1], true_original_image.shape[0])

    # Decide on the dimensions for the displayed image
    if true_original_image_dims[0] > MAX_IMAGE_DISPLAY_WIDTH:
        display_scale_factor = MAX_IMAGE_DISPLAY_WIDTH / true_original_image_dims[0]
        new_w = MAX_IMAGE_DISPLAY_WIDTH
        new_h = int(true_original_image_dims[1] * display_scale_factor)
        original_image = cv2.resize(true_original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        update_log(
            f"Image resized from {true_original_image_dims[0]}x{true_original_image_dims[1]} to {new_w}x{new_h} for display. Display Scale Factor: {display_scale_factor:.4f}")
    else:
        original_image = true_original_image.copy()  # Use a copy to avoid accidental modifications if true_original_image is later used directly
        update_log(
            f"Image loaded at original size: {true_original_image_dims[0]}x{true_original_image_dims[1]}. No display scaling applied.")

    opencv_window_shape = (original_image.shape[1], original_image.shape[0])

    zoom_level = MIN_ZOOM_LEVEL
    # view_rect always refers to the true_original_image coordinates
    view_rect = (0, 0, true_original_image_dims[0], true_original_image_dims[1])

    reset_points_action(clear_image=False)
    temp_clicked_2d_point = None
    label_2d_coord.config(text="Clicked 2D Point: N/A")
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    draw_points_on_image()
    update_log("Image loaded. Click to select points. Shift+Scroll to zoom.")

def add_point_pair_action():
    global temp_clicked_2d_point, collected_image_points, collected_lidar_points
    if temp_clicked_2d_point is None:
        messagebox.showerror("Error", "Please click a 2D point on the image first.")
        return
    try:
        lidar_x = float(entry_3d_x.get())
        lidar_y = float(entry_3d_y.get())
        lidar_z = float(entry_3d_z.get())

    except ValueError:
        messagebox.showerror("Error", "Invalid 3D coordinates. Please enter numbers.")
        return

    # temp_clicked_2d_point is ALREADY in true original image coordinates
    collected_image_points.append(list(temp_clicked_2d_point))
    collected_lidar_points.append([lidar_x, lidar_y, lidar_z])
    listbox_points.insert(tk.END, f"2D: {temp_clicked_2d_point} <-> 3D: ({lidar_x:.2f}, {lidar_y:.2f}, {lidar_z:.2f})")
    temp_clicked_2d_point = None
    draw_points_on_image()
    label_2d_coord.config(text="Clicked 2D Point: N/A")
    entry_3d_x.delete(0, tk.END);
    entry_3d_y.delete(0, tk.END);
    entry_3d_z.delete(0, tk.END)
    entry_3d_x.focus()
    update_log(f"Added point pair: {len(collected_image_points)}")

def run_calibration_action():
    global collected_image_points, collected_lidar_points
    global camera_matrix, dist_coeffs, last_calibration_results

    if len(collected_image_points) < 4:
        messagebox.showerror("Error", "Not enough point pairs. Please add at least 4 pairs.")
        return

        # ADD THIS NEW CODE:
        # Analyze point distribution before calibration
        if true_original_image is not None:
            dist_analysis = analyze_point_distribution(
                collected_image_points,
                collected_lidar_points,
                true_original_image.shape
            )

    LIDAR_3D_PTS_np = np.array(collected_lidar_points, dtype=np.float32)
    IMAGE_2D_PTS_np = np.array(collected_image_points, dtype=np.float32)

    current_camera_matrix = camera_matrix.copy()

    # === DIAGNOSTIC INFORMATION ===
    update_log("\n=== DIAGNOSTIC INFORMATION ===")
    update_log(f"Number of point pairs: {len(collected_image_points)}")
    update_log(f"3D points range:")
    update_log(f"  X: [{np.min(LIDAR_3D_PTS_np[:, 0]):.2f}, {np.max(LIDAR_3D_PTS_np[:, 0]):.2f}]")
    update_log(f"  Y: [{np.min(LIDAR_3D_PTS_np[:, 1]):.2f}, {np.max(LIDAR_3D_PTS_np[:, 1]):.2f}]")
    update_log(f"  Z: [{np.min(LIDAR_3D_PTS_np[:, 2]):.2f}, {np.max(LIDAR_3D_PTS_np[:, 2]):.2f}]")
    update_log(f"2D points range:")
    update_log(f"  X: [{np.min(IMAGE_2D_PTS_np[:, 0]):.0f}, {np.max(IMAGE_2D_PTS_np[:, 0]):.0f}]")
    update_log(f"  Y: [{np.min(IMAGE_2D_PTS_np[:, 1]):.0f}, {np.max(IMAGE_2D_PTS_np[:, 1]):.0f}]")
    update_log(f"Camera matrix:")
    update_log(f"  fx={current_camera_matrix[0, 0]:.2f}, fy={current_camera_matrix[1, 1]:.2f}")
    update_log(f"  cx={current_camera_matrix[0, 2]:.2f}, cy={current_camera_matrix[1, 2]:.2f}")

    # Show individual point pairs
    update_log("\nPoint correspondences:")
    for i, (lidar_pt, img_pt) in enumerate(zip(LIDAR_3D_PTS_np, IMAGE_2D_PTS_np)):
        update_log(
            f"  {i}: 3D({lidar_pt[0]:.2f}, {lidar_pt[1]:.2f}, {lidar_pt[2]:.2f}) -> 2D({img_pt[0]:.0f}, {img_pt[1]:.0f})")

    # === COORDINATE SYSTEM TRANSFORMATION ===
    transformations = {
        "no_transform": np.eye(3),
        "z_down_y_forward": np.array([
            [1, 0, 0],  # X_cam = X_lidar
            [0, 0, 1],  # Y_cam = Z_lidar (Z-down becomes Y-down)
            [0, 1, 0]  # Z_cam = Y_lidar (Y-forward becomes Z-forward)
        ]),
        "z_up_y_forward": np.array([
            [1, 0, 0],  # X_cam = X_lidar
            [0, 0, -1],  # Y_cam = -Z_lidar (Z-up becomes Y-down)
            [0, 1, 0]  # Z_cam = Y_lidar (Y-forward becomes Z-forward)
        ]),
        "standard_robotics": np.array([
            [0, 0, 1],  # X_cam = Z_lidar
            [-1, 0, 0],  # Y_cam = -X_lidar
            [0, -1, 0]  # Z_cam = -Y_lidar
        ])
    }

    # Test each transformation
    best_transform_name = None
    best_transform_error = float('inf')
    best_transformed_points = None

    update_log("\n=== TESTING COORDINATE TRANSFORMATIONS ===")

    for name, transform_matrix in transformations.items():
        transformed_pts = np.dot(LIDAR_3D_PTS_np, transform_matrix.T)

        # Check if all Z values are positive (points in front of camera)
        positive_z = np.all(transformed_pts[:, 2] > 0)

        update_log(f"\n{name}:")
        update_log(f"  Transformed range:")
        update_log(f"    X: [{np.min(transformed_pts[:, 0]):.2f}, {np.max(transformed_pts[:, 0]):.2f}]")
        update_log(f"    Y: [{np.min(transformed_pts[:, 1]):.2f}, {np.max(transformed_pts[:, 1]):.2f}]")
        update_log(f"    Z: [{np.min(transformed_pts[:, 2]):.2f}, {np.max(transformed_pts[:, 2]):.2f}]")
        update_log(f"  All Z positive: {positive_z}")

        if positive_z:
            try:
                success, rvec_test, tvec_test = cv2.solvePnP(
                    transformed_pts, IMAGE_2D_PTS_np,
                    current_camera_matrix, dist_coeffs
                )

                if success:
                    proj_pts = project_3d_to_2d(transformed_pts, current_camera_matrix, dist_coeffs,
                                                rvec_test, tvec_test)
                    error = np.mean(np.linalg.norm(IMAGE_2D_PTS_np - proj_pts, axis=1))
                    update_log(f"  solvePnP success: Mean error = {error:.2f} pixels")

                    if error < best_transform_error:
                        best_transform_error = error
                        best_transform_name = name
                        best_transformed_points = transformed_pts
                else:
                    update_log(f"  solvePnP failed")
            except Exception as e:
                update_log(f"  solvePnP error: {e}")
        else:
            update_log(f"  Skipped (negative Z values)")

    # Use the best transformation
    if best_transformed_points is not None:
        LIDAR_3D_FOR_CALIBRATION = best_transformed_points
        update_log(f"\n*** Using transformation: {best_transform_name} (error: {best_transform_error:.2f} px) ***")
    else:
        # Fallback
        transform_matrix = transformations["z_up_y_forward"]
        LIDAR_3D_FOR_CALIBRATION = np.dot(LIDAR_3D_PTS_np, transform_matrix.T)
        update_log(f"\n*** Fallback: Using z_up_y_forward transformation ***")

    # === OUTLIER DETECTION ===
    # Check for potential outliers using initial solvePnP result
    try:
        success, rvec_initial, tvec_initial = cv2.solvePnP(
            LIDAR_3D_FOR_CALIBRATION, IMAGE_2D_PTS_np,
            current_camera_matrix, dist_coeffs
        )

        if success:
            initial_proj = project_3d_to_2d(LIDAR_3D_FOR_CALIBRATION, current_camera_matrix, dist_coeffs,
                                            rvec_initial, tvec_initial)
            initial_errors = np.linalg.norm(IMAGE_2D_PTS_np - initial_proj, axis=1)

            update_log(f"\n=== OUTLIER ANALYSIS ===")
            update_log(f"Initial solvePnP errors per point:")
            for i, error in enumerate(initial_errors):
                status = ""
                if error > 20:
                    status = " ⚠ HIGH ERROR"
                elif error > 50:
                    status = " ❌ POTENTIAL OUTLIER"
                update_log(f"  Point {i}: {error:.2f} pixels{status}")

            mean_error = np.mean(initial_errors)
            std_error = np.std(initial_errors)
            update_log(f"Mean: {mean_error:.2f} px, Std: {std_error:.2f} px")

            # Identify potential outliers (> 2 standard deviations)
            outlier_threshold = mean_error + 2 * std_error
            outliers = np.where(initial_errors > outlier_threshold)[0]
            if len(outliers) > 0:
                update_log(f"Potential outliers (>{outlier_threshold:.1f} px): {outliers}")
    except:
        pass

    # === REFINED OPTIMIZATION ===
    initial_guesses = []

    # Primary guess: improved solvePnP
    try:
        success, rvec_cv, tvec_cv = cv2.solvePnP(
            LIDAR_3D_FOR_CALIBRATION, IMAGE_2D_PTS_np,
            current_camera_matrix, dist_coeffs
        )

        if success:
            # Try both SOLVEPNP_ITERATIVE and SOLVEPNP_EPNP for better accuracy
            methods = [cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP]
            if len(LIDAR_3D_FOR_CALIBRATION) >= 4:
                methods.append(cv2.SOLVEPNP_P3P)

            for method in methods:
                try:
                    success_iter, rvec_iter, tvec_iter = cv2.solvePnP(
                        LIDAR_3D_FOR_CALIBRATION, IMAGE_2D_PTS_np,
                        current_camera_matrix, dist_coeffs,
                        flags=method
                    )
                    if success_iter:
                        guess = np.concatenate([rvec_iter.flatten(), tvec_iter.flatten()])
                        initial_guesses.append((f"solvePnP_{method}", guess))
                except:
                    pass

            # Also add the basic solvePnP result
            solvepnp_guess = np.concatenate([rvec_cv.flatten(), tvec_cv.flatten()])
            initial_guesses.insert(0, ("solvePnP_basic", solvepnp_guess))

    except Exception as e:
        update_log(f"solvePnP error: {e}")

    # Add some refined guesses based on data characteristics
    if len(initial_guesses) > 0:
        base_guess = initial_guesses[0][1]
        # Small perturbations of the best guess
        for i in range(5):
            noise_r = np.random.normal(0, 0.05, 3)  # Very small rotation noise
            noise_t = np.random.normal(0, 0.5, 3)  # Small translation noise
            perturbed = base_guess + np.concatenate([noise_r, noise_t])
            initial_guesses.append((f"refined_perturbation_{i + 1}", perturbed))

    # === ADVANCED OPTIMIZATION ===
    best_result = None
    best_error = float('inf')

    update_log(f"\n=== ADVANCED OPTIMIZATION ({len(initial_guesses)} initial guesses) ===")

    for i, (name, initial_params) in enumerate(initial_guesses):
        update_log(f"\nTrying {name}:")

        # Calculate initial error
        initial_error = objective_function(initial_params, LIDAR_3D_FOR_CALIBRATION, IMAGE_2D_PTS_np,
                                           current_camera_matrix, dist_coeffs)

        if initial_error > 1e10:
            update_log(f"  Skipping (initial error too large: {initial_error:.2e})")
            continue

        update_log(f"  Initial error: {initial_error:.4e}")

        # Try multiple optimization methods with different tolerances
        optimization_configs = [
            ('SLSQP', {'ftol': 1e-9, 'maxiter': 20000}),
            ('L-BFGS-B', {'ftol': 1e-9, 'gtol': 1e-8, 'maxiter': 20000}),
            ('Powell', {'ftol': 1e-9, 'maxiter': 20000}),
        ]

        for method, options in optimization_configs:
            try:
                result = minimize(
                    objective_function,
                    initial_params,
                    args=(LIDAR_3D_FOR_CALIBRATION, IMAGE_2D_PTS_np, current_camera_matrix, dist_coeffs),
                    method=method,
                    options=dict(options, disp=False)
                )

                if result.success and result.fun < best_error:
                    best_result = result
                    best_error = result.fun
                    update_log(f"  *** NEW BEST: {method} - Error: {result.fun:.4e} ***")
                    break  # Found good result

            except Exception as e:
                continue

    # === STORE RESULTS AND DISPLAY ===
    if best_result is not None:
        optimized_params = best_result.x
        rvec_optimized = optimized_params[:3].reshape(3, 1)
        tvec_optimized = optimized_params[3:].reshape(3, 1)

        # Calculate final projections and errors
        final_projected_points = project_3d_to_2d(LIDAR_3D_FOR_CALIBRATION, current_camera_matrix, dist_coeffs,
                                                  rvec_optimized, tvec_optimized)
        reprojection_errors = np.linalg.norm(IMAGE_2D_PTS_np - final_projected_points, axis=1)

        # Store results globally for visualization
        last_calibration_results = {
            'rvec': rvec_optimized,
            'tvec': tvec_optimized,
            'transformed_3d_points': LIDAR_3D_FOR_CALIBRATION,
            'reprojection_errors': reprojection_errors,
            'projected_points': final_projected_points,
            'transform_name': best_transform_name,  # Store which transformation was used
            'camera_matrix': current_camera_matrix,
            'dist_coeffs': dist_coeffs
        }

        update_log("\n" + "=" * 50)
        update_log("FINAL CALIBRATION RESULTS")
        update_log("=" * 50)
        update_log(f"Optimization successful: {best_result.success}")
        update_log(f"Final SSE: {best_error:.4e}")
        update_log(
            f"Optimized rvec (radians): [{rvec_optimized[0, 0]:.6f}, {rvec_optimized[1, 0]:.6f}, {rvec_optimized[2, 0]:.6f}]")
        update_log(
            f"Optimized rvec (degrees): [{np.rad2deg(rvec_optimized[0, 0]):.2f}, {np.rad2deg(rvec_optimized[1, 0]):.2f}, {np.rad2deg(rvec_optimized[2, 0]):.2f}]")
        update_log(
            f"Optimized tvec (meters): [{tvec_optimized[0, 0]:.6f}, {tvec_optimized[1, 0]:.6f}, {tvec_optimized[2, 0]:.6f}]")

        # Rotation matrix for interpretation
        R_matrix, _ = cv2.Rodrigues(rvec_optimized)
        update_log(f"\nRotation Matrix:")
        for row in R_matrix:
            update_log(f"  [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}]")

        # Detailed reprojection analysis
        mean_error = np.mean(reprojection_errors)
        update_log(f"\nREPROJECTION ERROR ANALYSIS:")
        update_log(f"  Mean error: {mean_error:.4f} pixels")
        update_log(f"  Median error: {np.median(reprojection_errors):.4f} pixels")
        update_log(f"  Std deviation: {np.std(reprojection_errors):.4f} pixels")
        update_log(f"  Min error: {np.min(reprojection_errors):.4f} pixels")
        update_log(f"  Max error: {np.max(reprojection_errors):.4f} pixels")
        update_log(f"  RMS error: {np.sqrt(np.mean(reprojection_errors ** 2)):.4f} pixels")

        # Individual point analysis
        update_log(f"\nINDIVIDUAL POINT ERRORS:")
        for i, (error, orig_2d, proj_2d) in enumerate(
                zip(reprojection_errors, IMAGE_2D_PTS_np, final_projected_points)):
            status = ""
            if error > 10:
                status = " ⚠"
            elif error > 20:
                status = " ❌"
            update_log(f"  Point {i}: {error:.2f} px - Orig: ({orig_2d[0]:.1f}, {orig_2d[1]:.1f}), "
                       f"Proj: ({proj_2d[0]:.1f}, {proj_2d[1]:.1f}){status}")

        # Quality assessment
        update_log(f"\nQUALITY ASSESSMENT:")
        if mean_error < 2.0:
            update_log("  ✅ EXCELLENT calibration (< 2.0 px)")
        elif mean_error < 5.0:
            update_log("  ✅ VERY GOOD calibration (< 5.0 px)")
        elif mean_error < 10.0:
            update_log("  ✅ GOOD calibration (< 10.0 px)")
        elif mean_error < 20.0:
            update_log("  ⚠ ACCEPTABLE calibration (< 20.0 px)")
        else:
            update_log("  ❌ POOR calibration (> 20.0 px)")

        update_log(f"\n*** Calibration results stored for visualization ***")
        update_log("*** Click 'Visualize Projections' to see results overlaid on image ***")

        # Export format for use in other code
        update_log(f"\nFor use in your code:")
        update_log(
            f"rvec = np.array([{rvec_optimized[0, 0]:.6f}, {rvec_optimized[1, 0]:.6f}, {rvec_optimized[2, 0]:.6f}])")
        update_log(
            f"tvec = np.array([{tvec_optimized[0, 0]:.6f}, {tvec_optimized[1, 0]:.6f}, {tvec_optimized[2, 0]:.6f}])")

    else:
        update_log("\n❌ CALIBRATION FAILED - No results to visualize")
        update_log("All optimization attempts failed. Check:")
        update_log("1. Point correspondences are correct")
        update_log("2. Camera parameters are accurate")
        update_log("3. Coordinate system transformation is appropriate")
        update_log("4. Points are well-distributed in the image")

def analyze_point_contributions():
    """
    Analyze how each point contributes to the calibration error
    This helps identify problematic correspondences
    """
    global collected_image_points, collected_lidar_points, last_calibration_results

    if len(collected_image_points) < 4:
        update_log("Need at least 4 points for analysis")
        return

    LIDAR_3D_PTS_np = np.array(collected_lidar_points, dtype=np.float32)
    IMAGE_2D_PTS_np = np.array(collected_image_points, dtype=np.float32)

    # Apply the same coordinate transformation used in calibration
    transformations = {
        "z_up_y_forward": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        "z_down_y_forward": np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
        "no_transform": np.eye(3)
    }

    best_transform_name = "z_up_y_forward"  # Default, should match your last calibration
    if last_calibration_results:
        best_transform_name = last_calibration_results.get('transform_name', 'z_up_y_forward')

    transform_matrix = transformations[best_transform_name]
    LIDAR_3D_FOR_ANALYSIS = np.dot(LIDAR_3D_PTS_np, transform_matrix.T)

    update_log("\n=== POINT CONTRIBUTION ANALYSIS ===")
    update_log(f"Using coordinate transformation: {best_transform_name}")

    # Test calibration with each point removed (leave-one-out analysis)
    point_errors = []
    baseline_error = None

    for i in range(len(collected_image_points)):
        # Create dataset with point i removed
        mask = np.ones(len(collected_image_points), dtype=bool)
        mask[i] = False

        reduced_3d = LIDAR_3D_FOR_ANALYSIS[mask]
        reduced_2d = IMAGE_2D_PTS_np[mask]

        if len(reduced_3d) < 4:
            continue

        try:
            # Quick calibration with reduced dataset
            success, rvec_test, tvec_test = cv2.solvePnP(
                reduced_3d, reduced_2d, camera_matrix, dist_coeffs
            )

            if success:
                # Calculate error on the full dataset
                projected_all = project_3d_to_2d(LIDAR_3D_FOR_ANALYSIS, camera_matrix, dist_coeffs,
                                                 rvec_test, tvec_test)
                full_error = np.mean(np.linalg.norm(IMAGE_2D_PTS_np - projected_all, axis=1))

                # Calculate error without the removed point
                projected_reduced = project_3d_to_2d(reduced_3d, camera_matrix, dist_coeffs,
                                                     rvec_test, tvec_test)
                reduced_error = np.mean(np.linalg.norm(reduced_2d - projected_reduced, axis=1))

                point_errors.append((i, reduced_error, full_error))

                update_log(f"Without point {i}: Error = {reduced_error:.2f}px (full dataset: {full_error:.2f}px)")
            else:
                update_log(f"Without point {i}: solvePnP failed")

        except Exception as e:
            update_log(f"Without point {i}: Error - {e}")

    # Get baseline error with all points
    try:
        success, rvec_all, tvec_all = cv2.solvePnP(
            LIDAR_3D_FOR_ANALYSIS, IMAGE_2D_PTS_np, camera_matrix, dist_coeffs
        )
        if success:
            projected_all = project_3d_to_2d(LIDAR_3D_FOR_ANALYSIS, camera_matrix, dist_coeffs,
                                             rvec_all, tvec_all)
            baseline_error = np.mean(np.linalg.norm(IMAGE_2D_PTS_np - projected_all, axis=1))
            update_log(f"\nBaseline (all points): Error = {baseline_error:.2f}px")
    except:
        pass

    # Identify problematic points
    if point_errors and baseline_error:
        update_log(f"\n=== POINT QUALITY ANALYSIS ===")

        improvements = []
        for i, reduced_error, full_error in point_errors:
            improvement = baseline_error - reduced_error
            improvements.append((i, improvement, reduced_error))

        # Sort by improvement (largest improvement = most problematic point)
        improvements.sort(key=lambda x: x[1], reverse=True)

        update_log("Points ranked by potential problems (higher = more problematic):")
        for rank, (point_idx, improvement, error_without) in enumerate(improvements):
            status = ""
            if improvement > 5:
                status = " ❌ VERY PROBLEMATIC"
            elif improvement > 2:
                status = " ⚠ SUSPICIOUS"
            elif improvement > 0.5:
                status = " 🟡 MINOR ISSUE"
            else:
                status = " ✅ GOOD"

            lidar_pt = collected_lidar_points[point_idx]
            image_pt = collected_image_points[point_idx]
            update_log(f"  {rank + 1}. Point {point_idx}: Improvement={improvement:.2f}px{status}")
            update_log(f"     3D: ({lidar_pt[0]:.2f}, {lidar_pt[1]:.2f}, {lidar_pt[2]:.2f})")
            update_log(f"     2D: ({image_pt[0]:.0f}, {image_pt[1]:.0f})")

        # Suggest removing worst points
        worst_points = [idx for idx, imp, _ in improvements[:3] if imp > 2]
        if worst_points:
            update_log(f"\n💡 SUGGESTION: Consider removing points {worst_points}")
            update_log("   These points seem to be degrading the calibration quality")

def progressive_calibration_fixed():
    """
    Fixed progressive calibration that handles the 5-point solvePnP issue
    """
    global collected_image_points, collected_lidar_points

    if len(collected_image_points) < 4:
        update_log("Need at least 4 points for progressive calibration")
        return

    LIDAR_3D_PTS_np = np.array(collected_lidar_points, dtype=np.float32)
    IMAGE_2D_PTS_np = np.array(collected_image_points, dtype=np.float32)

    # Apply coordinate transformation
    transform_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # z_up_y_forward
    LIDAR_3D_TRANSFORMED = np.dot(LIDAR_3D_PTS_np, transform_matrix.T)

    update_log("\n=== FIXED PROGRESSIVE CALIBRATION ANALYSIS ===")

    # Find the best initial 4-point combination
    from itertools import combinations

    n_points = len(collected_image_points)
    best_4_error = float('inf')
    best_4_indices = None

    update_log("Testing all possible 4-point combinations...")

    for combo in combinations(range(n_points), 4):
        indices = list(combo)
        test_3d = LIDAR_3D_TRANSFORMED[indices]
        test_2d = IMAGE_2D_PTS_np[indices]

        try:
            # Use SOLVEPNP_ITERATIVE which works with 4 points
            success, rvec, tvec = cv2.solvePnP(
                test_3d, test_2d, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                projected = project_3d_to_2d(test_3d, camera_matrix, dist_coeffs, rvec, tvec)
                error = np.mean(np.linalg.norm(test_2d - projected, axis=1))

                if error < best_4_error:
                    best_4_error = error
                    best_4_indices = indices
        except Exception as e:
            continue

    if best_4_indices is None:
        update_log("Could not find a good 4-point combination")
        return

    update_log(f"Best 4-point combination: {best_4_indices} (error: {best_4_error:.2f}px)")

    # Show which points these are
    update_log("Best 4 points details:")
    for i, idx in enumerate(best_4_indices):
        lidar_pt = collected_lidar_points[idx]
        image_pt = collected_image_points[idx]
        update_log(
            f"  Point {idx}: 3D({lidar_pt[0]:.2f}, {lidar_pt[1]:.2f}, {lidar_pt[2]:.2f}) → 2D({image_pt[0]:.0f}, {image_pt[1]:.0f})")

    # Progressive addition with proper 6-point handling
    used_indices = set(best_4_indices)
    remaining_indices = [i for i in range(n_points) if i not in used_indices]

    current_indices = list(best_4_indices)
    current_error = best_4_error

    update_log(f"\nProgressive point addition:")
    update_log(f"Starting with 4 points: {current_indices}, Error: {current_error:.2f}px")

    # For 5-point combinations, we need to use a different approach
    # Let's try combinations of 6 points (original 4 + 2 new ones)
    if len(remaining_indices) >= 2:
        update_log("\nTesting 6-point combinations (4 base + 2 additional):")

        best_6_error = float('inf')
        best_6_indices = None

        for combo in combinations(remaining_indices, 2):
            test_indices = current_indices + list(combo)
            test_3d = LIDAR_3D_TRANSFORMED[test_indices]
            test_2d = IMAGE_2D_PTS_np[test_indices]

            try:
                success, rvec, tvec = cv2.solvePnP(test_3d, test_2d, camera_matrix, dist_coeffs)
                if success:
                    projected = project_3d_to_2d(test_3d, camera_matrix, dist_coeffs, rvec, tvec)
                    error = np.mean(np.linalg.norm(test_2d - projected, axis=1))

                    change = error - current_error
                    status = ""
                    if change < -1:
                        status = " ✅ SIGNIFICANT IMPROVEMENT"
                    elif change < -0.1:
                        status = " 🟢 IMPROVEMENT"
                    elif change < 1:
                        status = " 🟡 MINOR DEGRADATION"
                    else:
                        status = " ❌ SIGNIFICANT DEGRADATION"

                    update_log(f"Adding points {list(combo)}: Error {error:.2f}px (change: {change:+.2f}px){status}")

                    if error < best_6_error:
                        best_6_error = error
                        best_6_indices = test_indices

            except Exception as e:
                update_log(f"Adding points {list(combo)}: Error - {str(e)[:50]}...")

    # Now test individual remaining points with the current best set
    update_log(f"\nTesting individual point additions:")

    # Use the best 6-point set if it's better, otherwise stick with 4
    if best_6_indices and best_6_error < current_error + 1:
        current_indices = best_6_indices
        current_error = best_6_error
        update_log(f"Updated base set to: {current_indices}, Error: {current_error:.2f}px")
        used_indices = set(current_indices)
        remaining_indices = [i for i in range(n_points) if i not in used_indices]

    for remaining_idx in remaining_indices:
        test_indices = current_indices + [remaining_idx]
        test_3d = LIDAR_3D_TRANSFORMED[test_indices]
        test_2d = IMAGE_2D_PTS_np[test_indices]

        try:
            success, rvec, tvec = cv2.solvePnP(test_3d, test_2d, camera_matrix, dist_coeffs)
            if success:
                projected = project_3d_to_2d(test_3d, camera_matrix, dist_coeffs, rvec, tvec)
                error = np.mean(np.linalg.norm(test_2d - projected, axis=1))

                change = error - current_error
                status = ""
                if change < -1:
                    status = " ✅ SIGNIFICANT IMPROVEMENT"
                elif change < -0.1:
                    status = " 🟢 IMPROVEMENT"
                elif change < 1:
                    status = " 🟡 MINOR DEGRADATION"
                else:
                    status = " ❌ SIGNIFICANT DEGRADATION"

                # Show point details for problematic points
                if change > 2:
                    lidar_pt = collected_lidar_points[remaining_idx]
                    image_pt = collected_image_points[remaining_idx]
                    update_log(f"Adding point {remaining_idx}: Error {error:.2f}px (change: {change:+.2f}px){status}")
                    update_log(
                        f"  → Point {remaining_idx} details: 3D({lidar_pt[0]:.2f}, {lidar_pt[1]:.2f}, {lidar_pt[2]:.2f}) → 2D({image_pt[0]:.0f}, {image_pt[1]:.0f})")
                else:
                    update_log(f"Adding point {remaining_idx}: Error {error:.2f}px (change: {change:+.2f}px){status}")

                # Accept points that don't degrade too much
                if change < 2.0:  # Accept if degradation is less than 2 pixels
                    current_indices.append(remaining_idx)
                    current_error = error
                    update_log(f"  → Point {remaining_idx} ACCEPTED")
                else:
                    update_log(f"  → Point {remaining_idx} REJECTED (too much degradation)")
                    if change > 5:
                        update_log(f"      ⚠ This point likely has correspondence errors!")

        except Exception as e:
            update_log(f"Adding point {remaining_idx}: Error - {str(e)[:50]}...")

    update_log(f"\n📊 RECOMMENDED POINT SET: {current_indices}")
    update_log(f"📊 FINAL ERROR: {current_error:.2f}px")

    # Show problematic points
    if len(current_indices) < len(collected_image_points):
        rejected_points = [i for i in range(len(collected_image_points)) if i not in current_indices]
        update_log(f"📊 SUGGESTED REMOVAL: Points {rejected_points}")

        update_log(f"\nProblematic points to check:")
        for point_idx in rejected_points:
            lidar_pt = collected_lidar_points[point_idx]
            image_pt = collected_image_points[point_idx]
            update_log(
                f"  Point {point_idx}: 3D({lidar_pt[0]:.2f}, {lidar_pt[1]:.2f}, {lidar_pt[2]:.2f}) → 2D({image_pt[0]:.0f}, {image_pt[1]:.0f})")
            update_log(f"⚠ Check if this 2D-3D correspondence is correct!")

def quick_point_error_analysis():
    """
    Quick analysis to identify the most problematic points
    """
    global collected_image_points, collected_lidar_points

    if len(collected_image_points) < 4:
        update_log("Need at least 4 points for analysis")
        return

    LIDAR_3D_PTS_np = np.array(collected_lidar_points, dtype=np.float32)
    IMAGE_2D_PTS_np = np.array(collected_image_points, dtype=np.float32)

    # Apply coordinate transformation
    transform_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    LIDAR_3D_TRANSFORMED = np.dot(LIDAR_3D_PTS_np, transform_matrix.T)

    update_log("\n=== QUICK POINT ERROR ANALYSIS ===")

    # Get baseline calibration with all points
    try:
        success, rvec_all, tvec_all = cv2.solvePnP(
            LIDAR_3D_TRANSFORMED, IMAGE_2D_PTS_np, camera_matrix, dist_coeffs
        )

        if success:
            projected_all = project_3d_to_2d(LIDAR_3D_TRANSFORMED, camera_matrix, dist_coeffs,
                                             rvec_all, tvec_all)
            individual_errors = np.linalg.norm(IMAGE_2D_PTS_np - projected_all, axis=1)
            mean_error = np.mean(individual_errors)

            update_log(f"All points calibration:")
            update_log(f"  Mean error: {mean_error:.2f}px")
            update_log(f"  Individual errors:")

            # Sort points by error
            error_ranking = [(i, err) for i, err in enumerate(individual_errors)]
            error_ranking.sort(key=lambda x: x[1], reverse=True)

            for rank, (point_idx, error) in enumerate(error_ranking):
                status = ""
                if error > mean_error + 5:
                    status = " ❌ MAJOR PROBLEM"
                elif error > mean_error + 2:
                    status = " ⚠ SUSPICIOUS"
                elif error > mean_error:
                    status = " 🟡 ABOVE AVERAGE"
                else:
                    status = " ✅ GOOD"

                lidar_pt = collected_lidar_points[point_idx]
                image_pt = collected_image_points[point_idx]
                update_log(f"    {rank + 1}. Point {point_idx}: {error:.2f}px{status}")
                update_log(
                    f"       3D({lidar_pt[0]:.2f}, {lidar_pt[1]:.2f}, {lidar_pt[2]:.2f}) → 2D({image_pt[0]:.0f}, {image_pt[1]:.0f})")

            # Identify worst points
            worst_threshold = mean_error + 3
            worst_points = [idx for idx, err in error_ranking if err > worst_threshold]

            if worst_points:
                update_log(f"\n🚨 POINTS TO CHECK: {worst_points}")
                update_log("These points have significantly higher errors and should be verified.")

    except Exception as e:
        update_log(f"Error in analysis: {e}")

def attempt_without_worst_points():
    """
    Test calibration after removing the worst performing points
    """
    global collected_image_points, collected_lidar_points

    if len(collected_image_points) < 6:
        update_log("Need at least 6 points to test point removal")
        return

    LIDAR_3D_PTS_np = np.array(collected_lidar_points, dtype=np.float32)
    IMAGE_2D_PTS_np = np.array(collected_image_points, dtype=np.float32)

    # Apply coordinate transformation
    transform_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    LIDAR_3D_TRANSFORMED = np.dot(LIDAR_3D_PTS_np, transform_matrix.T)

    update_log("\n=== TESTING WITHOUT WORST POINTS ===")

    # Get individual errors
    try:
        success, rvec_all, tvec_all = cv2.solvePnP(
            LIDAR_3D_TRANSFORMED, IMAGE_2D_PTS_np, camera_matrix, dist_coeffs
        )

        if success:
            projected_all = project_3d_to_2d(LIDAR_3D_TRANSFORMED, camera_matrix, dist_coeffs,
                                             rvec_all, tvec_all)
            individual_errors = np.linalg.norm(IMAGE_2D_PTS_np - projected_all, axis=1)
            mean_error = np.mean(individual_errors)

            update_log(f"Baseline (all points): {mean_error:.2f}px")

            # Remove worst 1, 2, 3 points and test
            error_ranking = [(i, err) for i, err in enumerate(individual_errors)]
            error_ranking.sort(key=lambda x: x[1], reverse=True)

            for num_to_remove in [1, 2, 3]:
                if len(collected_image_points) - num_to_remove < 4:
                    break

                # Remove worst N points
                worst_indices = [idx for idx, _ in error_ranking[:num_to_remove]]
                keep_mask = np.ones(len(collected_image_points), dtype=bool)
                keep_mask[worst_indices] = False

                reduced_3d = LIDAR_3D_TRANSFORMED[keep_mask]
                reduced_2d = IMAGE_2D_PTS_np[keep_mask]

                try:
                    success_reduced, rvec_reduced, tvec_reduced = cv2.solvePnP(
                        reduced_3d, reduced_2d, camera_matrix, dist_coeffs
                    )

                    if success_reduced:
                        projected_reduced = project_3d_to_2d(reduced_3d, camera_matrix, dist_coeffs,
                                                             rvec_reduced, tvec_reduced)
                        reduced_error = np.mean(np.linalg.norm(reduced_2d - projected_reduced, axis=1))

                        improvement = mean_error - reduced_error

                        status = ""
                        if improvement > 2:
                            status = " 🎉 MAJOR IMPROVEMENT"
                        elif improvement > 0.5:
                            status = " ✅ GOOD IMPROVEMENT"
                        elif improvement > 0:
                            status = " 🟢 SLIGHT IMPROVEMENT"
                        else:
                            status = " 🔴 NO IMPROVEMENT"

                        update_log(
                            f"Remove {num_to_remove} worst point(s) {worst_indices}: {reduced_error:.2f}px (improvement: {improvement:+.2f}px){status}")

                except Exception as e:
                    update_log(f"Remove {num_to_remove} points: Error - {str(e)[:50]}...")

    except Exception as e:
        update_log(f"Error in analysis: {e}")

def robust_calibration_with_ransac():
    """
    RANSAC-based calibration to handle outliers automatically
    """
    global collected_image_points, collected_lidar_points

    if len(collected_image_points) < 6:  # Need extra points for RANSAC
        update_log("Need at least 6 points for RANSAC calibration")
        return

    LIDAR_3D_PTS_np = np.array(collected_lidar_points, dtype=np.float32)
    IMAGE_2D_PTS_np = np.array(collected_image_points, dtype=np.float32)

    # Apply coordinate transformation
    transform_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    LIDAR_3D_TRANSFORMED = np.dot(LIDAR_3D_PTS_np, transform_matrix.T)

    update_log("\n=== ROBUST RANSAC CALIBRATION ===")

    from itertools import combinations
    import random

    n_points = len(collected_image_points)
    max_iterations = min(1000, len(list(combinations(range(n_points), 4))))
    inlier_threshold = 3.0  # pixels

    best_inliers = []
    best_error = float('inf')
    best_model = None

    update_log(f"Running RANSAC with {max_iterations} iterations...")

    for iteration in range(max_iterations):
        # Randomly sample 4 points
        sample_indices = random.sample(range(n_points), 4)
        sample_3d = LIDAR_3D_TRANSFORMED[sample_indices]
        sample_2d = IMAGE_2D_PTS_np[sample_indices]

        try:
            # Fit model to sample
            success, rvec, tvec = cv2.solvePnP(sample_3d, sample_2d, camera_matrix, dist_coeffs)
            if not success:
                continue

            # Test model on all points
            projected_all = project_3d_to_2d(LIDAR_3D_TRANSFORMED, camera_matrix, dist_coeffs,
                                             rvec, tvec)
            errors = np.linalg.norm(IMAGE_2D_PTS_np - projected_all, axis=1)

            # Find inliers
            inliers = np.where(errors < inlier_threshold)[0]

            if len(inliers) > len(best_inliers):
                # Refit model using all inliers
                try:
                    inlier_3d = LIDAR_3D_TRANSFORMED[inliers]
                    inlier_2d = IMAGE_2D_PTS_np[inliers]

                    success_inlier, rvec_inlier, tvec_inlier = cv2.solvePnP(
                        inlier_3d, inlier_2d, camera_matrix, dist_coeffs
                    )

                    if success_inlier:
                        projected_inlier = project_3d_to_2d(inlier_3d, camera_matrix, dist_coeffs,
                                                            rvec_inlier, tvec_inlier)
                        inlier_error = np.mean(np.linalg.norm(inlier_2d - projected_inlier, axis=1))

                        best_inliers = inliers
                        best_error = inlier_error
                        best_model = (rvec_inlier, tvec_inlier)

                except:
                    continue

        except:
            continue

    if best_model is None:
        update_log("RANSAC failed to find a good model")
        return

    update_log(f"RANSAC Results:")
    update_log(f"  Inliers: {len(best_inliers)}/{n_points} points")
    update_log(f"  Inlier indices: {list(best_inliers)}")
    update_log(f"  Mean error (inliers only): {best_error:.2f}px")

    outliers = [i for i in range(n_points) if i not in best_inliers]
    if outliers:
        update_log(f"  Outliers detected: {outliers}")
        update_log("  These points should be checked for correspondence errors")

        for outlier_idx in outliers:
            lidar_pt = collected_lidar_points[outlier_idx]
            image_pt = collected_image_points[outlier_idx]
            update_log(
                f"    Point {outlier_idx}: 3D({lidar_pt[0]:.2f}, {lidar_pt[1]:.2f}, {lidar_pt[2]:.2f}) → 2D({image_pt[0]:.0f}, {image_pt[1]:.0f})")

    # Store results for visualization
    global last_calibration_results
    if last_calibration_results is None:
        last_calibration_results = {}

    rvec_opt, tvec_opt = best_model
    projected_all = project_3d_to_2d(LIDAR_3D_TRANSFORMED, camera_matrix, dist_coeffs,
                                     rvec_opt, tvec_opt)
    all_errors = np.linalg.norm(IMAGE_2D_PTS_np - projected_all, axis=1)

    last_calibration_results.update({
        'rvec': rvec_opt,
        'tvec': tvec_opt,
        'transformed_3d_points': LIDAR_3D_TRANSFORMED,
        'reprojection_errors': all_errors,
        'projected_points': projected_all,
        'inliers': best_inliers,
        'outliers': outliers
    })

    update_log("RANSAC calibration results stored for visualization")

def analyze_point_distribution(image_points, lidar_points, image_shape):
    """
    Analyze if points are well-distributed for calibration
    """
    if len(image_points) < 4:
        return None

    img_h, img_w = image_shape[:2]

    # Convert to numpy arrays
    img_pts = np.array(image_points)
    lidar_pts = np.array(lidar_points)

    # Check image coverage
    x_coords = img_pts[:, 0]
    y_coords = img_pts[:, 1]

    x_coverage = (np.max(x_coords) - np.min(x_coords)) / img_w
    y_coverage = (np.max(y_coords) - np.min(y_coords)) / img_h

    # Check 3D spread
    depth_values = lidar_pts[:, 2]  # Assuming Z is depth
    depth_range = np.max(depth_values) - np.min(depth_values)

    # Check spatial distribution using grid
    grid_size = 3
    grid_counts = np.zeros((grid_size, grid_size))

    for pt in img_pts:
        grid_x = int(pt[0] * grid_size / img_w)
        grid_y = int(pt[1] * grid_size / img_h)
        grid_x = min(grid_x, grid_size - 1)
        grid_y = min(grid_y, grid_size - 1)
        grid_counts[grid_y, grid_x] += 1

    empty_cells = np.sum(grid_counts == 0)
    distribution_score = 1.0 - (empty_cells / (grid_size * grid_size))

    # Generate report
    update_log("\n=== POINT DISTRIBUTION ANALYSIS ===")
    update_log(f"Total points: {len(image_points)}")
    update_log(f"Image coverage: X={x_coverage * 100:.1f}%, Y={y_coverage * 100:.1f}%")
    update_log(f"Depth range: {depth_range:.1f}m (min={np.min(depth_values):.1f}m, max={np.max(depth_values):.1f}m)")
    update_log(f"Spatial distribution score: {distribution_score * 100:.1f}%")
    update_log(f"Grid occupancy ({grid_size}x{grid_size}):")

    for i in range(grid_size):
        row_str = "  "
        for j in range(grid_size):
            count = int(grid_counts[i, j])
            if count == 0:
                row_str += "[  ] "
            else:
                row_str += f"[{count:2d}] "
        update_log(row_str)

    # Recommendations
    if x_coverage < 0.5 or y_coverage < 0.5:
        update_log("\n⚠️ WARNING: Insufficient image coverage!")
        update_log("   → Select points from image corners and edges")

    if depth_range < 5.0:
        update_log("\n⚠️ WARNING: Insufficient depth variation!")
        update_log("   → Select points at varying distances (near and far)")

    if distribution_score < 0.6:
        update_log("\n⚠️ WARNING: Poor spatial distribution!")
        update_log("   → Add points in empty grid cells shown above")

    if len(image_points) < 8:
        update_log("\n💡 TIP: Consider adding more points (8-15 recommended)")

    # Check for collinear points
    if len(image_points) >= 3:
        # Use SVD to check collinearity
        centered_pts = img_pts - np.mean(img_pts, axis=0)
        _, s, _ = np.linalg.svd(centered_pts)
        if s[1] < 0.1 * s[0]:  # Second singular value very small
            update_log("\n⚠️ WARNING: Points appear to be nearly collinear!")
            update_log("   → Select points that form a good 2D spread")

    return {
        'x_coverage': x_coverage,
        'y_coverage': y_coverage,
        'depth_range': depth_range,
        'distribution_score': distribution_score,
        'grid_counts': grid_counts
    }

def bundle_adjustment(points_3d, points_2d, camera_matrix, dist_coeffs, rvec_init, tvec_init):
    """
    Refine pose using bundle adjustment (Levenberg-Marquardt optimization)
    """

    def residuals(params):
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:6].reshape(3, 1)

        projected = project_3d_to_2d(points_3d, camera_matrix, dist_coeffs, rvec, tvec)
        return (projected - points_2d).flatten()

    def jacobian_sparsity():
        # Each 3D point affects only its corresponding 2D projection
        n_points = len(points_3d)
        n_params = 6  # rvec (3) + tvec (3)

        # Create sparsity pattern
        m = 2 * n_points  # 2D points have x,y
        sparsity = lil_matrix((m, n_params), dtype=int)

        for i in range(n_points):
            # Each point's x,y residual depends on all 6 params
            sparsity[2 * i:2 * i + 2, :] = 1

        return sparsity

    x0 = np.hstack([rvec_init.flatten(), tvec_init.flatten()])

    result = least_squares(
        residuals,
        x0,
        method='lm',  # Levenberg-Marquardt
        jac_sparsity=jacobian_sparsity(),
        max_nfev=1000,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8
    )

    rvec_opt = result.x[:3].reshape(3, 1)
    tvec_opt = result.x[3:6].reshape(3, 1)

    return rvec_opt, tvec_opt, result

def robust_pnp_calibration(lidar_points, image_points, camera_matrix, dist_coeffs):
    """
    Multi-stage calibration with outlier rejection using RANSAC
    """
    update_log("\n=== ROBUST CALIBRATION WITH RANSAC ===")

    # Convert to proper format
    lidar_pts = np.array(lidar_points, dtype=np.float32)
    image_pts = np.array(image_points, dtype=np.float32)

    # Stage 1: RANSAC-based PnP
    update_log("Stage 1: RANSAC outlier detection...")

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        lidar_pts,
        image_pts,
        camera_matrix,
        dist_coeffs,
        reprojectionError=3.0,  #pixels
        confidence=0.999,
        iterationsCount=2000,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success or inliers is None:
        update_log("❌ RANSAC failed to find a solution")
        return None

    inliers = inliers.flatten()
    outliers = np.setdiff1d(np.arange(len(lidar_pts)), inliers)

    update_log(f"  Inliers: {len(inliers)}/{len(lidar_pts)} points")
    update_log(f"  Outliers: {list(outliers)}")

    if len(outliers) > 0:
        update_log("  Outlier details:")
        for idx in outliers:
            update_log(
                f"    Point {idx}: 3D({lidar_points[idx][0]:.2f}, {lidar_points[idx][1]:.2f}, {lidar_points[idx][2]:.2f})")

    # Get inlier points
    inlier_3d = lidar_pts[inliers]
    inlier_2d = image_pts[inliers]

    # Stage 2: Try different PnP methods with inliers
    update_log("\nStage 2: Testing multiple PnP algorithms...")

    methods = [
        (cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"),
        (cv2.SOLVEPNP_EPNP, "EPNP"),
        (cv2.SOLVEPNP_P3P if len(inlier_3d) >= 4 else None, "P3P"),
        (cv2.SOLVEPNP_AP3P if len(inlier_3d) >= 4 else None, "AP3P"),
        (cv2.SOLVEPNP_IPPE if len(inlier_3d) >= 4 else None, "IPPE"),
    ]

    best_result = None
    best_error = float('inf')
    best_method = None

    for method_flag, method_name in methods:
        if method_flag is None:
            continue

        try:
            success, rvec_refined, tvec_refined = cv2.solvePnP(
                inlier_3d,
                inlier_2d,
                camera_matrix,
                dist_coeffs,
                rvec.copy(),
                tvec.copy(),
                useExtrinsicGuess=True,
                flags=method_flag
            )

            if success:
                # Evaluate
                proj = project_3d_to_2d(inlier_3d, camera_matrix, dist_coeffs, rvec_refined, tvec_refined)
                errors = np.linalg.norm(inlier_2d - proj, axis=1)
                mean_error = np.mean(errors)

                update_log(f"  {method_name}: Mean error = {mean_error:.3f} px")

                if mean_error < best_error:
                    best_error = mean_error
                    best_result = (rvec_refined.copy(), tvec_refined.copy())
                    best_method = method_name

        except Exception as e:
            update_log(f"  {method_name}: Failed - {str(e)[:50]}...")

    if best_result is None:
        update_log("❌ All PnP methods failed")
        return None

    update_log(f"\n  Best method: {best_method} (error: {best_error:.3f} px)")

    # Stage 3: Bundle adjustment refinement
    update_log("\nStage 3: Bundle adjustment refinement...")

    rvec_best, tvec_best = best_result

    try:
        rvec_ba, tvec_ba, ba_result = bundle_adjustment(
            inlier_3d,
            inlier_2d,
            camera_matrix,
            dist_coeffs,
            rvec_best,
            tvec_best
        )

        # Check if bundle adjustment improved the result
        proj_ba = project_3d_to_2d(inlier_3d, camera_matrix, dist_coeffs, rvec_ba, tvec_ba)
        error_ba = np.mean(np.linalg.norm(inlier_2d - proj_ba, axis=1))

        update_log(f"  Bundle adjustment: {best_error:.3f} → {error_ba:.3f} px")
        update_log(f"  Optimization iterations: {ba_result.nfev}")
        update_log(f"  Optimization status: {ba_result.message}")

        if error_ba < best_error:
            best_result = (rvec_ba, tvec_ba)
            best_error = error_ba
            update_log("Bundle adjustment improved the result")
        else:
            update_log("Bundle adjustment did not improve the result")

    except Exception as e:
        update_log(f"Bundle adjustment failed: {str(e)[:100]}...")

    # Return complete results
    rvec_final, tvec_final = best_result

    # Calculate final statistics with all points
    all_proj = project_3d_to_2d(lidar_pts, camera_matrix, dist_coeffs, rvec_final, tvec_final)
    all_errors = np.linalg.norm(image_pts - all_proj, axis=1)

    return {
        'rvec': rvec_final,
        'tvec': tvec_final,
        'inliers': inliers,
        'outliers': outliers,
        'all_errors': all_errors,
        'mean_error': np.mean(all_errors[inliers]),
        'method': best_method
    }

def show_point_selection_guide():
    """
    Show a guide overlay on the image for optimal point selection
    """
    global true_original_image

    if true_original_image is None:
        messagebox.showinfo("Info", "Please load an image first")
        return

    guide_image = true_original_image.copy()
    h, w = guide_image.shape[:2]

    # Semi-transparent overlay
    overlay = guide_image.copy()

    # Draw grid to show recommended point distribution
    grid_size = 3
    cell_w = w // grid_size
    cell_h = h // grid_size

    # Draw grid lines
    for i in range(1, grid_size):
        x = int(w * i / grid_size)
        y = int(h * i / grid_size)
        cv2.line(overlay, (x, 0), (x, h), (0, 255, 0), 3)
        cv2.line(overlay, (0, y), (w, y), (0, 255, 0), 3)

    # Add cell numbers and recommendations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(w, h) / 1000.0
    thickness = max(1, int(min(w, h) / 500))

    for i in range(grid_size):
        for j in range(grid_size):
            cell_num = i * grid_size + j + 1
            cx = j * cell_w + cell_w // 2
            cy = i * cell_h + cell_h // 2

            # Cell number
            text = f"{cell_num}"
            text_size = cv2.getTextSize(text, font, font_scale * 2, thickness * 2)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2

            # White background for text
            cv2.rectangle(overlay,
                          (text_x - 10, text_y - text_size[1] - 10),
                          (text_x + text_size[0] + 10, text_y + 10),
                          (255, 255, 255), -1)
            cv2.putText(overlay, text, (text_x, text_y),
                        font, font_scale * 2, (0, 0, 0), thickness * 2)

    # Blend overlay
    cv2.addWeighted(overlay, 0.7, guide_image, 0.3, 0, guide_image)

    # Add instructions at the top
    instruction_bg_h = int(h * 0.15)
    cv2.rectangle(guide_image, (0, 0), (w, instruction_bg_h), (0, 0, 0), -1)

    instructions = [
        "POINT SELECTION GUIDE:",
        "1. Select at least 1-2 points in each numbered grid cell",
        "2. Include objects at varying distances (near, medium, far)",
        "3. Avoid selecting only from edges or center",
        "4. Choose clearly identifiable features in both image and LiDAR"
    ]

    y_offset = int(instruction_bg_h * 0.2)
    line_height = int(instruction_bg_h * 0.18)

    for i, instruction in enumerate(instructions):
        cv2.putText(guide_image, instruction,
                    (int(w * 0.02), y_offset + i * line_height),
                    font, font_scale * 0.8, (255, 255, 255), thickness)

    # Display current point distribution if any
    if len(collected_image_points) > 0:
        img_pts = np.array(collected_image_points)
        grid_counts = np.zeros((grid_size, grid_size), dtype=int)

        for pt in img_pts:
            grid_x = int(pt[0] * grid_size / w)
            grid_y = int(pt[1] * grid_size / h)
            grid_x = min(grid_x, grid_size - 1)
            grid_y = min(grid_y, grid_size - 1)
            grid_counts[grid_y, grid_x] += 1

            # Draw existing points
            cv2.circle(guide_image, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)
            cv2.circle(guide_image, (int(pt[0]), int(pt[1])), 12, (255, 255, 255), 2)

        # Show count in each cell
        for i in range(grid_size):
            for j in range(grid_size):
                if grid_counts[i, j] > 0:
                    cx = j * cell_w + cell_w // 2
                    cy = i * cell_h + cell_h // 2 + int(cell_h * 0.3)

                    count_text = f"Points: {grid_counts[i, j]}"
                    cv2.putText(guide_image, count_text, (cx - 50, cy),
                                font, font_scale * 0.8, (255, 255, 0), thickness * 2)

    # Create window
    window_name = "Point Selection Guide"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Resize for display if too large
    max_display_width = 1280
    if w > max_display_width:
        scale = max_display_width / w
        display_h = int(h * scale)
        display_image = cv2.resize(guide_image, (max_display_width, display_h))
    else:
        display_image = guide_image

    cv2.imshow(window_name, display_image)

    update_log("\n📋 Point Selection Guide displayed")
    update_log("Press any key in the guide window to close")

    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def visualize_distribution_plot():
    """
    Create a matplotlib plot showing point distribution statistics
    """
    if len(collected_image_points) < 3:
        messagebox.showinfo("Info", "Need at least 3 points to visualize distribution")
        return

    # Create a new window
    plot_window = tk.Toplevel(root)
    plot_window.title("Point Distribution Visualization")
    plot_window.geometry("800x600")

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Calibration Point Distribution Analysis', fontsize=16)

    # Convert points to arrays
    img_pts = np.array(collected_image_points)
    lidar_pts = np.array(collected_lidar_points)

    # 1. Image point distribution
    ax1.scatter(img_pts[:, 0], img_pts[:, 1], c='red', s=50)
    ax1.set_xlim(0, true_original_image_dims[0])
    ax1.set_ylim(true_original_image_dims[1], 0)  # Invert Y for image coordinates
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_title('2D Point Distribution in Image')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 2. 3D point distribution (top view)
    ax2.scatter(lidar_pts[:, 0], lidar_pts[:, 1], c='blue', s=50)
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title('3D Point Distribution (Top View)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # 3. Depth distribution
    depths = lidar_pts[:, 2]
    ax3.hist(depths, bins=max(5, len(depths) // 3), color='green', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Depth (meters)')
    ax3.set_ylabel('Count')
    ax3.set_title('Depth Distribution')
    ax3.grid(True, alpha=0.3)

    # 4. Coverage heatmap
    if true_original_image is not None:
        h, w = true_original_image.shape[:2]
        heatmap, xedges, yedges = np.histogram2d(
            img_pts[:, 0], img_pts[:, 1],
            bins=[10, 10],
            range=[[0, w], [0, h]]
        )

        im = ax4.imshow(heatmap.T, extent=[0, w, h, 0],
                        cmap='hot', interpolation='bilinear', aspect='auto')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        ax4.set_title('Point Density Heatmap')
        plt.colorbar(im, ax=ax4, label='Point Count')

    # Add statistics text
    stats_text = f"Total Points: {len(img_pts)}\n"
    stats_text += f"Depth Range: {np.min(depths):.1f} - {np.max(depths):.1f} m\n"
    stats_text += f"Mean Depth: {np.mean(depths):.1f} m"

    fig.text(0.02, 0.02, stats_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

    plt.tight_layout()

    # Embed in tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Add close button
    close_btn = tk.Button(plot_window, text="Close", command=plot_window.destroy)
    close_btn.pack(pady=5)

def run_calibration_with_robust_optimization():
    """
    Enhanced calibration using the robust optimization strategy
    """
    global collected_image_points, collected_lidar_points
    global camera_matrix, dist_coeffs, last_calibration_results

    if len(collected_image_points) < 4:
        messagebox.showerror("Error", "Not enough point pairs. Please add at least 4 pairs.")
        return

    # First, analyze point distribution
    if true_original_image is not None:
        dist_analysis = analyze_point_distribution(
            collected_image_points,
            collected_lidar_points,
            true_original_image.shape
        )

        # Ask user if they want to continue if distribution is poor
        if dist_analysis and (dist_analysis['distribution_score'] < 0.5 or
                              dist_analysis['x_coverage'] < 0.4 or
                              dist_analysis['y_coverage'] < 0.4):
            response = messagebox.askyesno(
                "Poor Point Distribution",
                "The point distribution is suboptimal. This may affect calibration quality.\n\n"
                "Do you want to continue anyway?"
            )
            if not response:
                return

    LIDAR_3D_PTS_np = np.array(collected_lidar_points, dtype=np.float32)
    IMAGE_2D_PTS_np = np.array(collected_image_points, dtype=np.float32)

    # Use your existing coordinate transformation logic
    transformations = {
        "no_transform": np.eye(3),
        "z_down_y_forward": np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ]),
        "z_up_y_forward": np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ]),
        "standard_robotics": np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
    }

    # Test each transformation (keeping your existing logic)
    best_transform_name = None
    best_transform_error = float('inf')
    best_transformed_points = None

    update_log("\n=== TESTING COORDINATE TRANSFORMATIONS ===")

    for name, transform_matrix in transformations.items():
        transformed_pts = np.dot(LIDAR_3D_PTS_np, transform_matrix.T)
        positive_z = np.all(transformed_pts[:, 2] > 0)

        if positive_z:
            try:
                # Use robust calibration instead of regular solvePnP
                result = robust_pnp_calibration(
                    transformed_pts,
                    IMAGE_2D_PTS_np,
                    camera_matrix,
                    dist_coeffs
                )

                if result is not None:
                    error = result['mean_error']
                    update_log(f"{name}: Mean error = {error:.2f} pixels")

                    if error < best_transform_error:
                        best_transform_error = error
                        best_transform_name = name
                        best_transformed_points = transformed_pts
                else:
                    update_log(f"{name}: Calibration failed")
            except Exception as e:
                update_log(f"{name}: Error - {str(e)[:50]}...")
        else:
            update_log(f"{name}: Skipped (negative Z values)")

    if best_transformed_points is None:
        messagebox.showerror("Error", "Calibration failed for all coordinate transformations")
        return

    # Use the best transformation for final robust calibration
    update_log(f"\n*** Using transformation: {best_transform_name} ***")

    final_result = robust_pnp_calibration(
        best_transformed_points,
        IMAGE_2D_PTS_np,
        camera_matrix,
        dist_coeffs
    )

    if final_result is None:
        messagebox.showerror("Error", "Final calibration failed")
        return

    # Store results in your existing format
    last_calibration_results = {
        'rvec': final_result['rvec'],
        'tvec': final_result['tvec'],
        'transformed_3d_points': best_transformed_points,
        'reprojection_errors': final_result['all_errors'],
        'projected_points': project_3d_to_2d(
            best_transformed_points,
            camera_matrix,
            dist_coeffs,
            final_result['rvec'],
            final_result['tvec']
        ),
        'transform_name': best_transform_name,
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'inliers': final_result['inliers'],
        'outliers': final_result['outliers']
    }

    # Display results
    update_log("\n" + "=" * 50)
    update_log("ROBUST CALIBRATION RESULTS")
    update_log("=" * 50)
    update_log(f"Optimization method: {final_result['method']}")
    update_log(f"Inliers: {len(final_result['inliers'])}/{len(collected_image_points)}")

    rvec = final_result['rvec']
    tvec = final_result['tvec']

    update_log(
        f"Optimized rvec (degrees): [{np.rad2deg(rvec[0, 0]):.2f}, {np.rad2deg(rvec[1, 0]):.2f}, {np.rad2deg(rvec[2, 0]):.2f}]")
    update_log(f"Optimized tvec (meters): [{tvec[0, 0]:.6f}, {tvec[1, 0]:.6f}, {tvec[2, 0]:.6f}]")

    # Error analysis
    inlier_errors = final_result['all_errors'][final_result['inliers']]
    update_log(f"\nREPROJECTION ERROR (INLIERS):")
    update_log(f"  Mean: {np.mean(inlier_errors):.4f} pixels")
    update_log(f"  Median: {np.median(inlier_errors):.4f} pixels")
    update_log(f"  Max: {np.max(inlier_errors):.4f} pixels")

    if len(final_result['outliers']) > 0:
        update_log(f"\nOUTLIERS DETECTED: {list(final_result['outliers'])}")
        update_log("Consider checking these point correspondences")

    update_log("\n✅ Calibration complete! Use 'Visualize Projections' to see results.")

def manual_point_removal_tool():
    """
    Interactive tool to manually remove problematic points
    """
    global collected_image_points, collected_lidar_points

    if len(collected_image_points) < 5:
        messagebox.showinfo("Info", "Need at least 5 points to enable point removal")
        return

    # Create selection dialog
    dialog = tk.Toplevel(root)
    dialog.title("Remove Problematic Points")
    dialog.geometry("500x400")
    dialog.transient(root)
    dialog.grab_set()

    # Instructions
    tk.Label(dialog, text="Select points to remove (keep at least 4):",
             font=("Arial", 12, "bold")).pack(pady=10)

    # Listbox with checkboxes simulation
    frame = tk.Frame(dialog)
    frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # Point list with errors
    if last_calibration_results:
        errors = last_calibration_results.get('reprojection_errors', [0] * len(collected_image_points))
    else:
        errors = [0] * len(collected_image_points)

    checkboxes = []
    for i in range(len(collected_image_points)):
        var = tk.BooleanVar()
        lidar_pt = collected_lidar_points[i]
        image_pt = collected_image_points[i]
        error = errors[i] if i < len(errors) else 0

        text = f"Point {i}: 3D({lidar_pt[0]:.1f}, {lidar_pt[1]:.1f}, {lidar_pt[2]:.1f}) → 2D({image_pt[0]:.0f}, {image_pt[1]:.0f}) | Error: {error:.1f}px"

        cb = tk.Checkbutton(frame, text=text, variable=var, anchor='w')
        cb.pack(fill=tk.X, pady=1)
        checkboxes.append((i, var))

    # Buttons
    button_frame = tk.Frame(dialog)
    button_frame.pack(fill=tk.X, padx=20, pady=10)

    def remove_selected():
        # Get selected indices
        to_remove = []
        for i, var in checkboxes:
            if var.get():
                to_remove.append(i)

        if len(collected_image_points) - len(to_remove) < 4:
            messagebox.showerror("Error", "Must keep at least 4 points")
            return

        if not to_remove:
            messagebox.showinfo("Info", "No points selected for removal")
            dialog.destroy()
            return

        # Remove points (in reverse order to maintain indices)
        for i in sorted(to_remove, reverse=True):
            removed_2d = collected_image_points.pop(i)
            removed_3d = collected_lidar_points.pop(i)
            listbox_points.delete(i)
            update_log(f"Removed point {i}: 2D{removed_2d}, 3D{removed_3d}")

        update_log(f"Removed {len(to_remove)} points. {len(collected_image_points)} points remaining.")
        draw_points_on_image()
        dialog.destroy()

    tk.Button(button_frame, text="Remove Selected Points", command=remove_selected,
              bg="orange").pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

def visualize_projections():
    """
    Visualize the projected points on the image after calibration
    """
    global last_calibration_results, true_original_image, collected_image_points, collected_lidar_points
    global camera_matrix, dist_coeffs

    if last_calibration_results is None:
        messagebox.showerror("Error", "No calibration results available. Run calibration first.")
        return

    if true_original_image is None:
        messagebox.showerror("Error", "No image loaded.")
        return

    # Extract calibration results
    rvec_optimized = last_calibration_results['rvec']
    tvec_optimized = last_calibration_results['tvec']
    transformed_3d_points = last_calibration_results['transformed_3d_points']
    reprojection_errors = last_calibration_results['reprojection_errors']

    # Create a copy of the original image for visualization
    vis_image = true_original_image.copy()

    # Project the calibrated 3D points
    projected_points = project_3d_to_2d(
        transformed_3d_points, camera_matrix, dist_coeffs,
        rvec_optimized, tvec_optimized
    )

    update_log("\n=== VISUAL PROJECTION VERIFICATION ===")

    # Draw both original points and projected points
    for i, (original_2d, projected_2d, error) in enumerate(
            zip(collected_image_points, projected_points, reprojection_errors)):
        # Original point (green circle)
        orig_pt = (int(round(original_2d[0])), int(round(original_2d[1])))
        cv2.circle(vis_image, orig_pt, 8, (0, 255, 0), 2)  # Green circle
        cv2.putText(vis_image, f"O{i}", (orig_pt[0] + 12, orig_pt[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Projected point (red cross)
        proj_pt = (int(round(projected_2d[0])), int(round(projected_2d[1])))
        cv2.drawMarker(vis_image, proj_pt, (0, 0, 255), cv2.MARKER_CROSS, 15, 3)  # Red cross
        cv2.putText(vis_image, f"P{i}", (proj_pt[0] + 12, proj_pt[1] + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw line connecting original to projected
        cv2.line(vis_image, orig_pt, proj_pt, (255, 0, 255), 2)  # Magenta line

        # Add error text
        mid_pt = ((orig_pt[0] + proj_pt[0]) // 2, (orig_pt[1] + proj_pt[1]) // 2)
        cv2.putText(vis_image, f"{error:.1f}px", mid_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        update_log(
            f"Point {i}: Original({orig_pt[0]}, {orig_pt[1]}) -> Projected({proj_pt[0]}, {proj_pt[1]}) | Error: {error:.2f}px")

    if last_calibration_results.get('outliers') is not None:
        outliers = last_calibration_results['outliers']
        for outlier_idx in outliers:
            if outlier_idx < len(collected_image_points):
                pt = collected_image_points[outlier_idx]
                pt_int = (int(round(pt[0])), int(round(pt[1])))
                # Draw outlier with special marking
                cv2.circle(vis_image, pt_int, 15, (0, 0, 255), 3)  # Red thick circle
                cv2.putText(vis_image, "OUT", (pt_int[0] + 20, pt_int[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Add legend
    legend_y_start = 30
    cv2.putText(vis_image, "Legend:", (30, legend_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.circle(vis_image, (50, legend_y_start + 30), 8, (0, 255, 0), 2)
    cv2.putText(vis_image, "Original 2D points", (70, legend_y_start + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    cv2.drawMarker(vis_image, (50, legend_y_start + 60), (0, 0, 255), cv2.MARKER_CROSS, 15, 3)
    cv2.putText(vis_image, "Projected 3D points", (70, legend_y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    cv2.line(vis_image, (30, legend_y_start + 90), (60, legend_y_start + 90), (255, 0, 255), 2)
    cv2.putText(vis_image, "Error vector", (70, legend_y_start + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Add calibration statistics
    stats_y_start = legend_y_start + 130
    mean_error = np.mean(reprojection_errors)
    max_error = np.max(reprojection_errors)
    cv2.putText(vis_image, f"Mean Error: {mean_error:.2f}px", (30, stats_y_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis_image, f"Max Error: {max_error:.2f}px", (30, stats_y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the visualization
    # Scale down if image is too large for display
    display_image = vis_image
    if vis_image.shape[1] > 1920 or vis_image.shape[0] > 1080:
        scale_factor = min(1920 / vis_image.shape[1], 1080 / vis_image.shape[0])
        new_width = int(vis_image.shape[1] * scale_factor)
        new_height = int(vis_image.shape[0] * scale_factor)
        display_image = cv2.resize(vis_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        update_log(f"Display image scaled by {scale_factor:.3f} for viewing")

    cv2.namedWindow("Projection Verification", cv2.WINDOW_NORMAL)
    cv2.imshow("Projection Verification", display_image)

    # Save the visualization
    try:
        filepath = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")],
            title="Save Projection Visualization"
        )
        if filepath:
            cv2.imwrite(filepath, vis_image)
            update_log(f"Projection visualization saved to: {filepath}")
    except Exception as e:
        update_log(f"Could not save visualization: {e}")

    update_log("Projection visualization complete. Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyWindow("Projection Verification")

def project_all_lidar_points():
    """
    Project LiDAR points from various file formats with enhanced attribute support
    """
    global last_calibration_results, true_original_image, camera_matrix, dist_coeffs

    if last_calibration_results is None:
        messagebox.showerror("Error", "No calibration results available. Run calibration first.")
        return

    if true_original_image is None:
        messagebox.showerror("Error", "No image loaded.")
        return

    # Create file dialog with multiple format support
    file_types = [
        ("All Point Clouds", "*.las *.laz *.pcd *.csv *.txt *.ply"),
        ("LAS files", "*.las *.laz"),
        ("PCD files", "*.pcd"),
        ("CSV files", "*.csv"),
        ("Text files", "*.txt"),
        ("PLY files", "*.ply"),
        ("All files", "*.*")
    ]

    filepath = filedialog.askopenfilename(
        title="Select Point Cloud File",
        filetypes=file_types
    )

    if not filepath:
        return

    try:
        # Load the point cloud
        update_log(f"Loading point cloud from: {os.path.basename(filepath)}")
        point_data = load_point_cloud_file(filepath)

        # Show point cloud info
        pc_info = get_point_cloud_info_with_attributes(point_data)
        update_log(f"Point cloud info:\n{pc_info}")

        # Ask user for filtering options with enhanced dialog
        filter_dialog = EnhancedPointCloudFilterDialog(root, point_data)
        if filter_dialog.result is None:
            return

        # Apply filters
        filtered_data = filter_dialog.result
        filtered_points = filtered_data['points']
        update_log(f"After filtering: {len(filtered_points)} points")

        if len(filtered_points) == 0:
            messagebox.showwarning("Warning", "No points remaining after filtering.")
            return

        # Apply coordinate transformation
        transform_name = last_calibration_results.get('transform_name', 'z_up_y_forward')

        transformations = {
            "no_transform": np.eye(3),
            "z_down_y_forward": np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            "z_up_y_forward": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            "standard_robotics": np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        }

        transform_matrix = transformations[transform_name]
        transformed_lidar = np.dot(filtered_points, transform_matrix.T)

        # Filter points in front of camera
        in_front_mask = transformed_lidar[:, 2] > 0
        valid_points = transformed_lidar[in_front_mask]
        update_log(f"Points in front of camera: {len(valid_points)}")

        if len(valid_points) == 0:
            messagebox.showwarning("Warning", "No points are in front of the camera.")
            return

        # Project all points
        rvec = last_calibration_results['rvec']
        tvec = last_calibration_results['tvec']

        projected_points = project_3d_to_2d(valid_points, camera_matrix, dist_coeffs, rvec, tvec)

        # Filter points within image bounds
        img_h, img_w = true_original_image.shape[:2]
        in_bounds_mask = ((projected_points[:, 0] >= 0) & (projected_points[:, 0] < img_w) &
                          (projected_points[:, 1] >= 0) & (projected_points[:, 1] < img_h))

        valid_projected = projected_points[in_bounds_mask]
        valid_3d = valid_points[in_bounds_mask]

        # Filter attributes correspondingly
        final_mask = in_front_mask.copy()
        final_mask[in_front_mask] = in_bounds_mask  # Compound mask

        update_log(f"Points projecting within image: {len(valid_projected)}")

        if len(valid_projected) == 0:
            messagebox.showwarning("Warning", "No points project within the image bounds.")
            return

        # Determine coloring options
        color_options = ["Distance from camera", "Height (Z coordinate)"]

        # Add intensity option if available
        if filtered_data.get('intensity') is not None:
            color_options.insert(0, "Intensity")  # Make intensity the first option

        # Add RGB option if available
        if filtered_data.get('rgb') is not None:
            color_options.append("RGB (original colors)")

        # Ask user for color option
        color_dialog = ColorOptionDialog(root, color_options)
        if color_dialog.result is None:
            return

        color_choice = color_dialog.result

        # Create visualization
        vis_image = true_original_image.copy()

        # Determine color values based on choice
        if color_choice == "Intensity" and filtered_data.get('intensity') is not None:
            # Use intensity values
            intensity_values = filtered_data['intensity'][final_mask]
            color_values = intensity_values
            legend_text = "Intensity"
            unit = ""

        elif color_choice == "RGB (original colors)" and filtered_data.get('rgb') is not None:
            # Use original RGB colors
            rgb_values = filtered_data['rgb'][final_mask]
            # Normalize RGB values (LAS RGB is often 16-bit)
            if np.max(rgb_values) > 255:
                rgb_values = (rgb_values / 65535.0 * 255).astype(np.uint8)
            else:
                rgb_values = rgb_values.astype(np.uint8)

            # Plot with original colors
            for point_2d, rgb in zip(valid_projected, rgb_values):
                color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))  # BGR format for OpenCV
                pt = (int(round(point_2d[0])), int(round(point_2d[1])))
                cv2.circle(vis_image, pt, 1, color, -1)

            # Add simple legend
            legend_x, legend_y = 30, img_h - 80
            cv2.putText(vis_image, "RGB Colors", (legend_x, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, f"Points: {len(valid_projected):,}", (legend_x, legend_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif color_choice == "Distance from camera":
            distances = np.linalg.norm(valid_3d, axis=1)
            color_values = distances
            legend_text = "Distance"
            unit = "m"

        else:  # Height
            # Use original Z coordinates before transformation
            original_z = filtered_points[final_mask][:, 2]
            color_values = original_z
            legend_text = "Height"
            unit = "m"

        # Apply color mapping (unless RGB was used)
        if color_choice != "RGB (original colors)":
            min_val, max_val = np.min(color_values), np.max(color_values)

            for point_2d, color_val in zip(valid_projected, color_values):
                # Color mapping: blue = low value, red = high value
                if max_val > min_val:
                    normalized_val = (color_val - min_val) / (max_val - min_val)
                else:
                    normalized_val = 0.5

                color = (int(255 * normalized_val), 0, int(255 * (1 - normalized_val)))  # BGR

                pt = (int(round(point_2d[0])), int(round(point_2d[1])))
                cv2.circle(vis_image, pt, 1, color, -1)

            # Add legend
            legend_x, legend_y = 30, img_h - 120
            cv2.putText(vis_image, f"{legend_text}:", (legend_x, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, f"{min_val:.1f}{unit} (blue)", (legend_x, legend_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(vis_image, f"{max_val:.1f}{unit} (red)", (legend_x, legend_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(vis_image, f"Points: {len(valid_projected):,}", (legend_x, legend_y + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display
        display_image = vis_image
        if vis_image.shape[1] > 1920 or vis_image.shape[0] > 1080:
            scale_factor = min(1920 / vis_image.shape[1], 1080 / vis_image.shape[0])
            new_width = int(vis_image.shape[1] * scale_factor)
            new_height = int(vis_image.shape[0] * scale_factor)
            display_image = cv2.resize(vis_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.namedWindow("LiDAR Point Cloud Projection", cv2.WINDOW_NORMAL)
        cv2.imshow("LiDAR Point Cloud Projection", display_image)

        # Option to save the result
        save_result = messagebox.askyesno("Save Result", "Save the projected point cloud image?")
        if save_result:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")],
                title="Save Projected Point Cloud"
            )
            if save_path:
                cv2.imwrite(save_path, vis_image)
                update_log(f"Projected point cloud saved to: {save_path}")

        update_log("Point cloud projection complete. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyWindow("LiDAR Point Cloud Projection")

    except Exception as e:
        messagebox.showerror("Error", f"Could not load or project point cloud: {e}")
        update_log(f"Error: {e}")

def reset_points_action(clear_image=True):
    global collected_image_points, collected_lidar_points, temp_clicked_2d_point
    global true_original_image, zoom_level, view_rect, true_original_image_dims

    collected_image_points.clear()
    collected_lidar_points.clear()
    listbox_points.delete(0, tk.END)
    temp_clicked_2d_point = None
    label_2d_coord.config(text="Clicked 2D Point: N/A")

    if true_original_image is not None:
        zoom_level = MIN_ZOOM_LEVEL
        # view_rect refers to the true original image's coordinates
        view_rect = (0, 0, true_original_image_dims[0], true_original_image_dims[1])
        if clear_image:
            draw_points_on_image()
    update_log("All points cleared. Zoom reset to default.")

def save_points_action():
    global collected_image_points, collected_lidar_points
    if not collected_image_points or not collected_lidar_points:
        messagebox.showinfo("No Data", "No points collected to save.")
        update_log("Save cancelled: No points to save.")
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Save Points As CSV"
    )

    if not filepath:
        update_log("Save cancelled by user.")
        return

    try:
        with open(filepath, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header
            csv_writer.writerow(['image_x', 'image_y', 'lidar_x', 'lidar_y', 'lidar_z'])
            # Write data
            for img_pt, lidar_pt in zip(collected_image_points, collected_lidar_points):
                csv_writer.writerow([img_pt[0], img_pt[1], lidar_pt[0], lidar_pt[1], lidar_pt[2]])
        update_log(f"Points successfully saved to: {filepath}")
    except Exception as e:
        messagebox.showerror("Save Error", f"Could not save points to file: {e}")
        update_log(f"Error saving points: {e}")

def load_points_action():
    global collected_image_points, collected_lidar_points

    filepath = filedialog.askopenfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Load Points From CSV"
    )

    if not filepath:
        update_log("Load cancelled by user.")
        return

    try:
        # Clear existing points before loading new ones
        reset_points_action(clear_image=False)
        temp_collected_image_points = []
        temp_collected_lidar_points = []

        with open(filepath, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)  # Skip header row

            expected_header = ['image_x', 'image_y', 'lidar_x', 'lidar_y', 'lidar_z']
            if header != expected_header:
                messagebox.showwarning("Warning",
                                       "CSV header does not match expected format. Attempting to load anyway.")
                update_log("Warning: CSV header mismatch.")

            for row in csv_reader:
                if len(row) == 5:
                    try:
                        img_x, img_y = int(row[0]), int(row[1])
                        lidar_x, lidar_y, lidar_z = float(row[2]), float(row[3]), float(row[4])
                        temp_collected_image_points.append([img_x, img_y])
                        temp_collected_lidar_points.append([lidar_x, lidar_y, lidar_z])
                        listbox_points.insert(tk.END,
                                              f"2D: ({img_x}, {img_y}) <-> 3D: ({lidar_x:.2f}, {lidar_y:.2f}, {lidar_z:.2f})")
                    except ValueError:
                        update_log(f"Skipping row due to invalid data: {row}")
                else:
                    update_log(f"Skipping row due to incorrect number of columns: {row}")

        collected_image_points = temp_collected_image_points
        collected_lidar_points = temp_collected_lidar_points
        update_log(f"Successfully loaded {len(collected_image_points)} points from: {filepath}")
        draw_points_on_image()  # Redraw points after loading
    except Exception as e:
        messagebox.showerror("Load Error", f"Could not load points from file: {e}")
        update_log(f"Error loading points: {e}")

def delete_selected_point_pair_action():
    global collected_image_points, collected_lidar_points

    selected_indices = listbox_points.curselection()
    if not selected_indices:
        messagebox.showwarning("No Selection", "Please select a point pair to delete.")
        return

    # Delete in reverse order to avoid index issues
    for index in sorted(selected_indices, reverse=True):
        if 0 <= index < len(collected_image_points):
            removed_2d = collected_image_points.pop(index)
            removed_3d = collected_lidar_points.pop(index)
            listbox_points.delete(index)
            update_log(f"Removed point pair at index {index}: 2D {removed_2d}, 3D {removed_3d}")

    draw_points_on_image()  # Redraw to reflect point removal

def update_log(message):
    log_text.configure(state='normal')
    log_text.insert(tk.END, message + "\n")
    log_text.configure(state='disabled')
    log_text.see(tk.END)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        cv2.destroyAllWindows()
        root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    root.title("LiDAR-Camera Calibration Point Selector")

    controls_frame = tk.Frame(root, padx=10, pady=10)
    controls_frame.pack(side=tk.LEFT, fill=tk.Y)

    tk.Button(controls_frame, text="Load Image", command=load_image_action).pack(fill=tk.X, pady=2)
    label_2d_coord = tk.Label(controls_frame, text="Clicked 2D Point: N/A")
    label_2d_coord.pack(pady=2)

    tk.Label(controls_frame, text="LiDAR X:").pack(pady=(5, 0))
    entry_3d_x = tk.Entry(controls_frame, width=10);
    entry_3d_x.pack()
    tk.Label(controls_frame, text="LiDAR Y:").pack()
    entry_3d_y = tk.Entry(controls_frame, width=10);
    entry_3d_y.pack()
    tk.Label(controls_frame, text="LiDAR Z:").pack()
    entry_3d_z = tk.Entry(controls_frame, width=10);
    entry_3d_z.pack()

    tk.Button(controls_frame, text="Add Point Pair", command=add_point_pair_action).pack(fill=tk.X, pady=5)
    tk.Label(controls_frame, text="Collected Points:").pack(pady=(10, 0))
    listbox_points = tk.Listbox(controls_frame, height=10, width=40)
    listbox_points.pack(pady=2)

    tk.Label(controls_frame, text="Manage Correspondence", font=("Arial", 9, "bold")).pack(pady=(10, 2))
    tk.Button(controls_frame, text="Load Points from CSV", command=load_points_action, bg='lightgreen').pack(fill=tk.X, pady=2)
    tk.Button(controls_frame, text="Save Points to CSV", command=save_points_action, bg='lightblue').pack(fill=tk.X, pady=2)
    tk.Button(controls_frame, text="Delete Selected Point Pair", command=delete_selected_point_pair_action, bg="orange").pack(fill=tk.X, pady=2)
    tk.Button(controls_frame, text="Reset All Points", command=reset_points_action, bg="lightcoral").pack(fill=tk.X, pady=2)
    tk.Button(controls_frame, text="Remove Problem Points", command=manual_point_removal_tool, bg="mistyrose").pack(fill=tk.X, pady=1)

    tk.Label(controls_frame, text="Calibration Tools:", font=("Arial", 9, "bold")).pack(pady=(10, 2))

    tk.Button(controls_frame, text="Run Robust Calibration", command=run_calibration_with_robust_optimization, bg="lightblue").pack(fill=tk.X, pady=5)
    tk.Button(controls_frame, text="Show Selection Guide", command=show_point_selection_guide, bg="lightyellow").pack(fill=tk.X, pady=1)

    tk.Label(controls_frame, text="Visualization Tools:", font=("Arial", 9, "bold")).pack(pady=(10, 2))

    tk.Button(controls_frame, text="Visualize Projections", command=visualize_projections, bg="lightgreen").pack(fill=tk.X, pady=2)
    tk.Button(controls_frame, text="Project LiDAR Point Cloud", command=project_all_lidar_points, bg="lightblue").pack(fill=tk.X, pady=2)

    # Add a separator for debugging tools
    tk.Label(controls_frame, text="Miscellaneous:", font=("Arial", 9, "bold")).pack(pady=(10, 2))

    tk.Button(controls_frame, text="Quick Error Analysis", command=quick_point_error_analysis, bg="lightgreen").pack(fill=tk.X, pady=1)
    tk.Button(controls_frame, text="Test Without Worst Points", command=attempt_without_worst_points, bg="lightcyan").pack(fill=tk.X, pady=1)
    tk.Button(controls_frame, text="Analyze Point Quality", command=analyze_point_contributions, bg="lightyellow").pack(fill=tk.X, pady=1)
    tk.Button(controls_frame, text="Progressive Calibration", command=progressive_calibration_fixed, bg="lightyellow").pack(fill=tk.X, pady=1)
    tk.Button(controls_frame, text="RANSAC Calibration", command=robust_calibration_with_ransac, bg="lavender").pack(fill=tk.X, pady=1)



    log_frame = tk.Frame(root, padx=10, pady=10)
    log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    tk.Label(log_frame, text="Log / Results:").pack(anchor='w')
    log_text = scrolledtext.ScrolledText(log_frame, height=25, width=60, state='disabled')
    log_text.pack(fill=tk.BOTH, expand=True)

    update_log("Application started. Load an image to begin.")
    update_log("Use Shift + Mouse Wheel to zoom in/out on the image.")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()