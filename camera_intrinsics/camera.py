import numpy as np
import cv2

class CameraModel:
    """
    CameraModel encapsulates OpenCV camera intrinsics and provides mapping functions
    between 2D image pixels and 3D camera coordinates.
    """
    def __init__(self, camera_matrix, dist_coeffs=None):
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        else:
            self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)

    def pixel_to_ray(self, pixel):
        """
        Convert a 2D pixel (u, v) to a normalized 3D direction vector in camera coordinates.
        This is the inverse of the projection operation (ignoring distortion).
        Returns a unit vector in camera coordinates (x, y, 1) normalized.
        """
        pixel = np.asarray(pixel, dtype=np.float32).reshape(-1, 1, 2)
        # Get normalized camera coordinates (x, y, 1)
        undistorted = cv2.undistortPoints(pixel, self.camera_matrix, self.dist_coeffs, P=None)
        rays = np.concatenate([undistorted.squeeze(axis=1), np.ones((undistorted.shape[0], 1), dtype=np.float32)], axis=1)
        # Normalize direction vectors
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return rays.squeeze()

    def project_point(self, point3d, rvec=None, tvec=None):
        """
        Project a 3D point (or array of points) in world/camera coordinates to 2D image pixel coordinates.
        rvec, tvec: rotation and translation vectors (if projecting from world to camera)
        """
        point3d = np.asarray(point3d, dtype=np.float32).reshape(-1, 3)
        if rvec is None:
            rvec = np.zeros((3, 1), dtype=np.float32)
        if tvec is None:
            tvec = np.zeros((3, 1), dtype=np.float32)
        image_points, _ = cv2.projectPoints(point3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        return image_points.reshape(-1, 2)

    def unproject_pixel(self, pixel, depth):
        """
        Unproject a 2D pixel (u, v) with a given depth (Z) to a 3D point in camera coordinates.
        """
        ray = self.pixel_to_ray(pixel)
        ray = np.asarray(ray, dtype=np.float32)
        if ray.ndim == 1:
            return ray * depth / ray[2]
        else:
            return ray * (depth[:, np.newaxis] / ray[:, 2:3])

    def get_intrinsics(self):
        return self.camera_matrix.copy(), self.dist_coeffs.copy()

    def set_intrinsics(self, camera_matrix, dist_coeffs=None):
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        else:
            self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
