import numpy as np
import cv2
from scipy.optimize import minimize, least_squares
import logging

def project_3d_to_2d(LIDAR_3D_PTS, camera_matrix_input, dist_coeffs_input, rvec, tvec):
    rvec = np.asarray(rvec, dtype=np.float32)
    tvec = np.asarray(tvec, dtype=np.float32)
    points_2d, _ = cv2.projectPoints(LIDAR_3D_PTS, rvec, tvec, camera_matrix_input, dist_coeffs_input)
    points_2d = points_2d.reshape(-1, 2)
    return points_2d

def robust_objective_function(params, LIDAR_3D_PTS, IMAGE_2D_PTS, camera_matrix_input, dist_coeffs_input):
    try:
        rvec_opt = params[:3].reshape(3, 1)
        tvec_opt = params[3:].reshape(3, 1)
        if np.any(np.abs(rvec_opt) > 2 * np.pi):
            return 1e10
        if np.any(np.abs(tvec_opt) > 1000):
            return 1e10
        projected_2d = project_3d_to_2d(LIDAR_3D_PTS, camera_matrix_input, dist_coeffs_input, rvec_opt, tvec_opt)
        if np.any(np.isnan(projected_2d)) or np.any(np.isinf(projected_2d)):
            return 1e10
        if np.any(projected_2d < -1000) or np.any(projected_2d > 10000):
            return 1e10
        error = np.sum((projected_2d - IMAGE_2D_PTS) ** 2)
        return error
    except Exception:
        return 1e10

def run_calibration(image_points, lidar_points, camera_matrix, dist_coeffs):
    image_points = np.asarray(image_points, dtype=np.float32)
    lidar_points = np.asarray(lidar_points, dtype=np.float32)
    n = len(image_points)
    if n < 4 or len(lidar_points) < 4:
        logging.error('Need at least 4 point pairs.')
        return None, 'Need at least 4 point pairs.'
    # Choose solvePnP method based on number of points
    if n == 4:
        method = cv2.SOLVEPNP_P3P
    else:
        method = cv2.SOLVEPNP_ITERATIVE
    try:
        logging.info(f'Calling cv2.solvePnP with {n} points and method {method}')
        success, rvec, tvec = cv2.solvePnP(lidar_points, image_points, camera_matrix, dist_coeffs, flags=method)
    except Exception as e:
        logging.error(f'cv2.solvePnP failed: {e}')
        return None, f'cv2.solvePnP failed: {e}'
    if not success:
        logging.error('cv2.solvePnP failed.')
        return None, 'cv2.solvePnP failed.'
    logging.info(f'solvePnP success: {success}, rvec: {rvec.ravel()}, tvec: {tvec.ravel()}')
    initial_params = np.concatenate([rvec.flatten(), tvec.flatten()])
    initial_proj_2d = project_3d_to_2d(lidar_points, camera_matrix, dist_coeffs, rvec, tvec)
    initial_errors = np.linalg.norm(image_points - initial_proj_2d, axis=1)
    logging.info(f'Initial mean reprojection error: {np.mean(initial_errors):.4f}, max error: {np.max(initial_errors):.4f}')
    result = minimize(
        robust_objective_function,
        initial_params,
        args=(lidar_points, image_points, camera_matrix, dist_coeffs),
        method='L-BFGS-B',
        options={'ftol': 1e-9, 'maxiter': 10000, 'disp': False}
    )
    if not result.success:
        logging.warning(f'Optimisation failed: {result.message}. Returning initial pose from solvePnP.')
        # Fallback: return initial pose from solvePnP
        proj_2d = initial_proj_2d
        errors = initial_errors
        rot_matrix, _ = cv2.Rodrigues(rvec)
        return {
            'rvec': rvec,
            'rotation_matrix': rot_matrix,
            'tvec': tvec,
            'proj_2d': proj_2d,
            'errors': errors,
            'mean_error': np.mean(errors),
            'success': False,
            'fallback': True,
            'message': f'Optimisation failed: {result.message}'
        }, 'Optimisation failed, returned initial pose from solvePnP.'
    opt_params = result.x
    rvec_opt = opt_params[:3].reshape(3, 1)
    tvec_opt = opt_params[3:].reshape(3, 1)
    proj_2d = project_3d_to_2d(lidar_points, camera_matrix, dist_coeffs, rvec_opt, tvec_opt)
    errors = np.linalg.norm(image_points - proj_2d, axis=1)
    rot_matrix_opt, _ = cv2.Rodrigues(rvec_opt)
    logging.info(f'Optimized mean reprojection error: {np.mean(errors):.4f}, max error: {np.max(errors):.4f}')
    return {
        'rvec': rvec_opt,
        'rotation_matrix': rot_matrix_opt,
        'tvec': tvec_opt,
        'proj_2d': proj_2d,
        'errors': errors,
        'mean_error': np.mean(errors),
        'success': True,
        'fallback': False
    }, None

def analyze_outliers(image_points, lidar_points, camera_matrix, dist_coeffs):
    """
    Returns a dict with per-point errors, mean, std, and outlier threshold.
    """
    image_points = np.asarray(image_points, dtype=np.float32)
    lidar_points = np.asarray(lidar_points, dtype=np.float32)
    if len(image_points) < 4:
        return None, 'Need at least 4 points for outlier analysis.'
    result, err = run_calibration(image_points, lidar_points, camera_matrix, dist_coeffs)
    if result is None:
        return None, f'Calibration required for outlier analysis: {err}'
    errors = result['errors']
    mean = np.mean(errors)
    std = np.std(errors)
    outlier_thresh = mean + 2 * std
    return {
        'errors': errors,
        'mean': mean,
        'std': std,
        'outlier_thresh': outlier_thresh
    }, None

def analyze_point_contributions(image_points, lidar_points, camera_matrix, dist_coeffs):
    """
    Returns a list of dicts: for each point, the mean error when that point is left out, and the delta from the base error.
    """
    image_points = np.asarray(image_points, dtype=np.float32)
    lidar_points = np.asarray(lidar_points, dtype=np.float32)
    if len(image_points) < 5:
        return None, 'Need at least 5 points for point contribution analysis.'
    base_result, base_err = run_calibration(image_points, lidar_points, camera_matrix, dist_coeffs)
    if base_result is None:
        return None, f'Calibration failed for full set: {base_err}'
    base_error = base_result['mean_error']
    contributions = []
    for i in range(len(image_points)):
        mask = np.ones(len(image_points), dtype=bool)
        mask[i] = False
        sub_img = image_points[mask]
        sub_lidar = lidar_points[mask]
        result, err = run_calibration(sub_img, sub_lidar, camera_matrix, dist_coeffs)
        if result is not None:
            delta = result['mean_error'] - base_error
            contributions.append({'index': i, 'mean_error': result['mean_error'], 'delta': delta, 'success': True})
        else:
            contributions.append({'index': i, 'mean_error': None, 'delta': None, 'success': False, 'error': err})
    return contributions, None

def project_points(lidar_points, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Projects 3D points to 2D using given extrinsics and intrinsics.
    """
    return project_3d_to_2d(lidar_points, camera_matrix, dist_coeffs, rvec, tvec)

# --- Advanced Calibration Features ---
def huber_loss(residuals, delta=1.0):
    abs_r = np.abs(residuals)
    mask = abs_r <= delta
    out = np.empty_like(residuals)
    out[mask] = 0.5 * residuals[mask] ** 2
    out[~mask] = delta * (abs_r[~mask] - 0.5 * delta)
    return out

def run_robust_calibration(image_points, lidar_points, camera_matrix, dist_coeffs, loss='huber', ransac_thresh=8.0, ransac_trials=100):
    """
    RANSAC-based robust calibration. Returns best inlier set and calibration result.
    """
    image_points = np.asarray(image_points, dtype=np.float32)
    lidar_points = np.asarray(lidar_points, dtype=np.float32)
    n = len(image_points)
    if n < 4:
        return None, 'Need at least 4 point pairs.'
    best_inliers = None
    best_result = None
    best_err = np.inf
    rng = np.random.default_rng()
    for _ in range(ransac_trials):
        idx = rng.choice(n, 4, replace=False)
        res, err = run_calibration(image_points[idx], lidar_points[idx], camera_matrix, dist_coeffs)
        if res is None:
            continue
        proj_2d = project_3d_to_2d(lidar_points, camera_matrix, dist_coeffs, res['rvec'], res['tvec'])
        errors = np.linalg.norm(image_points - proj_2d, axis=1)
        inliers = errors < ransac_thresh
        if np.sum(inliers) >= 4:
            res_full, err_full = run_calibration(image_points[inliers], lidar_points[inliers], camera_matrix, dist_coeffs)
            if res_full is not None and res_full['mean_error'] < best_err:
                best_inliers = inliers
                best_result = res_full
                best_err = res_full['mean_error']
    if best_result is None:
        return None, 'RANSAC failed to find a valid model.'
    return {'result': best_result, 'inliers': best_inliers}, None

def run_bundle_adjustment(image_points_list, lidar_points_list, camera_matrix, dist_coeffs, rvec_init=None, tvec_init=None):
    """
    Bundle adjustment for multiple views (lists of image/3d points). Returns optimized rvec, tvec.
    """
    # image_points_list, lidar_points_list: list of arrays, one per view
    if not (isinstance(image_points_list, list) and isinstance(lidar_points_list, list)):
        return None, 'Input must be lists of arrays.'
    if len(image_points_list) < 1:
        return None, 'Need at least one view.'
    # Stack all points
    img_pts = np.vstack(image_points_list)
    lidar_pts = np.vstack(lidar_points_list)
    if rvec_init is None or tvec_init is None:
        res, err = run_calibration(img_pts, lidar_pts, camera_matrix, dist_coeffs)
        if res is None:
            return None, f'Initial calibration failed: {err}'
        rvec_init = res['rvec']
        tvec_init = res['tvec']
    def ba_residuals(params):
        rvec = params[:3].reshape(3,1)
        tvec = params[3:].reshape(3,1)
        proj = project_3d_to_2d(lidar_pts, camera_matrix, dist_coeffs, rvec, tvec)
        return (proj - img_pts).ravel()
    x0 = np.concatenate([rvec_init.flatten(), tvec_init.flatten()])
    result = least_squares(ba_residuals, x0, loss='huber', f_scale=1.0)
    if not result.success:
        return None, 'Bundle adjustment failed.'
    rvec_opt = result.x[:3].reshape(3,1)
    tvec_opt = result.x[3:].reshape(3,1)
    return {'rvec': rvec_opt, 'tvec': tvec_opt, 'success': True}, None
    tvec_opt = result.x[3:].reshape(3,1)
    return {'rvec': rvec_opt, 'tvec': tvec_opt, 'success': True}, None
