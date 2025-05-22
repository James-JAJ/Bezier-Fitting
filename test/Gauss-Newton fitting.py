import numpy as np

def chord_length_parameterize(points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.insert(np.cumsum(distances), 0, 0)
    t = cumulative / cumulative[-1]  # Normalize to [0, 1]
    return t
def bernstein_matrix(t: np.ndarray) -> np.ndarray:
    B = np.zeros((len(t), 4))
    B[:, 0] = (1 - t) ** 3
    B[:, 1] = 3 * (1 - t) ** 2 * t
    B[:, 2] = 3 * (1 - t) * t ** 2
    B[:, 3] = t ** 3
    return B
def fit_bezier_curve(points: np.ndarray) -> np.ndarray:
    t = chord_length_parameterize(points)
    B = bernstein_matrix(t)
    # Solve least squares: B @ C = points
    Cx, _, _, _ = np.linalg.lstsq(B, points[:, 0], rcond=None)
    Cy, _, _, _ = np.linalg.lstsq(B, points[:, 1], rcond=None)
    control_points = np.column_stack([Cx, Cy])
    return control_points
def bezier_eval(control_points: np.ndarray, t: np.ndarray) -> np.ndarray:
    B = bernstein_matrix(t)
    return B @ control_points
def compute_fitting_error(points: np.ndarray, control_points: np.ndarray) -> float:
    t = chord_length_parameterize(points)
    fitted_points = bezier_eval(control_points, t)
    errors = np.linalg.norm(points - fitted_points, axis=1)
    return np.max(errors), np.mean(errors)
def fit_and_evaluate_bezier(points: np.ndarray):
    control_points = fit_bezier_curve(points)
    max_err, mean_err = compute_fitting_error(points, control_points)
    return control_points, max_err, mean_err
points = np.array([
    [0, 0],
    [1, 2],
    [2, 3],
    [4, 3],
    [5, 2],
    [6, 0]
])

ctrl_pts, max_error, mean_error = fit_and_evaluate_bezier(points)
print("控制點：\n", ctrl_pts)
print("最大誤差：", max_error)
print("平均誤差：", mean_error)

