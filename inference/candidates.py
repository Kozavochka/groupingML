import numpy as np
import cv2


def _coords_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def extract_candidate_coords(
    rgb: np.ndarray,
    method: str = "non_white",
    white_threshold: int = 245,
    canny_threshold1: int = 80,
    canny_threshold2: int = 180,
    canny_aperture_size: int = 3,
    canny_l2gradient: bool = False,
    canny_dilate_iter: int = 1,
    max_candidate_points: int = 0,
) -> np.ndarray:
    """
    Returns candidate points in image coordinates as int32 array [x, y].
    """
    h, w = rgb.shape[:2]
    method = (method or "non_white").lower().strip()

    if method == "all_pixels":
        xs, ys = np.meshgrid(np.arange(w, dtype=np.int32), np.arange(h, dtype=np.int32))
        coords = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if method == "canny":
            edges = cv2.Canny(
                gray,
                threshold1=int(canny_threshold1),
                threshold2=int(canny_threshold2),
                apertureSize=int(canny_aperture_size),
                L2gradient=bool(canny_l2gradient),
            )
            if int(canny_dilate_iter) > 0:
                edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=int(canny_dilate_iter))
            coords = _coords_from_mask(edges > 0)
        else:
            # non_white default: suitable for charts on light background
            coords = _coords_from_mask(gray < int(white_threshold))

    if int(max_candidate_points) > 0 and len(coords) > int(max_candidate_points):
        # Uniform random subsampling to keep DBSCAN bounded on high-res images.
        idx = np.random.choice(len(coords), size=int(max_candidate_points), replace=False)
        coords = coords[idx]

    return coords.astype(np.int32, copy=False)

