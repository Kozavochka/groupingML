import json
from pathlib import Path
import numpy as np


def _pick_payloads_by_image(mapping: dict, target_image_name: str | None) -> list:
    if not isinstance(mapping, dict) or not mapping:
        return []

    if not target_image_name:
        return list(mapping.values())

    # 1) Exact key match.
    if target_image_name in mapping:
        return [mapping[target_image_name]]

    # 2) Basename match (when full path vs filename differs).
    target_base = Path(target_image_name).name
    for key, payload in mapping.items():
        if Path(str(key)).name == target_base:
            return [payload]

    # 3) Fallback: no match found, keep all to avoid dropping valid input.
    return list(mapping.values())


def parse_chartreader_markup(
    markup_raw,
    score_thr: float = 0.4,
    category_idx: int = 0,
    target_image_name: str | None = None,
) -> np.ndarray:
    """
    Parse ChartReader markup into Nx2 [x,y] int32 candidate coords.

    Supported payloads:
    1) {"result": {"image.png": [ [class0, class1, ...], ... ]}}
    2) {"image.png": [ {"0": [[score,cls,x,y], ...]}, ... ]}  # legacy-like
    3) JSON string of one of formats above.
    """
    if markup_raw is None:
        return np.zeros((0, 2), dtype=np.int32)

    if isinstance(markup_raw, str):
        markup_raw = markup_raw.strip()
        if not markup_raw:
            return np.zeros((0, 2), dtype=np.int32)
        try:
            obj = json.loads(markup_raw)
        except Exception:
            return np.zeros((0, 2), dtype=np.int32)
    else:
        obj = markup_raw

    points = []
    category_key = str(int(category_idx))

    def _append_point(p):
        if not isinstance(p, (list, tuple)) or len(p) < 4:
            return
        score = float(p[0])
        if score < float(score_thr):
            return
        x = int(round(float(p[2])))
        y = int(round(float(p[3])))
        points.append((x, y))

    def _parse_legacy_image_payload(image_payload):
        if not isinstance(image_payload, list):
            return
        for item in image_payload:
            if not isinstance(item, dict):
                continue
            arr = item.get(category_key, [])
            if not isinstance(arr, list):
                continue
            for p in arr:
                _append_point(p)

    if isinstance(obj, dict) and isinstance(obj.get("result"), dict):
        # ChartReader may return either:
        # 1) {"result": {"image.png": [ [class0, class1, ...], ... ]}}
        # 2) {"result": {"image.png": [ {"0": [[score,cls,x,y], ...]}, ... ]}}
        for image_groups in _pick_payloads_by_image(obj["result"], target_image_name):
            if not isinstance(image_groups, list):
                continue

            # Newer grouped format.
            for group in image_groups:
                if not isinstance(group, list):
                    continue
                if int(category_idx) < len(group) and isinstance(group[int(category_idx)], list):
                    for p in group[int(category_idx)]:
                        _append_point(p)

            # Legacy-per-image format wrapped into "result".
            _parse_legacy_image_payload(image_groups)

        if points:
            return np.array(points, dtype=np.int32)

    # Legacy-like format fallback
    if isinstance(obj, dict):
        values = _pick_payloads_by_image(obj, target_image_name)
    else:
        values = [obj]

    for image_obj in values:
        _parse_legacy_image_payload(image_obj)

    if not points:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(points, dtype=np.int32)
