from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from config import settings
from inference.candidates import extract_candidate_coords
from model import SmallUNet


@dataclass
class ClusterParams:
    eps: float = 0.3
    min_samples: int = 20
    l2_normalize: bool = True
    auto_eps: bool = False
    auto_eps_k: int = 10
    auto_eps_q: float = 0.90
    use_spatial: bool = False
    spatial_weight: float = 0.15

    candidate_method: str = "non_white"
    white_threshold: int = 245
    canny_threshold1: int = 80
    canny_threshold2: int = 180
    canny_aperture_size: int = 3
    canny_l2gradient: bool = False
    canny_dilate_iter: int = 1
    max_candidate_points: int = 0


def auto_eps_knn(feats: np.ndarray, k: int = 10, q: float = 0.90) -> float:
    n = feats.shape[0]
    if n <= 1:
        return 0.0
    k_eff = max(2, min(int(k), n))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(feats)
    dists, _ = nn.kneighbors(feats)
    kth = dists[:, -1]
    eps = float(np.quantile(kth, float(q)))
    return max(eps, 1e-6)


def _cluster_points_by_embedding(
    emb: np.ndarray,
    coords_xy: np.ndarray,
    params: ClusterParams,
) -> tuple[np.ndarray, np.ndarray, float]:
    _, h, w = emb.shape
    if len(coords_xy) == 0:
        return np.zeros((0,), dtype=np.int32), coords_xy, float(params.eps)

    xs = coords_xy[:, 0]
    ys = coords_xy[:, 1]
    ok = (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
    coords_used = coords_xy[ok]
    if len(coords_used) == 0:
        return np.zeros((0,), dtype=np.int32), coords_used, float(params.eps)

    xs = coords_used[:, 0]
    ys = coords_used[:, 1]
    feats = emb[:, ys, xs].T.astype(np.float32)

    if params.l2_normalize:
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    feats_cluster = feats
    if params.use_spatial:
        xyn = coords_used.astype(np.float32).copy()
        xyn[:, 0] = xyn[:, 0] / max(1.0, float(w - 1))
        xyn[:, 1] = xyn[:, 1] / max(1.0, float(h - 1))
        xyn = xyn - 0.5
        feats_cluster = np.concatenate([feats, float(params.spatial_weight) * xyn], axis=1)

    eps_used = (
        auto_eps_knn(feats_cluster, k=params.auto_eps_k, q=params.auto_eps_q)
        if params.auto_eps
        else float(params.eps)
    )
    labels = DBSCAN(eps=eps_used, min_samples=int(params.min_samples)).fit(feats_cluster).labels_.astype(np.int32)
    return labels, coords_used, float(eps_used)


def _remap_clusters(labels: np.ndarray) -> np.ndarray:
    uniq = [u for u in sorted(set(labels.tolist())) if int(u) != -1]
    remap = {int(u): i + 1 for i, u in enumerate(uniq)}
    return np.array([remap[int(lb)] if int(lb) != -1 else -1 for lb in labels], dtype=np.int32)


def _build_clusters(coords_used: np.ndarray, labels_remap: np.ndarray) -> dict[str, list[list[int]]]:
    clusters: dict[str, list[list[int]]] = {}
    for (x, y), lb in zip(coords_used.tolist(), labels_remap.tolist()):
        if int(lb) == -1:
            continue
        clusters.setdefault(str(int(lb)), []).append([int(x), int(y)])
    return clusters


def _build_cluster_pixels(
    rgb: np.ndarray,
    coords_used: np.ndarray,
    labels_remap: np.ndarray,
) -> dict[str, list[list[int]]]:
    pixel_values: dict[str, list[list[int]]] = {}
    for (x, y), lb in zip(coords_used.tolist(), labels_remap.tolist()):
        if int(lb) == -1:
            continue
        r, g, b = rgb[int(y), int(x)].tolist()
        pixel_values.setdefault(str(int(lb)), []).append([int(r), int(g), int(b)])
    return pixel_values


class InferenceService:
    def __init__(self, checkpoint_path: str | None = None, emb_dim: int | None = None, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path or settings.model_checkpoint_path)
        self.emb_dim = int(emb_dim if emb_dim is not None else settings.model_emb_dim)
        self.device = device or settings.model_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: SmallUNet | None = None

    def _load_model(self) -> None:
        if self.model is not None:
            return
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        model = SmallUNet(in_ch=3, emb_dim=self.emb_dim).to(self.device)
        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        model.eval()
        self.model = model

    @torch.no_grad()
    def _embedding_map(self, rgb: np.ndarray) -> np.ndarray:
        assert self.model is not None
        x = rgb.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        emb = self.model(x)[0].detach().cpu().numpy()
        return emb

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Invalid or unsupported image file")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def predict(self, image_bytes: bytes, params: ClusterParams) -> dict[str, Any]:
        self._load_model()
        rgb = self._decode_image(image_bytes)
        emb = self._embedding_map(rgb)

        coords = extract_candidate_coords(
            rgb=rgb,
            method=params.candidate_method,
            white_threshold=params.white_threshold,
            canny_threshold1=params.canny_threshold1,
            canny_threshold2=params.canny_threshold2,
            canny_aperture_size=params.canny_aperture_size,
            canny_l2gradient=params.canny_l2gradient,
            canny_dilate_iter=params.canny_dilate_iter,
            max_candidate_points=params.max_candidate_points,
        )

        labels, coords_used, eps_used = _cluster_points_by_embedding(emb=emb, coords_xy=coords, params=params)
        labels_remap = _remap_clusters(labels)
        clusters = _build_clusters(coords_used=coords_used, labels_remap=labels_remap)
        cluster_pixel_values = _build_cluster_pixels(rgb=rgb, coords_used=coords_used, labels_remap=labels_remap)

        noise = [
            [int(x), int(y)]
            for (x, y), lb in zip(coords_used.tolist(), labels_remap.tolist())
            if int(lb) == -1
        ]

        result = {
            "num_clusters": int(len(clusters)),
            "clusters": clusters,
            "cluster_pixel_values": cluster_pixel_values,
            "noise": noise,
            "meta": {
                "total_points": int(len(coords)),
                "used_points": int(len(coords_used)),
                "noise_points": int(np.sum(labels_remap == -1)),
                "eps_used": float(eps_used),
                "image_height": int(rgb.shape[0]),
                "image_width": int(rgb.shape[1]),
                "params": asdict(params),
            },
        }
        return result
