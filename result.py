import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from main import SmallUNet


PALETTE_BGR = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 128, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 255, 128),
]


def parse_args():
    p = argparse.ArgumentParser(description="Cluster candidate pixels by embedding and render visualization.")
    p.add_argument("--images_dir", required=True, type=str, help="Root folder with images.")
    p.add_argument("--candidates_json", required=True, type=str, help="JSON with candidate points per image.")
    p.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint (*.pt).")
    p.add_argument("--out_json", required=True, type=str, help="Output JSON path with clustering results.")
    p.add_argument("--out_viz_dir", required=True, type=str, help="Output directory for rendered images.")

    p.add_argument("--emb_dim", type=int, default=16)
    p.add_argument("--score_thr", type=float, default=0.4)
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--min_samples", type=int, default=20)
    p.add_argument("--radius", type=int, default=2, help="Draw radius for points.")
    p.add_argument("--draw_noise", action="store_true")

    p.add_argument("--auto_eps", action="store_true", help="Use per-image eps from kNN distances.")
    p.add_argument("--auto_eps_k", type=int, default=10)
    p.add_argument("--auto_eps_q", type=float, default=0.90)

    return p.parse_args()


def parse_candidates(obj_for_image, score_thr=0.0):
    """
    Expected format:
    list[dict["0"] -> list[[score, cls, x, y], ...]]
    Returns (N, 2) int32: [x, y].
    """
    pts = []
    if not isinstance(obj_for_image, list):
        return np.zeros((0, 2), dtype=np.int32)

    for item in obj_for_image:
        if not isinstance(item, dict):
            continue
        for _, arr in item.items():
            if not isinstance(arr, list):
                continue
            for p in arr:
                if not isinstance(p, (list, tuple)) or len(p) < 4:
                    continue
                score = float(p[0])
                if score < score_thr:
                    continue
                x = int(round(float(p[2])))
                y = int(round(float(p[3])))
                pts.append((x, y))

    if not pts:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(pts, dtype=np.int32)


def build_image_index(root: Path, exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp")):
    idx = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            idx[p.name] = p
    return idx


def load_model(ckpt_path: Path, emb_dim: int, device: str):
    model = SmallUNet(in_ch=3, emb_dim=emb_dim).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def get_embedding_map(model, rgb, device):
    x = rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    emb = model(x)[0].detach().cpu().numpy()
    return emb


def auto_eps_knn(feats, k=10, q=0.90):
    n = feats.shape[0]
    if n <= 1:
        return 0.0
    k_eff = max(2, min(k, n))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(feats)
    dists, _ = nn.kneighbors(feats)
    kth = dists[:, -1]
    eps = float(np.quantile(kth, q))
    return max(eps, 1e-6)


def cluster_points_by_embedding(
    emb, coords_xy, eps=0.3, min_samples=20, l2_normalize=True, auto_eps=False, auto_eps_k=10, auto_eps_q=0.90
):
    """
    emb: (D, h, w)
    coords_xy: (N, 2) image coords [x, y]
    Returns:
      labels: (M,) DBSCAN labels for filtered points
      coords_used: (M, 2)
      eps_used: float
    """
    _, h, w = emb.shape
    if len(coords_xy) == 0:
        return np.zeros((0,), dtype=np.int32), coords_xy, float(eps)

    xs = coords_xy[:, 0]
    ys = coords_xy[:, 1]
    ok = (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
    coords_used = coords_xy[ok]
    if len(coords_used) == 0:
        return np.zeros((0,), dtype=np.int32), coords_used, float(eps)

    xs = coords_used[:, 0]
    ys = coords_used[:, 1]
    feats = emb[:, ys, xs].T

    if l2_normalize:
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    eps_used = auto_eps_knn(feats, k=auto_eps_k, q=auto_eps_q) if auto_eps else float(eps)
    labels = DBSCAN(eps=eps_used, min_samples=min_samples).fit(feats).labels_.astype(np.int32)
    return labels, coords_used, eps_used


def remap_clusters(labels):
    uniq = [u for u in sorted(set(labels.tolist())) if u != -1]
    remap = {int(u): i + 1 for i, u in enumerate(uniq)}
    labels_remap = np.array([remap[int(lb)] if int(lb) != -1 else -1 for lb in labels], dtype=np.int32)
    return labels_remap


def build_clusters_dict(coords_used, labels_remap):
    clusters = {}
    for (x, y), lb in zip(coords_used.tolist(), labels_remap.tolist()):
        if lb == -1:
            continue
        clusters.setdefault(str(lb), []).append([int(x), int(y)])
    return clusters


def draw_clusters(image_bgr, clusters, noise=None, radius=2, draw_noise=False):
    out = image_bgr.copy()

    def _key(k):
        try:
            return int(k)
        except Exception:
            return k

    cluster_ids = sorted(clusters.keys(), key=_key)
    for i, cid in enumerate(cluster_ids):
        color = PALETTE_BGR[i % len(PALETTE_BGR)]
        for x, y in clusters[cid]:
            cv2.circle(out, (int(x), int(y)), radius + 1, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)

    if draw_noise and noise is not None:
        for x, y in noise:
            cv2.circle(out, (int(x), int(y)), radius + 1, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    return out


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    images_dir = Path(args.images_dir)
    out_json = Path(args.out_json)
    out_viz_dir = Path(args.out_viz_dir)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_viz_dir.mkdir(parents=True, exist_ok=True)

    with open(args.candidates_json, "r", encoding="utf-8") as f:
        cand_data = json.load(f)

    img_index = build_image_index(images_dir)
    model = load_model(Path(args.checkpoint), emb_dim=args.emb_dim, device=device)

    results = {}
    missing = []

    for file_name, obj_for_image in cand_data.items():
        img_path = img_index.get(Path(file_name).name)
        if img_path is None:
            missing.append(file_name)
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            missing.append(file_name)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        coords = parse_candidates(obj_for_image, score_thr=args.score_thr)
        emb = get_embedding_map(model, rgb, device)
        labels, coords_used, eps_used = cluster_points_by_embedding(
            emb,
            coords,
            eps=args.eps,
            min_samples=args.min_samples,
            l2_normalize=True,
            auto_eps=args.auto_eps,
            auto_eps_k=args.auto_eps_k,
            auto_eps_q=args.auto_eps_q,
        )

        labels_remap = remap_clusters(labels)
        clusters = build_clusters_dict(coords_used, labels_remap)
        noise = [[int(x), int(y)] for (x, y), lb in zip(coords_used.tolist(), labels_remap.tolist()) if lb == -1]

        num_clusters = len(clusters)
        results[file_name] = {
            "num_clusters": num_clusters,
            "clusters": clusters,  # dict: cluster_id -> [[x,y], ...]
            "noise": noise,
            "meta": {
                "total_points": int(len(coords)),
                "used_points": int(len(coords_used)),
                "noise_points": int(np.sum(labels_remap == -1)),
                "eps_used": float(eps_used),
                "min_samples": int(args.min_samples),
                "auto_eps": bool(args.auto_eps),
                "auto_eps_k": int(args.auto_eps_k),
                "auto_eps_q": float(args.auto_eps_q),
            },
        }

        vis = draw_clusters(
            bgr,
            clusters=clusters,
            noise=noise,
            radius=args.radius,
            draw_noise=args.draw_noise,
        )
        out_path = out_viz_dir / f"{Path(file_name).stem}_clusters.png"
        cv2.imwrite(str(out_path), vis)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Processed: {len(results)} images")
    print(f"Missing/unreadable: {len(missing)} images")
    if missing:
        print("First missing:", missing[:10])
    print(f"Saved results JSON: {out_json}")
    print(f"Saved visualizations: {out_viz_dir}")


if __name__ == "__main__":
    main()
