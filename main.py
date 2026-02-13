import os, json, random
import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

from model import SmallUNet

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_json", required=True)
    p.add_argument("--train_img_dir", required=True)
    p.add_argument("--val_json", required=True)
    p.add_argument("--val_img_dir", required=True)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=1, help="save last every N epochs")
    p.add_argument("--eval_every", type=int, default=1, help="eval val loss every N epochs")
    p.add_argument("--resume", type=str, default="", help="path to ckpt_last.pt to resume")

    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--thickness", type=int, default=2)
    p.add_argument("--line_category_id", type=int, default=2)

    p.add_argument("--max_steps", type=int, default=0, help="0 = full epoch; else limit steps per epoch")
    p.add_argument("--amp", action="store_true", help="enable mixed precision (CUDA only)")
    p.add_argument("--seed", type=int, default=42)

    # Discriminative loss hyperparameters.
    p.add_argument("--delta_v", type=float, default=0.5)
    p.add_argument("--delta_d", type=float, default=1.5)
    p.add_argument("--w_var", type=float, default=1.0)
    p.add_argument("--w_dist", type=float, default=1.0)
    p.add_argument("--w_reg", type=float, default=0.001)

    # Validation clustering metrics.
    p.add_argument("--disable_val_metrics", action="store_true")
    p.add_argument("--val_metric_eps", type=float, default=0.5)
    p.add_argument("--val_metric_min_samples", type=int, default=6)
    p.add_argument("--val_metric_max_points", type=int, default=6000)
    p.add_argument("--val_metric_pair_samples", type=int, default=20000)

    # Which metric defines "best checkpoint".
    p.add_argument("--best_metric", choices=["val_loss", "val_pair_f1"], default="val_loss")
    return p.parse_args()

def pad_to(x, H, W, pad_value=0):
    # x: (C,H,W) or (H,W)
    if x.dim() == 3:
        C, h, w = x.shape
        pad_h = H - h
        pad_w = W - w
        return F.pad(x, (0, pad_w, 0, pad_h), value=pad_value)
    else:
        h, w = x.shape
        pad_h = H - h
        pad_w = W - w
        return F.pad(x, (0, pad_w, 0, pad_h), value=pad_value)

def collate_pad(batch):
    # batch items: (img_t, inst_t, valid_t, image_id)
    imgs, insts, valids, ids = zip(*batch)

    maxH = max(im.shape[1] for im in imgs)
    maxW = max(im.shape[2] for im in imgs)

    imgs_p   = torch.stack([pad_to(im, maxH, maxW, pad_value=0.0) for im in imgs], dim=0)
    insts_p  = torch.stack([pad_to(m,  maxH, maxW, pad_value=0)   for m in insts], dim=0)
    valids_p = torch.stack([pad_to(v.to(torch.uint8), maxH, maxW, pad_value=0).bool() for v in valids], dim=0)

    return imgs_p, insts_p, valids_p, torch.tensor(ids, dtype=torch.long)

def load_coco_like(json_path):
    data = json.loads(Path(json_path).read_text())
    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    cats = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
    return data, images, anns_by_image, cats

def poly_to_pts(poly):
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    pts = np.round(pts).astype(np.int32)
    return pts

def build_masks_for_image(img_meta, anns, line_category_id=2, thickness=2):
    H, W = int(img_meta["height"]), int(img_meta["width"])
    candidate = np.zeros((H, W), dtype=np.uint8)
    instance  = np.zeros((H, W), dtype=np.int32)
    overlap   = np.zeros((H, W), dtype=np.uint8)

    anns = [a for a in anns if a.get("category_id") == line_category_id]

    k = 0
    for ann in anns:
        poly = ann.get("bbox", None)
        if not poly or len(poly) < 4:
            continue

        pts = poly_to_pts(poly)
        k += 1

        m = np.zeros((H, W), dtype=np.uint8)
        cv2.polylines(
            m, [pts],
            isClosed=False,
            color=1,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )

        overlap[(m > 0) & (candidate > 0)] = 1
        candidate[m > 0] = 1
        instance[m > 0] = k

    return candidate, instance, k, overlap

def discriminative_loss(emb, inst, mask, delta_v=0.5, delta_d=1.5, w_var=1.0, w_dist=1.0, w_reg=0.001):
    """
    emb: (B, D, H, W) float
    inst: (B, H, W) int32 (0 background, 1..K instances)
    mask: (B, H, W) bool (where valid line pixels)
    """
    B, D, H, W = emb.shape
    loss_var = emb.new_zeros(())
    loss_dist = emb.new_zeros(())
    loss_reg = emb.new_zeros(())

    for b in range(B):
        emb_b = emb[b].permute(1,2,0).reshape(-1, D)     # (H*W, D)
        inst_b = inst[b].reshape(-1)                     # (H*W,)
        mask_b = mask[b].reshape(-1)                     # (H*W,)

        emb_b = emb_b[mask_b]
        inst_b = inst_b[mask_b]

        if emb_b.numel() == 0:
            continue

        ids = torch.unique(inst_b)
        ids = ids[ids != 0]
        if len(ids) == 0:
            continue

        # centroids
        mus = []
        for i in ids:
            ei = emb_b[inst_b == i]
            mu = ei.mean(dim=0)
            mus.append(mu)

            # variance (pull)
            dist = torch.norm(ei - mu, dim=1)
            loss_var = loss_var + torch.mean(F.relu(dist - delta_v) ** 2)

        mus = torch.stack(mus, dim=0)  # (K, D)

        # distance (push)
        if mus.shape[0] > 1:
            K = mus.shape[0]
            mu_a = mus.unsqueeze(0).repeat(K,1,1)
            mu_b = mus.unsqueeze(1).repeat(1,K,1)
            diff = mu_a - mu_b
            dist_mu = torch.norm(diff, dim=2)  # (K,K)

            eye = torch.eye(K, device=emb.device).bool()
            dist_mu = dist_mu[~eye]
            loss_dist = loss_dist + torch.mean(F.relu(2*delta_d - dist_mu) ** 2)

        # regularization
        loss_reg = loss_reg + torch.mean(torch.norm(mus, dim=1))

    denom = max(B, 1)
    return (w_var * loss_var + w_dist * loss_dist + w_reg * loss_reg) / denom


class ChartLineDataset(Dataset):
    def __init__(self, JSON_PATH, IMAGES_DIR, thickness=2, line_category_id=2):
        self.data, self.images, self.anns_by_image, self.cats = load_coco_like(JSON_PATH)
        self.ids = sorted(self.images.keys())
        self.images_dir = Path(IMAGES_DIR)
        self.thickness = thickness
        self.line_category_id = line_category_id

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        meta = self.images[image_id]
        img_path = self.images_dir / meta["file_name"]

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        anns = self.anns_by_image.get(image_id, [])
        cand, inst, k, overlap = build_masks_for_image(meta, anns,
                                                       line_category_id=self.line_category_id,
                                                       thickness=self.thickness)
        # valid line pixels = candidate and not overlap
        valid = (cand > 0) & (overlap == 0)

        # to torch
        img_t = torch.from_numpy(img).permute(2,0,1).float()
        inst_t = torch.from_numpy(inst).long()
        valid_t = torch.from_numpy(valid).bool()

        return img_t, inst_t, valid_t, image_id

def save_ckpt(path, model, opt, epoch, best_val=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "best_val": best_val,
    }, str(path))

def load_ckpt(path, model, opt=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = ckpt.get("best_val", None)
    return start_epoch, best_val

@torch.no_grad()
def _pair_f1_from_sampled_pairs(gt_labels, pred_labels, num_pairs, rng):
    n = int(len(gt_labels))
    if n < 2 or num_pairs <= 0:
        return float("nan")

    i = rng.integers(0, n, size=int(num_pairs), dtype=np.int64)
    j = rng.integers(0, n, size=int(num_pairs), dtype=np.int64)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n, size=int(np.sum(same)), dtype=np.int64)
        same = i == j

    gt_same = gt_labels[i] == gt_labels[j]
    pred_same = (pred_labels[i] == pred_labels[j]) & (pred_labels[i] != -1)

    positives = int(np.sum(gt_same))
    if positives == 0:
        return float("nan")

    tp = int(np.sum(gt_same & pred_same))
    fp = int(np.sum((~gt_same) & pred_same))
    fn = int(np.sum(gt_same & (~pred_same)))
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return float("nan")
    return float((2 * tp) / denom)


@torch.no_grad()
def _sample_grouping_metrics(
    emb_sample: np.ndarray,
    inst_sample: np.ndarray,
    valid_sample: np.ndarray,
    metric_eps: float,
    metric_min_samples: int,
    metric_max_points: int,
    metric_pair_samples: int,
    rng,
):
    line_mask = (valid_sample > 0) & (inst_sample > 0)
    ys, xs = np.nonzero(line_mask)
    if len(xs) < 2:
        return None

    gt_labels = inst_sample[ys, xs].astype(np.int32, copy=False)
    if metric_max_points > 0 and len(xs) > metric_max_points:
        idx = rng.choice(len(xs), size=int(metric_max_points), replace=False)
        ys = ys[idx]
        xs = xs[idx]
        gt_labels = gt_labels[idx]

    feats = emb_sample[:, ys, xs].T.astype(np.float32, copy=False)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    min_samples = max(2, int(metric_min_samples))
    labels = DBSCAN(eps=float(metric_eps), min_samples=min_samples).fit(feats).labels_.astype(np.int32)

    ari = float(adjusted_rand_score(gt_labels, labels))
    pair_f1 = _pair_f1_from_sampled_pairs(gt_labels, labels, metric_pair_samples, rng)

    gt_num = int(np.unique(gt_labels).shape[0])
    pred_num = int(np.unique(labels[labels != -1]).shape[0]) if np.any(labels != -1) else 0
    count_mae = float(abs(pred_num - gt_num))
    noise_ratio = float(np.mean(labels == -1))

    return {
        "ari": ari,
        "pair_f1": pair_f1,
        "cluster_count_mae": count_mae,
        "noise_ratio": noise_ratio,
    }


@torch.no_grad()
def eval_val(
    model,
    dl_val,
    device,
    use_amp=False,
    loss_kwargs=None,
    metric_cfg=None,
):
    model.eval()
    loss_kwargs = loss_kwargs or {}

    total = 0.0
    n = 0
    metric_keys = ["ari", "pair_f1", "cluster_count_mae", "noise_ratio"]
    metric_values = {k: [] for k in metric_keys}
    metric_samples = 0
    rng = np.random.default_rng(int(metric_cfg["seed"])) if metric_cfg is not None else None

    for img, inst, valid, _ in dl_val:
        img = img.to(device); inst = inst.to(device); valid = valid.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            emb = model(img)
            h, w = emb.shape[-2], emb.shape[-1]
            inst  = inst[:, :h, :w]
            valid = valid[:, :h, :w]
            loss = discriminative_loss(emb, inst, valid, **loss_kwargs)
        total += float(loss.item())
        n += 1

        if metric_cfg is not None:
            emb_np = emb.detach().cpu().numpy()
            inst_np = inst.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()
            for b in range(emb_np.shape[0]):
                sample_m = _sample_grouping_metrics(
                    emb_sample=emb_np[b],
                    inst_sample=inst_np[b],
                    valid_sample=valid_np[b],
                    metric_eps=float(metric_cfg["eps"]),
                    metric_min_samples=int(metric_cfg["min_samples"]),
                    metric_max_points=int(metric_cfg["max_points"]),
                    metric_pair_samples=int(metric_cfg["pair_samples"]),
                    rng=rng,
                )
                if sample_m is None:
                    continue
                metric_samples += 1
                for k in metric_keys:
                    metric_values[k].append(sample_m[k])

    model.train()
    out = {"val_loss": total / max(1, n)}
    out["val_metric_samples"] = int(metric_samples)
    for k in metric_keys:
        vals = np.asarray(metric_values[k], dtype=np.float64)
        out[f"val_{k}"] = float(np.nanmean(vals)) if vals.size > 0 else float("nan")
    return out

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # data
    ds = ChartLineDataset(args.train_json, args.train_img_dir,
                          thickness=args.thickness, line_category_id=args.line_category_id)

    # persistent_workers имеет смысл только если num_workers > 0
    persistent = True if args.num_workers > 0 else False
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate_pad,
                    persistent_workers=persistent)

    ds_val = ChartLineDataset(args.val_json, args.val_img_dir,
                              thickness=args.thickness, line_category_id=args.line_category_id)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_pad)

    print("len(ds) =", len(ds))
    print("batch_size =", args.batch_size)
    print("len(dl) steps/epoch =", len(dl))

    # model
    model = SmallUNet(in_ch=3, emb_dim=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = bool(args.amp and device == "cuda")
    if args.amp and not use_amp:
        print("AMP requested, but CUDA is unavailable. Running in FP32.")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loss_kwargs = {
        "delta_v": float(args.delta_v),
        "delta_d": float(args.delta_d),
        "w_var": float(args.w_var),
        "w_dist": float(args.w_dist),
        "w_reg": float(args.w_reg),
    }
    metric_cfg = None if args.disable_val_metrics else {
        "eps": float(args.val_metric_eps),
        "min_samples": int(args.val_metric_min_samples),
        "max_points": int(args.val_metric_max_points),
        "pair_samples": int(args.val_metric_pair_samples),
        "seed": int(args.seed),
    }

    # resume
    start_epoch = 1
    best_val = None
    if args.resume:
        start_epoch, best_val = load_ckpt(args.resume, model, opt, device=device)
        print(f"Resumed from {args.resume} -> start_epoch={start_epoch}, best_val={best_val}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # train
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        steps_done = 0
        t0 = time.time()

        for step, (img, inst, valid, _) in enumerate(dl, start=1):
            img = img.to(device); inst = inst.to(device); valid = valid.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                emb = model(img)
                h, w = emb.shape[-2], emb.shape[-1]
                inst  = inst[:, :h, :w]
                valid = valid[:, :h, :w]

                loss = discriminative_loss(emb, inst, valid, **loss_kwargs)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += float(loss.item())
            steps_done = step

            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(dl)} loss={total/steps_done:.4f}")

            if args.max_steps and step >= args.max_steps:
                break

        train_loss = total / max(1, steps_done)
        print(f"epoch {epoch} DONE train_loss={train_loss:.4f} steps={steps_done} time={time.time()-t0:.1f}s")

        # --- save last every N epochs ---
        if epoch % args.save_every == 0:
            save_ckpt(save_dir / "ckpt_last.pt", model, opt, epoch, best_val=best_val)

        # --- eval & save best ---
        if epoch % args.eval_every == 0:
            val_stats = eval_val(
                model=model,
                dl_val=dl_val,
                device=device,
                use_amp=use_amp,
                loss_kwargs=loss_kwargs,
                metric_cfg=metric_cfg,
            )
            val_loss = float(val_stats["val_loss"])
            val_ari = float(val_stats["val_ari"])
            val_pair_f1 = float(val_stats["val_pair_f1"])
            val_count_mae = float(val_stats["val_cluster_count_mae"])
            val_noise_ratio = float(val_stats["val_noise_ratio"])
            val_metric_samples = int(val_stats["val_metric_samples"])

            print(
                f"epoch {epoch} val_loss={val_loss:.4f} "
                f"val_ari={val_ari:.4f} "
                f"val_pair_f1={val_pair_f1:.4f} "
                f"val_cluster_count_mae={val_count_mae:.4f} "
                f"val_noise_ratio={val_noise_ratio:.4f} "
                f"metric_samples={val_metric_samples}"
            )

            if args.best_metric == "val_pair_f1":
                score = val_pair_f1
                better = np.isfinite(score) and ((best_val is None) or (score > best_val))
            else:
                score = val_loss
                better = (best_val is None) or (score < best_val)

            if better:
                best_val = score
                save_ckpt(save_dir / "ckpt_best.pt", model, opt, epoch, best_val=best_val)
                print(
                    f"Saved BEST -> {save_dir/'ckpt_best.pt'} "
                    f"(best_metric={args.best_metric}, best_val={best_val:.4f})"
                )

    # финальный сейв на всякий
    save_ckpt(save_dir / "ckpt_final.pt", model, opt, args.epochs, best_val=best_val)
    print("Training finished. Saved final checkpoint.")

if __name__ == "__main__":
    main()

