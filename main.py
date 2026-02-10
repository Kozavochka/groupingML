import os, json, random
import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import DBSCAN
import torch
import torch.nn.functional as F

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


class SmallUNet(nn.Module):
    def __init__(self, in_ch=3, emb_dim=8):
        super().__init__()

        def C(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.down1 = C(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = C(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.mid  = C(64, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = C(128, 64)

        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = C(64, 32)

        self.emb_head = nn.Conv2d(32, emb_dim, 1)

    @staticmethod
    def _crop_like(src, tgt):
        """
        Crop src (B,C,H,W) to spatial size of tgt (B,C,h,w)
        """
        _, _, H, W = src.shape
        _, _, h, w = tgt.shape
        if H == h and W == w:
            return src
        dh = H - h
        dw = W - w
        top = dh // 2
        left = dw // 2
        return src[:, :, top:top+h, left:left+w]

    def forward(self, x):
        d1 = self.down1(x)          # (B,32,H,W)
        d2 = self.down2(self.pool1(d1))   # (B,64,H/2,W/2)

        m  = self.mid(self.pool2(d2))     # (B,128,H/4,W/4)

        u2 = self.up2(m)            # (B,64,~H/2,~W/2)
        d2c = self._crop_like(d2, u2)
        x2 = self.dec2(torch.cat([u2, d2c], dim=1))

        u1 = self.up1(x2)           # (B,32,~H,~W)
        d1c = self._crop_like(d1, u1)
        x1 = self.dec1(torch.cat([u1, d1c], dim=1))

        emb = self.emb_head(x1)     # (B,emb_dim,H,W) (примерно, после crop совпадает)
        return emb

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
def eval_val_loss(model, dl_val, device, use_amp=False):
    model.eval()
    total = 0.0
    n = 0
    for img, inst, valid, _ in dl_val:
        img = img.to(device); inst = inst.to(device); valid = valid.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            emb = model(img)
            h, w = emb.shape[-2], emb.shape[-1]
            inst  = inst[:, :h, :w]
            valid = valid[:, :h, :w]
            loss = discriminative_loss(emb, inst, valid)
        total += float(loss.item())
        n += 1
    model.train()
    return total / max(1, n)

def main():
    args = parse_args()

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

                loss = discriminative_loss(emb, inst, valid)

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
            val_loss = eval_val_loss(model, dl_val, device, use_amp=use_amp)
            print(f"epoch {epoch} val_loss={val_loss:.4f}")

            if (best_val is None) or (val_loss < best_val):
                best_val = val_loss
                save_ckpt(save_dir / "ckpt_best.pt", model, opt, epoch, best_val=best_val)
                print(f"Saved BEST -> {save_dir/'ckpt_best.pt'} (best_val={best_val:.4f})")

    # финальный сейв на всякий
    save_ckpt(save_dir / "ckpt_final.pt", model, opt, args.epochs, best_val=best_val)
    print("Training finished. Saved final checkpoint.")

if __name__ == "__main__":
    main()

