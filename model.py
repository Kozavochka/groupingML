import torch
import torch.nn as nn


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

        self.mid = C(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = C(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = C(64, 32)

        self.emb_head = nn.Conv2d(32, emb_dim, 1)

    @staticmethod
    def _match_spatial(a, b):
        # a,b: (B,C,H,W) -> crop to min(H,W) around center
        Ha, Wa = a.shape[-2:]
        Hb, Wb = b.shape[-2:]
        h = min(Ha, Hb)
        w = min(Wa, Wb)

        def center_crop(x, h, w):
            H, W = x.shape[-2:]
            top = (H - h) // 2
            left = (W - w) // 2
            return x[:, :, top : top + h, left : left + w]

        return center_crop(a, h, w), center_crop(b, h, w)

    def forward(self, x):
        d1 = self.down1(x)  # (B,32,H,W)
        d2 = self.down2(self.pool1(d1))  # (B,64,H/2,W/2)

        m = self.mid(self.pool2(d2))  # (B,128,H/4,W/4)

        u2 = self.up2(m)  # (B,64,~H/2,~W/2)
        u2, d2c = self._match_spatial(u2, d2)
        x2 = self.dec2(torch.cat([u2, d2c], dim=1))

        u1 = self.up1(x2)  # (B,32,~H,~W)
        u1, d1c = self._match_spatial(u1, d1)
        x1 = self.dec1(torch.cat([u1, d1c], dim=1))

        emb = self.emb_head(x1)  # (B,emb_dim,H,W)
        return emb
