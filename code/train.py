"""
TurtleVision ensemble training.

Run:   python train.py --data path/to/my_dataset
       python train.py --data path/to/my_dataset --weights weights/

Trains the four ensemble models one after another. Each is saved to --out
(default runs/) under the file name infer.py expects.

  from ImageNet (default)      up to 42 epochs, stops after 21 without a wGeoAcc gain
  from released (--weights)    13 epochs

Both start at learning rate 5e-5, halve it after 5 epochs without a gain, and
keep the epoch with the best validation wGeoAcc.
"""

import argparse
import math
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


# =============================================================================
# SETTINGS
# =============================================================================

MODELS = [  # (patch size, backbone, color), same as infer.py
    (320, "swin_s", "GREY"),
    (512, "efficientnet_b5", "RGB"),
    (512, "swin_s", "RGB"),
    (1024, "efficientnet_b5", "RGB"),
]

IMG_SIZE = 512
BATCH_SIZE = 16
LR = 5e-5
FN_WEIGHT = 2          # wGeoAcc = (TPR^2 * TNR)^(1/3)
IMAGE_LOSS_WEIGHT = 0.05
STRIDES = [4, 8, 16, 32, 64, 128]
NORMALIZE = {"RGB": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), "GREY": ([0.449] * 3, [0.226] * 3)}

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()


# =============================================================================
# DATA
# =============================================================================

class TurtleTracksDataset(Dataset):
    """root/images/{split}/*.jpg|png with labels in root/labels/{split}/*.txt.
    A label line is `class x1 y1 x2 y2 x3 y3 x4 y4`, corners scaled to [0, 1].
    A missing or empty label file means a patch with no tracks."""

    def __init__(self, root, split, transform):
        self.images = sorted(p for p in (root / "images" / split).iterdir() if p.suffix.lower() in (".jpg", ".png"))
        self.labels = root / "labels" / split
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(str(self.images[i])), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        label = self.labels / f"{self.images[i].stem}.txt"
        rows = [r.split() for r in label.read_text().splitlines()] if label.exists() else []
        corners = np.array([r[1:9] for r in rows if len(r) >= 9], np.float32).reshape(-1, 2) * (w, h)

        out = self.transform(image=image, keypoints=corners)
        rects = [cv2.minAreaRect(q) for q in np.float32(out["keypoints"]).reshape(-1, 4, 2)]
        boxes = [(cx, cy, bw, bh, math.radians(a)) for (cx, cy), (bw, bh), a in rects if bw > 1 and bh > 1]
        return out["image"], torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5)


def build_transform(color, train):
    augs = [A.Resize(IMG_SIZE, IMG_SIZE)]
    if train:
        augs += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.5, 1.5), translate_percent=(-0.1, 0.1), p=0.5),
            A.HueSaturationValue(int(0.015 * 180), int(0.7 * 255), int(0.4 * 255), p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
            A.OneOf([A.GaussianBlur(blur_limit=(5, 9)), A.MotionBlur(blur_limit=(5, 9)), A.MedianBlur(blur_limit=5)], p=0.3),
            A.GaussNoise(std_range=(0.05, 0.18), p=0.25),
            A.ImageCompression(quality_range=(25, 65), p=0.25),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(int(IMG_SIZE * 0.02), int(IMG_SIZE * 0.1)),
                            hole_width_range=(int(IMG_SIZE * 0.02), int(IMG_SIZE * 0.1)), fill=114, p=0.3),
        ]
    if color == "GREY":
        augs.append(A.ToGray(num_output_channels=3, p=1.0))
    augs += [A.Normalize(*NORMALIZE[color]), ToTensorV2()]
    return A.Compose(augs, keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))


def collate_fn(batch):
    images, boxes = zip(*batch)
    return torch.stack(images), list(boxes)


def make_loader(root, split, color):
    train = split == "train"
    return DataLoader(TurtleTracksDataset(root, split, build_transform(color, train)), batch_size=BATCH_SIZE,
                      shuffle=train, num_workers=8, persistent_workers=True, pin_memory=True, collate_fn=collate_fn)


# =============================================================================
# MODEL
# =============================================================================

class FPN(nn.Module):
    def __init__(self, in_channels, out=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList(nn.Conv2d(c, out, 1) for c in in_channels)
        self.output_convs = nn.ModuleList(nn.Conv2d(out, out, 3, padding=1) for _ in in_channels)
        self.extra_convs = nn.ModuleList(nn.Conv2d(out, out, 3, stride=2, padding=1) for _ in range(2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        lat = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        for i in range(len(lat) - 2, -1, -1):
            lat[i] = lat[i] + F.interpolate(lat[i + 1], size=lat[i].shape[-2:], mode="nearest")
        outs = [conv(x) for conv, x in zip(self.output_convs, lat)]
        outs.append(self.extra_convs[0](outs[-1]))
        outs.append(self.extra_convs[1](F.relu(outs[-1])))
        return outs


def conv_tower(ch, depth=4):
    layers = []
    for _ in range(depth):
        layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(32, ch), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class FCOSHead(nn.Module):
    def __init__(self, ch=256, num_levels=6):
        super().__init__()
        self.cls_convs = conv_tower(ch)
        self.reg_convs = conv_tower(ch)
        self.quality = nn.Conv2d(ch, 1, 3, padding=1)
        self.box_pred = nn.Conv2d(ch, 6, 3, padding=1)
        self.scales = nn.ParameterList(nn.Parameter(torch.ones(1)) for _ in range(num_levels))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.quality.bias, -math.log(99))  # every point starts at 1% confidence

    def forward(self, features):
        outs = []
        for x, scale in zip(features, self.scales):
            qual = self.quality(self.cls_convs(x))
            box = self.box_pred(self.reg_convs(x))
            angle = box[:, 4:6]
            angle = angle / (torch.linalg.vector_norm(angle, dim=1, keepdim=True) + 1e-6)
            outs.append((qual, torch.cat([box[:, :4] * scale, angle], dim=1)))
        return outs


class ImageClassifier(nn.Module):
    """Does this patch contain a track at all? Training signal only; inference ignores it."""

    def __init__(self, ch=256, num_levels=6):
        super().__init__()
        self.classifier = nn.Linear(ch * num_levels, 1)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features):
        return self.classifier(torch.cat([f.mean((2, 3)) for f in features], 1)).squeeze(-1)


BACKBONES = {
    "swin_s": dict(
        nodes={"features.1": "s0", "features.3": "s1", "features.5": "s2", "features.7": "s3"},
        channels=[96, 192, 384, 768], permute=True),
    "efficientnet_b5": dict(
        nodes={"features.2": "s0", "features.3": "s1", "features.5": "s2", "features.7": "s3"},
        channels=[40, 64, 176, 512], permute=False),
}


class El_Modelo(nn.Module):
    def __init__(self, backbone, imagenet=True):
        super().__init__()
        cfg = BACKBONES[backbone]
        self.permute = cfg["permute"]
        net = getattr(models, backbone)(weights="IMAGENET1K_V1" if imagenet else None)
        self.backbone = create_feature_extractor(net, return_nodes=cfg["nodes"])
        self.fpn = FPN(cfg["channels"])
        self.head = FCOSHead()
        self.image_cls = ImageClassifier()

    def forward(self, x):
        out = self.backbone(x)
        feats = [out[k] for k in ("s0", "s1", "s2", "s3")]
        if self.permute:
            feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        feats = self.fpn(feats)
        return self.head(feats), self.image_cls(feats)


# =============================================================================
# LOSS
# A box is (cx, cy, w, h, angle) in pixels of the 512 x 512 model input.
# =============================================================================

def assign_targets(points, boxes, stride):
    """Each grid point takes the smallest box whose center region contains it.
    The center region is the box, cut to at most 1.5 strides from its center.
    boxes: (B, M, 5), zero padded (a zero size box claims nothing)."""
    cx, cy, w, h, a = boxes.unbind(-1)
    dx = points[None, :, 0, None] - cx[:, None]          # (B, P, M)
    dy = points[None, :, 1, None] - cy[:, None]
    cos, sin = a.cos()[:, None], a.sin()[:, None]
    r = 1.5 * stride
    inside = ((dx * cos + dy * sin).abs() < (w / 2).clamp(max=r)[:, None]) & \
             ((dy * cos - dx * sin).abs() < (h / 2).clamp(max=r)[:, None])
    area, best = torch.where(inside, (w * h)[:, None], torch.inf).min(-1)
    pos = area.isfinite()                                 # (B, P)
    return boxes.gather(1, best[..., None].expand(-1, -1, 5))[pos], pos


def cov(w, h, a):
    """A box as a Gaussian: the (xx, xy, yy) entries of its 2x2 covariance."""
    c, s, w2, h2 = a.cos(), a.sin(), w * w / 4, h * h / 4
    return w2 * c * c + h2 * s * s, (w2 - h2) * c * s, w2 * s * s + h2 * c * c


def det(a, b, d):
    return (a * d - b * b).clamp_min(1e-6)


def kfiou_loss(pred, tgt):
    """Center loss + KFIoU loss for matched boxes. Also returns the KFIoU, the quality target."""
    a1, b1, d1 = cov(pred[:, 2].clamp_min(1), pred[:, 3].clamp_min(1), pred[:, 4])
    a2, b2, d2 = cov(tgt[:, 2].clamp_min(1), tgt[:, 3].clamp_min(1), tgt[:, 4])
    det1, det2 = det(a1, b1, d1), det(a2, b2, d2)

    # how far the true center sits from the predicted one, measured in the predicted box's own shape
    dx, dy = tgt[:, 0] - pred[:, 0], tgt[:, 1] - pred[:, 1]
    center = torch.log((d1 * dx * dx - 2 * b1 * dx * dy + a1 * dy * dy) / det1 + 1)

    # overlap of the Kalman fused Gaussian, whose determinant is det1 * det2 / det(sum)
    s1, s2 = det1.sqrt(), det2.sqrt()
    s12 = s1 * s2 / det(a1 + a2, b1 + b2, d1 + d2).sqrt()
    kfiou = (3 * s12 / (s1 + s2 - s12 + 1e-6)).clamp(0, 1)  # x3 stretches KFIoU's [0, 1/3] to [0, 1]
    return center + torch.exp(1 - kfiou) - 1, kfiou.detach()


def quality_focal_loss(logits, target, beta=2.0):
    weight = (target - logits.sigmoid()).abs().pow(beta)
    return (weight * F.binary_cross_entropy_with_logits(logits, target, reduction="none")).mean()


def loss_fn(outputs, image_logits, targets):
    """Per FPN level: quality focal loss everywhere, box loss at assigned points.
    Plus a small image level loss: does the patch have a track?"""
    device = image_logits.device
    boxes = torch.zeros(len(targets), max(1, *map(len, targets)), 5, device=device)
    for i, t in enumerate(targets):
        boxes[i, :len(t)] = t

    q_loss = box_loss = 0
    for stride, (qual, box) in zip(STRIDES, outputs):
        h, w = qual.shape[-2:]
        ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        points = (torch.stack([xs.flatten(), ys.flatten()], 1) + 0.5) * stride

        tgt, pos = assign_targets(points, boxes, stride)
        q_target = torch.zeros(pos.shape, device=device)
        if pos.any():
            p = box.flatten(2).transpose(1, 2)[pos]
            xy = points.expand(len(pos), -1, -1)[pos]
            pred = torch.cat([xy + p[:, :2] * stride, p[:, 2:4].clamp(-6, 6).exp() * stride,
                              torch.atan2(p[:, 4:5], p[:, 5:6])], 1)
            loss, iou = kfiou_loss(pred, tgt)
            box_loss = box_loss + loss.mean()
            q_target[pos] = iou
        q_loss = q_loss + quality_focal_loss(qual.flatten(1), q_target)

    has_track = torch.tensor([len(t) > 0 for t in targets], dtype=torch.float32, device=device)
    image_loss = F.binary_cross_entropy_with_logits(image_logits, has_track)
    return (q_loss + box_loss) / len(outputs) + IMAGE_LOSS_WEIGHT * image_loss


# =============================================================================
# VALIDATION
# =============================================================================

def decode(outputs, conf_thr):
    """Model output for a batch -> per sample (boxes (N,5) = cx cy w h angle, scores)."""
    batch = outputs[0][0].shape[0]
    parts = [[] for _ in range(batch)]
    for stride, (qual, box) in zip(STRIDES, outputs):
        conf = torch.sigmoid(qual[:, 0])
        bi, gy, gx = torch.where(conf > conf_thr)
        if len(bi) == 0:
            continue
        rb = box[bi, :, gy, gx].float()
        cx = (gx + 0.5 + rb[:, 0]) * stride
        cy = (gy + 0.5 + rb[:, 1]) * stride
        w = torch.exp(rb[:, 2].clamp(-4, 4)) * stride
        h = torch.exp(rb[:, 3].clamp(-4, 4)) * stride
        ang = torch.atan2(rb[:, 4], rb[:, 5])
        boxes = torch.stack([cx.clamp(0, IMG_SIZE), cy.clamp(0, IMG_SIZE),
                             w.clamp(1, IMG_SIZE), h.clamp(1, IMG_SIZE), ang], dim=1)
        bi, boxes, sc = bi.cpu().numpy(), boxes.cpu().numpy(), conf[bi, gy, gx].float().cpu().numpy()
        for i in np.unique(bi):
            parts[i].append((boxes[bi == i], sc[bi == i]))
    return [(np.concatenate([b for b, _ in p]), np.concatenate([s for _, s in p])) if p else (None, None)
            for p in parts]


def obb5_to_corners(b):
    return cv2.boxPoints(((b[0], b[1]), (b[2], b[3]), math.degrees(b[4])))


def quad_iou(a, b):
    if (a.max(0) < b.min(0)).any() or (b.max(0) < a.min(0)).any():
        return 0.0
    inter = cv2.intersectConvexConvex(a, b)[0]
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0.0


def obb_nms(quads, iou_thr=0.5):
    """quads sorted by score, best first."""
    keep = []
    for q in quads:
        if all(quad_iou(k, q) < iou_thr for k in keep):
            keep.append(q)
    return keep


@torch.no_grad()
def validate(model, loader):
    """Patch level TPR and TNR at confidence 0.25. A patch with tracks is a hit if a
    detection touches a track, otherwise a miss. Any detection on an empty patch is a false alarm."""
    model.eval()
    tp = tn = fp = fn = 0
    for images, targets in tqdm(loader, desc="val", leave=False):
        with torch.autocast("cuda", torch.bfloat16, enabled=BF16):
            outputs, _ = model(images.to(DEVICE))
        for (b5, scores), gt in zip(decode(outputs, 0.25), targets):
            dets = [] if b5 is None else obb_nms([obb5_to_corners(b) for b in b5[np.argsort(-scores)[:500]]])
            gts = [obb5_to_corners(g) for g in gt.numpy()]
            if gts:
                if dets and any(quad_iou(d, g) > 0 for d in dets for g in gts):
                    tp += 1
                else:
                    fn += 1
            elif dets:
                fp += 1
            else:
                tn += 1
    tpr, tnr = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return (tpr ** FN_WEIGHT * tnr) ** (1 / (FN_WEIGHT + 1)), tpr, tnr


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer):
    model.train()
    total = 0.0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        targets = [t.to(DEVICE) for t in targets]
        with torch.autocast("cuda", torch.bfloat16, enabled=BF16):
            loss = loss_fn(*model(images), targets)
        if not loss.isfinite():
            continue  # skip the rare bad batch
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def train_model(size, backbone, color, args):
    name = f"turtlevision_{'swin' if backbone == 'swin_s' else 'b5'}_{size}_{'gray' if color == 'GREY' else 'rgb'}"
    print(f"\n### {name}")
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)

    model = El_Modelo(backbone, imagenet=args.weights is None)
    if args.weights:
        sd = torch.load(args.weights / f"{name}.pth", map_location="cpu", weights_only=False)["model_state_dict"]
        model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
    model.to(DEVICE)

    root = args.data / f"patch_{size}"
    train_loader, val_loader = make_loader(root, "train", color), make_loader(root, "val", color)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    epochs, patience = (13, 13) if args.weights else (42, 21)

    best, best_epoch = -1.0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer)
        wga, tpr, tnr = validate(model, val_loader)
        scheduler.step(wga)
        print(f"epoch {epoch}/{epochs}  loss {loss:.4f}  TPR {tpr:.3f}  TNR {tnr:.3f}  wGeoAcc {wga:.4f}  "
              f"lr {optimizer.param_groups[0]['lr']:.1e}  {time.time() - t0:.0f}s")
        if wga > best:
            best, best_epoch = wga, epoch
            torch.save({"model_state_dict": model.state_dict()}, args.out / f"{name}.pth")
        elif epoch - best_epoch >= patience:
            break
    print(f"best wGeoAcc {best:.4f} at epoch {best_epoch} -> {args.out / name}.pth")


def main():
    parser = argparse.ArgumentParser(description="Train the four TurtleVision ensemble models.")
    parser.add_argument("--data", type=Path, required=True, help="folder holding patch_320/, patch_512/, patch_1024/")
    parser.add_argument("--out", type=Path, default=Path("runs"), help="where trained weights are saved")
    parser.add_argument("--weights", type=Path, help="start from the released weights in this folder, not ImageNet")
    args = parser.parse_args()
    if args.out.resolve() in (Path("weights").resolve(), (args.weights or Path("weights")).resolve()):
        parser.error("--out would overwrite the released weights; pick another folder")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")
    for size, backbone, color in MODELS:
        train_model(size, backbone, color, args)


if __name__ == "__main__":
    main()
