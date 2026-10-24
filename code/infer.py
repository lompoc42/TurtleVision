"""
TurtleVision ensemble inference.

Run:   python infer.py --data path/to/images
       python infer.py --data path/to/images --weights runs/ --out results/ --upscale 8000

--data is a folder of images, or a folder of folders of images. Each folder is
processed on its own. For every image:
  1. upscale so the short side is at least 1024 px (always) and, with --upscale N,
     the long side is at least N px
  2. cut into patches at three scales (320 / 512 / 1024 px), resize each to 512
  3. run the patches through the matching model(s), decode oriented boxes
  4. fuse boxes across models (WBF); a box must be seen by >= MIN_AGREE models
  5. route the image by its best fused confidence into
        high_confidence/   >= 0.51
        medium_confidence/ >= 0.31
        low_confidence/    >= 0.10
     writing an annotated JPEG and an OBB label file. Anything lower is skipped.
Running a folder again resumes after the last image that already has an output.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path.home() / ".cache/torchinductor"))
Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


# =============================================================================
# SETTINGS
# =============================================================================

MODEL_FILES = [
    "turtlevision_swin_320_gray.pth",
    "turtlevision_b5_512_rgb.pth",
    "turtlevision_swin_512_rgb.pth",
    "turtlevision_b5_1024_rgb.pth",
]

#### DO NOT CHANGE THIS
MIN_AGREE = 2
#### DO NOT CHANGE THIS

# Per patch size score threshold; overridden by applied_thresholds.json in the
# weights folder if present, whose keys look like {"B320": 0.37, "B512": 0.27, "B1024": 0.22}.
DEFAULT_CONF = {320: 0.37, 512: 0.27, 1024: 0.22}

# Max boxes kept per patch before NMS, by patch size.
PRECAP = {320: 500, 512: 1500, 1024: 4000}

STRIDES = [4, 8, 16, 32, 64, 128]
BOX_COLOR = (180, 50, 160)  # BGR purple
IMG_EXT = (".jpg", ".jpeg", ".png")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CUDA = DEVICE == "cuda"
COMPILE = CUDA and sys.platform == "linux"  # torch.compile needs Triton, which Windows lacks  # changed
if CUDA:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
NORMALIZE = {"RGB": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), "GREY": ([0.449] * 3, [0.226] * 3)}  # same as train.py


def load_thresholds(weights):
    path = weights / "applied_thresholds.json"
    if not path.exists():
        return dict(DEFAULT_CONF)
    raw = json.loads(path.read_text())
    return {**DEFAULT_CONF, **{int(re.sub(r"\D", "", k)): float(v) for k, v in raw.items()}}


# =============================================================================
# MODEL
# =============================================================================

class FPN(nn.Module):
    def __init__(self, in_channels, out=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList(nn.Conv2d(c, out, 1) for c in in_channels)
        self.output_convs = nn.ModuleList(nn.Conv2d(out, out, 3, padding=1) for _ in in_channels)
        self.extra_convs = nn.ModuleList(nn.Conv2d(out, out, 3, stride=2, padding=1) for _ in range(2))

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

    def forward(self, features):
        outs = []
        for x, scale in zip(features, self.scales):
            qual = self.quality(self.cls_convs(x))
            box = self.box_pred(self.reg_convs(x))
            angle = box[:, 4:6]
            angle = angle / (torch.linalg.vector_norm(angle, dim=1, keepdim=True) + 1e-6)
            outs.append((qual, torch.cat([box[:, :4] * scale, angle], dim=1)))
        return outs


BACKBONES = {
    "swin_s": dict(
        nodes={"features.1": "s0", "features.3": "s1", "features.5": "s2", "features.7": "s3"},
        channels=[96, 192, 384, 768], permute=True),
    "efficientnet_b5": dict(
        nodes={"features.2": "s0", "features.3": "s1", "features.5": "s2", "features.7": "s3"},
        channels=[40, 64, 176, 512], permute=False),
}


class El_Modelo(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        cfg = BACKBONES[backbone]
        self.permute = cfg["permute"]
        self.backbone = create_feature_extractor(getattr(models, backbone)(weights=None), return_nodes=cfg["nodes"])
        self.fpn = FPN(cfg["channels"])
        self.head = FCOSHead()

    def forward(self, x):
        out = self.backbone(x)
        feats = [out[k] for k in ("s0", "s1", "s2", "s3")]
        if self.permute:
            feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        return self.head(self.fpn(feats))


def parse_name(path):
    m = re.fullmatch(r"turtlevision_(swin|b5)_(320|512|1024)_(rgb|gray)", path.stem)
    return dict(
        name=path.stem,
        basesize=int(m[2]),
        resize=512,
        backbone={"swin": "swin_s", "b5": "efficientnet_b5"}[m[1]],
        color={"rgb": "RGB", "gray": "GREY"}[m[3]],
    )


def load_model(path, backbone):
    model = El_Modelo(backbone)
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)  # checkpoint also holds an unused image head
    model = model.eval().to(DEVICE)
    if COMPILE:  # changed
        model = torch.compile(model.to(memory_format=torch.channels_last), mode="reduce-overhead")
    return model


# =============================================================================
# PATCHES IN
# =============================================================================

def to_model_color(bgr, color):
    if color == "GREY":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return np.stack([g, g, g], axis=-1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def patch_image(image, ps):
    """Non overlapping ps x ps patches; the last row/col is shifted back to fit."""
    h, w = image.shape[:2]
    ys = list(range(0, max(1, h - ps + 1), ps))
    xs = list(range(0, max(1, w - ps + 1), ps))
    if ys[-1] + ps < h:
        ys.append(h - ps)
    if xs[-1] + ps < w:
        xs.append(w - ps)
    positions = [(x, y) for y in ys for x in xs]
    patches = np.stack([image[y:y + ps, x:x + ps] for x, y in positions])
    return patches, positions


def preprocess(patches, resize, color):
    """(N, H, W, 3) uint8 -> (N, 3, resize, resize) normalized float on DEVICE."""
    mean, std = (torch.tensor(v, device=DEVICE).view(1, 3, 1, 1) for v in NORMALIZE[color])
    out = torch.empty((len(patches), 3, resize, resize), device=DEVICE)
    for i in range(0, len(patches), 128):
        x = torch.from_numpy(patches[i:i + 128]).to(DEVICE).permute(0, 3, 1, 2).float() / 255
        if x.shape[-1] != resize:
            x = F.interpolate(x, size=(resize, resize), mode="bilinear", align_corners=False)
        out[i:i + 128] = (x - mean) / std
    return out


# =============================================================================
# BOXES OUT
# A box is a 9-tuple: (x1,y1, x2,y2, x3,y3, x4,y4, conf) in image pixels.
# =============================================================================

def decode(outputs, conf_thr, resize):
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
        boxes = torch.stack([cx.clamp(0, resize), cy.clamp(0, resize),
                             w.clamp(1, resize), h.clamp(1, resize), ang], dim=1)
        bi, boxes, sc = bi.cpu().numpy(), boxes.cpu().numpy(), conf[bi, gy, gx].float().cpu().numpy()
        for i in np.unique(bi):
            parts[i].append((boxes[bi == i], sc[bi == i]))
    return [(np.concatenate([b for b, _ in p]), np.concatenate([s for _, s in p])) if p else (None, None)
            for p in parts]


def to_global(boxes5, scores, xy, basesize, resize):
    """Patch space (cx cy w h angle) -> image space corner boxes."""
    s = basesize / resize
    cx, cy = boxes5[:, 0] * s + xy[0], boxes5[:, 1] * s + xy[1]
    hw, hh = boxes5[:, 2] * s / 2, boxes5[:, 3] * s / 2
    cos, sin = np.cos(boxes5[:, 4]), np.sin(boxes5[:, 4])
    dx = hw[:, None] * np.array([-1, 1, 1, -1])
    dy = hh[:, None] * np.array([-1, -1, 1, 1])
    gx = cx[:, None] + dx * cos[:, None] - dy * sin[:, None]
    gy = cy[:, None] + dx * sin[:, None] + dy * cos[:, None]
    coords = np.stack([gx, gy], axis=2).reshape(len(scores), 8)
    return [(*c, float(sc)) for c, sc in zip(coords, scores)]


def quad(box):
    return np.array(box[:8], np.float32).reshape(4, 2)


def quad_iou(a, b):
    if (a.max(0) < b.min(0)).any() or (b.max(0) < a.min(0)).any():
        return 0.0
    inter = cv2.intersectConvexConvex(a, b)[0]
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0.0


def obb_nms(boxes, iou_thr):
    boxes = sorted(boxes, key=lambda b: b[8], reverse=True)
    quads = [quad(b) for b in boxes]
    keep = []
    for i, q in enumerate(quads):
        if all(quad_iou(quads[k], q) < iou_thr for k in keep):
            keep.append(i)
    return [boxes[i] for i in keep]


def wbf(tagged, n_models, iou_thr=0.21):
    """Weighted box fusion. tagged = boxes with a 10th element: model name.
    Boxes overlapping a cluster by IoU > iou_thr join it. Fused corners are the
    conf weighted mean; fused conf is the mean conf scaled by (boxes / n_models),
    capped at 1. Clusters seen by fewer than MIN_AGREE models are dropped."""
    clusters = []  # each: [fused quad, [quads], [confs], {models}]
    for b in sorted(tagged, key=lambda b: b[8], reverse=True):
        q = quad(b)
        ious = [quad_iou(c[0], q) for c in clusters]
        best = int(np.argmax(ious)) if ious else -1
        if best >= 0 and ious[best] > iou_thr:
            c = clusters[best]
            c[1].append(q)
            c[2].append(b[8])
            c[3].add(b[9])
            w = np.array(c[2])[:, None, None]
            c[0] = ((w * np.stack(c[1])).sum(0) / w.sum()).astype(np.float32)
        else:
            clusters.append([q, [q], [b[8]], {b[9]}])
    fused = []
    for q, _, confs, model_set in clusters:
        if len(model_set) >= MIN_AGREE:
            fused.append((*q.flatten(), np.mean(confs) * min(len(confs), n_models) / n_models))
    return fused


# =============================================================================
# INFERENCE
# =============================================================================

@torch.inference_mode()
def infer(model, x, positions, basesize, resize, conf_thr):
    """Run one model over all preprocessed patches of one image -> boxes."""
    boxes = []
    for i in range(0, len(x), 64):
        xb = x[i:i + 64]
        if CUDA:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(xb.to(memory_format=torch.channels_last))
        else:
            out = model(xb)
        for (b5, sc), xy in zip(decode(out, conf_thr, resize), positions[i:i + 64]):
            if b5 is None:
                continue
            dets = to_global(b5, sc, xy, basesize, resize)
            dets = sorted(dets, key=lambda b: b[8], reverse=True)[:PRECAP[basesize]]
            boxes += obb_nms(dets, 0.5)
    boxes = sorted(boxes, key=lambda b: b[8], reverse=True)[:2000]
    return obb_nms(boxes, 0.3)[:300]


def detect(bgr, groups, conf):
    """Run the whole ensemble on one image -> fused boxes."""
    tagged = []
    for (basesize, color, resize), members in groups.items():
        patches, positions = patch_image(to_model_color(bgr, color), basesize)
        x = preprocess(patches, resize, color)
        for m in members:
            for b in infer(m["model"], x, positions, basesize, resize, conf[basesize]):
                tagged.append((*b, m["name"]))
        del x
    return wbf(tagged, sum(len(v) for v in groups.values()))


# =============================================================================
# OUTPUT
# =============================================================================

def merge_touching(boxes):
    """Merge boxes whose quads overlap into one axis aligned box each (for drawing)."""
    groups = []
    for b in boxes:
        hit = [g for g in groups if any(quad_iou(quad(o), quad(b)) > 0 for o in g)]
        groups = [g for g in groups if g not in hit] + [sum(hit, [b])]
    merged = []
    for g in groups:
        pts = np.concatenate([quad(b) for b in g])
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        merged.append((x0, y0, x1, y0, x1, y1, x0, y1, max(b[8] for b in g)))
    return merged


def draw(bgr, boxes, route):
    vis = bgr.copy()
    for b in merge_touching(boxes):
        pts = np.round(quad(b)).astype(np.int32)
        cv2.polylines(vis, [pts], True, BOX_COLOR, 9)
        cv2.putText(vis, f"{b[8]:.2f}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, BOX_COLOR, 2)
    text = f"{route.upper()} CONFIDENCE | det={len(boxes)} | max_conf={max(b[8] for b in boxes):.3f}"
    scale = max(1.0, vis.shape[1] / 1200)
    thick = int(scale * 2)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    strip = np.full((th + base + 30, vis.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, text, ((vis.shape[1] - tw) // 2, th + 15), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick)
    return np.vstack([strip, vis])


def write_label(path, boxes, h, w):
    """One line per box: class x1 y1 ... x4 y4, normalized to [0, 1]."""
    norm = np.array([w, h] * 4)
    Path(path).write_text("\n".join("0 " + " ".join(f"{v:.6f}" for v in np.array(b[:8]) / norm) for b in boxes))


def read_ahead(paths, pool, window=16):
    """Yield images in order while a thread pool reads `window` images ahead."""
    futures = deque()
    for p in paths:
        futures.append(pool.submit(cv2.imread, str(p)))
        if len(futures) > window:
            yield futures.popleft().result()
    for f in futures:
        yield f.result()


def upscale(bgr, long_side=0):
    """Upscale so the short side is at least 1024 (the largest patch size; smaller
    breaks it) and the long side is at least long_side."""
    h, w = bgr.shape[:2]
    s = max(1024 / min(h, w), long_side / max(h, w))
    if s <= 1:
        return bgr
    return cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


# =============================================================================
# PER FOLDER RUN
# =============================================================================

def route_of(conf):
    return "high" if conf >= 0.51 else "medium" if conf >= 0.31 else "low" if conf >= 0.10 else "below_low"


def images_in(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def process_folder(img_dir, out, groups, conf, long_side):
    paths = images_in(img_dir)
    tiers = {t: out / f"{t}_confidence" for t in ("high", "medium", "low")}
    for d in tiers.values():
        (d / "labels").mkdir(parents=True, exist_ok=True)

    done = {f.stem for d in tiers.values() for f in d.glob("*.jpg")}
    start = max([i + 1 for i, p in enumerate(paths) if p.stem in done], default=0)
    total, paths = len(paths), paths[start:]
    if start:
        print(f"RESUME: {len(done)} outputs found, skipping first {start} images.")
    print(f"{img_dir}: {len(paths)} images -> {out}")

    csv_path = out / "routing_log.csv"
    new_csv = not csv_path.exists()
    log = open(csv_path, "a", newline="")
    writer = csv.DictWriter(log, fieldnames=["image", "route", "num_det", "max_conf"])
    if new_csv:
        writer.writeheader()

    counts = defaultdict(int)
    t0 = time.time()
    pool = ThreadPoolExecutor(4)
    for i, (path, bgr) in enumerate(zip(paths, read_ahead(paths, pool)), start + 1):
        t_img = time.time()
        if bgr is None:
            route, boxes = "bad_read", []
        else:
            bgr = upscale(bgr, long_side)
            boxes = detect(bgr, groups, conf)
            route = route_of(max(b[8] for b in boxes)) if boxes else "negative"
        max_conf = max(b[8] for b in boxes) if boxes else 0
        counts[route] += 1
        writer.writerow(dict(image=path.name, route=route, num_det=len(boxes), max_conf=round(max_conf, 4)))
        log.flush()
        if route in tiers:
            pool.submit(cv2.imwrite, str(tiers[route] / f"{path.stem}.jpg"), draw(bgr, boxes, route),
                        [cv2.IMWRITE_JPEG_QUALITY, 75])
            pool.submit(write_label, tiers[route] / "labels" / f"{path.stem}.txt", boxes, *bgr.shape[:2])
        print(f"[{i}/{total}] {path.stem} {route.upper()} dets={len(boxes)} conf={max_conf:.3f} {time.time() - t_img:.1f}s")
    pool.shutdown(wait=True)
    log.close()

    elapsed = time.time() - t0
    summary = "\n".join([
        "=" * 60,
        f"RUN COMPLETE  {img_dir.name}  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Images: {total} total, {start} skipped (resume), {len(paths)} processed",
        *[f"  {k:<10} {counts[k]}" for k in ("high", "medium", "low", "negative", "below_low", "bad_read")],
        f"  Time: {elapsed:.0f}s ({elapsed / max(len(paths), 1):.1f}s/img)",
        f"  Thresholds: high 0.51, med 0.31, low 0.10, per patch {conf}",
        f"  Outputs: {out}",
        "=" * 60,
    ])
    (out / "run_summary.txt").write_text(summary)
    print(summary)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Find sea turtle tracks with the TurtleVision ensemble.")
    parser.add_argument("--data", type=Path, required=True, help="folder of images, or folder of folders of images")
    parser.add_argument("--weights", type=Path, default=Path("weights"), help="folder holding the four .pth files")
    parser.add_argument("--out", type=Path, default=Path("results"), help="where results are written")
    parser.add_argument("--upscale", type=int, default=0,
                        help="also upscale so the long side is at least this many px, e.g. 8000 (found after the paper)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    folders = [d for d in [args.data, *sorted(args.data.iterdir())] if d.is_dir() and images_in(d)]
    conf = load_thresholds(args.weights)
    print(f"{len(folders)} folders in {args.data}  |  device: {DEVICE}  |  per patch thresholds: {conf}")

    # Load models; group models sharing (basesize, color, resize) so each
    # image is patched and preprocessed once per group.
    groups = defaultdict(list)
    for name in MODEL_FILES:
        m = parse_name(args.weights / name)
        t = time.time()
        m["model"] = load_model(args.weights / name, m["backbone"])
        print(f"Loaded {name} ({time.time() - t:.0f}s)")
        groups[(m["basesize"], m["color"], m["resize"])].append(m)

    if COMPILE:  # changed
        print("Warmup (torch.compile, several minutes per model)...")
        for (_, _, resize), members in groups.items():
            dummy = torch.randn(1, 3, resize, resize, device=DEVICE).to(memory_format=torch.channels_last)
            for m in members:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    m["model"](dummy)
        torch.cuda.synchronize()

    t0 = time.time()
    for i, folder in enumerate(folders, 1):
        print(f"\n### FOLDER {i}/{len(folders)}: {folder.name}")
        process_folder(folder, args.out / folder.name, groups, conf, args.upscale)
    print(f"\nALL DONE: {len(folders)} folders in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
