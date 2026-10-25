# TurtleVision
TurtleVision: A Deployed Multiscale Ensemble for Sea Turtle Track Detection in Operational Aerial Surveys

## Quick Start

Follow these steps to download TurtleVision, install its dependencies, and run inference or training.

### 1. Download TurtleVision

Open a terminal and run:

```bash
git clone https://github.com/lompoc42/TurtleVision.git
cd TurtleVision
```

### 2. Create a Python environment

Create a separate environment for TurtleVision:

```bash
conda create -n turtlevision python=3.11 -y
conda activate turtlevision
```

Install the required packages:

```bash
pip install -r requirements.txt
```

**Note:** A compatible PyTorch installation and GPU configuration are required for GPU training and inference. See the installation requirements for your system.

### 3. Download the pretrained models

Download all four `.pth` files from the [TurtleVision Releases](https://github.com/lompoc42/TurtleVision/releases) page.

Create a folder named `weights` in the repository and place the downloaded files inside it:

```text
TurtleVision/
└── weights/
    ├── turtlevision_b5_1024_rgb.pth
    ├── turtlevision_b5_512_rgb.pth
    ├── turtlevision_swin_320_gray.pth
    └── turtlevision_swin_512_rgb.pth
```

If using a custom threshold configuration, place `applied_thresholds.json` in the same folder. Otherwise, the default thresholds are used automatically.

### 4. Run inference

Create a folder containing the images you want to analyze:

```bash
mkdir -p images
```

Copy your `.jpg`, `.jpeg`, or `.png` images into `images/`.

Run:

```bash
python code/infer.py --data images --weights weights --out results
```

Results will be saved in `results/images/`, organized by detection confidence.

### 5. Prepare training data

Training requires labeled image patches organized into three sizes: 320, 512, and 1024 pixels.

For each size, use the following directory structure:

```text
dataset/
└── patch_320/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

Repeat this structure for `patch_512/` and `patch_1024/`.

Each label file must have the same filename as its corresponding image, with a `.txt` extension. Labels use normalized oriented bounding box coordinates:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

Images without turtle tracks can have empty label files.

### 6. Train TurtleVision

To train the four models from ImageNet pretrained backbones:

```bash
python code/train.py --data dataset --out runs
```

To continue training from the released TurtleVision weights:

```bash
python code/train.py --data dataset --weights weights --out runs
```

Newly trained models will be saved in `runs/`. The original pretrained weights remain unchanged.
