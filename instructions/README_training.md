## Training

Train all four ensemble models from ImageNet weights:

```bash
python code/train.py --data path/to/my_dataset
```

Or start from the released weights:

```bash
python code/train.py --data path/to/my_dataset --weights weights/
```

Trained models are saved to `runs/` (change it with `--out`) under the same filenames as the released weights. To use them:

```bash
python code/infer.py --data path/to/images --weights runs/
```

| Start | Epochs | Early stopping |
|---|---|---|
| ImageNet | Up to 42 | After 21 epochs without a wGeoAcc improvement |
| Released weights (`--weights`) | 13 | Runs all 13 epochs |

Each model keeps the checkpoint with the best validation wGeoAcc.

The paper's models started from DOTA pretraining, so training from ImageNet will not reproduce them exactly. The inference thresholds were calibrated for the released weights, so retrained models may require different thresholds.

### Data

```text
my_dataset/
├── patch_320/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── patch_512/     same structure as patch_320
└── patch_1024/    same structure as patch_320
```

`patch_320/` contains square 320 × 320 image patches, and likewise for the other sizes. Images must be `.jpg` or `.png`. Each image's label file has the same filename, with a `.txt` extension.

Each label file contains one line per turtle track:

```text
0 x1 y1 x2 y2 x3 y3 x4 y4
```

`0` identifies the class. The remaining values represent the four corners of the bounding box, normalized by the patch width and height to fall between 0 and 1.

Patches without tracks are equally important. Their label files may be empty or absent. The paper used approximately five negative patches for every positive patch, although the script trains on whatever is provided.

Both `train` and `val` should contain positive and negative examples, since validation measures both TPR and TNR.

If a bounding box crosses a patch boundary, its label must describe only the portion inside the patch. Simply moving the outside corners onto the patch boundary is insufficient.
