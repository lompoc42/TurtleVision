## Training

Train all four ensemble models from ImageNet weights:

```bash
python train.py --data path/to/my_dataset
```

Or start from the released weights:

```bash
python train.py --data path/to/my_dataset --weights weights/
```

Trained models are saved to `runs/` (change it with `--out`) under the same file names as the released weights. To use them:

```bash
python infer.py --data path/to/images --weights runs/
```

| Start       | Epochs   | Stops early                             |
|-------------|----------|-----------------------------------------|
| ImageNet    | up to 42 | after 21 epochs without a wGeoAcc gain  |
| `--weights` | 13       | never                                   |

Each model keeps its epoch with the best validation wGeoAcc.

The paper's models started from DOTA pretraining, so training from ImageNet won't reproduce them exactly. The inference thresholds were tuned for the released weights, so retrained models may need different ones.

### Data

```
my_dataset/
├── patch_320/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── patch_512/     same as patch_320
└── patch_1024/    same as patch_320
```

`patch_320/` holds square 320 × 320 patches cut from your images, and so on. Images are `.jpg` or `.png`. Each image's label file has the same name, ending in `.txt`.

A label file has one line per track box:

```
0 x1 y1 x2 y2 x3 y3 x4 y4
```

`0` is the class (always 0). The rest are the box's four corners, divided by the patch width and height so they fall between 0 and 1.

Patches with no tracks matter as much as patches with them. Give them an empty label file, or none. The paper used about 5 patches without tracks for every patch with one; the script uses whatever is in the folders. Both `train` and `val` need both kinds, since validation scores TPR and TNR.

If a box crosses a patch edge, its label should cover only the part inside the patch. Moving the outside corners onto the edge is not enough.
