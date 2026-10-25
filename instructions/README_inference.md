## Inference

```bash
python code/infer.py --data path/to/images
```

`--data` accepts a folder of survey images (`.jpg`, `.jpeg`, or `.png`) or a folder containing multiple datasets. Each folder is processed independently. Images with a short side smaller than 1024 pixels are automatically upscaled to accommodate the largest patch size.

The four pretrained model weights are loaded from `weights/` by default. Download the weights from the GitHub release and place them in this folder without changing their filenames. Use `--weights` to specify a different location.

Results are saved to `results/` by default. Use `--out` to specify another location.

The optional `--upscale X` argument upscales images so their longest side is at least X pixels. This may improve detection on some datasets.

### Output

Each dataset receives its own results folder:

```text
results/<folder>/
├── high_confidence/      Confidence ≥ 0.51
├── medium_confidence/    Confidence ≥ 0.31
├── low_confidence/       Confidence ≥ 0.10
├── routing_log.csv       Detection results for every image
└── run_summary.txt       Processing summary
```

Each confidence folder contains annotated images and a `labels/` subfolder with corresponding `.txt` files.

Labels use normalized oriented bounding box coordinates:

```text
0 x1 y1 x2 y2 x3 y3 x4 y4
```

Coordinates are divided by the image width and height. Boxes extending beyond image boundaries may have coordinates outside the range [0, 1].

Images without qualifying detections appear only in `routing_log.csv`.

When rerunning a dataset, the script attempts to resume from the last image with a saved annotated result. Images without detections may be processed again.

### Detection Thresholds

Default confidence thresholds are:

- 320 px patches: `0.37`
- 512 px patches: `0.27`
- 1024 px patches: `0.22`

To override these values, place `applied_thresholds.json` in the weights directory, using the keys `B320`, `B512`, and `B1024`.

### Performance

On Linux systems with NVIDIA GPUs, model compilation introduces additional startup time. The first image may also take considerably longer than subsequent images.
