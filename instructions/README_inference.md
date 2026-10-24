## Inference

```bash
python infer.py --data path/to/images
```

`--data` is a folder of survey images (`.jpg`, `.jpeg` or `.png`), or a folder of such folders. Each folder is processed on its own. Any image whose short side is under 1024 px is upscaled so its short side is 1024, because the largest patches are 1024 px and smaller images would break them.

The four released model files are read from `weights/` (change it with `--weights`); download them there without renaming them, since `infer.py` loads them by their exact names. Results are written to `results/` (change it with `--out`).

`--upscale 8000` also upscales every image so its long side is at least 8000 px. This helped on small images in work after the paper, and is off by default.

Each folder gets its own results:

```
results/<folder>/
├── high_confidence/      best fused confidence of at least 0.51
├── medium_confidence/    at least 0.31
├── low_confidence/       at least 0.10
├── routing_log.csv       every image: route, number of detections, best confidence
└── run_summary.txt
```

Each confidence folder holds annotated copies of its images, plus a `labels/` folder with one `.txt` per image. A label line is `0 x1 y1 x2 y2 x3 y3 x4 y4`: the four corners of a track box, divided by the image width and height so they fall between 0 and 1, except boxes at the image edge, which can extend past it. Images with nothing found appear only in `routing_log.csv`.

Running a folder again picks up after the last image that already has a result.

If the weights folder holds an `applied_thresholds.json` (keys `B320`, `B512`, `B1024`), its per patch thresholds replace the defaults of 0.37, 0.27 and 0.22.

On Linux with an NVIDIA GPU, the first run spends a few minutes per model compiling before it starts.
