# Benchmark — BlazePose Lite/Full/Heavy on Ego-Exo4D val (exo views)

Diversity-filtered held-out 3D body-pose benchmark.  See
[results/RESULTS.md](results/RESULTS.md) for the headline numbers and full
audit.

## Protocol

This is **not the official Ego-Exo4D body-pose benchmark** (that one is
egocentric-only and BlazePose can't see the camera-wearer in their own
Aria feed). We define our own protocol on the same data:

- **Source**: Ego-Exo4D **val split**, body-pose annotated takes (218 in
  the release).
- **Inputs**: exo cameras only (`cam01..cam05` / `gp01..gp05`); Aria ego
  excluded.
- **Diversity filter** (see `select_frames.py`):
  - Stratified across (scenario × university), max 3 takes per combo.
  - Per take: ≤ 5 frames per cam, with ≥ 0.5 sec temporal stride and
    ≥ 2 cm root-relative pose displacement between consecutive picks.
  - Drop takes that can't yield ≥ 2 diverse frames.
- **Final benchmark size**: 24 takes / 510 frame-evaluations.
- **Pipeline**: end-to-end (MediaPipe Pose detector → landmarker, no GT bbox).
- **Metrics**: MPJPE (centred), **PA-MPJPE** (Procrustes-aligned, primary),
  PCK3D@50/150 mm, AUC over 0-200 mm, end-to-end detection rate.

## Persistent layout (~1.1 GB)

```
benchmark/
├── README.md                       this file
├── results/RESULTS.md              headline numbers + audit + per-take table
├── results/<variant>_summary.json  per-variant breakdown
├── results/comparison.csv          one row per variant
│
├── frames_manifest.json            {take_uid: {cam: [frame_idx, ...]}}
├── diversity_audit.json            scenario/university coverage + motion stats
├── subset.json                     original random 50-take pool (for reproducibility)
│
├── raw/annotations/                trimmed to the 24 manifest takes
│   ├── splits.json
│   └── ego_pose/val/{body,camera_pose}/<uid>.json   (×24 each)
│
├── frames/<take_uid>/<cam>/<frame>.jpg              the actual images BlazePose sees
├── predictions/<variant>/<take_uid>.json            cached model output
│
├── select_subset.py                step 1: random 50 from the 218 val takes
├── stream_extract.py               step 2: pull videos → JPEGs → delete videos
├── select_frames.py                step 3: diversity filter -> manifest
├── run_eval.py                     step 4: BlazePose × manifest -> predictions
├── compute_metrics.py              step 5: predictions × GT -> reports
└── lib/                            shared loaders / metrics / projection
```

## Reproducing the benchmark from scratch

```bash
# (one-time) annotations download — ~1.3 GB before trim, ~150 MB after
egoexo -o benchmark/raw --parts annotations --benchmarks egopose --splits val

# 1. Pick 50 random val body takes (seed=42)
python select_subset.py --n 50

# 2. Stream-download exo videos, extract GT-annotated frames, delete videos
#    Peak transient: ~300 MB per take.  Final: ~2 GB (before trim).
python stream_extract.py

# 3. Build the diversity-filtered manifest
python select_frames.py --max-per-combo 3 --frames-per-take 5 \
                        --min-stride 5 --min-pose-disp-m 0.02

# 4. Run BlazePose × 3 variants on manifest frames (~1 min total CPU)
python run_eval.py --variant lite
python run_eval.py --variant full
python run_eval.py --variant heavy

# 5. Score
python compute_metrics.py
```

## Scoring a new model variant

The manifest + cached frames let us evaluate any new model with two commands.
Future BlazePose v2 student:

```bash
python run_eval.py --variant student_v2     # extend MODEL_PATHS to include it
python compute_metrics.py --variants lite full heavy student_v2
```

PA-MPJPE delta vs Heavy is the headline metric for the v2 student.

## Files we explicitly preserved (don't delete)

- `frames_manifest.json` + `diversity_audit.json` — define the benchmark.
- `frames/` — pre-decoded JPEGs (re-running extract is expensive: ~100 min download).
- `raw/annotations/{body,camera_pose}/<uid>.json` for the 24 manifest takes.
- `predictions/<variant>/` — cached for reference; safe to wipe and rerun.

## Caveats baked into the protocol

1. **Bouldering is absent**: the random 50-take seed had zero bouldering takes;
   we did not re-extract to add them. Worth fixing before final ship-ready
   numbers — re-extract a few `*climb*` val takes and rerun `select_frames.py`.
2. **Heavy doesn't strictly beat Full in PA-MPJPE.** Each variant scores on
   its own detection set; Heavy's higher detection rate (81 % vs 78 %)
   includes harder frames that drag the average up. For a like-for-like
   number, a future enhancement to `compute_metrics.py` should support
   `--intersection-only` mode.
3. **Hip definition mismatch**: BlazePose hips are femoral-head (anatomical),
   Ego-Exo4D hips are surface annotations. ~2-4 cm systematic offset
   contaminates root-relative MPJPE on every joint. PA-MPJPE is the cleaner
   number, and even there the bias remains at the 1-2 mm level after
   Procrustes scale-fit.
4. **Joint visibility is implicit**: GT joints absent from `annotation3D` are
   treated as occluded/unannotated and masked out of MPJPE.
