# ldp-har
Geometry Aware LDP with Temporal Unlinkability for HAR

## src/
```
    __init__.py        # optional, can be empty

    data.py            # Step 1: dataset loading and basic preprocessing
    models.py          # Step 1: all neural and classical models
    train.py           # Step 1: generic training and evaluation loops
    eval.py            # Step 1: metrics and plotting-friendly evaluation

    features.py        # Step 2: feature extraction from raw sequences
    clip.py            # Step 3: geometry-aware clipping
    ldp.py             # Step 4: Laplace LDP mechanisms

    temporal.py        # Step 5: bout detection + shuffling + pseudonym rotation + timestamp perturbation
    unlinkability.py   # Step 6: attack dataset construction and AUC computation

    pipeline.py        # Steps 7-8: combined pipeline / ablations
    overhead.py        # Step 9: overhead measurement -- profiling, timing, cost
    utils.py           # shared helpers -- seed setting, device selection, logging, path helpers
```
