# Model-selection illustrations

Regenerate all 14 illustrations from any working directory:

```bash
python docs/tools/generate_split_diagrams.py
```

The generator uses only the Python standard library. Output is deterministic SVG
under `docs/source/_static/splits/`. Every image uses the same 1200 x 480 canvas,
16 x 18 sample marks, font sizes and palette. Document images use 100% width;
short diagrams are centered on the common canvas rather than enlarged.

Visible text is limited to Subject, Trial, Fold, Samples and the Train/Test legend.
SVG titles provide the API name for accessibility without adding a visible title.
Purple marks with white dashes identify test samples; gray marks identify training
samples. Counts and split ratios are illustrative, not API defaults.

Check the implementation when adding diagrams: KFoldCrossTrial groups trial IDs
across the dataset, whereas train_test_split_cross_trial splits whole trials
within each subject. GroupbyTrial methods split samples within each trial, so
both partitions share trials. PerSubject methods illustrate one subject at a
time; experiments are performed separately for each selected subject.
