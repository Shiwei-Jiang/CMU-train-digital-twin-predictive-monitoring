# Train Digital Twin Predictive Monitoring

Context-aware digital twin for train vibration prediction, operating-condition classification, and interpretable decision support under varying load, placement, material, and speed conditions.

This project was developed for Carnegie Mellon University's `12-831 Digital Twin and AI` final project. It combines synchronized Arduino motion telemetry and DAQ vibration sensing to learn an expected vibration baseline, measure context-aware residuals, and translate those signals into monitoring actions.

![Measured vs predicted vibration](report_figures/fig_expected_vs_measured.png)

## Overview

Traditional threshold alarms treat all vibration increases the same way. This project instead asks a more useful question:

> Is the measured vibration unusual given the train's current operating context?

The pipeline models expected vibration from context, compares measured and predicted behavior, classifies the operating condition, and maps the result to a human-readable action recommendation.

`Sensors -> Synchronization -> Feature Engineering -> Digital Twin Regression -> Residual Analysis -> AI Classification -> Decision Logic`

## Problem Setting

The train test bed was intentionally exercised under multiple operating regimes:

- 3 material configurations
- 12 operating conditions
- 108 runs total
- 1-second analysis windows
- 4,239 labeled windows after preprocessing

The system must distinguish normal variation caused by speed, load, and placement from behavior that remains unexplained after those factors are accounted for.

## Technical Approach

### 1. Multi-sensor alignment

- Arduino motion telemetry and DAQ vibration signals are aligned on a shared `25 Hz` timeline.
- Two accelerometer channels are fused into mean vibration and channel-difference features.

### 2. Window-level feature engineering

For each non-overlapping 1-second window, the pipeline extracts:

- Time-domain vibration features: `g_rms`, `g_ptp`, `g_mean_std`, `g_diff_mean`
- Frequency-domain features: `dom_freq`, `spec_centroid`
- Motion context features: velocity, acceleration, phase
- Encoded operating context: material, speed, placement, experiment group, estimated load

### 3. Context-aware digital twin

A `RandomForestRegressor` predicts expected RMS vibration from operating context only:

`expected_vibration = f(material, thickness, phase, speed, placement, load, velocity, acceleration, experiment_group)`

Residual outputs:

- `residual = measured - expected`
- `abs_residual = |residual|`

### 4. AI classification

A `RandomForestClassifier` uses vibration features, motion context, and digital twin residuals to classify:

- Binary state: `baseline` vs `loaded`
- Fine-grained state: 12 operating conditions

### 5. Decision support layer

Residual thresholds are calibrated from the empirical baseline distribution:

- Warning threshold: `0.01625`
- Critical threshold: `0.03024`

Action recommendations:

- `continue`
- `adjust_speed`
- `slow_down`
- `stop`

## Key Results

### Digital twin regression

- `R^2 = 0.9142`
- `MAE = 0.0048 g`
- `RMSE = 0.0069 g`
- Mean residual bias: `-0.0002 g`

### Binary classification

- Overall accuracy: `79.7%`
- Balanced accuracy: `55.2%`
- Matthews correlation coefficient: `0.077`

### Multi-class classification

- 12-class accuracy: `40.6%`
- Macro F1: `0.40`

### Decision logic

- `continue`: `96.9%`
- `adjust_speed`: `2.2%`
- `stop`: `0.8%`
- `slow_down`: `0.2%`

## Evaluation Visuals

| Digital Twin Fidelity | Binary Evaluation |
|---|---|
| ![Digital twin fidelity](report_figures/fig_expected_vs_measured.png) | ![Binary evaluation](report_figures/fig_eval_binary_confusion_and_condition_accuracy.png) |

| Multi-class Confusion | Decision Distribution |
|---|---|
| ![Multiclass confusion](report_figures/fig_eval_multiclass_confusion.png) | ![Decision distribution](report_figures/fig_eval_decision_distribution_and_residual_sorted.png) |

| Residual by Condition | Feature Importance |
|---|---|
| ![Residual by condition](report_figures/fig_residual_by_condition.png) | ![Feature importance](top10features_driving_the_AIclarification.JPG) |

## Repository Contents

```text
.
├── 12831digital_twin_final code.ipynb
├── 12831final_project_report.md
├── 12831final_project_report.pdf
├── 12831_final_presentation_updated.pptx
├── digital_twin_full_dashboard.html
├── run_metadata.csv
├── aligned_runs.csv
├── window_features.csv
├── digital_twin_predictions.csv
└── report_figures/
```

## Main Deliverables

- [Final notebook](./12831digital_twin_final%20code.ipynb)
- [Project report in Markdown](./12831final_project_report.md)
- [Project report PDF](./12831final_project_report.pdf)
- [Final presentation](./12831_final_presentation_updated.pptx)
- [Interactive dashboard HTML](./digital_twin_full_dashboard.html)

## Reproducibility

### Environment

- Python `3.11.5`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run order

1. Open `12831digital_twin_final code.ipynb`
2. Update the dataset path if needed
3. Run the notebook from top to bottom
4. Export figures and CSV artifacts

## Notes

- The repository focuses on processed artifacts and project deliverables.
- The raw train-run dataset is intentionally excluded from version control because of repository size and portability concerns.
- The current dataset contains controlled operating conditions rather than confirmed physical fault labels, so non-`continue` recommendations should be interpreted as operator-review signals rather than verified failures.
