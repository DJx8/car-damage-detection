# Evaluation and Test Results

This document records the available held-out test results for the vehicle-part and damage segmentation models. It separates model-level segmentation metrics from the behaviour of the complete application.

## Evaluation scope

The project contains three models:

| Model | Purpose | Evaluation information |
|---|---|---|
| Vehicle model | Detects and isolates the car using the COCO car class | No separate held-out test evaluation is available |
| Vehicle-parts model | Segments 10 vehicle-part classes | Held-out test metrics available |
| Damage model | Segments 5 damage classes | Held-out test metrics available |

The parts and damage models were evaluated on their respective test splits. The notebook records 101 test images for the parts dataset and 374 test images for the damage dataset.

The reported precision, recall, mAP and F1 values are model-level results. They are not measurements of end-to-end application accuracy, damage-to-part attribution accuracy, or severity-estimation accuracy.

## Held-out test metrics

<small>Values are percentages. The notebook labels these as held-out test metrics from mask evaluation.</small>

| Metric | Vehicle-parts segmentation | Damage segmentation |
|---|---:|---:|
| Precision | 93.47% | 75.50% |
| Recall | 85.79% | 63.79% |
| mAP50 | 90.48% | 70.57% |
| mAP50-95 | 70.30% | 48.00% |
| F1-score | 89.46% | 69.18% |

### What the metrics indicate

- **Precision** measures how many predicted detections are correct.
- **Recall** measures how many annotated objects are found by the model.
- **mAP50** summarizes average precision at an intersection-over-union threshold of 0.50.
- **mAP50-95** averages performance over IoU thresholds from 0.50 to 0.95.
- **F1-score** is the harmonic mean of precision and recall.

The parts model has higher reported scores than the damage model across these metrics. The damage model's lower recall and mAP50-95 indicate that missed damage instances and mask-boundary accuracy remain areas to improve.

## Training information

- The vehicle model is a pretrained YOLO11n-seg model using the COCO car class; it was not trained by the project author.
- The vehicle-parts model was fine-tuned for 35 epochs on a selected subset of the Humans in the Loop car-parts dataset.
- The damage model was fine-tuned for 45 epochs using the CarDD dataset.
- The fine-tuning runs were performed on a Google Colab Tesla T4 GPU.

The available project notes do not provide a complete reproducibility record here (for example, all hyperparameters, random seeds, exact data versions, and environment details). Do not treat this short summary as a full training log.

## Qualitative test observations

The notebook records the following sample-level observations:

| Sample | Ground-truth / expected content | Recorded model behaviour |
|---|---|---|
| `000016.jpg` | Dents and scratches | Successful predictions recorded |
| `000013.jpg` | Cracks | The model missed the damage |
| `000103.jpg` | Shattered glass | Successful prediction recorded |
| `000034.jpg` | Broken lamps | Two detections recorded, with confidence scores 0.681 and 0.225 |
| Clean vehicle sample | Undamaged vehicle | No damage detections recorded |

These are selected examples, not a complete qualitative benchmark. In particular, `000013.jpg` must be treated as a missed crack detection, not a successful result. No per-class test metrics are reported in this document because the available evaluation notes do not provide them.

## Limitations and interpretation

1. **Vehicle isolation is not independently benchmarked here.** The pretrained vehicle model has no separate held-out test metrics in the available evaluation notes.
2. **No end-to-end metric is reported.** The model metrics do not measure the combined pipeline from raw image to final damage report.
3. **Attribution is rule-based.** Damage-to-part mapping uses mask overlap and a 10% threshold; its accuracy has not been separately measured against a labelled attribution test set.
4. **Severity is heuristic.** Severity labels and repair/replace recommendations are based on application rules, not a validated automotive inspection standard.
5. **Selected examples do not represent all conditions.** Performance may vary with lighting, camera angle, image quality, occlusion, vehicle type, and damage appearance.

This system should be treated as an experimental computer-vision prototype, not as a professional vehicle inspection, repair quotation, or insurance assessment.

## Dataset and model distribution note

The project uses datasets with separate licensing conditions. Review the relevant dataset terms and obtain any required permissions before redistributing dataset content or derived model weights. See [models/README.md](../models/README.md) for the current model provenance and distribution notes.
