# Car Damage Detection and Attribution System

A computer vision project that detects visible vehicle damage, segments vehicle components, associates damage regions with the affected components, and estimates damage severity.

> **Project status:** Research prototype. Results are experimental and should not be treated as a professional vehicle inspection, repair estimate, or insurance assessment.

## Overview

The system uses a three-stage instance-segmentation pipeline:

1. **Vehicle isolation** — a COCO-pretrained YOLO11n-seg model identifies the car (COCO class ID 2) and isolates its region. This model was used as a pretrained model; it was not trained as part of this project.
2. **Vehicle-part segmentation** — a YOLO11n-seg model fine-tuned by me identifies ten exterior vehicle components.
3. **Damage segmentation** — a separate YOLO11n-seg model fine-tuned by me identifies five visible damage categories.

The part and damage models are trained separately because the two source datasets provide different annotation types. Their predicted masks are combined to associate a damage region with a vehicle part using mask overlap.

## Pipeline

```text
Input image
    |
    v
Vehicle detection and isolation
    |
    v
Vehicle-part segmentation  +  Damage segmentation
    |                              |
    +---------- mask overlap ------+
                   |
                   v
       Damage-to-part attribution
                   |
                   v
       Coverage and severity rules
                   |
                   v
     Visual results and JSON report
```

### Damage attribution

For each detected damage instance, the system compares its mask with the predicted part masks and assigns it to the part with the greatest overlap, subject to the implementation's overlap threshold. If no qualifying part is found, the damage is marked **Unassigned**.

### Severity estimation

Severity is estimated using the percentage of a part covered by the damage mask. The current prototype uses heuristic thresholds, with shattered glass and broken lamps treated as critical cases. These thresholds are project-specific rules, not validated automotive repair standards.

## Datasets and acknowledgements

This project uses two separately sourced datasets. **The datasets themselves are not included in this repository.** Please obtain them only from their official sources and follow their current terms.

### 1. Car Parts dataset — Humans in the Loop

- Official dataset page: https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/
- The publisher describes the dataset as dedicated to the public domain under **CC0 1.0**.
- This project used a selected car-parts subset and selected ten component classes for training.
- Dataset source and terms should be checked at the official page before reuse.

### 2. CarDD — Car Damage Detection Dataset

- Official project page: https://cardd-ustc.github.io/
- Paper: Xinkuang Wang, Wenjing Li, and Zhongcheng Wu, “CarDD: A New Dataset for Vision-Based Car Damage Detection,” *IEEE Transactions on Intelligent Transportation Systems*, 2023, 24(7), 7202–7214.
- DOI: https://doi.org/10.1109/TITS.2023.3258480
- Official licensing document: https://cardd-ustc.github.io/docs/CarDD_license.pdf

**Important CarDD notice:** The dataset's licensing document states that prior consent is required for research use and that distribution or broadcasting of all or part of the dataset to third parties requires prior authorization. This repository therefore does not host CarDD images, annotations, archives, or download mirrors. Public availability of this code does not grant access to CarDD or permission to redistribute it. Any use of trained weights derived from CarDD must be considered separately under the dataset terms and any written permission received from the dataset owner.

### Dataset citation

```bibtex
@article{CarDD,
  author  = {Wang, Xinkuang and Li, Wenjing and Wu, Zhongcheng},
  title   = {CarDD: A New Dataset for Vision-Based Car Damage Detection},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2023},
  volume  = {24},
  number  = {7},
  pages   = {7202--7214},
  doi     = {10.1109/TITS.2023.3258480}
}
```

## Model training and evaluation

The two task-specific models were trained using Google Colab with a Tesla T4 GPU. The parts model was trained for 35 epochs and the damage model for 45 epochs. Both use the YOLO11n-seg architecture.

### Held-out test results

| Metric | Vehicle-parts model | Damage model |
|---|---:|---:|
| Precision | 93.47% | 75.50% |
| Recall | 85.79% | 63.79% |
| mAP@50 | 90.48% | 70.57% |
| mAP@50–95 | 70.30% | 48.00% |
| F1 score | 89.46% | 69.18% |

Test set sizes were 101 images for the parts model and 374 images for the damage model. These metrics describe the individual segmentation models on their respective test splits; they are not an end-to-end system accuracy score.

The damage model's lower recall means some real damage may be missed. Performance can also be affected by lighting, reflections, occlusion, image quality, and damage that is small or difficult to distinguish.

## Technology

- Python
- Ultralytics YOLO11n-seg
- PyTorch
- OpenCV
- NumPy
- Gradio (for the interactive demo)

## Repository contents

The repository is being published in stages. Dataset files and trained weights are intentionally excluded while their applicable permissions are reviewed.

- `app.py` — interactive application (to be added)
- `requirements.txt` — Python dependencies
- `models/` — model-weight availability and licensing notes
- `docs/` — architecture and evaluation documentation
- `sample_images/` — reserved for images that are confirmed safe to publish

## Reproducibility and usage

The complete runnable application and model weights are not yet included in this repository. Setup instructions will be added after the public-release and licensing review is complete. Do not assume that cloning this repository alone provides access to the datasets or trained weights.

## Limitations

- This is an experimental prototype, not a production-grade inspection system.
- The damage model can miss visible damage, especially subtle cracks and small defects.
- Part attribution depends on the quality and overlap of predicted masks; some damage may remain unassigned.
- Severity levels are heuristic estimates and are not calibrated against repair costs or certified inspection criteria.
- The system has not been validated across all vehicle types, camera conditions, or real-world inspection settings.

## Ultralytics acknowledgement

This project uses the Ultralytics YOLO11 implementation and models. Refer to the official documentation and licensing information:

- Documentation: https://docs.ultralytics.com/models/yolo11/
- License: https://www.ultralytics.com/license
- Source: https://github.com/ultralytics/ultralytics

Ultralytics publishes AGPL-3.0 and Enterprise licensing options. Review the applicable terms for the software, pretrained weights, fine-tuned weights, and the complete application before choosing a repository licence or distributing artifacts. No separate project licence is asserted by this README.

## Author

**Dhananjay A V**

- GitHub: https://github.com/DJx8
- LinkedIn: https://www.linkedin.com/in/dhananjay-av/

---

If you use or discuss this work, please cite the original dataset and model sources and respect their licences.