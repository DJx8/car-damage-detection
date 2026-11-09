# Automated Car Damage Detection and Attribution System

A computer-vision research prototype that isolates a vehicle, segments its exterior parts, detects visible damage, links damage regions to vehicle parts, and estimates severity using rule-based heuristics.

<details>
<summary><strong>Project status and important notice</strong></summary>

This is an experimental prototype, not a professional vehicle inspection, repair estimate, or insurance assessment. Severity labels are heuristic outputs and are not validated automotive repair standards.

</details>

## Live demo

Try the hosted Gradio application on Hugging Face Spaces:

**[Open the Car Damage Detection demo](https://huggingface.co/spaces/Deejay888/car-damage-detection)**

The GitHub repository contains the application source and project documentation. The trained model-weight files and source datasets are not included in this repository.

## How it works

The application uses three YOLO11n-seg models in a staged pipeline:

1. **Vehicle isolation:** a COCO-pretrained YOLO11n-seg model detects the car (COCO class ID 2) and isolates the largest detected vehicle. This model was used pretrained; it was not trained as part of this project.
2. **Vehicle-part segmentation:** a separate YOLO11n-seg model, fine-tuned by me, predicts ten exterior vehicle-part classes.
3. **Damage segmentation:** another YOLO11n-seg model, fine-tuned by me, predicts five visible damage classes.
4. **Damage-to-part attribution:** the application compares the damage and part masks. It assigns a damage instance to the part with the greatest overlap when that overlap covers at least 10% of the full damage mask. Otherwise, it marks the damage as **Unassigned**.
5. **Severity estimation:** a rule-based heuristic estimates severity from the percentage of the assigned part covered by the damage mask. Shattered glass and broken lamps are handled as critical cases.

The part and damage models are trained separately because their source datasets use different annotation types and class definitions.

## Pipeline

```text
Input vehicle image
        |
        v
Vehicle detection and isolation
        |
        +----------------------+
        |                      |
        v                      v
Vehicle-part segmentation   Damage segmentation
        |                      |
        +----------+-----------+
                   |
                   v
       Damage-to-part attribution
         (mask-overlap rule)
                   |
                   v
       Coverage-based severity
             estimation
                   |
                   v
       Visual outputs + JSON report
```

## Models and training

| Model | Purpose | Training / provenance |
|---|---|---|
| Vehicle model | Detect and isolate the car | COCO-pretrained YOLO11n-seg; not trained by me |
| Parts model | Segment ten exterior vehicle parts | YOLO11n-seg fine-tuned by me |
| Damage model | Segment five visible damage categories | YOLO11n-seg fine-tuned by me |

The two task-specific models were trained in Google Colab using a Tesla T4 GPU:

- Parts model: 35 epochs.
- Damage model: 45 epochs.

### Held-out test results

| Metric | Vehicle-parts model | Damage model |
|---|---:|---:|
| Precision | 93.47% | 75.50% |
| Recall | 85.79% | 63.79% |
| mAP@50 | 90.48% | 70.57% |
| mAP@50–95 | 70.30% | 48.00% |
| F1 score | 89.46% | 69.18% |

The test splits contained 101 images for the parts model and 374 images for the damage model. These are model-level results on their respective test sets, **not an end-to-end system accuracy score**. The damage model's lower recall means that some real damage may be missed.

## Datasets and acknowledgements

The datasets are not included in this repository. Obtain them only from their official sources and follow their current terms.

### Humans in the Loop — Car Parts dataset

- Official dataset page: https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/
- The publisher describes the dataset as dedicated to the public domain under CC0 1.0.
- I used a selected subset and trained the ten vehicle-part classes used by this project.
- Check the official page for the current dataset details and terms.

### CarDD — Car Damage Detection Dataset

- Official project page: https://cardd-ustc.github.io/
- Paper: Xinkuang Wang, Wenjing Li, and Zhongcheng Wu, “CarDD: A New Dataset for Vision-Based Car Damage Detection,” *IEEE Transactions on Intelligent Transportation Systems*, 2023, 24(7), 7202–7214.
- DOI: https://doi.org/10.1109/TITS.2023.3258480
- Official licensing document: https://cardd-ustc.github.io/docs/CarDD_license.pdf

**CarDD licensing notice:** Its licensing document states that prior consent is required for research use and that distribution or broadcasting of all or part of the dataset to third parties requires prior authorization. This repository does not host CarDD images, annotations, archives, or download mirrors. Permission to access or use a dataset for training should not be assumed to include permission to publish derived model weights. The weights trained using CarDD are therefore not included here pending confirmation of the applicable permissions.

### CarDD citation

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

## Technology stack

- Python
- Ultralytics YOLO11n-seg
- PyTorch
- OpenCV
- NumPy
- Pillow
- Gradio
- Hugging Face Spaces

## Repository structure

```text
car-damage-detection/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── README.md
├── docs/
│   └── README.md
└── sample_images/
    └── README.md
```

## Run locally

The application source is available here, but the three trained model-weight files are not distributed in this repository. A local run requires you to obtain the model files lawfully and place them at the following paths:

```text
models/
├── vehicle_model.pt
├── parts_model.pt
└── damage_model.pt
```

Then install the dependencies and start the Gradio app:

```bash
pip install -r requirements.txt
python app.py
```

Without the model-weight files, the application cannot perform inference. The hosted Hugging Face demo is the easiest way to try the project.

## Limitations

- This is a research prototype, not a production-grade vehicle inspection system.
- The damage model may miss subtle, small, occluded, or visually ambiguous damage.
- Attribution depends on the quality and overlap of predicted masks; some detections may remain unassigned.
- If a damage region spans multiple vehicle parts, the current logic assigns it to only the single part with the greatest qualifying overlap.
- Severity is a fixed heuristic based on mask coverage, not a learned repair-cost model or certified assessment.
- The system has not been validated across all vehicle types, camera conditions, or real-world inspection settings.
- Model-level test metrics do not represent the accuracy of the complete end-to-end pipeline.

## Ultralytics acknowledgement and licensing

This project uses the Ultralytics YOLO11 implementation and models:

- Documentation: https://docs.ultralytics.com/models/yolo11/
- Licensing: https://www.ultralytics.com/license
- Source code: https://github.com/ultralytics/ultralytics

Ultralytics publishes AGPL-3.0 and Enterprise licensing options. Review the applicable terms for the software, pretrained weights, fine-tuned weights, and the complete application before redistributing artifacts. No separate project licence is asserted by this README.

## Author

**Dhananjay A V**

- GitHub: https://github.com/DJx8
- LinkedIn: https://www.linkedin.com/in/dhananjay-av/

---

If you use or discuss this work, please cite the original dataset and model sources and respect their licences.
