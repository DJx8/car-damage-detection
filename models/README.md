# Model weights

Model weight files are intentionally not included at this stage.

| Model | Role | Provenance | Public distribution |
|---|---|---|---|
| Vehicle model | Vehicle isolation (COCO car class) | Pretrained YOLO11n-seg; not trained by the project author | Pending review of the applicable Ultralytics terms |
| Parts model | Ten-class vehicle-part segmentation | Fine-tuned by the project author on a selected Humans in the Loop car-parts subset | Not included pending a complete licensing review |
| Damage model | Five-class damage segmentation | Fine-tuned by the project author using CarDD | Not included; CarDD terms restrict redistribution without prior authorization |

CarDD's official licensing document requires prior authorization before distributing or broadcasting all or part of the dataset. Permission to access or train on a dataset should not be assumed to include permission to publish derived model weights.

Do not upload model files here unless the applicable dataset and model licences, plus any written permissions, have been checked and allow that distribution.

Official references:
- Humans in the Loop dataset: https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/
- CarDD project and licensing: https://cardd-ustc.github.io/
- CarDD licence: https://cardd-ustc.github.io/docs/CarDD_license.pdf
- Ultralytics licence: https://www.ultralytics.com/license