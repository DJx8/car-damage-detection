import os

# Hugging Face ZeroGPU support (optional for local execution)
try:
    import spaces
except ImportError:

    class LocalSpaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

    spaces = LocalSpaces()

import json
import uuid
import tempfile
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEVICE = "cuda" if os.getenv("SPACES_ZERO_GPU") == "1" else "cpu"

VEHICLE_MODEL_PATH = os.path.join(MODEL_DIR, "vehicle_model.pt")
PARTS_MODEL_PATH = os.path.join(MODEL_DIR, "parts_model.pt")
DAMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "damage_model.pt")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

print("Loading vehicle detection model...")
vehicle_model = YOLO(VEHICLE_MODEL_PATH)

print("Loading vehicle parts segmentation model...")
parts_model = YOLO(PARTS_MODEL_PATH)

print("Loading damage segmentation model...")
damage_model = YOLO(DAMAGE_MODEL_PATH)

vehicle_model.to(DEVICE)
parts_model.to(DEVICE)
damage_model.to(DEVICE)

print(f"All three models loaded successfully on {DEVICE}!")


# --------------------------------------------------
# HELPER: NORMALIZE MASK SIZE
# --------------------------------------------------


def resize_mask(mask, target_shape):
    height, width = target_shape

    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    return mask.astype(bool)


# --------------------------------------------------
# STAGE 1: VEHICLE ISOLATION
# --------------------------------------------------


def isolate_vehicle(image, conf=0.25, device=DEVICE):

    image = np.array(image.convert("RGB")).copy()

    result = vehicle_model.predict(
        source=image,
        classes=[2],
        conf=conf,
        retina_masks=True,
        device=device,
        verbose=False,
    )[0]

    if result.masks is None or len(result.masks.data) == 0:
        raise ValueError("No vehicle detected. Please try another image.")

    masks = result.masks.data.cpu().numpy()

    areas = [np.sum(mask > 0.5) for mask in masks]
    best_idx = int(np.argmax(areas))

    vehicle_mask = resize_mask(masks[best_idx] > 0.5, image.shape[:2])

    isolated = image.copy()
    isolated[~vehicle_mask] = 255

    confidence = float(result.boxes.conf[best_idx].item())

    return isolated, vehicle_mask, confidence


# --------------------------------------------------
# STAGE 2 & 3: PART AND DAMAGE SEGMENTATION
# --------------------------------------------------


def run_segmentation(model, image, device=DEVICE, confidence=0.25):

    result = model.predict(
        source=image,
        conf=confidence,
        imgsz=640,
        retina_masks=True,
        device=device,
        verbose=False,
    )[0]

    instances = []

    if result.masks is not None:

        masks = result.masks.data.cpu().numpy()

        for i, mask in enumerate(masks):

            box = result.boxes[i]
            class_id = int(box.cls[0].item())

            instances.append(
                {
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": float(box.conf[0].item()),
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                    "mask": resize_mask(mask > 0.5, image.shape[:2]),
                }
            )

    return result, instances


# --------------------------------------------------
# DAMAGE-TO-PART ATTRIBUTION
# --------------------------------------------------


def attribute_damage(part_instances, damage_instances, vehicle_mask):

    attributed = []

    for damage in damage_instances:

        damage_mask = resize_mask(damage["mask"], vehicle_mask.shape)

        damage_pixels = int(damage_mask.sum())

        # Keep every detected damage, even if it falls
        # outside the detected vehicle region.
        vehicle_damage_mask = damage_mask & vehicle_mask
        vehicle_overlap = int(vehicle_damage_mask.sum())

        best_part = None
        best_overlap = 0
        best_part_pixels = 0

        for part in part_instances:

            part_mask = resize_mask(part["mask"], vehicle_mask.shape)

            overlap = int(np.logical_and(vehicle_damage_mask, part_mask).sum())

            if overlap > best_overlap:
                best_overlap = overlap
                best_part = part
                best_part_pixels = int(part_mask.sum())

        if (
            best_part is None
            or damage_pixels == 0
            or best_overlap / max(damage_pixels, 1) < 0.10
        ):
            part_name = "Unassigned"
            coverage = 0.0
        else:
            part_name = best_part["class_name"]
            coverage = (best_overlap / max(best_part_pixels, 1)) * 100

        damage_type = damage["class_name"]

        if damage_type in ["Shattered_Glass", "Broken_Lamps"]:
            severity = "Critical"
            recommendation = "Replace"
        elif coverage < 2:
            severity = "Light"
            recommendation = "Repair"
        elif coverage < 10:
            severity = "Medium"
            recommendation = "Repair"
        elif coverage <= 25:
            severity = "Heavy"
            recommendation = "Repair"
        else:
            severity = "Critical"
            recommendation = "Replace"

        attributed.append(
            {
                "damage_type": damage_type,
                "part": part_name,
                "confidence": round(damage["confidence"], 4),
                "overlap_pixels": best_overlap,
                "part_coverage_percent": round(coverage, 3),
                "severity": severity,
                "recommendation": recommendation,
            }
        )

    return attributed


# --------------------------------------------------
# COMPLETE INFERENCE PIPELINE
# --------------------------------------------------


def process_car_image(image, device=DEVICE):

    if image is None:
        raise gr.Error("Please upload a car image.")

    image = image.convert("RGB")

    isolated, vehicle_mask, vehicle_confidence = isolate_vehicle(image, device=device)

    _, parts = run_segmentation(parts_model, isolated, device=device)

    _, damages = run_segmentation(damage_model, isolated, device=device)

    attributions = attribute_damage(parts, damages, vehicle_mask)

    return {
        "original": np.array(image),
        "isolated": isolated,
        "parts": parts,
        "damages": damages,
        "attributions": attributions,
        "vehicle_confidence": vehicle_confidence,
    }


# --------------------------------------------------
# VISUALIZATION AND JSON OUTPUT
# --------------------------------------------------


@spaces.GPU(duration=60)
def run_app(image, device=DEVICE):

    result = process_car_image(image, device=device)

    # Part segmentation visualization
    part_visual = result["isolated"].copy()

    for part in result["parts"]:

        mask = part["mask"]

        color = np.array([0, 255, 0])

        part_visual[mask] = (0.55 * part_visual[mask] + 0.45 * color).astype(np.uint8)

        x1, y1, x2, y2 = map(int, part["bbox"])

        cv2.rectangle(part_visual, (x1, y1), (x2, y2), (0, 180, 0), 2)

        cv2.putText(
            part_visual,
            part["class_name"],
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 100, 0),
            2,
        )

    # Damage segmentation visualization
    damage_visual = result["isolated"].copy()

    for damage in result["damages"]:

        mask = damage["mask"]

        color = np.array([255, 0, 0])

        damage_visual[mask] = (0.5 * damage_visual[mask] + 0.5 * color).astype(np.uint8)

        x1, y1, x2, y2 = map(int, damage["bbox"])

        cv2.rectangle(damage_visual, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.putText(
            damage_visual,
            damage["class_name"],
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 0, 0),
            2,
        )

    # Attribution report
    report = []

    for i, item in enumerate(result["attributions"], start=1):

        report.append(
            f"{i}. {item['damage_type']} | "
            f"Part: {item['part']} | "
            f"Coverage: {item['part_coverage_percent']:.2f}% | "
            f"Severity: {item['severity']} | "
            f"Recommendation: {item['recommendation']}"
        )

    if not report:
        report = ["No damage detected."]

    report_text = (
        f"Vehicle confidence: "
        f"{result['vehicle_confidence']:.2%}\n\n" + "\n".join(report)
    )

    # Structured JSON output
    json_data = {
        "vehicle_confidence": result["vehicle_confidence"],
        "parts_detected": [
            {
                "class_name": p["class_name"],
                "confidence": p["confidence"],
                "bbox": p["bbox"],
            }
            for p in result["parts"]
        ],
        "damages_detected": result["attributions"],
    }

    output_path = os.path.join(
        tempfile.gettempdir(), f"car_damage_report_{uuid.uuid4().hex[:8]}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    return (
        result["original"],
        result["isolated"],
        part_visual,
        damage_visual,
        report_text,
        output_path,
    )


# --------------------------------------------------
# GRADIO INTERFACE
# --------------------------------------------------

with gr.Blocks(title="Car Damage Detection System") as demo:

    gr.Markdown("""
        # Automated Car Damage Detection and Attribution System

        Upload a vehicle image to isolate the vehicle, segment its
        parts, detect damage, and estimate damage severity.
        """)

    with gr.Row():

        input_image = gr.Image(label="Upload Vehicle Image", type="pil")

    analyze_button = gr.Button("Analyze Vehicle", variant="primary")

    with gr.Row():

        original_output = gr.Image(label="Original Image")
        isolated_output = gr.Image(label="Vehicle Isolation")

    with gr.Row():

        parts_output = gr.Image(label="Vehicle Part Segmentation")

        damage_output = gr.Image(label="Damage Segmentation")

    report_output = gr.Textbox(label="Damage Attribution and Severity", lines=10)

    json_output = gr.File(label="Download JSON Report")

    analyze_button.click(
        fn=run_app,
        inputs=input_image,
        outputs=[
            original_output,
            isolated_output,
            parts_output,
            damage_output,
            report_output,
            json_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()