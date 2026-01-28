

import cv2
import numpy as np
from PIL import Image
import torch
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarDamageDetector:

    def __init__(self, model_path=None, confidence_threshold=0.5):

        self.confidence_threshold = confidence_threshold
        self.model_path = (
            model_path or
            r"C:\Users\athar\Car Damage  Detection\yolov8\models\best_model.pt"
        )

        # DEVICE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # LOAD YOLO MODEL
        self.model = YOLO(self.model_path)
        print("MODEL LOADED:", self.model_path)
        print("MODEL CLASSES:", self.model.names)

        self.class_names = self.model.names

        # SEVERITY LEVELS
        self.severity_mapping = {
            "crack": {"light": 4, "moderate": 12},
            "crash": {"light": 5, "moderate": 15},
            "dent": {"light": 3, "moderate": 10},
            "dislocated part": {"light": 2, "moderate": 6},
            "glass shatter": {"light": 3, "moderate": 8},
            "lamp broken": {"light": 2, "moderate": 6},
            "no part": {"light": 5, "moderate": 15},
            "rub": {"light": 3, "moderate": 10},
            "scratch": {"light": 5, "moderate": 12},
            "tire flat": {"light": 2, "moderate": 5},
        }

        # COST ESTIMATION
        self.cost_estimates = {
            "crack": {"light": 400, "moderate": 900, "severe": 1800},
            "crash": {"light": 50000, "moderate": 10000, "severe": 18000},
            "dent": {"light": 2000, "moderate": 5000, "severe": 10000},
            "dislocated part": {"light": 350, "moderate": 900, "severe": 2000},
            "glass shatter": {"light": 700, "moderate": 1500, "severe": 3000},
            "lamp broken": {"light": 300, "moderate": 700, "severe": 1500},
            "no part": {"light": 2000, "moderate": 7000, "severe": 15000},
            "rub": {"light": 150, "moderate": 400, "severe": 900},
            "scratch": {"light": 700, "moderate": 3500, "severe": 14000},
            "tire flat": {"light": 100, "moderate": 250, "severe": 600},
        }

    # ----------------------------------------------------
    # YOLO Detection
    # ----------------------------------------------------
    def detect_damage(self, image):

        try:
            # Convert PIL → RGB numpy (DO NOT CONVERT TO BGR)
            img_rgb = np.array(image)
            h, w = img_rgb.shape[:2]

            # BEST YOLO CALL
            results = self.model.predict(
                img_rgb,
                conf=self.confidence_threshold,
                imgsz=640,
                verbose=False
            )

            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                print("🔥 YOLO returned NO detections.")
                return {"damages": [], "total_damages": 0}

            detections = []

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls.cpu().numpy())
                conf = float(box.conf.cpu().numpy())

                damage_type = self.class_names.get(cls, "unknown")

                area_pct = ((x2 - x1) * (y2 - y1)) / (h * w) * 100

                severity = self._classify_severity(damage_type, area_pct)
                cost = self._estimate_cost(damage_type, severity)

                detections.append({
                    "type": damage_type,
                    "severity": severity.capitalize(),
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                    "area_percentage": round(area_pct, 2),
                    "estimated_cost": cost,
                    "location": self._describe_location([x1, y1, x2, y2], (h, w)),
                })

            return {
                "damages": detections,
                "total_damages": len(detections),
                "image_shape": (h, w)
            }

        except Exception as e:
            print("🔥🔥 ERROR IN DETECT_DAMAGE:", e)
            raise

    # ----------------------------------------------------
    def _classify_severity(self, damage_type, area_pct):
        if damage_type not in self.severity_mapping:
            return "moderate"

        light = self.severity_mapping[damage_type]["light"]
        moderate = self.severity_mapping[damage_type]["moderate"]

        if area_pct <= light:
            return "light"
        elif area_pct <= moderate:
            return "moderate"
        return "severe"

    # ----------------------------------------------------
    def _estimate_cost(self, damage_type, severity):
        if damage_type not in self.cost_estimates:
            return 0
        return self.cost_estimates[damage_type][severity]

    # ----------------------------------------------------
    def _describe_location(self, bbox, shape):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h, w = shape

        horiz = "left" if cx < w/3 else "right" if cx > 2*w/3 else "center"
        vert = "upper" if cy < h/3 else "lower" if cy > 2*h/3 else "middle"

        return f"{vert} {horiz}"

    # ----------------------------------------------------
    def annotate_image(self, image, detections):

        img = np.array(image).copy()

        COLORS = {
            "crack": (255, 0, 0),
            "crash": (255, 50, 50),
            "dent": (255, 165, 0),
            "dislocated part": (0, 0, 255),
            "glass shatter": (0, 255, 255),
            "lamp broken": (255, 0, 255),
            "no part": (180, 180, 180),
            "rub": (0, 255, 0),
            "scratch": (0, 150, 255),
            "tire flat": (128, 0, 128),
        }

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = f"{d['type'].title()} ({d['confidence']})"
            color = COLORS.get(d["type"], (255, 255, 0))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img


if __name__ == "__main__":
    det = CarDamageDetector()
    print("Detector Ready")
