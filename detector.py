"""
streetlitter/detector.py
Core detection and incident evaluation engine for Street Litter & Person Detection.
"""

import os
import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Bounding box colors (BGR for OpenCV)
COLOR_PERSON = (0, 0, 255)   # Red for people
COLOR_LITTER = (0, 255, 0)   # Green for litter
TEXT_COLOR = (255, 255, 255) # White text

INCIDENT_STATUS_PENDING = "Pending Verification"

class LitterPersonDetector:
    def __init__(
        self,
        litter_model_path: str = "best.pt",
        person_model_path: str = "yolov8n.pt",
        incidents_dir: str = "incidents",
        preferred_device: Optional[str] = None
    ):
        self.litter_model_path = Path(litter_model_path)
        self.person_model_path = Path(person_model_path)
        self.incidents_dir = Path(incidents_dir)
        self.incidents_dir.mkdir(parents=True, exist_ok=True)
        self.incidents_json_path = self.incidents_dir / "incidents.json"

        # Determine device with CPU fallback
        self.device = self._resolve_device(preferred_device)
        print(f"[Detector] Initializing models on device: {self.device}")

        # Load litter detection model (reuse best.pt)
        if not self.litter_model_path.exists():
            raise FileNotFoundError(f"Litter model not found at: {self.litter_model_path}")
        self.litter_model = YOLO(str(self.litter_model_path))

        # Load person detection model (yolov8n.pt)
        # Ultralytics will auto-download if not present and internet is available
        self.person_model = YOLO(str(self.person_model_path))

        print(f"[Detector] Litter model loaded: {len(self.litter_model.names)} classes")
        print(f"[Detector] Person model loaded (targeting COCO class 0: person)")

    def _resolve_device(self, preferred: Optional[str] = None) -> Any:
        if preferred is not None:
            if preferred == "cpu":
                return "cpu"
            if torch.cuda.is_available():
                return preferred
            print(f"[Detector] Preferred device '{preferred}' unavailable. Falling back to CPU.")
            return "cpu"

        if torch.cuda.is_available():
            try:
                # Test basic CUDA allocation
                _ = torch.zeros(1, device="cuda:0")
                return 0
            except Exception as e:
                print(f"[Detector] CUDA initialization warning ({e}). Falling back to CPU.")
                return "cpu"
        return "cpu"

    def get_device_info(self) -> Dict[str, Any]:
        is_cuda = (self.device == 0 or (isinstance(self.device, str) and "cuda" in self.device))
        device_name = "CPU"
        vram_info = None
        if is_cuda and torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                vram_info = f"{total_mem:.1f} GB"
            except Exception:
                device_name = "CUDA Device"
        return {
            "device": "CUDA" if is_cuda else "CPU",
            "device_name": device_name,
            "vram": vram_info
        }

    def predict(
        self,
        image_path: str,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45
    ) -> Dict[str, Any]:
        """
        Runs both models on an input image, applies decision logic,
        and annotates the image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read original image via PIL (handles JPG, JPEG, PNG, WEBP, AVIF, and unicode paths on Windows)
        try:
            from PIL import Image, ImageOps
            pil_img = Image.open(image_path)
            # Correct orientation if EXIF tags exist
            pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.convert("RGB")
            # Convert to OpenCV BGR
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError(f"Could not load/decode image at {image_path}: {e}")

        orig_h, orig_w = img_bgr.shape[:2]

        # Run Litter Model (best.pt)
        litter_results = self.litter_model.predict(
            source=img_bgr,
            conf=conf_thresh,
            iou=iou_thresh,
            device=self.device,
            verbose=False
        )[0]

        # Run Person Model (yolov8n.pt, class 0 = person only)
        # Note: Do not introduce facial recognition or identify individuals
        person_results = self.person_model.predict(
            source=img_bgr,
            classes=[0],
            conf=conf_thresh,
            iou=iou_thresh,
            device=self.device,
            verbose=False
        )[0]

        # Parse litter detections
        litter_detections = []
        litter_categories = []
        for box in litter_results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.litter_model.names.get(cls_id, f"Litter-{cls_id}")
            conf = float(box.conf[0])
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            litter_detections.append({
                "class_id": cls_id,
                "label": cls_name,
                "confidence": round(conf, 4),
                "bbox": xyxy
            })
            litter_categories.append(cls_name)

        # Parse person detections
        person_detections = []
        for box in person_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            person_detections.append({
                "class_id": cls_id,
                "label": "Person",
                "confidence": round(conf, 4),
                "bbox": xyxy
            })

        # Decision logic
        has_litter = len(litter_detections) > 0
        has_person = len(person_detections) > 0

        if has_person and has_litter:
            decision_code = "POTENTIAL_INCIDENT"
            decision_label = "Potential Littering Incident"
            is_potential_incident = True
        elif has_litter and not has_person:
            decision_code = "LITTER_ONLY"
            decision_label = "Litter-Only Detection"
            is_potential_incident = False
        elif has_person and not has_litter:
            decision_code = "PERSON_ONLY"
            decision_label = "Person-Only Detection"
            is_potential_incident = False
        else:
            decision_code = "NO_DETECTIONS"
            decision_label = "No Detections"
            is_potential_incident = False

        # Create annotated image
        annotated_bgr = img_bgr.copy()

        # Draw people first (Red)
        for det in person_detections:
            self._draw_box(
                annotated_bgr,
                det["bbox"],
                f"Person {det['confidence']*100:.1f}%",
                COLOR_PERSON
            )

        # Draw litter on top (Green)
        for det in litter_detections:
            self._draw_box(
                annotated_bgr,
                det["bbox"],
                f"{det['label']} {det['confidence']*100:.1f}%",
                COLOR_LITTER
            )

        unique_litter_categories = sorted(list(set(litter_categories)))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result = {
            "timestamp": timestamp,
            "image_size": {"width": orig_w, "height": orig_h},
            "decision": {
                "code": decision_code,
                "label": decision_label,
                "is_potential_incident": is_potential_incident,
                "status": INCIDENT_STATUS_PENDING if is_potential_incident else "Informational"
            },
            "summary": {
                "person_count": len(person_detections),
                "litter_count": len(litter_detections),
                "litter_categories": unique_litter_categories
            },
            "detections": {
                "people": person_detections,
                "litter": litter_detections
            }
        }

        # If flagged as a potential incident, store it
        stored_incident_info = None
        if is_potential_incident:
            stored_incident_info = self._store_incident(
                original_bgr=img_bgr,
                annotated_bgr=annotated_bgr,
                timestamp=timestamp,
                litter_categories=unique_litter_categories,
                person_count=len(person_detections),
                detections_summary=result
            )
            result["stored_incident"] = stored_incident_info

        return result, annotated_bgr

    def _draw_box(
        self,
        img: np.ndarray,
        bbox: List[int],
        label: str,
        color: Tuple[int, int, int],
        thickness: int = 2
    ):
        x1, y1, x2, y2 = bbox
        # Draw bounding rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Text label configuration
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        text_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

        # Place label badge above bbox if room, otherwise inside
        label_y1 = max(0, y1 - text_h - baseline - 4)
        label_y2 = y1 if y1 - text_h - baseline - 4 >= 0 else y1 + text_h + baseline + 4
        label_x2 = min(img.shape[1], x1 + text_w + 6)

        # Background rectangle for crisp contrast
        cv2.rectangle(img, (x1, label_y1), (label_x2, label_y2), color, -1)
        # Put text
        text_pos = (x1 + 3, label_y2 - baseline - 2 if y1 - text_h - baseline - 4 >= 0 else label_y2 - 2)
        cv2.putText(img, label, text_pos, font, font_scale, TEXT_COLOR, text_thickness, cv2.LINE_AA)

    def _store_incident(
        self,
        original_bgr: np.ndarray,
        annotated_bgr: np.ndarray,
        timestamp: str,
        litter_categories: List[str],
        person_count: int,
        detections_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stores potential incidents:
        - Original image
        - Annotated image
        - Timestamp
        - Detected litter categories
        - Person detection status
        - Incident status: Pending Verification
        """
        incident_id = f"inc_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        incident_folder = self.incidents_dir / incident_id
        incident_folder.mkdir(parents=True, exist_ok=True)

        orig_filename = "original.jpg"
        annot_filename = "annotated.jpg"
        orig_path = incident_folder / orig_filename
        annot_path = incident_folder / annot_filename

        cv2.imwrite(str(orig_path), original_bgr)
        cv2.imwrite(str(annot_path), annotated_bgr)

        incident_record = {
            "incident_id": incident_id,
            "timestamp": timestamp,
            "incident_status": INCIDENT_STATUS_PENDING,
            "person_detection_status": f"Detected ({person_count} person{'s' if person_count > 1 else ''})",
            "person_count": person_count,
            "detected_litter_categories": litter_categories,
            "litter_count": len(detections_summary["detections"]["litter"]),
            "original_image_path": str(orig_path.relative_to(self.incidents_dir)).replace("\\", "/"),
            "annotated_image_path": str(annot_path.relative_to(self.incidents_dir)).replace("\\", "/"),
            "review_notes": "",
            "fine_issued": False # Explicitly not issuing fines
        }

        # Update JSON index
        self._append_incident_index(incident_record)
        return incident_record

    def _append_incident_index(self, record: Dict[str, Any]):
        incidents = self.list_incidents()
        # Prepend latest
        incidents.insert(0, record)
        with open(self.incidents_json_path, "w", encoding="utf-8") as f:
            json.dump(incidents, f, indent=2)

    def list_incidents(self) -> List[Dict[str, Any]]:
        if not self.incidents_json_path.exists():
            return []
        try:
            with open(self.incidents_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Street Litter & Person Detector")
    parser.add_argument("--image", type=str, default="test_images/badalona-spain-july-overflowing-garbage-bin-badalona-spain-july-overflowing-trash-bin-city-street-illustrating-urban-465877532.webp", help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Device to use (0, cpu)")
    parser.add_argument("--output", type=str, default="annotated_output.jpg", help="Path to save annotated image")
    args = parser.parse_args()

    detector = LitterPersonDetector(preferred_device=args.device)
    print("Device Info:", detector.get_device_info())
    res, annot = detector.predict(args.image, conf_thresh=args.conf)
    cv2.imwrite(args.output, annot)
    print(f"Decision: {res['decision']['label']} ({res['decision']['code']})")
    print(f"People detected: {res['summary']['person_count']}")
    print(f"Litter items detected: {res['summary']['litter_count']}")
    print(f"Litter categories: {res['summary']['litter_categories']}")
    print(f"Annotated output saved to: {args.output}")

