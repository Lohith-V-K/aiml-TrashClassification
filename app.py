"""
streetlitter/app.py
Flask Web Interface for YOLOv8 Street Litter & Person Detection.
"""

import os
import uuid
import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from detector import LitterPersonDetector

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "avif"}

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
INCIDENTS_DIR = BASE_DIR / "incidents"
TEST_IMAGES_DIR = BASE_DIR / "test_images"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024 # 32 MB max upload

# Initialize detector once (reuses GPU memory, avoids reloading)
detector = LitterPersonDetector(
    litter_model_path=str(BASE_DIR / "best.pt"),
    person_model_path=str(BASE_DIR / "yolov8n.pt"),
    incidents_dir=str(INCIDENTS_DIR)
)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/system_info", methods=["GET"])
def system_info():
    info = detector.get_device_info()
    return jsonify({
        "status": "online",
        "device": info["device"],
        "device_name": info["device_name"],
        "vram": info["vram"],
        "litter_model": "best.pt (18 classes)",
        "person_model": "yolov8n.pt (COCO Person)",
        "incidents_count": len(detector.list_incidents())
    })

@app.route("/api/sample_images", methods=["GET"])
def list_sample_images():
    samples = []
    if TEST_IMAGES_DIR.exists():
        for f in TEST_IMAGES_DIR.iterdir():
            if f.is_file() and f.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS:
                samples.append(f.name)
    return jsonify({"samples": samples})

@app.route("/api/detect", methods=["POST"])
def detect_image():
    # Check if a file was uploaded or a sample image was selected
    file = request.files.get("image")
    sample_name = request.form.get("sample_image")
    conf = float(request.form.get("conf", 0.25))

    upload_id = f"up_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    dest_dir = UPLOAD_DIR / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    input_path = None
    original_filename = ""

    if file and file.filename:
        if not allowed_file(file.filename):
            return jsonify({"error": f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400
        original_filename = secure_filename(file.filename)
        input_path = dest_dir / original_filename
        file.save(str(input_path))
    elif sample_name:
        sample_path = TEST_IMAGES_DIR / secure_filename(sample_name)
        if not sample_path.exists():
            return jsonify({"error": f"Sample image '{sample_name}' not found."}), 404
        original_filename = sample_name
        input_path = dest_dir / original_filename
        # Copy to upload directory
        with open(sample_path, "rb") as src, open(input_path, "wb") as dst:
            dst.write(src.read())
    else:
        return jsonify({"error": "No image file uploaded or sample selected."}), 400

    try:
        # Run dual detection
        result, annotated_bgr = detector.predict(str(input_path), conf_thresh=conf)

        # Save annotated image
        annotated_path = dest_dir / "annotated.jpg"
        import cv2
        cv2.imwrite(str(annotated_path), annotated_bgr)

        # Build accessible URLs
        result["original_image_url"] = f"/uploads/{upload_id}/{original_filename}"
        result["annotated_image_url"] = f"/uploads/{upload_id}/annotated.jpg"

        return jsonify({
            "success": True,
            "data": result
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/incidents", methods=["GET"])
def get_incidents():
    incidents = detector.list_incidents()
    return jsonify({"incidents": incidents})

@app.route("/api/incidents/<incident_id>/review", methods=["POST"])
def update_incident_review(incident_id):
    """
    Human verification review action: update status and notes.
    Strictly human verification - no automated fines.
    """
    data = request.get_json() or {}
    new_status = data.get("status", "Pending Verification")
    notes = data.get("notes", "")

    incidents = detector.list_incidents()
    updated = False
    for inc in incidents:
        if inc.get("incident_id") == incident_id:
            inc["incident_status"] = new_status
            inc["review_notes"] = notes
            inc["reviewed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            updated = True
            break

    if updated:
        import json
        with open(detector.incidents_json_path, "w", encoding="utf-8") as f:
            json.dump(incidents, f, indent=2)
        return jsonify({"success": True, "message": "Incident review updated."})
    return jsonify({"error": "Incident not found."}), 404

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)

@app.route("/incidents/<path:filename>")
def serve_incident_file(filename):
    return send_from_directory(str(INCIDENTS_DIR), filename)

@app.route("/test_images/<path:filename>")
def serve_test_image(filename):
    return send_from_directory(str(TEST_IMAGES_DIR), filename)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Street Litter Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    parser.add_argument("--port", type=int, default=5000, help="Port")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f" Street Litter & Person Detection System")
    print(f" Web UI available at: http://{args.host}:{args.port}")
    print(f"=======================================================\n")
    app.run(host=args.host, port=args.port, debug=False)

