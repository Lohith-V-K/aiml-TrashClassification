"""
test_system.py
Comprehensive verification test for dual YOLOv8 Street Litter & Person Detection.
"""

import os
import json
import numpy as np
import cv2
from detector import LitterPersonDetector
from app import app

def run_tests():
    print("==================================================")
    print(" 1. Initializing Detector Engine")
    print("==================================================")
    detector = LitterPersonDetector()
    info = detector.get_device_info()
    print("Device info:", info)
    assert info["device"] in ["CUDA", "CPU"], f"Invalid device: {info['device']}"

    print("\n==================================================")
    print(" 2. Testing Sample Image Detection (Litter Only)")
    print("==================================================")
    test_img = "test_images/badalona-spain-july-overflowing-garbage-bin-badalona-spain-july-overflowing-trash-bin-city-street-illustrating-urban-465877532.webp"
    res, annot = detector.predict(test_img, conf_thresh=0.25)
    print("Decision:", res["decision"])
    print("Litter items:", res["summary"]["litter_count"])
    print("People detected:", res["summary"]["person_count"])
    assert res["summary"]["litter_count"] > 0, "Expected litter detections"
    assert res["decision"]["code"] == "LITTER_ONLY", f"Expected LITTER_ONLY, got {res['decision']['code']}"

    print("\n==================================================")
    print(" 3. Testing Potential Incident Decision Logic & Storage")
    print("==================================================")
    # Simulate a potential incident with synthetic detections to verify incident storage
    h, w = 480, 640
    test_frame = np.full((h, w, 3), 128, dtype=np.uint8)
    dummy_path = "test_synthetic.jpg"
    cv2.imwrite(dummy_path, test_frame)

    # Let's test the detector's internal storage method directly
    test_record = detector._store_incident(
        original_bgr=test_frame,
        annotated_bgr=test_frame,
        timestamp="2026-11-09 15:30:00",
        litter_categories=["Drink Can", "Cardboard"],
        person_count=1,
        detections_summary={
            "detections": {
                "litter": [{"label": "Drink Can"}, {"label": "Cardboard"}],
                "people": [{"label": "Person"}]
            }
        }
    )
    print("Stored incident record:", test_record)
    assert test_record["incident_status"] == "Pending Verification"
    assert os.path.exists(detector.incidents_dir / test_record["original_image_path"])
    assert os.path.exists(detector.incidents_dir / test_record["annotated_image_path"])

    incidents = detector.list_incidents()
    print("Total incidents in log:", len(incidents))
    assert len(incidents) >= 1
    assert incidents[0]["incident_id"] == test_record["incident_id"]
    print("Incident storage verified successfully!")

    print("\n==================================================")
    print(" 4. Testing Flask Endpoints (Test Client)")
    print("==================================================")
    client = app.test_client()

    # Index page
    r = client.get("/")
    assert r.status_code == 200
    assert b"Street Litter &amp; Person Detection" in r.data or b"Street Litter & Person Detection" in r.data
    print("GET / -> 200 OK")

    # System info
    r = client.get("/api/system_info")
    assert r.status_code == 200
    info_data = r.get_json()
    print("GET /api/system_info ->", info_data)
    assert "device" in info_data

    # Sample images
    r = client.get("/api/sample_images")
    assert r.status_code == 200
    samples_data = r.get_json()
    print("GET /api/sample_images ->", samples_data)
    assert len(samples_data["samples"]) > 0

    # Test detection via sample
    sample_file = samples_data["samples"][0]
    r = client.post("/api/detect", data={"sample_image": sample_file, "conf": 0.25})
    assert r.status_code == 200
    detect_res = r.get_json()
    assert detect_res["success"] is True
    print("POST /api/detect (sample) -> Success! Decision:", detect_res["data"]["decision"]["label"])

    # Test detection via uploaded file
    with open(dummy_path, "rb") as f:
        r = client.post("/api/detect", data={"image": (f, "test_upload.jpg"), "conf": 0.25})
    assert r.status_code == 200
    upload_res = r.get_json()
    assert upload_res["success"] is True
    print("POST /api/detect (upload) -> Success! Decision:", upload_res["data"]["decision"]["label"])

    # Test incident list endpoint
    r = client.get("/api/incidents")
    assert r.status_code == 200
    inc_list = r.get_json()["incidents"]
    print(f"GET /api/incidents -> {len(inc_list)} incidents found")

    # Test incident human review endpoint
    incident_id = test_record["incident_id"]
    r = client.post(
        f"/api/incidents/{incident_id}/review",
        json={"status": "Verified Incident", "notes": "Reviewed and confirmed by human inspector."}
    )
    assert r.status_code == 200
    review_res = r.get_json()
    print("POST /api/incidents/<id>/review ->", review_res)

    # Clean up dummy test file
    if os.path.exists(dummy_path):
        os.remove(dummy_path)

    print("\n==================================================")
    print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()

