# Street Litter & Person Detection System (Dual YOLOv8)

An AI-assisted street litter and potential littering incident detection system integrating an existing trained YOLOv8 litter detection model (`best.pt`) with a pretrained YOLOv8 person detector (`yolov8n.pt`).

---

## Overview

This project extends an existing 18-class street litter detection model by pairing it with a person detection model to identify individuals and litter items simultaneously in individual uploaded photographs.

The system applies automated decision logic to flag **potential littering incidents** for human verification, automatically archives incidents with both original and annotated imagery, and presents results in a modern, clean web interface as well as via a command-line interface (CLI).

---

## Key Features

1. **Dual YOLOv8 Model Architecture**:
   - **Street Litter Detector (`best.pt`)**: Trained custom YOLOv8 model supporting 18 litter classes:
     - `Cardboard`, `Clear Plastic Bottle`, `Crisp Packet`, `Disposable Plastic Cup`, `Drink Can`, `Glass Bottle`, `Metal Bottle Cap`, `Paper Cup`, `Plastic Bottle Cap`, `Plastic Straw`, `Polystyrene`, and numerical category identifiers (`21`, `31`, `40`, `50`, `55`, `58`, `i`).
   - **Person Detector (`yolov8n.pt`)**: Pretrained YOLOv8 nano model targeting COCO Class 0 (`Person`).
2. **Color-Coded Visual Annotations**:
   - 🔴 **Red Bounding Boxes**: People detections (`Person` with confidence score).
   - 🟢 **Green Bounding Boxes**: Litter detections (Class name with confidence score).
   - High-contrast background badges ensure labels are legible across varied outdoor street lighting.
3. **Automated Incident Decision Logic**:
   - ⚠️ **Potential Littering Incident**: Both person(s) and litter item(s) are detected. Flagged and archived as `Pending Verification`.
   - 🌿 **Litter-Only Detection**: Street litter detected, but no people are present.
   - 👤 **Person-Only Detection**: Person(s) detected, but no street litter is present.
   - ⚪ **No Detections**: Neither people nor litter detected above the confidence threshold.
4. **Human Review & Verification Workflow**:
   - Automatically archives all potential incidents:
     - Original photograph (`original.jpg`)
     - Annotated photograph (`annotated.jpg`)
     - Timestamp
     - Detected litter categories
     - Person detection status
     - Status: **`Pending Verification`**
   - **Incident Log & Review Interface**: Reviewers can inspect logged incidents, add remarks, and confirm or dismiss flags.
   - **Strict Privacy & Safety Guardrails**:
     - **No facial recognition or identity tracking**: Object-level bounding boxes only.
     - **No automated fines**: Designed strictly as an incident-flagging and human-review support tool.
5. **Hardware Optimized**:
   - Optimized for NVIDIA GeForce RTX 3050 (4GB VRAM) via PyTorch CUDA with automatic, transparent fallback to CPU.
   - In-memory model caching prevents GPU memory fragmentation and redundant load times.

---

## Project Structure

```
streetlitter/
├── best.pt                       # Trained 18-class street litter detection model
├── yolov8n.pt                    # Pretrained YOLOv8 nano model (Person detection)
├── detector.py                   # Core dual-model detector & incident logic engine
├── app.py                        # Flask web application & REST API server
├── templates/
│   └── index.html                # Responsive web upload & incident review dashboard
├── static/
│   ├── css/style.css             # Modern styling, theme classes & badges
│   └── js/app.js                 # Drag-and-drop, API communication, comparison viewer
├── incidents/
│   ├── incidents.json            # Central audit log of potential incidents
│   └── inc_<timestamp>_<id>/     # Stored original and annotated images per incident
├── test_images/                  # Sample test photographs covering all 4 decision cases
├── test_system.py                # Automated system verification test suite
├── hello.py                      # Baseline script (preserved)
└── README.md                     # Project documentation
```

---

## Installation & Environment Setup

### 1. Prerequisites
- Windows 10/11
- Python virtual environment (already set up in `venv`)
- NVIDIA GPU with CUDA support (e.g. RTX 3050 4GB) or CPU fallback

### 2. Activate Virtual Environment
Open PowerShell in the project directory:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Web Interface Dependencies
Install `flask` into your virtual environment:

```powershell
pip install flask
```

*(Core dependencies — `ultralytics`, `torch`, `torchvision`, `opencv-python`, `pillow`, `numpy` — are already installed in your virtual environment).*

---

## Usage

### Option 1: Web Interface (Recommended)

1. Start the Flask application:
   ```powershell
   python app.py
   ```
2. Open your web browser and navigate to:
   👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

**Web UI Features**:
- **Drag-and-Drop / File Upload**: Supports `JPG`, `JPEG`, `PNG`, and `WEBP`.
- **Sample Image Picker**: Quick one-click testing using built-in samples in `test_images/`.
- **Confidence Slider**: Interactively tune the detection threshold (0.10 to 0.90, default 0.25).
- **Multi-View Modes**: Toggle between Annotated View, Original Photograph, or Side-by-Side comparison.
- **Incident Review Tab**: View stored potential incidents requiring verification and submit human review decisions (`Verified Incident` / `False Positive - Dismissed`) along with reviewer notes.

---

### Option 2: Command-Line Interface (CLI)

You can run detections directly on individual photographs without starting the web server:

```powershell
# Run with automatic GPU/CUDA detection (default conf: 0.25)
python detector.py --image "test_images/potential_incident_street_sample.jpg"

# Specify a custom confidence threshold and output destination
python detector.py --image "test_images/badalona-spain-july-overflowing-garbage-bin-badalona-spain-july-overflowing-trash-bin-city-street-illustrating-urban-465877532.webp" --conf 0.30 --output "annotated_result.jpg"

# Force CPU execution (useful for testing fallback mode)
python detector.py --image "test_images/street_pedestrians.jpg" --device cpu
```

---

## REST API Endpoints

The Flask application also exposes REST endpoints for headless integration:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/system_info` | Returns device info (CUDA/CPU, GPU name, VRAM, model statuses). |
| `GET` | `/api/sample_images` | Returns list of available sample images in `test_images/`. |
| `POST` | `/api/detect` | Multipart form upload (`image` or `sample_image`, optional `conf`). Returns detection JSON. |
| `GET` | `/api/incidents` | Returns all stored potential incidents logged in `incidents.json`. |
| `POST` | `/api/incidents/<id>/review` | Submits a human verification update (`status`, `notes`). |

---

## Decision Logic Matrix

| Person Detected? | Litter Detected? | Classification | Flag Potential Incident? | Stored in Incident Log? |
|---|---|---|---|---|
| **Yes** | **Yes** | **Potential Littering Incident** | **Yes** | **Yes** (`Pending Verification`) |
| **No** | **Yes** | **Litter-Only Detection** | No | No |
| **Yes** | **No** | **Person-Only Detection** | No | No |
| **No** | **No** | **No Detections** | No | No |

---

## Automated Verification Suite

To verify all model inferences, decision logic branches, incident storage, and Flask endpoints:

```powershell
python test_system.py
```
