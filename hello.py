from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="test_images",
    imgsz=640,
    conf=0.25,
    device=0,
    save=True
)

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"{model.names[cls]}: {conf:.2%}")

print("Detection completed!")