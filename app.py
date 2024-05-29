from ultralytics import YOLO
import cv2
import logging
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src_code.sort.sort import Sort
from src_code.util import get_car, read_license_plate

# Set device to CPU
device = 'cpu'

# Load models
try:
    yolo_model = YOLO('yolov8n.pt').to(device)
    license_plate_detector = YOLO('./model/best.pt').to(device)
except Exception as e:
    logging.error(f"Error loading YOLO models: {e}")
    raise

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount("/static", StaticFiles(directory="./static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("./static/index.html")

logging.basicConfig(level=logging.INFO)

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logging.error("Failed to decode image.")
            raise HTTPException(status_code=400, detail="Invalid image format.")
        logging.info("Image successfully read and decoded.")

        # Detect car
        car_detections = yolo_model.predict(image, conf=0.5, classes=[2])[0]
        detections_ = []
        for detection in car_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if class_id == 2:
                detections_.append([x1, y1, x2, y2, score])
        logging.info(f"Car detections: {detections_}")

        # Track car
        mot_tracker = Sort()
        track_ids = mot_tracker.update(np.asarray(detections_))
        logging.info(f"Track IDs: {track_ids}")

        # Detect license plate
        lisence_plates = license_plate_detector.predict(image, conf=0.5, classes=0)[0]
        results = {}
        show_results = {}
        for license_plate in lisence_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Process license plate
                license_plate_crop = image[int(y1):int(y2), int(x1):int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 65, 255, cv2.THRESH_BINARY_INV)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[car_id] = {
                        'car': {
                            'bbox': [xcar1, ycar1, xcar2, ycar2], 
                            },
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }
                    show_results[car_id] = {
                        'license_plate': {
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

        logging.info(f"Detection results: {show_results}")
        return show_results

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
