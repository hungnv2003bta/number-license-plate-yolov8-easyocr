from ultralytics import YOLO
import cv2
import PIL

from sort.sort import *
from util import get_car, read_license_plate, write_csv

def main():

    results = {}

    mot_tracker = Sort()

    # Load model
    license_plate_detector = YOLO('/Users/hungnguyen/TaiLieu/NhanDang/model/best.pt')
    yolo_model = YOLO('yolov8n.pt')
    # Load image
    image = cv2.imread('/Users/hungnguyen/TaiLieu/NhanDang/model/bienngang.jpg')

    # Detect car , motorbike, bus, truck: [2, 3, 5, 7]
    car_detections = yolo_model.predict(image,conf=0.5, classes=[2] )[0]

    # for r in car_detections:
    #     boxes = r.boxes
    #     for box in boxes:
    #         b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
    #         c = box.cls
    #         d = box.conf
    #         # Use the Annotator to draw boxes and labels
    #         annotator = r.plot(boxes=True, labels=True, conf=True)  
    # x1, y1, x2, y2 = b
    # class_id = c
    # conf = d

    # cv2.imshow('YOLO V8 Detection', annotator)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    detections_ = []
    for detection in car_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if(class_id == 2):
            detections_.append([x1, y1, x2, y2, score])

    # track car 
    track_ids = mot_tracker.update(np.asarray(detections_))

    # detect license plate
    lisence_plates = license_plate_detector.predict(image, conf= 0.5, classes= 0)[0]
    for license_plate in lisence_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        
        # assign license plate to car 
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1 : 

            # crop license plate
            license_plate_crop = image[int(y1):int(y2), int(x1): int(x2), :]

            # process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 65, 255, cv2.THRESH_BINARY_INV)
            # contours, _ = cv2.findContours(license_plate_crop_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the original crop
            # license_plate_crop_contours = license_plate_crop.copy()
            # cv2.drawContours(license_plate_crop_contours, contours, -1, (0, 255, 0), 2)

            # cv2.imshow('original_crop', license_plate_crop)
            # cv2.imshow('threshold', license_plate_crop_thresh)
            # cv2.imshow('contours', license_plate_crop_contours)
            # cv2.waitKey(0)

            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            print(f"license_plate_text: {license_plate_text}")
            print(f"license_plate_text_score: {license_plate_text_score}")

            # if license_plate_text is not None:
            #     results[car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
            #                                 'license_plate': {'bbox': [x1, y1, x2, y2],
            #                                                 'text': license_plate_text,
            #                                                 'bbox_score': score,
            #                                                 'text_score': license_plate_text_score}} 
            

    # write_csv(results, '/Users/hungnguyen/TaiLieu/NhanDang/test.csv')


if __name__ == '__main__':
    main()
