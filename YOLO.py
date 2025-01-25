import os
import yolov5
import cv2
import matplotlib.pyplot as plt
import nbimporter 
import numpy as np
import pygame
from pathlib import Path



 

# Load model
model = yolov5.load('keremberke/yolov5m-license-plate')
  
# Set model parameters
model.conf = 0.10  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# Load all german license plates 
plates = []
path_license_plates = Path('/Users/marlon/Desktop/sem/vs/vs_proj/VCS_Project/data/actual')
plates.extend(list(path_license_plates.glob('*.jpg')))

n_license_plates = len(plates)


# For testing purposes, select some random license plates
IMAGE_PATHS = []
for SAMPLE_ID in range(n_license_plates):
    IMAGE_PATHS.append(str(plates[SAMPLE_ID]))


for SAMPLE_ID, IMAGE_PATH in enumerate(IMAGE_PATHS):
    # Perform inference with test-time augmentation
    results = model(IMAGE_PATH, augment=True)

    # Parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # Load the image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise ValueError(f"Image not found at path: {IMAGE_PATH}")
    image = image.copy()  # Ensure it's writable

    # Draw bounding boxes on the image
    for box, score, category in zip(boxes, scores, categories):
        x1, y1, x2, y2 = map(int, box)
        label = f"ID {int(category)}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    license_plate_detected = len(boxes) > 0
    

    os.makedirs('VCS_Project/results/predictions', exist_ok=True)
    output_path = f'VCS_Project/results/predictions/predictions.{SAMPLE_ID}.png'
    cv2.imwrite(output_path, image)
    print("-"*80)
    print(f"Saved manually drawn predictions to {output_path}")


    if license_plate_detected:    
        # Load the image and show the cropped license plate
        image = cv2.imread(IMAGE_PATH)
        license_plate = image[int(boxes[0][1]):int(boxes[0][3]), int(boxes[0][0]):int(boxes[0][2])]

        # Create a new folder "results/license_plates" if it does not exist
        if not os.path.exists('VCS_Project/results/license_plates'):
            os.makedirs('VCS_Project/results/license_plates')
            
        # Save the cropped license plate to "results/license_plates" folder
        cv2.imwrite(f'VCS_Project/results/license_plates/license_plate.{SAMPLE_ID}.png', license_plate)
        print("-"*75)
        print(f"Saved cropped license plate to results/license_plates/license_plate.{SAMPLE_ID}.png")

        # Display the cropped license plate
        plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()



