import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

folder_path = 'images'  

image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg'))],
                     key=lambda f: int(f.split('_')[1].split('.')[0]))  # Sorting by image index


model = YOLO('models/best.pt')  # Path to your trained YOLOv8 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

class_colors = {
    0: (255, 0, 0),  # Red for gravel
    1: (61, 245, 61),  # Green for grass
    2: (50, 183, 255),  # Blue for asphalt
}

# Alpha value for transparency
alpha = 0.5  


for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)  # Full path to the image
    img = cv2.imread(image_path)  # Load the image

    # create an overlay with the same dimensions as the original image
    overlay = np.zeros_like(img)

    # inference
    results = model.predict(img, conf=0.3)  # Adjust confidence threshold as needed

    # add segmentation masks with transparency
    for result in results:
        for mask, box in zip(result.masks.xy, result.boxes):
            # extract the points from the segmentation mask
            points = np.int32([mask])

            # determine the color for the current class
            class_index = int(box.cls[0])  # Get the class index
            color = class_colors.get(class_index, (255, 255, 255))  # Default to white if not mapped

            # fill the polygon in the overlay with the class-specific color
            cv2.fillPoly(overlay, points, color)

    # blend the overlay with the original image to apply transparency
    img = ((1 - alpha) * img + alpha * overlay).astype(np.uint8)  # Apply transparency to the overlay

    # Display the processed image
    cv2.imshow("YOLOv8 Inference", img)  
    cv2.waitKey(20)  # Wait 1 second to view each result

cv2.destroyAllWindows()  # Close the image display windows
