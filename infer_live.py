# infer.py

import sys
import os
import logging
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from train import WasteDataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random

import cv2
from PIL import Image
import numpy as np

# Set up logging
logger = logging.getLogger('infer_logger')
logger.setLevel(logging.DEBUG)

# Create handlers for console and file logging
c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('infer.log')

c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Constants
# CLASSES = ['__background__', 'recycling', 'nonrecycling']  # Include background
# CLASSES = ['__background__', 'nonplastic', 'plastic']
CLASSES = ['__background__', 'cardboard', 'glass', 'metal', 'paper', 'plastic']

NUM_CLASSES = len(CLASSES)

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    logger.info("Initialized Faster R-CNN model with MobileNetV3 backbone.")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    logger.info("Modified the model's box predictor to accommodate the new number of classes.")

    return model

def infer_frame(model, frame, device):
    transform = transforms.ToTensor()
    input_tensor = transform(frame).to(device)

    with torch.no_grad():
        outputs = model([input_tensor])[0]
    return outputs

def draw_bounding_boxes(frame, boxes, labels, scores, threshold=0.5): 
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            xmin, ymin, xmax, ymax = map(int, box)
            class_name = CLASSES[label]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {score:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main(model_name, confidence_threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model_instance_segmentation(NUM_CLASSES)
    # model.load_state_dict(torch.load('fasterrcnn_model.pth', map_location=device))
    model.load_state_dict(torch.load(model_name, map_location=device))

    model.eval().to(device)

    # Open a connection to the webcam
    # cv2.VideoCapture()
    # for i in range(5):  # Testing indices from 0 to 4
    #     cap = cv2.VideoCapture(i)
    #     if cap.isOpened():
    #         print(f"Camera found at index {i}")
    #         # break
    #     cap.release()
    # else:
    #     print("No camera found")

    # cap = cv2.VideoCapture(2)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Failed to open webcam.")
        sys.exit(1)

    logger.info("Starting real-time object detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame.")
            break

        # Convert frame to RGB and PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Run inference
        outputs = infer_frame(model, pil_image, device)
        
        # Get the boxes, labels, and scores
        boxes = outputs['boxes'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()

        # Draw the bounding boxes and labels on the frame
        frame = draw_bounding_boxes(frame, boxes, labels, scores, threshold=confidence_threshold)

        # Display the frame
        cv2.imshow('Real-Time Waste Detector', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Real-time detection ended.")

# def infer(image_path):
#     logger.info(f"Starting inference on image: {image_path}")

#     # Data transformation
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     # Load model
#     device = torch.device('cpu')
#     model = get_model_instance_segmentation(NUM_CLASSES)
#     try:
#         model.load_state_dict(torch.load('fasterrcnn_model.pth', map_location=device))
#         logger.info("Model loaded successfully.")
#     except Exception as e:
#         logger.error(f"Failed to load model: {e}")
#         sys.exit(1)
#     model.eval()
#     model.to(device)
#     logger.info(f"Model moved to device: {device}")

#     # Load and preprocess image
#     try:
#         image = Image.open(image_path).convert('RGB')
#         logger.info(f"Image {image_path} loaded successfully.")
#     except Exception as e:
#         logger.error(f"Failed to load image {image_path}: {e}")
#         sys.exit(1)

#     input_tensor = transform(image).to(device)
#     logger.debug(f"Image transformed and tensor created with shape {input_tensor.shape}")

#     # Forward pass
#     with torch.no_grad():
#         outputs = model([input_tensor])
#     logger.debug("Model inference completed.")

#     # Process predictions
#     outputs = outputs[0]
#     boxes = outputs['boxes']
#     labels = outputs['labels']
#     scores = outputs['scores']

#     # Set a confidence threshold
#     confidence_threshold = 0.5

#     # Display detections
#     fig, ax = plt.subplots(1)
#     ax.imshow(image)

#     num_detections = 0
#     for box, label, score in zip(boxes, labels, scores):
#         if score > confidence_threshold:
#             num_detections += 1
#             xmin, ymin, xmax, ymax = box
#             class_name = CLASSES[label]
#             rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                      linewidth=2, edgecolor='r', facecolor='none')
#             ax.add_patch(rect)
#             ax.text(xmin, ymin - 10, f'{class_name}: {score:.2f}', color='red', fontsize=12)
#             logger.info(f"Detected object: {class_name}, Confidence {score:.2f}, "
#                         f"Box [{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}]")

#     if num_detections == 0:
#         logger.warning("No objects detected with confidence > 0.5.")
#     else:
#         logger.info(f"Total objects detected: {num_detections}")

#     # plt.show()
#     logger.info("Inference completed and results displayed.")


if __name__ == '__main__':

    model_name = 'models/mobilenet_ss_18_wd_0001/fasterrcnn_mobilenet_ss_18_wd_0001.pth'
    confidence_threshold = 0.7
    main(model_name, confidence_threshold)
