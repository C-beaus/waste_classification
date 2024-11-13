# train.py

import os
import sys
import logging
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from torchvision.ops import box_iou
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


# Set up logging
logger = logging.getLogger('train_logger')
logger.setLevel(logging.DEBUG)

# Create handlers for console and file logging
c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('train.log')

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

# Dataset class
class WasteDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images_dir = os.path.join(root, 'images')
        self.labels_dir = os.path.join(root, 'labels')
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.jpg')])
        logger.info(f"Initialized dataset with {len(self.image_files)} images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load images and masks
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + '.txt')

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise

        # Get bounding boxes and labels
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        class_id, cx, cy, w, h = map(float, line.strip().split())
                        class_id = int(class_id) + 1  # Adjust for background class at index 0
                        labels.append(class_id)

                        # Convert from normalized center coordinates to absolute coordinates
                        img_width, img_height = img.size
                        x_center = cx * img_width
                        y_center = cy * img_height
                        box_width = w * img_width
                        box_height = h * img_height

                        xmin = x_center - box_width / 2
                        ymin = y_center - box_height / 2
                        xmax = x_center + box_width / 2
                        ymax = y_center + box_height / 2

                        boxes.append([xmin, ymin, xmax, ymax])
                    except Exception as e:
                        logger.error(f"Error processing label in {label_path}: {e}")
        else:
            logger.warning(f"Label file not found for image {img_name}")

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img = self.transforms(img)

        return img, target

# Training function
def train():
    # Hyperparameters
    num_epochs = 100 # original is 3
    batch_size = 4
    learning_rate = 0.001 # original is 0.005
    momentum = 0.9
    weight_decay = 0.0005

    logger.info(f"Starting training: batch_size={batch_size}, learning_rate={learning_rate}, num_epochs={num_epochs}")

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Dataset and DataLoader 
    ###################### Change the dataset path here ######################
    dataset = WasteDataset(
        # root='c:/Users/chase/OneDrive/Documents/Grad/Robots_for_Recycling/waste_detector/waste_detector_repo/dataset/train',
        # root='c:/Users/chase/OneDrive/Documents/Grad/Robots_for_Recycling/waste_detector/waste_detector_repo/our_dataset/train',
        root='c:/Users/chase/OneDrive/Documents/Grad/Robots_for_Recycling/waste_detector/waste_detector_repo/plastic_and_metal_dataset/train',

        transforms=transform
    )

    val_dataset = WasteDataset(
        # root='c:/Users/chase/OneDrive/Documents/Grad/Robots_for_Recycling/waste_detector/waste_detector_repo/dataset/valid',
        # root='c:/Users/chase/OneDrive/Documents/Grad/Robots_for_Recycling/waste_detector/waste_detector_repo/our_dataset/valid',
        root='c:/Users/chase/OneDrive/Documents/Grad/Robots_for_Recycling/waste_detector/waste_detector_repo/plastic_and_metal_dataset/valid',


        transforms=transform
    )

    # Split dataset into train and test sets
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices)

    data_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=custom_collate_fn # lambda x: tuple(zip(*x))
    )

    val_data_loader = DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle = False, num_workers=4,
        collate_fn=custom_collate_fn
    )

    logger.info("DataLoader created.")

    # Get the model using our helper function
    model = get_model_instance_segmentation(NUM_CLASSES)
    device = torch.device('cpu')
    model.to(device)
    logger.info(f"Model loaded and moved to device: {device}")

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in data_loader: 
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            except Exception as e:
                logger.error(f"Error during model training: {e}")
                continue

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_loss = losses.item()
            epoch_loss += batch_loss
            logger.debug(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {batch_loss:.4f}")

        # Update the learning rate
        lr_scheduler.step()
        avg_loss = epoch_loss / len(data_loader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        writer_train = SummaryWriter('runs/train')
        writer_train.add_scalar('average_training_loss', epoch_loss, epoch)

        # Get accuracy for this epoch
        model.eval()
        Path(os.path.join("runs/", 'validation')).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter('runs/validation')
        compute_accuracy(model, val_data_loader, device, epoch, writer)

    # Save the trained model
        torch.save(model.state_dict(), 'fasterrcnn_model.pth')
    logger.info("Training completed and model saved to 'fasterrcnn_model.pth'.")

# def get_final_trained_model_accuracy(model, data_loader):

def compute_iou(box1, box2):
    # Compute IoU between two sets of boxes
    iou = box_iou(box1, box2)
    return iou

def compute_accuracy(model, data_loader, device, epoch, writer, iou_threshold=0.5):
    model.to(device)
    correct_detections = 0
    total_detections = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)  # Get predictions

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes']
                gt_boxes = target['boxes']
                pred_labels = output['labels']
                gt_labels = target['labels']

                # Calculate IoU for each prediction-gt pair
                ious = compute_iou(pred_boxes, gt_boxes)

                for i, iou_row in enumerate(ious): #ENSURE LOOPS OVER LENGTH OF IOUS, NOT EACH ELEMENT IN EACH ROW OF IOUS
                    max_iou, max_iou_idx = iou_row.max(0)  # Max IoU for the i-th predicted box

                    # Should handle if extra boxes are made
                    if max_iou > iou_threshold and pred_labels[i] == gt_labels[max_iou_idx]:
                        correct_detections += 1  # Correct label and IoU match
                    total_detections += 1

    # Calculate accuracy based on accuracy of drawn box and classification made
    accuracy = correct_detections / total_detections if total_detections > 0 else 0
    writer.add_scalar('validation_accuracy',
                                    100*accuracy,
                                    epoch)

    return accuracy

def custom_collate_fn(batch):
    return tuple(zip(*batch))

def get_model_instance_segmentation(num_classes):
    # Load a pre-trained model for classification and return
    # a model ready for fine-tuning

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

    logger.info("Loaded pre-trained Faster R-CNN model with MobileNetV3 backbone.")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    logger.info("Modified the model's box predictor to accommodate the new number of classes.")

    return model

if __name__ == '__main__':
    train()
