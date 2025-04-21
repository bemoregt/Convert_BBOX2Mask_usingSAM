import os
import csv
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import time

# Record start time
start_time = time.time()

# Path settings
base_dir = "/Users/m1_4k/Pictures/car_bbox_1001"  # Modify to match your image path
csv_path = os.path.join(base_dir, "train_solution_bounding_boxes (1).csv")
image_dir = os.path.join(base_dir, "training_images")
output_dir = os.path.join(base_dir, "segment")

# Create segment folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Device setup: Use MPS (Apple Silicon GPU acceleration)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Use lighter vit_b model
model_type = "vit_b"  # Using lighter model instead of vit_h
sam_checkpoint = "sam_vit_b_01ec64.pth"  # Modified model path to vit_b checkpoint

print(f"Loading SAM model: {model_type} - {sam_checkpoint}")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Reading CSV file
print("Reading CSV file...")
bbox_data = {}
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        image_name, xmin, ymin, xmax, ymax = row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4])
        if image_name not in bbox_data:
            bbox_data[image_name] = []
        bbox_data[image_name].append((xmin, ymin, xmax, ymax))

print(f"Total {len(bbox_data)} images to process")

# Batch size - adjust according to memory capacity
batch_size = 1  # Increase if memory is sufficient

# Processed image count
processed_count = 0

# Process images
for image_name, bboxes in bbox_data.items():
    # Image file path
    image_path = os.path.join(image_dir, image_name)
    
    # Display progress
    processed_count += 1
    print(f"[{processed_count}/{len(bbox_data)}] Processing {image_name}...")
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Warning: Cannot find file {image_path}. Skipping.")
        continue
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Cannot load image {image_path}. Skipping.")
        continue
    
    # Convert to RGB (SAM uses RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set image for SAM prediction
    predictor.set_image(image_rgb)
    
    # Create empty mask of the same size as the image (for accumulating result masks)
    final_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Memory optimization: Process multiple bounding boxes in batches
    for i in range(0, len(bboxes), batch_size):
        batch_bboxes = bboxes[i:i+batch_size]
        
        for bbox in batch_bboxes:
            xmin, ymin, xmax, ymax = bbox
            
            # Calculate center point (positive prompt)
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            
            # Calculate corners (negative prompts)
            corners = [
                (xmin, ymin),  # Top-left
                (xmax, ymin),  # Top-right
                (xmin, ymax),  # Bottom-left
                (xmax, ymax)   # Bottom-right
            ]
            
            # Prepare prompt points and labels
            input_points = np.array([[center_x, center_y]] + corners)
            input_labels = np.array([1, 0, 0, 0, 0])  # Center point is positive(1), corners are negative(0)
            
            # Predict mask with SAM - set to return single mask to save memory
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False  # Only output single mask to increase speed
            )
            
            # Add current object's mask to final mask (OR operation)
            mask = masks[0]  # Use first mask only since multimask_output=False
            final_mask = np.logical_or(final_mask, mask).astype(np.uint8) * 255
        
        # Clear memory after batch processing
        if device.type == "cuda" or device.type == "mps":
            torch.cuda.empty_cache() if device.type == "cuda" else torch.mps.empty_cache()
    
    # Save mask
    output_filename = os.path.splitext(image_name)[0] + "_mask.png"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, final_mask)
    
    # Display progress time
    elapsed_time = time.time() - start_time
    avg_time_per_image = elapsed_time / processed_count
    remaining_images = len(bbox_data) - processed_count
    estimated_time_remaining = avg_time_per_image * remaining_images
    
    print(f"{image_name} processing complete: Mask saved to {output_path}")
    print(f"Progress: {processed_count}/{len(bbox_data)} images completed")
    print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds ({estimated_time_remaining/60:.2f} minutes)")

total_time = time.time() - start_time
print(f"All images processed! Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")