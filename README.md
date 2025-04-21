# Convert_BBOX2Mask_usingSAM

A tool that converts bounding box information to segmentation masks using [SAM (Segment Anything Model)](https://github.com/facebookresearch/segment-anything).

## Example Results

![SAM Mask Conversion Process](ScrShot%2047.png)
![Bounding Box Based Segmentation](ScrShot%2048.png)
![Vehicle Image Segmentation Result 1](ScrShot%2049.png)
![Vehicle Image Segmentation Result 2](ScrShot%2050.png)

## Key Features

- Generate segmentation masks for images using bounding box information from CSV files
- Support for Apple Silicon (M1/M2) GPU acceleration (MPS) and NVIDIA GPU (CUDA)
- Batch processing and single mask output options for memory optimization
- Progress status and estimated time remaining display

## How It Works

1. Load bounding box information for each image from a CSV file
2. For each bounding box:
   - Use the center point as a positive prompt
   - Use the four corner points as negative prompts
   - Predict segmentation masks using the SAM model
3. Combine all object masks in an image into one
4. Save the resulting mask as a PNG file

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- OpenCV
- NumPy
- [segment-anything](https://github.com/facebookresearch/segment-anything) package
- SAM model checkpoint file (`sam_vit_b_01ec64.pth`)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/bemoregt/Convert_BBOX2Mask_usingSAM.git
cd Convert_BBOX2Mask_usingSAM

# 2. Install required packages
pip install torch opencv-python numpy
pip install git+https://github.com/facebookresearch/segment-anything.git

# 3. Download SAM model
# Download 'ViT-B SAM model' from
# https://github.com/facebookresearch/segment-anything#model-checkpoints
```

## Usage

1. Open the `bbox_to_mask.py` file and modify the following paths to match your environment:
   ```python
   base_dir = "/path/to/your/data"  # Data directory
   csv_path = os.path.join(base_dir, "your_bounding_boxes.csv")  # Bounding box CSV file
   image_dir = os.path.join(base_dir, "images")  # Image directory
   output_dir = os.path.join(base_dir, "segment")  # Output directory
   
   # SAM model file path
   sam_checkpoint = "path/to/sam_vit_b_01ec64.pth"
   ```

2. Run the script:
   ```bash
   python bbox_to_mask.py
   ```

## CSV File Format

The CSV file should have the following format:
```
filename,xmin,ymin,xmax,ymax
image1.jpg,100,150,300,400
image2.jpg,200,250,500,600
...
```

## Performance Optimization

- Adjust the `batch_size` variable to control memory usage and processing speed
- Increase `batch_size` if you have sufficient memory to improve processing speed
- Automatically detects and utilizes Apple Silicon GPU (MPS) or NVIDIA GPU (CUDA)

## License

MIT License

## Author

[bemoregt](https://github.com/bemoregt)