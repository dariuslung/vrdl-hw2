# NYCU Visual Recognition using Deep Learning - Spring 2026 - Homework 2

## Introduction
This repository contains the training and inference pipeline for detecting digits (10 classes, 0-9) using a highly customized Two-Stage Deformable DETR architecture. 

The implementation uses a ResNet-50 backbone, but the standard Deformable DETR Transformer has been optimized for the specific task of digit detection to prevent overfitting and improve inference speed. Key architectural configurations include:
* **Reduced Transformer Depth:** 4 encoder layers and 4 decoder layers (down from the default 6).
* **Targeted Queries:** 50 object queries (down from 300) to minimize overlapping bounding box predictions on images with low object counts.
* **Two-Stage Processing:** Enabled with bounding box refinement (`with_box_refine=True`) to dynamically generate region proposals from the encoder rather than relying on static spatial queries.
* **Bounded Relative Scaling:** Custom data augmentation that scales images proportionally while preventing severe feature map collapse on extremely small digits (enforces a minimum dimension of 400 pixels).

## Environment Setup
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment (e.g., Conda or venv).

### 1. Install Dependencies
Install PyTorch and Torchvision compatible with your CUDA version. Then, install the required Python packages:
```bash
pip install torch torchvision
pip install pycocotools tqdm tensorboard Pillow
````

### 2. Setup Deformable DETR

The scripts rely on the official Deformable DETR model architecture.

1. Clone the [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) repository into the root of this project. The folder must be named `Deformable-DETR`.
2. Compile the custom CUDA operators required by Deformable Attention. Navigate into the `Deformable-DETR/models/ops` directory and run the compilation script according to their official documentation (typically `python setup.py build install` or `sh make.sh`).

### 3. Data Directory Structure

The scripts expect the dataset to follow the standard COCO format and be organized in a `data/` directory at the project root:

```
project_root/
├── train.py
├── inference.py
├── Deformable-DETR/        # Cloned repository
└── data/
    ├── train/              # Training images
    ├── train.json          # COCO format training annotations
    ├── valid/              # Validation images
    ├── valid.json          # COCO format validation annotations
    └── test/               # Test images for inference
```

## Usage

### Training

To train the model, run `train.py`. The script will automatically handle model initialization, data augmentation, and logging. Checkpoints will be saved in the `checkpoints/` directory.

**Basic Training Command:**

Bash

```
python train.py --epochs 30 --batch_size 4 --num_workers 4
```

**Available Arguments:**

- `--epochs`: Total number of training epochs (default: 30)
- `--batch_size`: Batch size per GPU (default: 4)
- `--num_workers`: Number of dataloader workers (default: 4)
- `--lr`: Learning rate for the transformer (default: 1e-4)
- `--lr_backbone`: Learning rate for the ResNet-50 backbone (default: 1e-5)
- `--weight_decay`: Weight decay for the optimizer (default: 1e-3)
- `--output_dir`: Directory to save model checkpoints (default: `checkpoints`)

_Note: The learning rate scheduler is currently hardcoded to drop by a factor of 10 at epochs 15 and 25._

**Monitoring Training:** Training metrics (Losses, mAP, and Learning Rates) are logged via TensorBoard. To view them, run:

Bash

```
tensorboard --logdir runs
```

### Inference

To run inference on the test dataset, use `inference.py`. The script will process the images, apply a confidence threshold, sort the predictions by confidence, and output a COCO-compliant JSON file.

**Basic Inference Command:**

Bash

```
python inference.py --model_path checkpoints/best_model.pth --test_dir data/test --output_file pred.json
```

**Available Arguments:**

- `--model_path`: Path to the trained `.pth` weight file (default: `checkpoints/best_model.pth`)
- `--test_dir`: Directory containing the images to evaluate (default: `data/test`)
- `--output_file`: Path to save the resulting JSON file (default: `pred.json`)
- `--batch_size`: Inference batch size (default: 4)
- `--num_workers`: Number of dataloader workers (default: 4)
- `--threshold`: Minimum confidence score to keep a bounding box (default: 0.05)

The resulting `pred.json` is formatted as a list of dictionaries (`[{"image_id": int, "bbox": [x, y, w, h], "score": float, "category_id": int}, ...]`) and is ready for COCO metric evaluation or leaderboard submission.

## Performance Snapshot
<img width="931" height="512" alt="Screenshot 2026-04-21 165211" src="https://github.com/user-attachments/assets/6216b8ce-e1e3-4b49-9a9c-c55181049c17" />

