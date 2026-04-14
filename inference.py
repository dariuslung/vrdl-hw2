import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.amp import autocast
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchvision.ops import box_convert
from tqdm import tqdm

# Import the model architecture from the training script
from train import DigitDETR


@torch.no_grad()
def generate_submission(
    model: nn.Module, 
    test_img_dir: str, 
    transform: transforms.Compose, 
    device: torch.device, 
    output_file: str = "pred.json"
):
    """
    Evaluates the test directory and formats outputs exactly to the assignment requirements.
    """
    model.eval()
    test_dir = Path(test_img_dir)
    predictions = []
    
    # Parses all images in the test directory
    image_paths = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
    progress_bar = tqdm(image_paths, desc="[Inference] Generating Submission", dynamic_ncols=True)
    
    for img_path in progress_bar:
        try:
            # Assumes the filename '1234.png' corresponds to image_id 1234
            image_id = int(img_path.stem) 
        except ValueError:
            continue
            
        original_image = Image.open(img_path).convert("RGB")
        img_w, img_h = original_image.size
        
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        
        with autocast('cuda'):
            outputs = model(image_tensor)
        
        pred_logits = outputs['pred_logits'][0]
        pred_boxes = outputs['pred_boxes'][0]
        
        pred_scores, pred_labels = pred_logits.sigmoid().max(dim=-1)
        
        # Scale normalized coordinates back to the original unpadded image resolution
        scale_tensor = torch.tensor([img_w, img_h, img_w, img_h], device=device)
        pred_boxes_xyxy = box_convert(pred_boxes, 'cxcywh', 'xyxy') * scale_tensor
        pred_boxes_xywh = box_convert(pred_boxes_xyxy, 'xyxy', 'xywh')
        
        keep = pred_scores > 0.3
        boxes = pred_boxes_xywh[keep].cpu().numpy()
        scores = pred_scores[keep].cpu().numpy()
        labels = pred_labels[keep].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            predictions.append({
                "image_id": image_id,
                "bbox": box.tolist(),
                "score": float(score),
                "category_id": int(label) + 1  # Map 0-9 network output back to 1-10 labels
            })
            
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
        
    print(f"--> Successfully saved test predictions to {output_file}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Instantiating model...")
    num_classes = 10
    model = DigitDETR(num_classes=num_classes)
    model.to(device)

    # Recreate the EMA configuration to ensure keys match during weight loading
    ema_avg_fn = get_ema_multi_avg_fn(0.999)
    ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)
    
    save_dir = "checkpoints"
    checkpoint_path = os.path.join(save_dir, "detr_best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}. Train the model first.")
        return

    print("\nLoading best weights for test set inference...")
    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ema_model.load_state_dict(best_checkpoint['ema_model_state_dict'])
    
    # We use the standard validation transform for consistent sizing during testing
    test_transform = transforms.Compose([
        transforms.Resize(400, max_size=800),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    generate_submission(
        model=ema_model,
        test_img_dir="./data/test",
        transform=test_transform,
        device=device,
        output_file="pred.json"
    )


if __name__ == '__main__':
    main()