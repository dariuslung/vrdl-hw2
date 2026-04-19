import os
import argparse
import json
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import sys

# Point to the Deformable-DETR directory
sys.path.append("Deformable-DETR")

from models.deformable_transformer import DeformableTransformer
from models.deformable_detr import DeformableDETR
from models.backbone import build_backbone


def get_inference_model(model_path, device):
    """Initializes the Deformable DETR model architecture and loads the trained weights."""
    args = argparse.Namespace(
        lr_backbone=1e-5,
        masks=False,
        backbone='resnet50',
        dilation=False,
        position_embedding='sine',
        hidden_dim=256,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=1024,
        dropout=0.2,
        nheads=8,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=False,
        num_classes=10
    )

    backbone = build_backbone(args)

    transformer = DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
    )

    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=300,
        num_feature_levels=args.num_feature_levels,
    )

    # Load the trained weights
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    return model


class TestDataset(Dataset):
    """
    Dataset loader for test images. 
    Infers the image_id directly from the filename (e.g., '123.jpg' -> 123).
    """
    def __init__(self, test_dir, max_size=600):
        self.test_dir = test_dir
        self.max_size = max_size
        self.image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        
        # Extract ID from filename by stripping the extension
        image_id = int(os.path.splitext(img_name)[0])
        
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        # Exact bounded scaling used in training/validation
        scale_factor = 1.0
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # Limit 1: Prevent feature map collapse for small digits
        min_dim = min(new_w, new_h)
        if min_dim < 400:
            fix_scale = 400.0 / min_dim
            new_w = int(new_w * fix_scale)
            new_h = int(new_h * fix_scale)

        # Limit 2: Prevent large images from exceeding max_size
        max_dim = max(new_w, new_h)
        if max_dim > self.max_size:
            fix_scale = self.max_size / max_dim
            new_w = int(new_w * fix_scale)
            new_h = int(new_h * fix_scale)

        image_tensor = TF.resize(image, (new_h, new_w))
        image_tensor = TF.to_tensor(image_tensor)
        image_tensor = TF.normalize(
            image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        return image_tensor, image_id, (h, w)


def collate_fn(batch):
    images = [item[0] for item in batch]
    image_ids = [item[1] for item in batch]
    orig_sizes = [item[2] for item in batch]
    return images, image_ids, orig_sizes


def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), w, h]
    return torch.stack(b, dim=-1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Inference script for Deformable DETR")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to the trained weights")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Directory containing test images")
    parser.add_argument("--output_file", type=str, default="pred.json", help="Path to save the JSON predictions")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.05, help="Confidence threshold for saving boxes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_inference_model(args.model_path, device)

    dataset = TestDataset(args.test_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    predictions = []
    progress_bar = tqdm(dataloader, desc="Running Inference")

    for images, image_ids, orig_sizes in progress_bar:
        images = list(image.to(device) for image in images)
        orig_sizes = torch.tensor(orig_sizes, dtype=torch.int64, device=device)

        # Standard FP32 forward pass
        outputs = model(images)

        out_logits = outputs['pred_logits']
        out_boxes = outputs['pred_boxes']

        # Deformable DETR uses Focal Loss, so we use sigmoid
        prob = out_logits.sigmoid()
        scores, labels = prob.max(-1)

        # Convert normalized [cx, cy, w, h] to absolute [x_min, y_min, w, h]
        boxes = box_cxcywh_to_xywh(out_boxes)
        img_h, img_w = orig_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        for i, image_id in enumerate(image_ids):
            keep = scores[i] > args.threshold
            
            p_boxes = boxes[i][keep].cpu().tolist()
            p_scores = scores[i][keep].cpu().tolist()
            p_labels = labels[i][keep].cpu().tolist()

            for box, score, label in zip(p_boxes, p_scores, p_labels):
                predictions.append({
                    "image_id": image_id,
                    "bbox": box,
                    "score": score,
                    "category_id": label + 1  # Revert to 1-indexed category
                })

    # Sort descending by score so COCO evaluates your best guesses first
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # Save to JSON
    with open(args.output_file, "w") as f:
        json.dump(predictions, f, indent=4)
    
    print(f"\nSaved {len(predictions)} predictions to {args.output_file}")

if __name__ == "__main__":
    main()