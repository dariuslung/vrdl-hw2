import os
import argparse
import random
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch.nn.functional as F
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.datasets import CocoDetection
from torchvision.models.resnet import ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
# Point to the new Deformable-DETR directory
sys.path.append("Deformable-DETR")

from models.deformable_transformer import DeformableTransformer
from models.deformable_detr import DeformableDETR, SetCriterion
from models.matcher import HungarianMatcher
from models.backbone import build_backbone


def get_deformable_detr_model(device):
    # 1. Setup minimal arguments required by the Deformable DETR builder
    args = argparse.Namespace(
        lr_backbone=1e-5,
        masks=False,
        backbone='resnet50',
        dilation=False,
        position_embedding='sine',
        hidden_dim=128,
        enc_layers=3,
        dec_layers=3,
        dim_feedforward=1024,
        dropout=0.2,
        nheads=8,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=False,
        num_classes=10
    )

    # 2. Build the backbone (handles the multi-scale feature extraction)
    backbone = build_backbone(args)

    # 3. Build the Deformable Transformer
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

    # 4. Initialize Deformable DETR
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=20, # Deformable DETR typically uses 300 queries but reduce this for simpler dataset
        num_feature_levels=args.num_feature_levels,
    )
    
    # Initialize the specific matcher and criterion for Deformable DETR
    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=5)
    weight_dict = {'loss_ce': 2, 'loss_bbox': 5, 'loss_giou': 5}
    losses = ['labels', 'boxes', 'cardinality']
    
    criterion = SetCriterion(
        args.num_classes, 
        matcher=matcher, 
        weight_dict=weight_dict, 
        losses=losses,
        focal_alpha=0.25 # Deformable DETR uses Focal Loss for classification
    )
    
    return model.to(device), criterion.to(device)


class DetrTransform:
    """
    Transforms PIL Images and COCO bounding boxes into the format expected by DETR.
    Includes random scaling and cropping during training to prevent overfitting.
    """
    def __init__(self, max_size=600, train=True):
        self.max_size = max_size
        self.train = train

    def __call__(self, image, target):
        import random
import torch
import torchvision.transforms.functional as TF

class DetrTransform:
    """
    Transforms PIL Images and COCO bounding boxes into the format expected by DETR.
    Includes photometric augmentations and bounded relative scaling during training.
    """
    def __init__(self, max_size=600, train=True):
        self.max_size = max_size
        self.train = train

    def __call__(self, image, target):
        # 1. Photometric and Quality Augmentations
        if self.train:
            # Brightness & Contrast
            if random.random() < 0.5:
                bright_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, bright_factor)
            
            if random.random() < 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_contrast(image, contrast_factor)
                
            # Saturation
            if random.random() < 0.5:
                saturation_factor = random.uniform(0.5, 1.5)
                image = TF.adjust_saturation(image, saturation_factor)
                
            # Gaussian Blur (simulates out-of-focus cameras)
            if random.random() < 0.3:
                # Kernel size must be odd. 3 or 5 is safe for digits.
                image = TF.gaussian_blur(image, kernel_size=[3, 3])

        # Safely extract image_id
        image_id = target[0]["image_id"] if len(target) > 0 else -1

        # Extract absolute COCO boxes [x, y, w, h] and labels
        valid_targets = [t for t in target if t["bbox"][2] > 0 and t["bbox"][3] > 0]
        boxes = [t["bbox"] for t in valid_targets]
        labels = [t["category_id"] - 1 for t in valid_targets]

        w, h = image.size

        # 2. Bounded Relative Scaling
        if self.train:
            scale_factor = random.uniform(0.8, 1.2)
        else:
            scale_factor = 1.0

        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # Limit 1: Prevent feature map collapse for small digits
        min_dim = min(new_w, new_h)
        if min_dim < 400:
            fix_scale = 400.0 / min_dim
            new_w = int(new_w * fix_scale)
            new_h = int(new_h * fix_scale)

        # Limit 2: Prevent OOM and slowdowns on large images
        max_dim = max(new_w, new_h)
        if max_dim > self.max_size:
            fix_scale = self.max_size / max_dim
            new_w = int(new_w * fix_scale)
            new_h = int(new_h * fix_scale)

        image = TF.resize(image, (new_h, new_w))
        image = TF.to_tensor(image)
        image = TF.normalize(
            image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # 3. Final Conversion to DETR Format (Relative [cx, cy, w, h])
        normalized_boxes = []
        for box in boxes:
            bx, by, bw, bh = box
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            norm_w = bw / w
            norm_h = bh / h
            normalized_boxes.append([cx, cy, norm_w, norm_h])

        if len(normalized_boxes) > 0:
            normalized_boxes = torch.as_tensor(normalized_boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            normalized_boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        target_dict = {
            "boxes": normalized_boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "orig_size": torch.tensor([h, w], dtype=torch.int64)
        }
        
        return image, target_dict


def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), w, h]
    return torch.stack(b, dim=-1)


def collate_fn(batch):
    """
    Collate function to handle variable-sized lists of tensors.
    Torchvision DETR handles the internal padding/NestedTensors automatically.
    """
    return tuple(zip(*batch))


def get_dataloaders(batch_size, num_workers):
    # Pass train=True for the training dataset
    train_transform = DetrTransform(max_size=600, train=True)
    
    # Pass train=False for the validation dataset
    valid_transform = DetrTransform(max_size=600, train=False)

    train_dataset = CocoDetection(
        root="data/train",
        annFile="data/train.json",
        transforms=train_transform
    )
    
    valid_dataset = CocoDetection(
        root="data/valid",
        annFile="data/valid.json",
        transforms=valid_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, valid_loader


def train_one_epoch(model, criterion, dataloader, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for step, (images, targets) in enumerate(progress_bar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Run forward pass in standard FP32. 
        # Autocast must be removed to support the custom Deformable Attention CUDA kernel.
        outputs = model(images)

        # Calculate loss directly
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += loss.item()
        
        # Log to TensorBoard
        global_step = (epoch - 1) * len(dataloader) + step
        writer.add_scalar("Train/Total_Loss", loss.item(), global_step)
        for k, v in loss_dict.items():
            writer.add_scalar(f"Train/{k}", v.item(), global_step)

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate_and_eval(
    model, 
    criterion, 
    dataloader, 
    device, 
    epoch, 
    writer, 
    valid_json_path="data/valid.json"
):
    model.eval()
    
    coco_gt = COCO(valid_json_path)
    coco_results = []
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Valid]", leave=False)

    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
        orig_sizes = torch.stack([t["orig_size"] for t in targets_device])
        
        # Standard FP32 forward pass
        outputs = model(images)
            
        loss_dict = criterion(outputs, targets_device)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
        total_loss += loss.item()
        
        # Compute mAP predictions
        out_logits = outputs['pred_logits']
        out_boxes = outputs['pred_boxes']
        
        # Deformable DETR uses Focal Loss, meaning probabilities are evaluated via Sigmoid.
        # No background class slicing is required.
        prob = out_logits.sigmoid()
        scores, labels = prob.max(-1)
        
        boxes = box_cxcywh_to_xywh(out_boxes)
        img_h, img_w = orig_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        
        for i, target in enumerate(targets):
            image_id = target["image_id"].item()
            keep = scores[i] > 0.05
            
            p_boxes = boxes[i][keep].cpu().tolist()
            p_scores = scores[i][keep].cpu().tolist()
            p_labels = labels[i][keep].cpu().tolist()
            
            for box, score, label in zip(p_boxes, p_scores, p_labels):
                coco_results.append({
                    "image_id": image_id,
                    "category_id": label + 1,
                    "bbox": box,
                    "score": score
                })

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Valid/Total_Loss", avg_loss, epoch)
    
    if not coco_results:
        map_50_95 = 0.0
    else:
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map_50_95 = coco_eval.stats[0] 
    
    writer.add_scalar("Valid/mAP", map_50_95, epoch)
    
    return avg_loss, map_50_95


def main():
    parser = argparse.ArgumentParser(description="Train DETR for Digit Detection")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir="runs/deformable_detr_digits")

    train_loader, valid_loader = get_dataloaders(args.batch_size, args.num_workers)

    # Initialize model: 10 classes for the digits (0-9). The model will be built with a ResNet-50 backbone and a Deformable Transformer head.
    model, criterion = get_deformable_detr_model(device)

    # Standard DETR optimization: train the transformer parameters with a higher LR
    # than the backbone parameters to avoid destroying pretrained feature representations.
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=14, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[15, 25], 
        gamma=0.1
    )

    best_map = 0.0

    for epoch in range(1, args.epochs + 1):
        # param_groups[0] is the transformer/main model
        # param_groups[1] is the backbone
        current_lr = optimizer.param_groups[0]['lr']
        current_lr_backbone = optimizer.param_groups[1]['lr']
        
        writer.add_scalar("Hyperparameters/LR_Main", current_lr, epoch)
        writer.add_scalar("Hyperparameters/LR_Backbone", current_lr_backbone, epoch)

        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, writer)        
        val_loss, current_map = validate_and_eval(model, criterion, valid_loader, device, epoch, writer)
        
        lr_scheduler.step()
        
        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mAP: {current_map:.4f} | LR: {current_lr:.6f}")

        if current_map > best_map:
            best_map = current_map
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

        # Save latest checkpoint periodically
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch}.pth"))

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    main()