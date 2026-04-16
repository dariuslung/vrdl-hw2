import os
import argparse
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

from models.matcher import HungarianMatcher
from models.detr import SetCriterion


class DetrTransform:
    """
    Transforms PIL Images and COCO bounding boxes into the format expected by DETR.
    Resizes images to ensure they fit within 8GB VRAM while preserving aspect ratios.
    """
    def __init__(self, max_size=600):
        self.max_size = max_size

    def __call__(self, image, target):
        w, h = image.size
        scale = self.max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        image = TF.resize(image, (new_h, new_w))
        image = TF.to_tensor(image)
        image = TF.normalize(
            image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # 1. Safely extract the image_id. 
        # If an image has no digits (empty target list), we provide a fallback value.
        if len(target) > 0:
            image_id = target[0]["image_id"]
        else:
            image_id = -1 

        boxes = []
        labels = []
        for t in target:
            x, y, bw, bh = t["bbox"]
            
            # Convert COCO [x_min, y_min, w, h] to normalized [cx, cy, w, h]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            norm_w = bw / w
            norm_h = bh / h
            
            boxes.append([cx, cy, norm_w, norm_h])
            labels.append(t["category_id"] - 1)

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        # 2. Add image_id to the output dictionary as a tensor
        target_dict = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64) 
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
    transform = DetrTransform(max_size=600)

    train_dataset = CocoDetection(
        root="data/train",
        annFile="data/train.json",
        transforms=transform
    )
    
    valid_dataset = CocoDetection(
        root="data/valid",
        annFile="data/valid.json",
        transforms=transform
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


def train_one_epoch(model, criterion, dataloader, optimizer, scaler, device, epoch, writer):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for step, (images, targets) in enumerate(progress_bar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Mixed precision for speed and memory efficiency
        with autocast():
            # 1. Forward pass only takes images
            outputs = model(images)
            
            # 2. Criterion takes the outputs and targets to compute bipartite matching loss
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            
            # 3. Sum the weighted loss components
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        scaler.scale(loss).backward()
        
        # Gradient clipping is standard for DETR stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        scaler.step(optimizer)
        scaler.update()

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
        # Ensure targets are on the correct device for the criterion
        targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
        img_sizes = torch.tensor([img.shape[-2:] for img in images]).to(device)
        
        with autocast():
            outputs = model(images)
            
            # 1. Compute Loss
            loss_dict = criterion(outputs, targets_device)
            weight_dict = criterion.weight_dict
            loss = sum(
                loss_dict[k] * weight_dict[k] 
                for k in loss_dict.keys() if k in weight_dict
            )
            
        total_loss += loss.item()
        
        # 2. Compute mAP predictions
        out_logits = outputs['pred_logits']
        out_boxes = outputs['pred_boxes']
        
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :10].max(-1)
        
        boxes = box_cxcywh_to_xywh(out_boxes)
        img_h, img_w = img_sizes.unbind(1)
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

    # Log Loss
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Valid/Total_Loss", avg_loss, epoch)
    
    # Calculate and Log mAP
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir="runs/detr_digit_detection")

    train_loader, valid_loader = get_dataloaders(args.batch_size, args.num_workers)

    # Initialize model: 10 classes for the digits (0-9). 
    # DETR will automatically add the 11th class internally for the "background".
    model = torch.hub.load(
        'facebookresearch/detr:main', 
        'detr_resnet50', 
        pretrained=False, 
        num_classes=10
    )
    model.to(device)

    # Initialize DETR Matcher and Criterion
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes', 'cardinality']
    
    criterion = SetCriterion(
        num_classes=10, 
        matcher=matcher, 
        weight_dict=weight_dict, 
        eos_coef=0.1, 
        losses=losses
    )
    criterion.to(device)

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
    
    # Step scheduler reduces learning rate by a factor of 10 at epoch 40 (standard DETR behavior)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    scaler = GradScaler()

    best_map = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scaler, device, epoch, writer)
        
        val_loss, current_map = validate_and_eval(model, criterion, valid_loader, device, epoch, writer)
        
        lr_scheduler.step()
        
        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mAP: {current_map:.4f}")

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