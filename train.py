import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, SequentialLR, StepLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import box_convert, generalized_box_iou, sigmoid_focal_loss
from tqdm import tqdm
from PIL import Image

from dataset import setup_dataloaders
from models.detr import SetCriterion


class DigitDETR(nn.Module):
    """
    DETR model adapted to mimic Deformable DETR structural choices.
    """

    def __init__(self, num_classes: int = 10, num_queries: int = 100):
        super().__init__()

        self.detr = torch.hub.load(
            'facebookresearch/detr:main',
            'detr_resnet50',
            pretrained=False
        )

        pretrained_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.detr.backbone[0].body.load_state_dict(
            pretrained_resnet.state_dict(),
            strict=False
        )

        # Modification 1: Focal Loss does not require a +1 background class.
        in_features = self.detr.class_embed.in_features
        self.detr.class_embed = nn.Linear(in_features, num_classes)
        
        # Initialize bias for Focal Loss to prevent early instability
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.detr.class_embed.bias, bias_value)

        # Modification 2: Set queries to the specified number (default 100)
        # Fix: Must completely replace the embedding layer to match the new size
        self.detr.num_queries = num_queries
        hidden_dim = self.detr.transformer.d_model
        self.detr.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, images):
        # The facebookresearch/detr model natively accepts a list of tensors
        # and dynamically packs them into a NestedTensor with padding masks.
        return self.detr(images)


class FocalHungarianMatcher(nn.Module):
    """
    Custom Matcher that uses Focal Loss probabilities for bipartite matching,
    aligning with the Deformable DETR paper.
    """

    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0,
                 cost_giou: float = 2.0, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if len(tgt_ids) == 0:
            return [(torch.as_tensor([], dtype=torch.int64),
                     torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]

        neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_convert(out_bbox, 'cxcywh', 'xyxy'),
            box_convert(tgt_bbox, 'cxcywh', 'xyxy')
        )

        cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost_matrix = cost_matrix.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class DeformableSetCriterion(SetCriterion):
    """
    Overrides the standard DETR Cross-Entropy calculation with Sigmoid Focal Loss.
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, alpha=0.25, gamma=2.0):
        super().__init__(num_classes, matcher, weight_dict, eos_coef=0.1, losses=losses)
        self.alpha = alpha
        self.gamma = gamma

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot,
            alpha=self.alpha, gamma=self.gamma, reduction="none"
        )
        loss_ce = loss_ce.sum() / num_boxes

        losses = {'loss_ce': loss_ce}
        return losses


def train_one_epoch(
    model: nn.Module,
    ema_model: AveragedModel,
    criterion: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler
) -> float:
    model.train()
    criterion.train()

    epoch_loss = 0.0
    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} [Train]",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout
    )

    for step, (images, targets) in enumerate(progress_bar):
        # Optimization: non_blocking=True allows overlap of data transfer and computation
        images = list(image.to(device, non_blocking=True) for image in images)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        # Optimization: set_to_none=True is slightly faster than zeroing out gradients
        optimizer.zero_grad(set_to_none=True)

        # AMP: Run the forward pass and loss computation with updated autocast API
        with autocast('cuda'):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # AMP: Scale the loss and call backward
        scaler.scale(losses).backward()

        # Gradients must be unscaled before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

        # AMP: Step the optimizer and update the scaler
        scaler.step(optimizer)
        scaler.update()
        
        # EMA: Update the exponential moving average of the model weights
        ema_model.update_parameters(model)

        loss_val = losses.item()
        epoch_loss += loss_val
        progress_bar.set_postfix(loss=f"{loss_val:.4f}")

        global_step = epoch * len(dataloader) + step
        writer.add_scalar('Loss/Train_Batch', loss_val, global_step)
        for loss_name, loss_value in loss_dict.items():
            writer.add_scalar(f'Loss_Components/Train_{loss_name}', loss_value.item(), global_step)

    return epoch_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    ema_model: AveragedModel,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> tuple[float, float]:
    ema_model.eval()
    criterion.eval()

    epoch_loss = 0.0
    map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox').to(device)
    
    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} [Val]",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout
    )

    for step, (images, targets) in enumerate(progress_bar):
        images = list(image.to(device, non_blocking=True) for image in images)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        # AMP: Apply updated autocast API during evaluation for faster inference speed
        with autocast('cuda'):
            # Evaluate using the Exponential Moving Average weights
            outputs = ema_model(images)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_val = losses.item()
        epoch_loss += loss_val
        progress_bar.set_postfix(loss=f"{loss_val:.4f}")
        
        # Prepare targets and predictions for mAP calculation
        preds_list = []
        targets_list = []
        
        for i in range(len(images)):
            img_h, img_w = images[i].shape[-2:]
            scale_tensor = torch.tensor([img_w, img_h, img_w, img_h], device=device)
            
            # Convert prediction logic
            pred_logits = outputs['pred_logits'][i]
            pred_boxes = outputs['pred_boxes'][i]
            
            pred_scores, pred_labels = pred_logits.sigmoid().max(dim=-1)
            pred_boxes_xyxy = box_convert(pred_boxes, 'cxcywh', 'xyxy') * scale_tensor
            
            # Filter low confidence predictions to speed up mAP calculation
            keep = pred_scores > 0.05
            preds_list.append({
                "boxes": pred_boxes_xyxy[keep],
                "scores": pred_scores[keep],
                "labels": pred_labels[keep]
            })
            
            # Convert target logic
            tgt_boxes_xyxy = box_convert(targets[i]['boxes'], 'cxcywh', 'xyxy') * scale_tensor
            targets_list.append({
                "boxes": tgt_boxes_xyxy,
                "labels": targets[i]['labels']
            })
            
        map_metric.update(preds_list, targets_list)

    avg_val_loss = epoch_loss / len(dataloader)
    
    # Compute final mAP metric
    map_dict = map_metric.compute()
    val_map = map_dict['map'].item()
    
    writer.add_scalar('Loss/Val_Epoch', avg_val_loss, epoch)
    writer.add_scalar('Metrics/mAP_Val', val_map, epoch)
    
    map_metric.reset()

    return avg_val_loss, val_map


def main():
    # Optimization: Set modern TF32 execution for PyTorch >= 1.12
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    log_dir = os.path.join("runs", "detr_digit_detection")
    writer = SummaryWriter(log_dir=log_dir)

    print("Setting up datasets...")
    train_loader, val_loader = setup_dataloaders()

    print("Instantiating model...")
    num_classes = 10
    model = DigitDETR(num_classes=num_classes)
    model.to(device)

    ema_avg_fn = get_ema_multi_avg_fn(0.999)
    ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)

    matcher = FocalHungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    
    # OPTIMIZATION: Deep Supervision / Auxiliary Loss weights added
    base_weight_dict = {'loss_ce': 2.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
    weight_dict = base_weight_dict.copy()
    
    # DETR has 6 decoder layers, meaning 5 auxiliary outputs (index 0 to 4)
    aux_weight_dict = {}
    for i in range(5):
        aux_weight_dict.update({f'{k}_{i}': v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    criterion = DeformableSetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=['labels', 'boxes', 'cardinality']
    )
    criterion.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)

    num_epochs = 300
    
    # Warmup + StepLR Schedulers
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
    main_scheduler = StepLR(optimizer, step_size=200, gamma=0.1)
    lr_scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[5]
    )

    # Initialize the AMP GradScaler with the updated syntax
    scaler = GradScaler('cuda')

    # Replaced best_val_loss with best_map tracking
    best_map = 0.0
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model=model,
            ema_model=ema_model,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer,
            scaler=scaler
        )

        # Evaluate using the EMA model, tracking mAP
        val_loss, val_map = evaluate(
            ema_model=ema_model,
            criterion=criterion,
            dataloader=val_loader,
            device=device,
            epoch=epoch,
            writer=writer
        )

        lr_scheduler.step()

        writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
        writer.add_scalar('Hyperparameters/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        print(f"Epoch {epoch} Summary - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}")

        # Checkpoint is now saved based on best mAP, not lowest validation loss
        if val_map > best_map:
            best_map = val_map
            checkpoint_path = os.path.join(save_dir, "detr_best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_map': val_map,
            }, checkpoint_path)
            print(f"--> Saved new best model checkpoint (mAP: {val_map:.4f}) to {checkpoint_path}")

    writer.close()


if __name__ == '__main__':
    main()