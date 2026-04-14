import json
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class DigitCocoDataset(Dataset):
    """
    Custom Dataset class to parse COCO-format JSON files and prepare 
    the data specifically for the DETR architecture.
    """
    def __init__(self, image_dir: str, annotation_file: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
            
        self.images = {img['id']: img for img in coco_data['images']}
        
        self.img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        self.image_ids = list(self.images.keys())
        
        # Optimization: Pre-compute targets to eliminate per-epoch processing overhead
        self.precomputed_targets = {}
        for img_id in self.image_ids:
            img_info = self.images[img_id]
            w = img_info['width']
            h = img_info['height']
            
            anns = self.img_to_anns.get(img_id, [])
            boxes = []
            labels = []
            
            for ann in anns:
                x_min, y_min, box_w, box_h = ann['bbox']
                
                center_x = (x_min + box_w / 2.0) / w
                center_y = (y_min + box_h / 2.0) / h
                norm_w = box_w / w
                norm_h = box_h / h
                
                boxes.append([center_x, center_y, norm_w, norm_h])
                labels.append(ann['category_id'] - 1)
                
            self.precomputed_targets[img_id] = {
                'boxes': boxes,
                'labels': labels,
                'image_id': [img_id]
            }

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        image = Image.open(img_path).convert("RGB")
        
        target_data = self.precomputed_targets[img_id]
        target = {
            'boxes': torch.tensor(target_data['boxes'], dtype=torch.float32),
            'labels': torch.tensor(target_data['labels'], dtype=torch.int64),
            'image_id': torch.tensor(target_data['image_id'], dtype=torch.int64)
        }
        
        if self.transform:
            image = self.transform(image)
            
        return image, target


def custom_collate_fn(batch: list) -> tuple:
    """
    Pairs images and target dictionaries into tuples instead of stacking.
    This allows batches of dynamically-sized images to be passed safely.
    """
    return tuple(zip(*batch))


def setup_dataloaders():
    train_img_dir = "./data/train"
    train_json = "./data/train.json"
    
    valid_img_dir = "./data/valid"
    valid_json = "./data/valid.json"
    
    # Adjusted scales: Lowered range for originally small (100-200px) images
    # to avoid excessive blurriness while maintaining a sufficient feature map size
    scales = [256, 288, 320, 352, 384, 416, 448, 480]
    
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomChoice([
            transforms.Resize(size, max_size=800) for size in scales
        ]), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Use 400 for consistent validation evaluation, balancing resolution and speed
    valid_transform = transforms.Compose([
        transforms.Resize(400, max_size=800),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = DigitCocoDataset(
        image_dir=train_img_dir, 
        annotation_file=train_json, 
        transform=train_transform
    )
    
    val_dataset = DigitCocoDataset(
        image_dir=valid_img_dir, 
        annotation_file=valid_json, 
        transform=valid_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=8,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=8,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    return train_loader, val_loader