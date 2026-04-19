import json
import random
import os
from PIL import Image, ImageDraw

def visualize_predictions(test_dir, json_path, num_samples=10):
    with open(json_path, 'r') as f:
        preds = json.load(f)
        
    # Group predictions by image_id
    pred_dict = {}
    for p in preds:
        img_id = p['image_id']
        if img_id not in pred_dict:
            pred_dict[img_id] = []
        pred_dict[img_id].append(p)
        
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    for filename in sample_files:
        img_id = int(os.path.splitext(filename)[0])
        img_path = os.path.join(test_dir, filename)
        
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        if img_id in pred_dict:
            for p in pred_dict[img_id]:
                # Only draw high confidence boxes for visualization
                if p['score'] > 0.3:  
                    x, y, w, h = p['bbox']
                    label = p['category_id']
                    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
                    draw.text((x, y), str(label), fill="red")
                    
        # img.show()
        img.save(f"debug_{filename}")

visualize_predictions("data/test", "pred.json")