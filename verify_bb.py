import json

def validate_coco_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    invalid_boxes = 0
    categories = set()
    
    for ann in data.get('annotations', []):
        bbox = ann.get('bbox')
        cat_id = ann.get('category_id')
        
        categories.add(cat_id)
        
        if bbox:
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                invalid_boxes += 1
                
    print(f"Validation for {json_path}:")
    print(f"Unique category IDs: {sorted(list(categories))}")
    print(f"Number of annotations with width or height <= 0: {invalid_boxes}")

if __name__ == "__main__":
    validate_coco_json('data/train.json')