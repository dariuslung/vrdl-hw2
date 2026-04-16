def generate_predictions(model, test_img_dir, output_file="pred.json"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    transform = get_transforms()
    predictions = []
    
    for img_name in os.listdir(test_img_dir):
        # Extract image_id from filename depending on your dataset structure
        # Assuming filename is something like "00001.jpg"
        image_id = int(os.path.splitext(img_name)[0])
        img_path = os.path.join(test_img_dir, img_name)
        
        orig_image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = orig_image.size
        
        image_np = np.array(orig_image)
        # Apply transform without bboxes
        transformed = transform(image=image_np)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values=input_tensor)
            
        # Post-process to original image size
        # Albumentations padded the image. We need to account for the scale factor.
        scale = 512 / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.5 # Confidence threshold
        
        boxes = outputs.pred_boxes[0, keep].cpu().numpy()
        scores = probas[keep].max(-1).values.cpu().numpy()
        labels = probas[keep].argmax(-1).cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            cx, cy, w, h = box
            
            # Convert normalized 512x512 coordinates to absolute padded coordinates
            abs_cx, abs_cy = cx * 512, cy * 512
            abs_w, abs_h = w * 512, h * 512
            
            # Map back to original unpadded dimensions
            orig_cx = abs_cx / scale
            orig_cy = abs_cy / scale
            orig_box_w = abs_w / scale
            orig_box_h = abs_h / scale
            
            x_min = orig_cx - (orig_box_w / 2)
            y_min = orig_cy - (orig_box_h / 2)
            
            predictions.append({
                "image_id": image_id,
                "bbox": [float(x_min), float(y_min), float(orig_box_w), float(orig_box_h)],
                "score": float(score),
                "category_id": int(label)
            })
            
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=4)