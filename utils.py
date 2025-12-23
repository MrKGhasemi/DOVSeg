import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
from config import configs


def filter_synonyms(nouns, nlp, threshold=None):
    """
    Filters a list of nouns, removing synonyms based on vector similarity.
    """
    if threshold is None:
        threshold = configs.SYNONYM_THRESHOLD
        
    noun_tokens = [nlp(noun) for noun in nouns]
    final_nouns = []
    
    for i in range(len(noun_tokens)):
        token_a = noun_tokens[i]
        
        if token_a.vector_norm == 0:
            if token_a.text not in final_nouns:
                final_nouns.append(token_a.text)
            continue
            
        is_synonym = False
        for j in range(len(final_nouns)):
            token_b = nlp(final_nouns[j])
            if token_b.vector_norm == 0:
                continue
                
            similarity = token_a.similarity(token_b)
            if similarity > threshold:
                is_synonym = True
                print(f"--- Filtering '{token_a.text}' (similar to '{token_b.text}', score: {similarity:.2f})")
                break
                
        if not is_synonym:
            final_nouns.append(token_a.text)
            
    return final_nouns

def normalize_person_classes(nouns):
    """
    Merges all person-related nouns into a single 'person' class.
    """
    replace_list = configs.PERSON_REPLACE_LIST
    final_nouns = []
    person_added = False
    for noun in nouns:
        if noun in replace_list:
            if not person_added:
                final_nouns.append('person')
                person_added = True
        else:
            final_nouns.append(noun)
    return final_nouns

def parse_and_clean_nouns(raw_text, nlp):
    """
    Full NLP pipeline: parses text, extracts nouns, filters, and normalizes.
    """
    doc = nlp(raw_text)
    
    # Extract head nouns
    all_noun_phrases = []
    for chunk in doc.noun_chunks:
        head = chunk.root.lemma_.lower()
        if head.isalpha() and head not in {"a", "an", "the", "color", "thing", "someone", "something"}:
            all_noun_phrases.append(head)
            
    # Simple de-duplication
    seen = set()
    raw_noun_list = [x for x in all_noun_phrases if not (x in seen or seen.add(x))]
    
    # Semantic de-duplication
    print(f"Raw nouns: {raw_noun_list}")
    filtered_nouns = filter_synonyms(raw_noun_list, nlp)
    
    # Normalize person classes
    final_nouns = normalize_person_classes(filtered_nouns)
    
    print(f"Parsed clean object nouns: {final_nouns}")
    return final_nouns

def get_image_crops(raw_image):
    """
    Splits the image into 25 crops (16 patch, 4 quadrants, 2 Horizontal halves, 2 Vertical halves, 1 full)
    """
    x, y = raw_image.size
    p1 = raw_image.crop(box=(0, 0, x / 2, y / 2))
    p2 = raw_image.crop(box=(x / 2, 0, x, y / 2))
    p3 = raw_image.crop(box=(0, y / 2, x / 2, y))
    p4 = raw_image.crop(box=(x / 2, y / 2, x, y))
    p5 = raw_image.crop(box=(0, 0, x / 2, y))
    p6 = raw_image.crop(box=(x / 2, 0, x, y))
    p7 = raw_image.crop(box=(0, 0, x, y / 2))
    p8 = raw_image.crop(box=(0, y / 2, x, y))
    images = [p1, p2, p3, p4, p5, p6, p7, p8, raw_image]    
    x_step = x / 4
    y_step = y / 4
    for i in range(4): 
        for j in range(4):
            x1 = j * x_step
            y1 = i * y_step
            x2 = (j + 1) * x_step
            y2 = (i + 1) * y_step
            patch = raw_image.crop(box=(x1, y1, x2, y2))
            images.append(patch)
            
    return images

def encode_image(image_path):
    """Encodes an image file in Base64 for API calls."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _create_maps(image, classified_masks, class_colors):
    """Internal helper to generate the visualization images."""
    output_image_with_labels = image.copy()
    color_overlay = output_image_with_labels.copy()
    semantic_map = np.zeros(image.shape[:2], dtype=np.uint16)
    
    image_height = image.shape[0]
    font_scale = max(0.4, image_height / 1200.0)
    font_thickness = max(1, int(font_scale * 2))
    
    # Draw masks
    for mask_data in sorted(classified_masks, key=lambda x: x.get('area', 0), reverse=True):
        class_id = mask_data.get('class_id', 0)
        color = class_colors[class_id]
        
        mask = mask_data['segmentation'].astype(np.uint8)
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        color_overlay[mask_resized] = color
        semantic_map[mask_resized] = class_id 
    
    # Blend instance map
    alpha = configs.MASK_ALPHA
    output_image_with_labels = cv2.addWeighted(color_overlay, alpha, output_image_with_labels, 1 - alpha, 0)
    
    # Draw labels
    for mask_data in classified_masks:
        class_id = mask_data['class_id']
        class_name = mask_data['class_name']
        
        mask = mask_data['segmentation'].astype(np.uint8)
        mask_resized_for_centroid = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        M = cv2.moments(mask_resized_for_centroid)
        if M["m00"] == 0:
            bbox = mask_data['bbox']
            center_x = int(bbox[0] + (bbox[2] - bbox[0]) / 2)
            center_y = int(bbox[1] + (bbox[3] - bbox[1]) / 2)
        else:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

        # Get text size using the new dynamic scale
        text_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size
        
        text_x = center_x - text_w // 2
        text_y = center_y + text_h // 2
        
        padding = int(font_scale * 5) # Scale padding with font
        
        rect_x1 = text_x - padding
        rect_y1 = text_y - text_h - padding
        rect_x2 = text_x + text_w + padding
        rect_y2 = text_y + padding
        
        cv2.rectangle(output_image_with_labels, (rect_x1, rect_y1), (rect_x2, rect_y2), class_colors[class_id], -1)
        cv2.putText(output_image_with_labels, class_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        
    # Create solid color semantic map
    semantic_color_map = np.zeros_like(image)
    for i, color in enumerate(class_colors):
        semantic_color_map[semantic_map == i] = color
        
    return output_image_with_labels, semantic_color_map

def save_visualizations(output_image_with_labels, semantic_color_map, output_path_prefix):
    """Saves instance and semantic maps to disk."""
    instance_map_bgr = cv2.cvtColor(output_image_with_labels, cv2.COLOR_RGB2BGR)
    semantic_map_bgr = cv2.cvtColor(semantic_color_map, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(f"{output_path_prefix}_instance.jpg", instance_map_bgr)
    cv2.imwrite(f"{output_path_prefix}_semantic.png", semantic_map_bgr)
    print(f"Saved visualizations to {output_path_prefix}_*.jpg/png")

def show_anns(anns):
    """Helper for plotting masks in a notebook."""
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_visualizations(image_rgb, classified_masks, class_colors):
    """Displays instance and semantic maps in a notebook."""
    instance_map, semantic_map = _create_maps(image_rgb, classified_masks, class_colors)
    
    plt.figure(figsize=(30, 20))
    plt.subplot(3,1,1)
    plt.imshow(image_rgb)
    plt.title("raw image")
    plt.axis('off')
    
    plt.subplot(3,1,2)
    plt.imshow(instance_map)
    plt.title("Instance Labels and Masks")
    plt.axis('off')
    
    plt.subplot(3,1,3)
    plt.imshow(semantic_map)
    plt.title("Semantic Map")
    plt.axis('off')
    
    plt.subplots_adjust(hspace=0.15)
    

    plt.show()

