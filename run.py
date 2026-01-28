# run.py
import os
import argparse
import sys
import numpy as np
import cv2
import torch
from PIL import Image
from groundingdino.util.inference import load_image, predict, preprocess_caption
import torchvision

from config import configs
import models
import utils
import class_generators


def process_image(image_path, models, args):
    """
    Runs the full segmentation pipeline on a single image.
    """
    print(f"\n--- Processing {image_path} ---")
    try:
        image_data = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        raw_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None

    # Get Class Names
    if args.mode == "blip":
        noun_phrases, text_prompt = class_generators.get_classes_blip(
            raw_image, models, models['device']
        )
    elif args.mode == "llm":
        noun_phrases, text_prompt = class_generators.get_classes_llm(
            raw_image, models, args.api_key, args.base_url
        )

    if not noun_phrases:
        print("Warning: No class names found. Skipping image.")
        return None, None, None

    # Setup Dynamic Classes
    CLASS_COLORS = [(np.random.randint(0, 255), np.random.randint(
        0, 255), np.random.randint(0, 255)) for _ in noun_phrases]
    noun_phrases_with_bg = ["background"] + noun_phrases
    CLASS_COLORS_with_bg = [(0, 0, 0)] + CLASS_COLORS
    CLASS_NAME_TO_ID = {name: i for i, name in enumerate(noun_phrases_with_bg)}

    # GroundingDINO Prediction
    _, image_tensor = load_image(image_path)
    caption = preprocess_caption(caption=text_prompt)

    boxes, logits, phrases = predict(
        model=models['grounding_dino_model'],
        image=image_tensor,
        caption=caption,
        box_threshold=configs.BOX_THRESHOLD,
        text_threshold=configs.TEXT_THRESHOLD,
        device=models['device']
    )

    print(f'Pre-NMS prediction: {logits}')
    print(f'Pre-NMS phrases({len(phrases)}): {phrases}')

    if boxes.size(0) == 0:
        print("Warning: Grounding DINO found no objects. Skipping image.")
        return None, None, None

    # NMS
    boxes_cxcywh = boxes
    cx = boxes_cxcywh[:, 0]
    cy = boxes_cxcywh[:, 1]
    w = boxes_cxcywh[:, 2]
    h = boxes_cxcywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    boxes_xyxy_norm = torch.stack([x1, y1, x2, y2], dim=1)

    nms_indices = torchvision.ops.nms(
        boxes_xyxy_norm,
        logits,
        iou_threshold=configs.NMS_THRESHOLD
    )

    boxes_xyxy_norm_after_nms = boxes_xyxy_norm[nms_indices]
    logits_after_nms = logits[nms_indices]
    phrases_after_nms = [phrases[i] for i in nms_indices.cpu().numpy()]

    print(f"Post-NMS phrases({len(phrases_after_nms)}): {phrases_after_nms}")

    print(f"Post-NMS phrases: {phrases_after_nms}")
    # SAM Prediction
    models['sam_predictor'].set_image(image_rgb)
    img_h, img_w = image_rgb.shape[:2]

    denorm_array = torch.tensor([img_w, img_h, img_w, img_h], device=models['device'])
    boxes_xyxy_norm_after_nms = boxes_xyxy_norm_after_nms.to(models['device'])
    boxes_for_sam = boxes_xyxy_norm_after_nms * denorm_array

    if boxes_for_sam.size(0) == 0:
        print("Warning: No objects remained after NMS. Skipping image.")
        return None, None, None

    transformed_boxes = models['sam_predictor'].transform.apply_boxes_torch(
        boxes_for_sam, image_rgb.shape[:2])

    sam_masks, _, _ = models['sam_predictor'].predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # Mask Processing
    masks = []
    for i in range(sam_masks.shape[0]):
        mask_np = sam_masks[i].squeeze().cpu().numpy()
        bbox_tensor = boxes_for_sam[i]

        class_name = phrases_after_nms[i]
        confidence = logits_after_nms[i].item()

        if confidence > configs.MASK_CONFIDENCE_THRESHOLD:
            matched_id = None
            for noun in CLASS_NAME_TO_ID.keys():
                if noun in class_name.lower():
                    matched_id = CLASS_NAME_TO_ID[noun]
                    class_name = noun
                    break

            if matched_id is None:
                print(f"Warning: '{class_name}' not found in CLASS_NAME_TO_ID. Skipping.")
                continue

            if matched_id == 0:
                continue

            masks.append({
                'segmentation': mask_np,
                'bbox': bbox_tensor.cpu().tolist(),
                'area': int(mask_np.sum()),
                'score': confidence,
                'class_name': class_name,
                'class_id': matched_id
            })

    print(f"Generated {len(masks)} high-quality masks.")

    if not masks:
        print("No masks passed the confidence and matching filter.")
        return None, None, None

    # Save Visualizations
    base_filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_filename)[0]
    output_prefix = os.path.join(args.output, name_without_ext)

    if configs.SAVE_FIGURES:
        print("Generating visualization maps...")
        instance_map, semantic_map = utils._create_maps(image_rgb, masks, CLASS_COLORS_with_bg)
        utils.save_visualizations(instance_map, semantic_map, output_prefix, args.mode)

    return masks, image_rgb, CLASS_COLORS_with_bg


def main():
    parser = argparse.ArgumentParser(description="Grounded-SAM Segmentation Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["blip", "llm"],
                        help="Mode for generating class names ('blip' or 'llm').")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a single image or a directory of images.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output directory to save visualizations.")

    # LLM-specific arguments
    parser.add_argument("--api-key", type=str, default=os.environ.get("AVALAI_API_KEY"),
                        help="API key for the LLM. Defaults to AVALAI_API_KEY env var.")
    parser.add_argument("--base-url", type=str, default="https://api.avalai.ir/v1",
                        help="Base URL for the LLM API.")

    args = parser.parse_args()

    # Argument Validation
    if args.mode == "llm" and not args.api_key:
        print("Error: --mode 'llm' requires --api-key")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input path not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dict = models.load_all_models(device)

    # Find Images
    image_paths = []
    if os.path.isdir(args.input):
        for fname in os.listdir(args.input):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(args.input, fname))
    elif os.path.isfile(args.input):
        image_paths.append(args.input)

    print(f"Found {len(image_paths)} images to process.")

    # --- Run Pipeline ---
    for image_path in image_paths:
        process_image(image_path, models_dict, args)


if __name__ == "__main__":
    main()
