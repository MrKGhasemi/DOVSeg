import torch
import utils
from openai import OpenAI
from config import configs
import io
import base64
import time
import random
from PIL import Image


def get_classes_blip(raw_image, models, device="cuda"):
    """
    Generates noun list using BLIP-Large captioning on 9 image crops.
    """
    blip_model = models['blip_model']
    blip_processor = models['blip_processor']
    nlp = models['nlp']

    images = utils.get_image_crops(raw_image)

    all_captions = []

    print(f"Generating {len(images)} captions with BLIP-Large...")
    for i, img_crop in enumerate(images):
        blip_inputs = blip_processor(images=img_crop, return_tensors="pt").to(device, torch.float16)

        with torch.no_grad():
            blip_out = blip_model.generate(**blip_inputs, max_new_tokens=configs.BLIP_MAX_TOKENS)

        caption_text = blip_processor.decode(blip_out[0], skip_special_tokens=True)
        print(f"BLIP Caption {i}: '{caption_text}'")
        all_captions.append(caption_text)

    full_caption_text = " . ".join(all_captions)

    clean_nouns = utils.parse_and_clean_nouns(full_caption_text, nlp)

    text_prompt = ". ".join(clean_nouns)

    return clean_nouns, text_prompt


def get_classes_llm(raw_image, models, api_key, base_url):
    """
    Generates noun list using the external LLM API.
    Refined to prevent hallucinations by resizing images and forcing correct JSON structure.
    """
    nlp = models['nlp']

    # 1. Get Crops from the raw PIL image
    images = utils.get_image_crops(raw_image)
    all_captions = []
    print(f"Generating classes with external LLM ({len(images)} crops)...")

    client = OpenAI(api_key=api_key, base_url=base_url)

    for i, img_crop in enumerate(images):
        try:
            # --- CRITICAL FIXES FROM COPY FILE ---

            # 1. Resize to prevent API timeouts/rejections (Max 512x512)
            img_crop.thumbnail((512, 512))

            # 2. Force JPEG Encoding in memory
            buffered = io.BytesIO()
            img_crop.save(buffered, format="JPEG", quality=75)
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{base64_image}"

            # 3. Add random seed to system prompt to prevent caching previous hallucinations
            current_time = str(time.time())
            random_seed = str(random.randint(0, 10000))

            respond = client.chat.completions.create(
                model=configs.LLM_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": f"{configs.LLM_SYSTEM_PROMPT} [Ref: {current_time}-{random_seed}]. If you cannot see the image, reply with 'NO_IMAGE'."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": configs.LLM_USER_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                    "detail": "low"  # Forces low detail to save tokens/bandwidth
                                }
                            }
                        ]
                    }
                ],
                temperature=0.2,
                max_tokens=500
            )

            respond_crop = respond.choices[0].message.content

            # Check if LLM explicitly says it can't see the image
            if "NO_IMAGE" in respond_crop:
                print(f"LLM Warning (Crop {i}): Image rejected by API.")
                continue

            print(f"LLM Response (Crop {i}): '{respond_crop}'")

        except Exception as e:
            print(f"Error calling LLM API for crop {i}: {e}")
            continue

        all_captions.append(respond_crop)

    if not all_captions:
        print("Error: No captions generated.")
        return [], ""

    full_response = " . ".join(all_captions)
    clean_nouns = utils.parse_and_clean_nouns(full_response, nlp)
    text_prompt = ". ".join(clean_nouns)

    return clean_nouns, text_prompt
