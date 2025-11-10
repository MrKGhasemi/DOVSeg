import torch
import utils 
from openai import OpenAI
from config import configs

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

def get_classes_llm(image_path, models, api_key, base_url):
    """
    Generates noun list using the external LLM API.
    """
    nlp = models['nlp']
    
    print("Generating classes with external LLM...")
    try:
        image_data = utils.encode_image(image_path)
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        respond = client.chat.completions.create(
            model=configs.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": configs.LLM_SYSTEM_PROMPT},
                {"role": "user", "content": configs.LLM_USER_PROMPT},
                {"role": "user", "content": {"type": "image_url", "image_url": f'data:image/png;base64,{image_data}'}}
            ]
        )
        classes_str = respond.choices[0].message.content
        print(f"LLM Response: '{classes_str}'")
        
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return [], ""
        
    clean_nouns = utils.parse_and_clean_nouns(classes_str, nlp)
    text_prompt = ". ".join(clean_nouns)
    
    return clean_nouns, text_prompt