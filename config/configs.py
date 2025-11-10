# Models
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
SPACY_MODEL = "en_core_web_lg"

GROUNDING_DINO_CONFIG_PATH = "config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"
SAM_ENCODER_VERSION = "vit_h"

# Detection Thresholds
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8
MASK_CONFIDENCE_THRESHOLD = 0.2 # The `if confidence > 0.3` check

# NLP & Prompting
SYNONYM_THRESHOLD = 0.8
BLIP_MAX_TOKENS = 50
PERSON_REPLACE_LIST = ['man', 'woman', 'child', 'boy', 'girl', 'people']

# LLM API Settings (for 'llm' mode)
LLM_MODEL_NAME = "gemini-2.0-flash"
LLM_SYSTEM_PROMPT = """You are an image captioner model; at starting point of a semantic segmenation workflow. 
you should deeply look at the image and find all destinct and unique objects inside;  
these names will give to Grounding_DINO and SAM to find place of object and then segment them. 
just bring the saparated-camma list of 'class_names' you found. nothing else.
if you found human inside, mention as 'person'."""
LLM_USER_PROMPT = "bring objects list inside the image. "

# Visualization & Output
SAVE_FIGURES = True
MASK_ALPHA = 0.5 # Transparency
