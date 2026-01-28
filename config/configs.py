# Models
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
SPACY_MODEL = "en_core_web_lg"

GROUNDING_DINO_CONFIG_PATH = "config/GroundingDINO_SwinB_cfg.py"  # or "config/GroundingDINO_SwinT_OGC.py"
# or "weights/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CHECKPOINT_PATH = "weights/groundingdino_swinb_cogcoor.pth"
SAM_CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"
SAM_ENCODER_VERSION = "vit_h"

# Detection Thresholds
BOX_THRESHOLD = 0.1
TEXT_THRESHOLD = 0.1
NMS_THRESHOLD = 0.7
MASK_CONFIDENCE_THRESHOLD = 0.1

# NLP & Prompting
SYNONYM_THRESHOLD = 0.7
BLIP_MAX_TOKENS = 50
PERSON_REPLACE_LIST = ['man', 'woman', 'child', 'boy', 'girl', 'people']

# LLM API Settings (for 'llm' mode)
LLM_MODEL_NAME = "gemini-3-flash-preview"
LLM_SYSTEM_PROMPT = """You are an object detection assistant. 
generate a detailed caption for the provided image, describing all distinct, trackable physical objects in the foreground (e.g., person, car, ball). 
Ignore background scenery (sky, ground, wall, building), clothing parts (pocket, sleeve), and abstract concepts.
Return ONLY the caption without any additional text."""
LLM_USER_PROMPT = "give the caption for this image including all objects"

# Visualization & Output
SAVE_FIGURES = True
MASK_ALPHA = 0.5  # Transparency
