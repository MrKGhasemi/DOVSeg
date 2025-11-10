import spacy
from transformers import BlipProcessor, BlipForConditionalGeneration
from groundingdino.util.inference import load_model as load_grounding_dino_model
from segment_anything import sam_model_registry, SamPredictor
from config import configs

def load_blip_model(device="cuda"):
    """
    Loads the BLIP-Large captioning model.
    """
    print("Loading BLIP-Large model...")
    blip_model_id = configs.BLIP_MODEL_ID
    blip_processor = BlipProcessor.from_pretrained(blip_model_id)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_id).to(device)
    blip_model.eval()
    print("BLIP-Large model loaded.")
    return blip_model, blip_processor

def load_grounding_dino(device="cuda"):
    """
    Loads the GroundingDINO model from local paths.
    """
    print("Loading Grounding DINO model...")
    grounding_dino_model = load_grounding_dino_model(
        configs.GROUNDING_DINO_CONFIG_PATH, 
        configs.GROUNDING_DINO_CHECKPOINT_PATH
    )
    grounding_dino_model.to(device)
    print("Grounding DINO model loaded.")
    return grounding_dino_model

def load_sam_model(device="cuda"):
    """
    Loads the SAM model from a local path.
    """
    print("Loading SAM model...")
    sam = sam_model_registry[configs.SAM_ENCODER_VERSION](
        checkpoint=configs.SAM_CHECKPOINT_PATH
    )
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    print("SAM model loaded.")
    return sam_predictor

def load_spacy_model():
    """
    Loads the large spaCy model, downloading if necessary.
    """
    model_name = configs.SPACY_MODEL
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model: {model_name}")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        nlp = spacy.load(model_name)
    print("spaCy model loaded.")
    return nlp

def load_all_models(device="cuda"):
    """
    Loads and returns all models in a dictionary.
    """
    blip_model, blip_processor = load_blip_model(device)
    return {
        "device": device,
        "blip_model": blip_model,
        "blip_processor": blip_processor,
        "grounding_dino_model": load_grounding_dino(device),
        "sam_predictor": load_sam_model(device),
        "nlp": load_spacy_model()
    }