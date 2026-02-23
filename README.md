# DOVSeg: Dynamic Open-Vocabulary Semantic Segmentation Pipeline

![DOVSeg Example Output](./assets/output.png)
---
This project implements a flexible pipeline for open-set semantic segmentation. It dynamically identifies objects in an image using a captioning model (BLIP) or a generative LLM, and then uses [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) to generate precise segmentation masks for those objects.

The core workflow is:

* **Mode 1 (BLIP):** `Image` $\rightarrow$ `Captions (from 9 crops)` $\rightarrow$ `spaCy` $\rightarrow$ `Clean Noun List` $\rightarrow$ `GroundingDINO` $\rightarrow$ `Boxes` $\rightarrow$ `SAM` $\rightarrow$ `Masks`
* **Mode 2 (LLM):** `Image` $\rightarrow$ `Clean Noun List (from API)` $\rightarrow$ `GroundingDINO` $\rightarrow$ `Boxes` $\rightarrow$ `SAM` $\rightarrow$ `Masks`

---

## Core Components & Models

This pipeline is built by combining several state-of-the-art models:

1.  **Class Generation (2 Modes):**

     **`blip` mode:** Uses **BLIP-Large** (`Salesforce/blip-image-captioning-large`) to generate 9 captions (from 4 quadrants, 4 halves, and 1 full image).
     
     **`llm` mode:** Uses an external API (configured for `gemini-2.0-flash` via `api.avalai.ir`) to generate a comma-separated list of objects.

2.  **Noun Filtering:**

    **spaCy** (`en_core_web_lg`): Parses the generated text, extracts key nouns, and intelligently filters synonyms based on vector similarity to create a clean, unique list of objects for detection.

3.  **Object Detection:**

    **GroundingDINO:** A powerful open-set detector that finds bounding boxes for any given text prompt (the class list from the previous step).

4.  **Segmentation:**

    **Segment Anything Model (SAM):** Generates high-quality segmentation masks for the bounding boxes provided by GroundingDINO.

---

## Getting Started

Follow these steps to set up the project and download all necessary models and weights.

### 1. Clone Repositories

This project assumes you have local clones of GroundingDINO and SAM, as defined in `config.py`.

```bash
# Clone this repository
git clone https://github.com/MrKGhasemi/DOVSeg.git
cd DOVSeg

# 1. Clone GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git

# 2. Clone Segment Anything
git clone https://github.com/facebookresearch/segment-anything.git
```
2. Set Up Python Environment
recommended to use a Conda environment.
```bash
# Create and activate the environment
conda create -n git python=3.10
conda activate git

# Install all Python packages from requirements.txt
pip install -r requirements.txt

# Install the spaCy model
python -m spacy download en_core_web_lg

# Install Segment Anything
pip install git+[https://github.com/facebookresearch/segment-anything.git]
```
3. Download Model Weights
Download the checkpoint files for `GroundingDINO` and `SAM`.

*`GroundingDINO (Swin-T)`*:
Download `groundingdino_swint_ogc.pth` from this [link](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) and this [link](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth).

*`SAM (ViT-H)`*:
Download `sam_vit_h_4b8939.pth` from this [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

Place in the `weights` directory.

4. Configure Your Paths
In config.py, update the file paths to match where you cloned the repos and downloaded the weights.
```bash
# config.py
GROUNDING_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"
```
---
## Usage
You can run the pipeline either from the command line (for batch processing) or an interactive notebook (for testing).

1. Command Line (CLI)
The run.py script is the main entry point for processing images.

Mode 1: `blip` runs entirely on your local machine.

--mode `blip`: Uses the `BLIP-Large` + `spaCy` pipeline.

--`input`: Can be a path to a single image or a directory of images.

--`output`: The directory where segmented images will be saved.

Mode 2: llm (API-based).

This mode requires an API key.

```bash
python run.py --mode llm --input ./image.jpg --output ./results --api-key "YOUR_API_KEY_HERE"
```
--`mode` `llm`: Uses the external LLM to get the class list.

--`api-key`: Your secret API key.

2. Jupyter Notebook
The `example.ipynb` file provides a step-by-step walkthrough of the entire pipeline, showing the outputs of each model.

Update the `IMAGE_PATH` variable in the second code cell to point to your test image.

Run the cells one by one to see the models load, captions generate, and visualizations appear.
```bash
python run.py --mode blip --input ./path/to/your/images --output ./results
```
