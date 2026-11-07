from docsynth.generate import Generator
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

gen = Generator()

# First generate the documents
gen.generate()

# Then publish to Hugging Face
gen.publish_to_huggingface(
    repo_id="zachwel/MESA-Docsynth-Cancer",
    hf_token=HF_TOKEN
)