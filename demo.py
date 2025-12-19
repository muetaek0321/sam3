import os

from PIL import Image
import matplotlib.pyplot as plt
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("data/2007_007414.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="pylon")

plot_results(image, output)
plt.show()
plt.close()
