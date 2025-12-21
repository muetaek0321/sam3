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

# モデルの準備
model = build_sam3_image_model()
processor = Sam3Processor(model)
# 画像の読み込み
image = Image.open("data/1624777685449_985774_photo1.jpeg")
inference_state = processor.set_image(image)
# テキストプロンプトを設定して推論を実行
output = processor.set_text_prompt(state=inference_state, prompt="tomato")

plot_results(image, output)
plt.show()
plt.close()
