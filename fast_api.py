import io

from fastapi import FastAPI
import numpy as np
from PIL import Image
import torch

from src.dataset.torch.transforms import DefaultTransform


model = torch.load("saved_models/torch/torch_resnet50_2024-07-18-22_20_02.773146.pt")

transform = DefaultTransform()

app = FastAPI()

@app.post("/classify")
def classify(image):
    print(image)

    im = Image.open(io.BytesIO(image.getvalue()))
    pix = np.array(im)
    pix = transform(pix).unsqueeze(dim=0)
    output = model(image=pix)
    no, yes = tuple(output['logits'][0].tolist())

    return {
        "no": no,
        "yes": yes
    }


