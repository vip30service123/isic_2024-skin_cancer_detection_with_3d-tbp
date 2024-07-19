import io
import json

import numpy as np
from PIL import Image
import requests
import streamlit as st
import torch

from src.dataset.torch.transforms import DefaultTransform
from src.models.torch.resnet50 import Resnet50


model = Resnet50()
model.load_state_dict(torch.load("saved_models/torch/torch_resnet50_2024-07-18-22_20_02.773146.pt"))


transform = DefaultTransform()

if 'model' not in st.session_state:
    st.session_state['model'] = model

if "transform" not in st.session_state:
    st.session_state['transform'] = transform

file_image = st.file_uploader("Choose an image.")

if file_image:
    st.image(file_image)

    im = Image.open(io.BytesIO(file_image.getvalue()))
    pix = np.array(im)
    pix = transform(pix).unsqueeze(dim=0)
    print(pix)
    output = model(image=pix)
    no, yes = tuple(output['logits'][0].tolist())
    st.write(f"No: {no:0.2f}")
    st.write(f"Yes: {yes:0.2f}")

if file_image:
    res = requests.post(url="http://127.0.0.1:8000/classify", data=file_image)

    print(res)