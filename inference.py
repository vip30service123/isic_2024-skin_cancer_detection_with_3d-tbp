import io

import h5py
import numpy as np
from PIL import Image
import torch
import torchvision
import streamlit as st

from src.dataset.torch.transforms import DefaultTransform


model = torch.load("saved_models/torch/torch_resnet50_2024-07-18-17_12_32.311898.pt")


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

    output = model(image=pix)

    no, yes = tuple(output['logits'][0].tolist())

    st.write(f"No: {no:0.2f}")
    st.write(f"Yes: {yes:0.2f}")
