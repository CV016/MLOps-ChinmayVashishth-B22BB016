import streamlit as st
import gdown
import zipfile
import os
import torch
import cv2
import numpy as np
from train import train_pipeline
from model import UNet

def download_data(url, dest_folder="data"):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        zip_path = os.path.join(dest_folder, "dataset.zip")
        gdown.download(url, zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        os.remove(zip_path)

def inference(image, model, device):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 96), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
    return pred_mask

st.title("Cityscapes Segmentation Pipeline")

mode = st.sidebar.radio("Select Mode", ["Train Model", "Inference"])

if mode == "Train Model":
    drive_link = st.text_input("Enter Google Drive Dataset Link:")
    if st.button("Start Training"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with st.spinner("Downloading and extracting data..."):
            download_data(drive_link)
            
        with st.spinner(f"Training on {device} for 15 epochs..."):
            train_pipeline(data_dir="data/MLDLOPs_2026_Major_Exam", epochs=15, device=device)
            
        st.success("Training Complete! Model saved as unet_cityscapes.pth")
        st.image("training_metrics.png", caption="Training Metrics (mIoU, mDice, Loss)")

elif mode == "Inference":
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None and os.path.exists("unet_cityscapes.pth"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = UNet(in_channels=3, num_classes=23).to(device)
        model.load_state_dict(torch.load("unet_cityscapes.pth", map_location=device))
        model.eval()

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        mask = inference(image, model, device)
        
        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
        col2.image((mask * (255/23)).astype(np.uint8), caption="Predicted Mask", use_container_width=True, clamp=True)
    elif not os.path.exists("unet_cityscapes.pth"):
        st.warning("Please train the model first to generate unet_cityscapes.pth")