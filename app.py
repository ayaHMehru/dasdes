import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import UNetMobileNet

# Fungsi untuk memuat model
def load_model(model_path):
    # Inisialisasi model
    model = UNetMobileNet()  # Sesuaikan dengan model yang Anda gunakan

    # Muat state_dict dari file model
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Jika state_dict adalah objek model langsung, ekstrak state_dict-nya
    if isinstance(state_dict, torch.nn.Module):
        state_dict = state_dict.state_dict()

    # Set state_dict ke model
    model.model.load_state_dict(state_dict)
    model.eval()
    return model

# Fungsi untuk melakukan segmentasi pada gambar
def perform_segmentation(model, image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    segmentation_map = torch.argmax(output, dim=1).squeeze().numpy()
    return segmentation_map

def main():
    st.title("Aplikasi Segmentasi Semantik")

    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Dipilih", use_column_width=True)

        model_path = "Unet-Mobilenet.pt"  # Ganti sesuai dengan nama model Anda
        model = load_model(model_path)

        if st.button("Lakukan Segmentasi"):
            st.text("Sedang melakukan segmentasi...")
            segmentation_map = perform_segmentation(model, image)

            st.image(segmentation_map, caption="Hasil Segmentasi", use_column_width=True, cmap="viridis")

if __name__ == "__main__":
    main()
