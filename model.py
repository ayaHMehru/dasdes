import torch
import torch.nn as nn
import segmentation_models_pytorch as smp  # Pastikan Anda sudah menginstal paket ini

class UNetMobileNet(nn.Module):
    def __init__(self, num_classes=21):
        super(UNetMobileNet, self).__init__()
        # Inisialisasi model dari segmentation_models_pytorch
        self.model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=num_classes)

    def forward(self, x):
        return self.model(x)
