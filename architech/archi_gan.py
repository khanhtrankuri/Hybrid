# architech/archi_gan.py
import torch.nn as nn
import torch

class Generator(nn.Module):
    """Mô hình Generator MLP cho MNIST/GAN cơ bản"""
    def __init__(self, nz: int = 100, img_dim: int = 28*28, hidden: int = 256):
        super().__init__()
        self.nz = nz
        self.img_dim = img_dim
        self.model = nn.Sequential(
            nn.Linear(nz, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden*2, img_dim),
            nn.Tanh(), # Đầu ra trong phạm vi [-1, 1]
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.model(z)
        # Reshape thành (Batch, Channels, Height, Width)
        img = img.view(z.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    """Mô hình Discriminator MLP cho MNIST/GAN cơ bản"""
    def __init__(self, img_dim: int = 28*28, hidden: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_dim, hidden*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden*2, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(), # Đầu ra là xác suất [0, 1]
        )
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)
