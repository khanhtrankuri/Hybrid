# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import random
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# Import kiến trúc mới
from architech.archi_gan import Generator, Discriminator 

# ----------------------
# 1. Thiết lập & Tiện ích
# ----------------------
seed = 42
random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
OUT_DIR = Path('checkpoints') # Đổi sang folder checkpoints
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------
# 2. Dữ liệu & DataLoader
# ----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # scale to [-1,1]
])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

def get_label_indices(dataset, allowed_labels):
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)
    selected = []
    for lab in allowed_labels:
        selected.extend(label_to_indices[lab])
    return selected

# ----------------------
# 3. Hàm Huấn luyện
# ----------------------

def train_gan(dataloader, nz=100, epochs=50, lr=2e-4, betas=(0.5, 0.999), save_path='generator.pkl'):
    # Sử dụng Generator/Discriminator đã import
    gen = Generator(nz=nz).to(device)
    disc = Discriminator().to(device)

    optimG = optim.Adam(gen.parameters(), lr=lr, betas=betas)
    optimD = optim.Adam(disc.parameters(), lr=lr, betas=betas)
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(16, nz, device=device)

    for epoch in range(1, epochs+1):
        pbar = tqdm.tqdm(dataloader, desc=f'Epoch {epoch}/{epochs}', leave=False)
        for real_imgs, _ in pbar:
            bs = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            # ... (Phần code huấn luyện D và G y hệt như Cell 4)
            
            # Train Discriminator
            optimD.zero_grad()
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)
            outputs_real = disc(real_imgs)
            lossD_real = criterion(outputs_real, real_labels)
            noise = torch.randn(bs, nz, device=device)
            fake_imgs = gen(noise)
            outputs_fake = disc(fake_imgs.detach())
            lossD_fake = criterion(outputs_fake, fake_labels)
            lossD = (lossD_real + lossD_fake) * 0.5
            lossD.backward()
            optimD.step()

            # Train Generator
            optimG.zero_grad()
            outputs_fake_forG = disc(fake_imgs)
            lossG = criterion(outputs_fake_forG, real_labels)
            lossG.backward()
            optimG.step()
            
            pbar.set_postfix({'lossD': lossD.item(), 'lossG': lossG.item()})

        # end epoch
        with torch.no_grad():
            sample = gen(fixed_noise)
        print(f'Epoch {epoch}: lossD={lossD.item():.4f}, lossG={lossG.item():.4f}')

    # Save the full generator object
    save_full_path = OUT_DIR / save_path
    # Chỉ lưu state_dict để dễ tải lại và merge
    torch.save(gen.state_dict(), str(save_full_path))
    print('Saved generator state_dict to', save_full_path)
    return gen, disc


if __name__ == '__main__':
    # Prepare dataloaders for digits 0-4 and 5-9
    batch_size = 128
    indices_0_4 = get_label_indices(mnist_train, allowed_labels=list(range(0,5)))
    indices_5_9 = get_label_indices(mnist_train, allowed_labels=list(range(5,10)))
    sub_0_4 = Subset(mnist_train, indices_0_4)
    sub_5_9 = Subset(mnist_train, indices_5_9)
    loader_0_4 = DataLoader(sub_0_4, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    loader_5_9 = DataLoader(sub_5_9, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Train both GANs
    EPOCHS = 50 
    print('Training GAN for digits 0-4...')
    # Đổi tên file checkpoint
    gen_0_4, _ = train_gan(loader_0_4, epochs=EPOCHS, save_path='generator_gan_0_4.pth')

    print('\nTraining GAN for digits 5-9...')
    gen_5_9, _ = train_gan(loader_5_9, epochs=EPOCHS, save_path='generator_gan_5_9.pth')
    
    print('Training Complete.')