# predict.py

import argparse
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Import Generator mới
from architech.archi_gan import Generator 

def load_generator(checkpoint_path: str, device: str):
    """Tải mô hình Generator đã hợp nhất"""
    model = Generator().to(device)
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def generate_images(model, nz: int, num_images: int, device: str, out_path: str):
    print(f"Generating {num_images} images from the merged generator...")
    
    # Tạo nhiễu ngẫu nhiên
    noise = torch.randn(num_images, nz, device=device)
    
    with torch.no_grad():
        # Sinh ảnh
        imgs = model(noise).cpu()
    
    # Đưa ảnh về phạm vi [0, 1] và chuyển sang numpy
    imgs = (imgs + 1) / 2
    
    # Vẽ và lưu ảnh
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.2, rows*1.2))
    axes = axes.flatten()
    
    for i in range(num_images):
        if i < len(axes):
            ax = axes[i]
            # Loại bỏ kênh (vì là ảnh grayscale 1x28x28)
            ax.imshow(imgs[i].squeeze(), cmap='gray')
            ax.axis('off')
            
    # Tắt các subplot thừa
    for i in range(num_images, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved generated images to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate images using the merged GAN generator.")
    parser.add_argument("--checkpoint", default="checkpoints/generator_gan_merged_0_9.pth", help="Path to the merged generator checkpoint")
    parser.add_argument("--out", default="generated_0_9_merged.png", help="Path to save generated image grid")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    parser.add_argument("--nz", type=int, default=100, help="Latent dimension size (nz)")
    parser.add_argument("--num_images", type=int, default=36, help="Number of images to generate")
    args = parser.parse_args()

    device = args.device
    
    # Tải model
    model = load_generator(args.checkpoint, device)
    
    # Sinh ảnh
    generate_images(model, args.nz, args.num_images, device, args.out)

if __name__ == "__main__":
    # Thay thế phần này bằng code main()
    # Để chạy trong môi trường Jupyter/Kaggle mà không cần argparse:
    
    # Khởi tạo giả định (nếu không dùng terminal)
    class Args:
        checkpoint = "checkpoints/generator_gan_merged_0_9.pth"
        out = "generated_0_9_merged.png"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nz = 100
        num_images = 36
        
    args = Args()
    
    # Tải và sinh ảnh (Sử dụng code main() nhưng thay thế argparse)
    try:
        model = load_generator(args.checkpoint, args.device)
        generate_images(model, args.nz, args.num_images, args.device, args.out)
    except FileNotFoundError as e:
        print(f"ERROR: {e}. Vui lòng chạy main.py trước để tạo checkpoint.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")