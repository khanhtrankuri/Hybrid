import argparse
import os
from typing import Optional

import torch
import torchvision.transforms as transforms

from architech.archi import TextToImageModel, tokenize_batch


def load_model(checkpoint_path: str, device: str):
    model = TextToImageModel(
        vocab_size=4096,
        text_embed_dim=128,
        text_hidden_dim=256,
        latent_dim=64,
        base_channels=128,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.text_encoder.load_state_dict(state["text_encoder"])
    model.generator.load_state_dict(state["generator"])
    model.to(device)
    model.eval()
    return model


def save_tensor_image(tensor: torch.Tensor, path: str, out_size: int | None = None):
    # tensor expected shape (3, H, W), range [-1, 1]
    img = tensor.clamp(-1, 1)
    img = (img + 1) / 2.0
    to_pil = transforms.ToPILImage()
    pil = to_pil(img.cpu())
    if out_size is not None:
        pil = pil.resize((out_size, out_size))
    pil.save(path)


def predict(prompt: str, checkpoint: str, device: str, seed: Optional[int], out_path: str, out_size: Optional[int]):
    if seed is not None:
        torch.manual_seed(seed)

    model = load_model(checkpoint, device)
    token_ids, lengths = tokenize_batch([prompt], vocab_size=4096, max_len=16)
    token_ids = token_ids.to(device)
    lengths = lengths.to(device)

    noise = torch.randn(1, model.latent_dim, device=device)
    with torch.no_grad():
        img = model(token_ids, lengths, noise)[0]
    save_tensor_image(img, out_path, out_size)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate an image from text using the tiny text-to-image model.")
    parser.add_argument("--text", required=True, help="Input prompt text")
    parser.add_argument("--checkpoint", default="checkpoints/text2img_small.pth", help="Path to trained checkpoint")
    parser.add_argument("--out", default="generated.png", help="Path to save generated image")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--out-size", type=int, default=None, help="Resize output image to this square size (pixels); default keeps model output size")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    out_path = predict(args.text, args.checkpoint, args.device, args.seed, args.out, args.out_size)
    print(f"Saved generated image to: {out_path}")


if __name__ == "__main__":
    main()
