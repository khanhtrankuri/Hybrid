import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from datasets import load_dataset

from architech.archi import tokenize_batch
from architech.archi_large import LargeTextToImageModel, LargeT2IConfig

# Keep a single source of truth for image size to avoid mismatches; lower for faster convergence
IMAGE_SIZE = 256


def compute_mean_std(dataset, image_key="image", max_samples=128):
    """
    Compute per-channel mean/std over a subset of the dataset.
    """
    sums = torch.zeros(3)
    sq_sums = torch.zeros(3)
    count = 0
    to_tensor = transforms.ToTensor()
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        img = item[image_key]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        t = to_tensor(img)  # [0,1]
        sums += t.view(3, -1).mean(dim=1)
        sq_sums += (t.view(3, -1) ** 2).mean(dim=1)
        count += 1
    mean = (sums / count).tolist()
    std = (sq_sums / count - torch.tensor(mean) ** 2).sqrt().tolist()
    return mean, std


def build_transform(mean, std):
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class ImageTextCollator:
    """
    Top-level collate to stay picklable on Windows. Applies transform and tokenization.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        images = torch.stack([self.transform(item["image"].convert("RGB")) for item in batch])
        texts = [item["text"] for item in batch]
        token_ids, lengths = tokenize_batch(texts, vocab_size=4096, max_len=32)
        return images, token_ids


def get_dataloaders(batch_size=128):
    """
    Uses HF dataset with captions: wangherr/coco2017_train_512x_image_caption_canny (image, text).
    """
    dataset = load_dataset("wangherr/coco2017_train_512x_image_caption_canny")

    # Compute normalization stats from a subset for faster convergence
    mean, std = compute_mean_std(dataset["train"], image_key="image", max_samples=256)
    transform_image = build_transform(mean, std)

    collate = ImageTextCollator(transform_image)

    num_workers = 0 if os.name == "nt" else 4
    trainloader = torch.utils.data.DataLoader(
        dataset['train'], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0), collate_fn=collate
    )
    # The dataset has no official val split; use a small held-out subset for quick eval
    val_subset = dataset['train'].select(range(0, min(1000, len(dataset['train']))))
    testloader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0), collate_fn=collate
    )

    # Debug info to ensure consistent epoch length across GPU counts
    print(f"[Data] train samples: {len(dataset['train'])}, val samples: {len(val_subset)}, "
          f"batch_size: {batch_size}, steps/epoch (train): {len(trainloader)}")

    return trainloader, testloader


def train(model, trainloader, criterion, optimizer, device, epoch, latent_dim):
    model.train()
    running_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))
    for i, (images, token_ids) in enumerate(trainloader):
        images = images.to(device)
        token_ids = token_ids.to(device)
        noise = torch.randn(images.size(0), latent_dim, device=device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
            outputs = model(token_ids, noise)
            if outputs.shape[-2:] != images.shape[-2:]:
                outputs = F.interpolate(outputs, size=images.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, images)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if i % 100 == 99:
            print(f'[Epoch {epoch}, Batch {i + 1}] L1 loss: {running_loss / 100:.4f}')
            running_loss = 0.0


def evaluate(model, testloader, criterion, device, latent_dim):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, token_ids in testloader:
            images = images.to(device)
            token_ids = token_ids.to(device)
            noise = torch.randn(images.size(0), latent_dim, device=device)
            with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
                outputs = model(token_ids, noise)
                if outputs.shape[-2:] != images.shape[-2:]:
                    outputs = F.interpolate(outputs, size=images.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(outputs, images)
            test_loss += loss.item()

    avg_loss = test_loss / len(testloader)
    print(f'Val L1 Loss: {avg_loss:.4f}')
    return -avg_loss  # higher is better for checkpointing


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    os.makedirs('checkpoints', exist_ok=True)

    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 3e-4
    
    # Data
    trainloader, testloader = get_dataloaders(BATCH_SIZE)

    print("Building larger text-to-image generator (Transformer text encoder + ResUpsample generator)...")
    cfg = LargeT2IConfig(
        vocab_size=4096,
        max_len=32,
        text_embed_dim=512,
        text_heads=8,
        text_layers=6,
        text_ff=1024,
        latent_dim=128,
        base_channels=256,
        target_size=IMAGE_SIZE,
    )
    model = LargeTextToImageModel(cfg).to(device)

    # Optional: simple DataParallel across all visible GPUs (helps utilize 2×T4)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # latent_dim needed for noise sampling; handle DataParallel wrapper
    latent_dim = model.module.latent_dim if isinstance(model, torch.nn.DataParallel) else model.latent_dim

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_score = float("-inf")
    
    for epoch in range(EPOCHS):
        train(model, trainloader, criterion, optimizer, device, epoch, latent_dim)
        score = evaluate(model, testloader, criterion, device, latent_dim)
        scheduler.step()

        if score > best_score:
            print(f'Saving best model (score: {score:.4f})')
            target = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save({
                "text_encoder": target.text_encoder.state_dict(),
                "generator": target.generator.state_dict(),
            }, "checkpoints/text2img_small.pth")
            best_score = score

    print('Finished Training')


if __name__ == '__main__':
    main()
