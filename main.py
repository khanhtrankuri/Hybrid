import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from datasets import load_dataset

from architech.archi import TextToImageModel, tokenize_batch

# Keep a single source of truth for image size to avoid mismatches; lower for faster convergence
IMAGE_SIZE = 128


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


def get_dataloaders(batch_size=128):
    """
    Uses HF dataset with captions: wangherr/coco2017_train_512x_image_caption_canny (image, text).
    """
    dataset = load_dataset("wangherr/coco2017_train_512x_image_caption_canny")

    # Compute normalization stats from a subset for faster convergence
    mean, std = compute_mean_std(dataset["train"], image_key="image", max_samples=256)
    transform_image = build_transform(mean, std)

    def collate_fn(batch):
        images = torch.stack([transform_image(item["image"].convert("RGB")) for item in batch])
        texts = [item["text"] for item in batch]
        token_ids, lengths = tokenize_batch(texts, vocab_size=4096, max_len=16)
        return images, token_ids, lengths

    trainloader = torch.utils.data.DataLoader(
        dataset['train'], batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, collate_fn=collate_fn
    )
    # The dataset has no official val split; use a small held-out subset for quick eval
    val_subset = dataset['train'].select(range(0, min(1000, len(dataset['train']))))
    testloader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, collate_fn=collate_fn
    )

    return trainloader, testloader


def train(model, trainloader, criterion, optimizer, device, epoch, latent_dim):
    model.train()
    running_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))
    for i, (images, token_ids, lengths) in enumerate(trainloader):
        images = images.to(device)
        token_ids = token_ids.to(device)
        lengths = lengths.to(device)
        noise = torch.randn(images.size(0), latent_dim, device=device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
            outputs = model(token_ids, lengths, noise)
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
        for images, token_ids, lengths in testloader:
            images = images.to(device)
            token_ids = token_ids.to(device)
            lengths = lengths.to(device)
            noise = torch.randn(images.size(0), latent_dim, device=device)
            with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
                outputs = model(token_ids, lengths, noise)
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
    EPOCHS = 30
    LR = 3e-4
    
    # Data
    trainloader, testloader = get_dataloaders(BATCH_SIZE)

    # Model
    print("Building tiny text-to-image generator...")
    model = TextToImageModel(
        vocab_size=4096,
        text_embed_dim=128,
        text_hidden_dim=192,
        latent_dim=32,
        base_channels=64,
        target_size=IMAGE_SIZE,
    ).to(device)

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
