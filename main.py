import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from datasets import load_dataset

from architech.archi import TextToImageModel, tokenize_batch

# Image preprocessing for generator (range [-1, 1]); now 256x256
transform_image = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def get_dataloaders(batch_size=128):
    """
    Uses a small HF dataset with captions: UCSC-VLAA/Recap-COCO-30K.
    Fields: image (PIL), recaption (caption).
    """
    dataset = load_dataset("UCSC-VLAA/Recap-COCO-30K")

    def collate_fn(batch):
        images = torch.stack([transform_image(item["image"].convert("RGB")) for item in batch])
        texts = [item["recaption"] for item in batch]
        token_ids, lengths = tokenize_batch(texts, vocab_size=4096, max_len=16)
        return images, token_ids, lengths

    trainloader = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    # The dataset has no official val split; use a small held-out subset for quick eval
    val_subset = dataset['train'].select(range(0, min(500, len(dataset['train']))))
    testloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    return trainloader, testloader


def train(model, trainloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for i, (images, token_ids, lengths) in enumerate(trainloader):
        images = images.to(device)
        token_ids = token_ids.to(device)
        lengths = lengths.to(device)
        noise = torch.randn(images.size(0), model.latent_dim, device=device)

        optimizer.zero_grad()

        outputs = model(token_ids, lengths, noise)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print(f'[Epoch {epoch}, Batch {i + 1}] L1 loss: {running_loss / 100:.4f}')
            running_loss = 0.0


def evaluate(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, token_ids, lengths in testloader:
            images = images.to(device)
            token_ids = token_ids.to(device)
            lengths = lengths.to(device)
            noise = torch.randn(images.size(0), model.latent_dim, device=device)
            outputs = model(token_ids, lengths, noise)
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
        text_hidden_dim=256,
        latent_dim=64,
        base_channels=128,
        target_size=256,
    ).to(device)

    # Optional: simple DataParallel across all visible GPUs (helps utilize 2×T4)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_score = float("-inf")
    
    for epoch in range(EPOCHS):
        train(model, trainloader, criterion, optimizer, device, epoch)
        score = evaluate(model, testloader, criterion, device)
        scheduler.step()

        if score > best_score:
            print(f'Saving best model (score: {score:.4f})')
            torch.save({
                "text_encoder": model.text_encoder.state_dict(),
                "generator": model.generator.state_dict(),
            }, "checkpoints/text2img_small.pth")
            best_score = score

    print('Finished Training')


if __name__ == '__main__':
    main()
