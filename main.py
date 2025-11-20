import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from architech.archi import VisionMoEViT, HybridBackboneMoE, HybridClassifier
import os

from datasets import load_dataset

def get_dataloaders(batch_size=128):
    print("Preparing data with HF datasets...")
    dataset = load_dataset("uoft-cs/cifar100")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def train_transforms(examples):
        pixel_values = [transform_train(image.convert("RGB")) for image in examples['img']]
        return {'pixel_values': pixel_values, 'label': examples['fine_label']}

    def test_transforms(examples):
        pixel_values = [transform_test(image.convert("RGB")) for image in examples['img']]
        return {'pixel_values': pixel_values, 'label': examples['fine_label']}

    dataset['train'].set_transform(train_transforms)
    dataset['test'].set_transform(test_transforms)

    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return pixel_values, labels

    trainloader = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    return trainloader, testloader

def train(model, trainloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 100 == 99:
            print(f'[Epoch {epoch}, Batch {i + 1}] loss: {running_loss / 100:.3f} | acc: {100.*correct/total:.3f}%')
            running_loss = 0.0

def evaluate(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    avg_loss = test_loss / len(testloader)
    print(f'Test Loss: {avg_loss:.3f} | Test Acc: {acc:.3f}%')
    return acc

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    os.makedirs('checkpoints', exist_ok=True)

    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 10
    LR = 1e-4
    
    # Data
    trainloader, testloader = get_dataloaders(BATCH_SIZE)

    # Model
    print("Initializing VisionMoEViT expert...")
    vit_model = VisionMoEViT(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        num_experts=4,
        moe_layers=(2, 3),
        dropout=0.1
    )

    print("Initializing HybridBackboneMoE...")
    backbone = HybridBackboneMoE(
        img_size=32,
        in_chans=3,
        embed_dim=192,
        stem_dim=64,
        bert_depth=4,
        bert_heads=3,
        vit_model=vit_model
    )

    model = HybridClassifier(backbone, num_classes=100, embed_dim=192).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        train(model, trainloader, criterion, optimizer, device, epoch)
        acc = evaluate(model, testloader, criterion, device)
        scheduler.step()

        if acc > best_acc:
            print(f'Saving best model (acc: {acc:.3f}%)')
            torch.save({
                "backbone": model.backbone.state_dict(),
                "head": model.head.state_dict(),
            }, "checkpoints/cifar10_hybrid_moe.pth")
            best_acc = acc

    print('Finished Training')

if __name__ == '__main__':
    main()
