import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from datasets import load_dataset
from architech.archi import VisionMoEViT, HybridBackboneMoE, HybridClassifier

def get_cifar10_testloader(batch_size=128):
    print("Preparing CIFAR-10 test data...")
    # We use CIFAR-10 because the merged model head is from CIFAR-10
    dataset = load_dataset("uoft-cs/cifar10", split="test")
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def test_transforms(examples):
        pixel_values = [transform_test(image.convert("RGB")) for image in examples['img']]
        return {'pixel_values': pixel_values, 'label': examples['label']}

    dataset.set_transform(test_transforms)

    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return pixel_values, labels

    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    return testloader, dataset

def load_merged_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    
    # Initialize model structure (CIFAR-10 head)
    vit_model = VisionMoEViT(
        img_size=32, patch_size=4, in_chans=3, num_classes=10,
        embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
        num_experts=4, moe_layers=(2, 3), dropout=0.0
    )
    backbone = HybridBackboneMoE(
        img_size=32, in_chans=3, embed_dim=192, stem_dim=64,
        bert_depth=4, bert_heads=3, vit_model=vit_model
    )
    model = HybridClassifier(backbone, num_classes=10, embed_dim=192)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "backbone" in checkpoint:
        model.backbone.load_state_dict(checkpoint["backbone"])
        if "head_c10" in checkpoint:
            print("Detected multi-task checkpoint. Loading head_c10.")
            model.head.load_state_dict(checkpoint["head_c10"])
        elif "head" in checkpoint:
            model.head.load_state_dict(checkpoint["head"])
        else:
            print("Warning: No head found in checkpoint dict.")
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def evaluate_accuracy(model, testloader, device):
    print("Evaluating accuracy...")
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f'Test Accuracy: {acc:.3f}%')
    return acc

def visualize_results(model, dataset, device, output_file='evaluation_results.png', num_images=16):
    print(f"Generating visualization to {output_file}...")
    
    # Classes for CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Get random indices
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    model.eval()
    
    # Transform for prediction (normalized)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            item = dataset[int(idx)] # Access raw item (before transform if set_transform wasn't applied to the dataset object directly but it is)
            # dataset has set_transform applied, so item is {'pixel_values': ..., 'label': ...}
            
            image_tensor = item['pixel_values'].unsqueeze(0).to(device)
            label_idx = item['label']
            
            output = model(image_tensor)
            _, pred_idx = output.max(1)
            
            # For display, we need to un-normalize or use the original image if possible.
            # Since dataset returns transformed tensors, we can approximate un-normalization
            # Mean: (0.4914, 0.4822, 0.4465), Std: (0.2023, 0.1994, 0.2010)
            img_display = item['pixel_values'].permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            img_display = std * img_display + mean
            img_display = np.clip(img_display, 0, 1)
            
            ax = axes[i]
            ax.imshow(img_display)
            ax.set_title(f"True: {classes[label_idx]}\nPred: {classes[pred_idx.item()]}", 
                         color=("green" if label_idx == pred_idx.item() else "red"))
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(output_file)
    print("Visualization saved.")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    checkpoint_path = 'checkpoints/cifar_multitask_finetuned.pth'
    
    # Load Data
    testloader, dataset = get_cifar10_testloader()
    
    # Load Model
    try:
        model = load_merged_model(checkpoint_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Evaluate
    evaluate_accuracy(model, testloader, device)
    
    # Visualize
    visualize_results(model, dataset, device)

if __name__ == '__main__':
    main()
