import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from datasets import load_dataset
from architech.archi import VisionMoEViT, HybridBackboneMoE, HybridClassifier
import os

class MultiTaskHybridModel(nn.Module):
    def __init__(self, backbone, num_classes_1=10, num_classes_2=100, embed_dim=192):
        super().__init__()
        self.backbone = backbone
        self.head1 = nn.Linear(embed_dim, num_classes_1) # CIFAR-10
        self.head2 = nn.Linear(embed_dim, num_classes_2) # CIFAR-100

    def forward(self, x, task_id):
        # backbone returns (embedding, gate_probs)
        embedding, _ = self.backbone(x)
        if task_id == 0:
            return self.head1(embedding)
        elif task_id == 1:
            return self.head2(embedding)
        else:
            raise ValueError("Invalid task_id")

def get_dataloaders(batch_size=128):
    print("Preparing data with HF datasets...")
    
    # CIFAR-10
    ds_c10 = load_dataset("uoft-cs/cifar10")
    # CIFAR-100
    ds_c100 = load_dataset("uoft-cs/cifar100")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Use same normalization for simplicity, or specific ones if needed. 
    # CIFAR-100 mean/std is similar but slightly different. Using CIFAR-10 stats is usually fine for transfer.

    def train_transforms_c10(examples):
        pixel_values = [transform_train(image.convert("RGB")) for image in examples['img']]
        return {'pixel_values': pixel_values, 'label': examples['label']}

    def train_transforms_c100(examples):
        pixel_values = [transform_train(image.convert("RGB")) for image in examples['img']]
        return {'pixel_values': pixel_values, 'label': examples['fine_label']}

    ds_c10['train'].set_transform(train_transforms_c10)
    ds_c100['train'].set_transform(train_transforms_c100)

    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return pixel_values, labels

    loader_c10 = torch.utils.data.DataLoader(ds_c10['train'], batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    loader_c100 = torch.utils.data.DataLoader(ds_c100['train'], batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    
    return loader_c10, loader_c100

def load_merged_backbone(checkpoint_path, device):
    print(f"Loading merged backbone from {checkpoint_path}...")
    
    vit_model = VisionMoEViT(
        img_size=32, patch_size=4, in_chans=3, num_classes=10,
        embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
        num_experts=4, moe_layers=(2, 3), dropout=0.0
    )
    backbone = HybridBackboneMoE(
        img_size=32, in_chans=3, embed_dim=192, stem_dim=64,
        bert_depth=4, bert_heads=3, vit_model=vit_model
    )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "backbone" in checkpoint:
        backbone.load_state_dict(checkpoint["backbone"])
    else:
        # Try loading directly if it's just backbone, or extract if it's full model state dict
        # Assuming the checkpoint structure from merging_model.py which saves dict
        print("Warning: Checkpoint format might not match expected dict['backbone']. Trying direct load...")
        backbone.load_state_dict(checkpoint, strict=False)
        
    return backbone

def train_multitask(model, loader1, loader2, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    
    iter1 = iter(loader1)
    iter2 = iter(loader2)
    
    # Number of batches is min of both or max? 
    # Let's go with min for 1:1 mixing per epoch, or iterate until one exhausts.
    # For simplicity, let's use the length of the smaller loader.
    num_batches = min(len(loader1), len(loader2))
    
    for i in range(num_batches):
        try:
            inputs1, labels1 = next(iter1)
            inputs2, labels2 = next(iter2)
        except StopIteration:
            break
            
        inputs1, labels1 = inputs1.to(device), labels1.to(device)
        inputs2, labels2 = inputs2.to(device), labels2.to(device)

        optimizer.zero_grad()

        # Task 1: CIFAR-10
        outputs1 = model(inputs1, task_id=0)
        loss1 = criterion(outputs1, labels1)
        
        # Task 2: CIFAR-100
        outputs2 = model(inputs2, task_id=1)
        loss2 = criterion(outputs2, labels2)
        
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print(f'[Epoch {epoch}, Batch {i + 1}] Loss: {running_loss / 100:.3f} (L1+L2)')
            running_loss = 0.0

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    checkpoint_path = 'checkpoints/cifar_hybrid_moe_merged.pth'
    output_path = 'checkpoints/cifar_multitask_finetuned.pth'
    
    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 5 # User asked for 5-10
    LR = 5e-5 # Lower LR for fine-tuning
    
    # Data
    loader_c10, loader_c100 = get_dataloaders(BATCH_SIZE)
    
    # Model
    backbone = load_merged_backbone(checkpoint_path, device)
    model = MultiTaskHybridModel(backbone, num_classes_1=10, num_classes_2=100).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    for epoch in range(EPOCHS):
        print(f"Starting Epoch {epoch}")
        train_multitask(model, loader_c10, loader_c100, criterion, optimizer, device, epoch)
        scheduler.step()
        
        # Save checkpoint
        print(f"Saving checkpoint to {output_path}")
        torch.save({
            "backbone": model.backbone.state_dict(),
            "head_c10": model.head1.state_dict(),
            "head_c100": model.head2.state_dict(),
        }, output_path)

    print('Finished Fine-tuning')

if __name__ == '__main__':
    main()
