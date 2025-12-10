import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from architecture.archi_MMoMoE import ArchiMMoMoE
from dataset_loader import get_dataloader

def train():
    # Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    MAX_STEPS = 10000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_PATH = "checkpoints/mmomoe_laion.pth"
    LOG_INTERVAL = 10
    
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Training on {DEVICE}")

    # Initialize Model
    # Using the config compatible with the dataset loader (Bert Tokenizer)
    model = ArchiMMoMoE(
        img_size=224,
        patch_size=16,
        in_chans=3,
        vocab_size=30522, # Bert-base-uncased vocab size
        max_len=77,       # Standard CLIP context length
        embed_dim=512,
        depth=6,
        num_heads=8,
        num_experts=4
    ).to(DEVICE)

    # Loss Function (CLIP-style Contrastive Loss)
    # Learnable temperature parameter
    # Create tensor on device first, then wrap in Parameter to keep it as a leaf
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), device=DEVICE))
    
    # Optimizer - Weight Decay separation
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.1}, # Strong weight decay for weights
        {'params': nodecay_params, 'weight_decay': 0.0}, # No decay for biases/norms
        {'params': [logit_scale], 'weight_decay': 0.0}   # Logit scale (no decay)
    ]
    
    # Increased learning rate slightly for OneCycle/Cosine
    lr_peak = 5e-4 
    optimizer = optim.AdamW(optim_groups, lr=lr_peak, betas=(0.9, 0.95))

    # Dataloader
    # Note: num_workers=0 is safer for windows/streaming datasets
    dataloader = get_dataloader(batch_size=BATCH_SIZE, num_workers=0)
    
    # Scheduler
    # Assuming MAX_STEPS is accurate, we use a OneCycle or Cosine Schedule
    # If dataset size is unknown, we rely on step-based scheduler
    warmup_steps = int(MAX_STEPS * 0.1)
    
    # Simple Manual Cosine Schedule or torch's OneCycleLR
    # We'll use a LambdaLR for explicit control or OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr_peak, 
        total_steps=MAX_STEPS, 
        pct_start=0.1,    # 10% warmup
        anneal_strategy='cos',
        cycle_momentum=False, # AdamW usually better without cycle momentum
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision

    model.train()
    step = 0
    running_loss = 0.0
    
    print("Starting training loop with Improved Convergence (AMP + Scheduler + Optimized Init)...")
    
    try:
        for batch in dataloader:
            if step >= MAX_STEPS:
                break
                
            images = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Context manager for Mixed Precision
            with torch.amp.autocast('cuda', enabled=(DEVICE=="cuda")):
                outputs = model(image=images, text=input_ids)
                
                image_features = outputs["image_features"] # [B, D]
                text_features = outputs["text_features"]   # [B, D]
                
                # Normalize features
                image_features = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-6)
                text_features = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-6)
                
                # Clip logit scale
                logit_scale_clamped = logit_scale.exp().clamp(max=100)
                
                # Calculate Logits
                logits_per_image = logit_scale_clamped * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                
                batch_size_curr = images.shape[0]
                labels = torch.arange(batch_size_curr, device=DEVICE)
                
                loss_i = criterion(logits_per_image, labels)
                loss_t = criterion(logits_per_text, labels)
                loss = (loss_i + loss_t) / 2
            
            # Backward with Scaler
            scaler.scale(loss).backward()
            
            # Gradient Clipping (Unscale first)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item()
            
            if step % LOG_INTERVAL == 0:
                avg_loss = running_loss / LOG_INTERVAL if step > 0 else loss.item()
                curr_lr = optimizer.param_groups[0]['lr']
                
                # Calculate feature std for monitoring (on current batch)
                with torch.no_grad():
                    img_std = image_features.std(dim=0).mean().item()
                    txt_std = text_features.std(dim=0).mean().item()
                
                print(f"Step [{step}/{MAX_STEPS}] | Loss: {avg_loss:.4f} | Scale: {logit_scale_clamped.item():.2f} | LR: {curr_lr:.2e} | Grad: {grad_norm:.2f} | Std: {img_std:.4f}/{txt_std:.4f}")
                running_loss = 0.0
                
            step += 1
            
            if step % 500 == 0:
                 torch.save(model.state_dict(), SAVE_PATH)
                 print(f"Checkpoint saved to {SAVE_PATH}")
            
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

    # Save Final Model
    print(f"Saving final model to {SAVE_PATH}")
    torch.save(model.state_dict(), SAVE_PATH)
    print("Training complete.")

if __name__ == "__main__":
    train()
