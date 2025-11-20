import torch
import numpy as np
import copy
import os
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from architech.archi import VisionMoEViT, HybridBackboneMoE, HybridClassifier

def slerp(v0, v1, t, dot_threshold=0.9995):
    """
    Spherical Linear Interpolation
    """
    # Normalize vectors
    v0_norm = v0 / torch.norm(v0)
    v1_norm = v1 / torch.norm(v1)
    
    dot = torch.sum(v0_norm * v1_norm)
    
    # If vectors are too close, use linear interpolation
    if torch.abs(dot) > dot_threshold:
        return (1 - t) * v0 + t * v1
    
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    
    return s0 * v0 + s1 * v1

def flatten_params(model):
    """Flatten all parameters of a model into a single 1D tensor."""
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    return torch.cat(params)

def unflatten_params(model, flat_params):
    """Load flattened parameters back into a model."""
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data = flat_params[offset:offset + numel].view(param.shape)
        offset += numel

def m2n2_crossover_backbones(backbone_a, backbone_b, split_ratio=0.5, mix_ratio=0.5):
    """
    Thực hiện M2N2 crossover chỉ trên các tham số chung (same name + same shape)
    giữa backbone A và backbone B.
    Trả về backbone_merged có kiến trúc như backbone_b.
    """
    print("Identifying common parameters...")
    common_params_a = []
    common_params_b = []
    param_names = []
    
    dict_a = dict(backbone_a.named_parameters())
    dict_b = dict(backbone_b.named_parameters())
    
    for name, param_b in dict_b.items():
        if name in dict_a:
            param_a = dict_a[name]
            if param_a.shape == param_b.shape:
                common_params_a.append(param_a.view(-1))
                common_params_b.append(param_b.view(-1))
                param_names.append(name)
            else:
                print(f"Skipping {name}: shape mismatch {param_a.shape} vs {param_b.shape}")
        else:
            print(f"Skipping {name}: not found in backbone A")

    if not common_params_a:
        print("No common parameters found!")
        return copy.deepcopy(backbone_b)

    print(f"Flattening {len(common_params_a)} common parameters...")
    flat_a = torch.cat(common_params_a)
    flat_b = torch.cat(common_params_b)
    
    num_params = flat_a.numel()
    split_idx = int(num_params * split_ratio)
    
    print(f"Total common parameters: {num_params}")
    print(f"Split index: {split_idx} (Ratio: {split_ratio})")
    
    # Part 1
    print("Merging Part 1...")
    part1 = slerp(flat_a[:split_idx], flat_b[:split_idx], mix_ratio)
    
    # Part 2
    print("Merging Part 2...")
    part2 = slerp(flat_a[split_idx:], flat_b[split_idx:], 1.0 - mix_ratio)
    
    flat_merged = torch.cat([part1, part2])
    
    # Create merged backbone as copy of B
    merged_backbone = copy.deepcopy(backbone_b)
    
    print("Unflattening parameters into merged backbone...")
    offset = 0
    merged_dict = dict(merged_backbone.named_parameters())
    
    for name in param_names:
        param = merged_dict[name]
        numel = param.numel()
        param.data = flat_merged[offset:offset + numel].view(param.shape)
        offset += numel
        
    return merged_backbone

def main():
    # Paths
    ckpt_a_path = 'checkpoints/cifar100_hybrid_moe.pth'
    ckpt_b_path = 'checkpoints/cifar10_hybrid_moe.pth'
    output_path = 'checkpoints/cifar_hybrid_moe_merged.pth'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model structure
    def create_model(num_classes):
        vit_model = VisionMoEViT(
            img_size=32, patch_size=4, in_chans=3, num_classes=10, # Always 10 as per checkpoint
            embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
            num_experts=4, moe_layers=(2, 3), dropout=0.0
        )
        backbone = HybridBackboneMoE(
            img_size=32, in_chans=3, embed_dim=192, stem_dim=64,
            bert_depth=4, bert_heads=3, vit_model=vit_model
        )
        model = HybridClassifier(backbone, num_classes=num_classes, embed_dim=192)
        return model

    print("Initializing models...")
    model_a = create_model(100) # CIFAR-100
    model_b = create_model(10)  # CIFAR-10
    
    # Load checkpoints
    def load_checkpoint(model, path):
        print(f"Loading checkpoint: {path}")
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            return False
        
        checkpoint = torch.load(path, map_location=device)
        
        # Check if checkpoint is a dict with "backbone" and "head"
        if isinstance(checkpoint, dict) and "backbone" in checkpoint:
            print("Detected dictionary checkpoint format.")
            model.backbone.load_state_dict(checkpoint["backbone"])
            
            # Try to load head, but catch mismatch
            try:
                model.head.load_state_dict(checkpoint["head"])
            except RuntimeError as e:
                print(f"Warning: Could not load head weights due to mismatch: {e}")
                print("Skipping head loading (this is fine if we only need the backbone for merging).")
        else:
            print("Detected standard state_dict format.")
            # Try to load, but allow strict=False or catch mismatch if we only need backbone
            # However, standard format mixes backbone and head. 
            # If we have a mismatch in head, strict=False might be needed.
            try:
                model.load_state_dict(checkpoint)
            except RuntimeError as e:
                print(f"Warning: Could not load state_dict strictly: {e}")
                print("Attempting with strict=False...")
                model.load_state_dict(checkpoint, strict=False)
        return True

    if not load_checkpoint(model_a, ckpt_a_path): return
    if not load_checkpoint(model_b, ckpt_b_path): return
    
    # Perform Merge
    print("Starting M2N2 Crossover on Backbones...")
    # Pass backbones to the merge function
    merged_backbone = m2n2_crossover_backbones(model_a.backbone, model_b.backbone, split_ratio=0.5, mix_ratio=0.5)
    
    # Construct full merged model
    # We use model_b's head (CIFAR-10)
    merged_model = HybridClassifier(merged_backbone, num_classes=10, embed_dim=192)
    merged_model.head.load_state_dict(model_b.head.state_dict())
    
    # Save
    print(f"Saving merged model to {output_path}")
    # Save in the same format as main.py
    torch.save({
        "backbone": merged_model.backbone.state_dict(),
        "head": merged_model.head.state_dict(),
    }, output_path)
    print("Done.")

if __name__ == '__main__':
    main()
