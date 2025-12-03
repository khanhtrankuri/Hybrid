# merging_model.py

import torch
import copy
import os
import sys
from architech.archi_gan import Generator
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def slerp(v0, v1, t, dot_threshold=0.9995):
    """Spherical Linear Interpolation"""
    v0_norm = v0 / torch.norm(v0)
    v1_norm = v1 / torch.norm(v1)
    dot = torch.sum(v0_norm * v1_norm)
    if torch.abs(dot) > dot_threshold:
        return (1 - t) * v0 + t * v1
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    return s0 * v0 + s1 * v1

def m2n2_crossover(model_a, model_b, split_ratio=0.5, mix_ratio=0.5):
    print("Flattening parameters...")
    
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    
    param_names = sorted(state_a.keys())
    
    flat_a_list = [state_a[name].view(-1) for name in param_names]
    flat_b_list = [state_b[name].view(-1) for name in param_names]
    
    flat_a = torch.cat(flat_a_list)
    flat_b = torch.cat(flat_b_list)
    
    num_params = flat_a.numel()
    split_idx = int(num_params * split_ratio)
    
    print(f"Total parameters: {num_params}. Split index: {split_idx}")
    
    part1 = slerp(flat_a[:split_idx], flat_b[:split_idx], mix_ratio)
    
    part2 = slerp(flat_a[split_idx:], flat_b[split_idx:], 1.0 - mix_ratio)
    
    flat_merged = torch.cat([part1, part2])
    
    merged_model = copy.deepcopy(model_b) # Kiến trúc B (Generator)
    merged_state_dict = merged_model.state_dict()
    
    print("Unflattening parameters into merged model...")
    offset = 0
    
    for name in param_names:
        param = merged_state_dict[name]
        numel = param.numel()
        param.data = flat_merged[offset:offset + numel].view(param.shape)
        offset += numel
        
    return merged_model

def main():
    # Paths
    ckpt_a_path = 'checkpoints/generator_gan_0_4.pth'
    ckpt_b_path = 'checkpoints/generator_gan_5_9.pth'
    output_path = 'checkpoints/generator_gan_merged_0_9.pth'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Initializing Generator architectures...")
    gen_a = Generator().to(device)
    gen_b = Generator().to(device)

    print(f"Loading checkpoint A: {ckpt_a_path}")
    state_a = torch.load(ckpt_a_path, map_location=device)
    gen_a.load_state_dict(state_a)

    print(f"Loading checkpoint B: {ckpt_b_path}")
    state_b = torch.load(ckpt_b_path, map_location=device)
    gen_b.load_state_dict(state_b)

    print("Starting M2N2 Crossover on Generators...")
    merged_generator = m2n2_crossover(gen_a, gen_b, split_ratio=0.5, mix_ratio=0.5)
    
    print(f"Saving merged generator to {output_path}")
    torch.save(merged_generator.state_dict(), output_path)
    print("Done.")

if __name__ == '__main__':
    Path('checkpoints').mkdir(parents=True, exist_ok=True)
    main()
