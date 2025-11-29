# Hybrid Text-to-Image Experiments

Mã nguồn này chứa hai kiến trúc chính:
- `architech/archi.py`: mô hình nhỏ/tối giản (hybrid MoE backbone + text-to-image toy).
- `architech/archi_large.py`: mô hình lớn hơn (encoder Transformer MoE + generator FiLM/ResUpsample) phù hợp train trên 1–2 GPU.

### Cấu trúc file chính
- `architech/archi.py`: backbone MoE cho phân loại và một text-to-image nhỏ (GRU + conv-transpose).
- `architech/archi_large.py`: text encoder Transformer dùng MoE FFN (có map expert-to-GPU), positional encoding + CLS pooling; generator upsample kèm FiLM (điều kiện theo text).
- `main.py`: script train text-to-image; mặc định dùng `LargeTextToImageModel` từ `archi_large.py`. Có tính mean/std động, AMP, DataParallel (nếu >1 GPU).
- `predict.py`: suy luận/generate ảnh với checkpoint (cần chỉnh cho model bạn dùng).
- `finetune.py`, `evaluate.py`: tiện ích/khung tham khảo (tùy chỉnh thêm theo nhu cầu).

### Yêu cầu môi trường
- Python + PyTorch + torchvision + datasets (HF) + Pillow + requests (nếu tải URL).
- GPU khuyến nghị. Với model lớn, 1–2 × T4/V100/3090; điều chỉnh batch/IMAGE_SIZE để tránh OOM.

### Chạy train (model lớn hiện tại trong main.py)
1. Cài gói:
   ```bash
   pip install torch torchvision datasets pillow requests
   ```
2. Chỉnh hyper/đường dẫn dataset trong `main.py` nếu cần:
   - `IMAGE_SIZE` (mặc định 128; tăng/giảm theo VRAM).
   - Dataset HF: hiện dùng `wangherr/coco2017_train_512x_image_caption_canny` (trường `image`, `text`). Thay tên dataset và key nếu khác.
   - Batch/Epoch/LR ở phần Hyperparameters.
3. Train:
   ```bash
   python main.py
   ```
   - Tự tính mean/std từ subset, bật AMP, DataParallel nếu có >1 GPU.
   - Checkpoint lưu ở `checkpoints/text2img_small.pth` (gồm `text_encoder`, `generator`).

### Suy luận
- `predict.py` mặc định cho mô hình nhỏ; nếu dùng mô hình lớn, cần chỉnh `load_model` và khởi tạo `LargeTextToImageModel` phù hợp checkpoint của bạn.
- Ví dụ tổng quát (sau khi chỉnh `predict.py`):
   ```bash
   python predict.py --text "a photo of a cat" --checkpoint checkpoints/text2img_small.pth --out cat.png --device cuda
   ```

### Map expert sang nhiều GPU (MoE text encoder)
- Trong `main.py`, khi khởi tạo config:
  ```python
  cfg = LargeT2IConfig(
      ...,
      text_expert_devices=["cuda:0","cuda:0","cuda:1","cuda:1"],  # ví dụ 4 expert chia 2 GPU
  )
  model = LargeTextToImageModel(cfg).to("cuda:0")
  ```

### Tips cải thiện chất lượng/tốc độ
- Giảm `IMAGE_SIZE`, `latent_dim`, `base_channels` nếu cần tốc độ/VRAM; tăng để chất lượng cao hơn.
- Dùng AMP (đã bật sẵn) và pin_memory/num_workers>0 trên Linux để tăng throughput.
- Loss: hiện tại L1, có thể thêm perceptual/GAN/CLIP loss để ảnh sắc nét hơn.

### Lưu ý
- Dataset HF cần kết nối mạng hoặc cache sẵn. Nếu chạy offline, thay bằng dataset cục bộ (ImageFolder + caption).
- `predict.py` chưa khớp sẵn với model lớn; chỉnh tay nếu bạn muốn generate từ checkpoint mới.
