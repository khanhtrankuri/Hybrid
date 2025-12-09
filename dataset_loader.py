import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoTokenizer
import requests
from PIL import Image
import io
import re

class CleanedLaionDataset(IterableDataset):
    def __init__(
        self, 
        dataset_name="linyq/laion_text_debiased_100M", 
        split="train", 
        batch_size=32,
        min_image_size=200,
        max_aspect_ratio=3.0,
        tokenizer_name="bert-base-uncased",
        max_length=77,
        safe_mode=True
    ):
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.batch_size = batch_size
        self.min_image_size = min_image_size
        self.max_aspect_ratio = max_aspect_ratio
        self.safe_mode = safe_mode
        
        # Text Processing
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Image Processing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)), # Resize to fixed size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def is_valid_image_metadata(self, item):
        """
        Filter based on metadata columns before downloading image.
        """
        # 1. NSFW Check
        if self.safe_mode:
            nsfw_status = item.get("NSFW", "UNKNOWN")
            if nsfw_status in ["NSFW", "Likely"]:
                return False
        
        # 2. Size Check
        width = item.get("WIDTH")
        height = item.get("HEIGHT")
        if width is None or height is None:
            return False
            
        if width < self.min_image_size or height < self.min_image_size:
            return False
            
        # 3. Aspect Ratio Check (avoid extremely wide/tall images)
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio > self.max_aspect_ratio or aspect_ratio < (1 / self.max_aspect_ratio):
            return False
            
        # 4. Text Check
        text = item.get("TEXT", "")
        if not text or len(text.strip()) < 5:
            return False
            
        return True

    def download_and_process_image(self, url):
        try:
            response = requests.get(url, timeout=5, stream=True)
            if response.status_code != 200:
                return None
                
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return self.image_transform(image)
        except Exception:
            return None

    def process_text(self, text):
        # Basic cleaning
        text = re.sub(r"\s+", " ", text).strip()
        
        # Tokenization
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)

    def __iter__(self):
        for item in self.dataset:
            # 1. Metadata Filtering (Fast)
            if not self.is_valid_image_metadata(item):
                continue
                
            url = item.get("URL")
            text = item.get("TEXT")
            
            # 2. Image Download & Processing (Slow)
            pixel_values = self.download_and_process_image(url)
            if pixel_values is None:
                continue
                
            # 3. Text Processing
            input_ids, attention_mask = self.process_text(text)
            
            yield {
                "image": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "original_text": text
            }

def get_dataloader(batch_size=32, num_workers=0):
    dataset = CleanedLaionDataset(batch_size=batch_size)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

if __name__ == "__main__":
    # Demo usage
    print("Initializing dataset...")
    dl = get_dataloader(batch_size=4)
    
    print("Fetching first batch (this may take a moment to download images)...")
    for i, batch in enumerate(dl):
        print(f"Batch {i+1}:")
        print(f" - Image Shape: {batch['image'].shape}")
        print(f" - Text Shape: {batch['input_ids'].shape}")
        print(f" - Sample Text: {batch['original_text'][0]}")
        
        if i >= 2: # Stop after 3 batches
            break
