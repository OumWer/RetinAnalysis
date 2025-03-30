import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
from tqdm import tqdm
import multiprocessing

# Define constants
CLASS_TEXTS = [
    "Dry AMD", "Glaucoma", "Normal Fundus", "Wet AMD", "Mild DR",
    "Moderate DR", "Severe DR", "Proliferate DR", "Cataract",
    "Hypertensive Retinopathy", "Pathological Myopia"
]

class RetinaDataset(Dataset):
    def __init__(self, image_urls, labels, processor):
        self.image_urls = image_urls
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, idx):
        try:
            response = requests.get(self.image_urls[idx], timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0), self.labels[idx]
        except Exception as e:
            print(f"Error loading {self.image_urls[idx]}: {str(e)}")
            return torch.zeros((3, 224, 224)), -1  # Return dummy data

def collate_fn(batch):
    images, labels = zip(*[(i, l) for i, l in batch if l != -1])
    return torch.stack(images), torch.tensor(labels)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Load model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load dataset
    dataset = pd.read_excel("Labeled Retinology Images.xlsx")
    image_urls = dataset['ImageURL'].tolist()
    labels = [CLASS_TEXTS.index(label) for label in dataset['Label'].tolist()]

    # Create DataLoader
    train_dataset = RetinaDataset(image_urls, labels, processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0 if multiprocessing.get_start_method() == 'spawn' else 4
    )

    # Preprocess text inputs
    text_inputs = processor(
        text=CLASS_TEXTS,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.cuda.amp.autocast():
                image_features = model.get_image_features(pixel_values=images)
                text_features = model.get_text_features(**text_inputs)
                logits_per_image = image_features @ text_features.T * model.logit_scale.exp()
            
            loss = criterion(logits_per_image, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            correct += (logits_per_image.argmax(dim=1) == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        torch.save(model.state_dict(), f"clip_retina_epoch_{epoch+1}.pt")
