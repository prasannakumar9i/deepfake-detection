import logging
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from deepfake.deepfakeai.utils import DeepfakeDataset
from deepfake.deepfakeai.models.base_model import CNN_ViT_LSTM

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enhanced image transforms (same, but can reduce size if needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),                 # Standardize input size
    transforms.RandomHorizontalFlip(p=0.5),        # Helps generalization
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def train_model(config: dict[str, dict]):
    """
    Main function for training the Model
    """
    image_dataset = config["imgdir"] / "train"
    models_dir = config["modelsdir"]
    model_name = f"{config['model_name']}.pth"
    model_path = os.path.join(str(models_dir), model_name)

    # Use smaller backbones for limited GPU memory
    cnn_backbone = config.get("models", {}).get("cnn_backbone")  
    vit_backbone = config.get("models", {}).get("vit_backbone")

    # Set batch size to 1 or 2 due to 4GB GPU
    train_dataset = DeepfakeDataset(
        root_dir=str(image_dataset),
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,  # Reduced batch size
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )

    model = CNN_ViT_LSTM(
        cnn_backbone=cnn_backbone,
        vit_backbone=vit_backbone,
        hidden_dim=256,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Check if unsqueeze(1) is required — if model expects 5D input
            # If not needed, remove or comment out
            imgs = imgs.unsqueeze(1)  # [B, 1, C, H, W] for temporal modeling (keep if model requires)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    # Save model weights
    torch.save(model.state_dict(), model_path)
