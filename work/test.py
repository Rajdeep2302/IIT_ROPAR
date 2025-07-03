import os
import cv2
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import segmentation_models_pytorch as smp
from segment_anything import sam_model_registry, SamPredictor

# Set device (CPU, CUDA, or MPS)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Configuration
IMAGE_DIR = '/Users/akashsaha/Desktop/Ropar_Computer_Vision_Research/g-5-2SMI/JPEG'  # path to JPG images
BATCH_SIZE = 4
EPOCHS = 10
IMAGE_SIZE = 256
SAM_CHECKPOINT = '/Users/akashsaha/Downloads/sam_vit_h_4b8939.pth'  # path to MedSAM checkpoint

# Load MedSAM model
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam.to(DEVICE)
predictor = SamPredictor(sam)

# Dataset class to handle images and pseudo-labels from MedSAM
class BoneDatasetSAM(Dataset):
    def __init__(self, image_dir):
        # Get image paths from the directory
        self.image_paths = glob.glob(os.path.join(image_dir, '*.jpeg'))  # Adjusted for .jpeg extension as well
        print(f"Found {len(self.image_paths)} images.")
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),  # MedSAM expects 3 channels
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None  # Skip problematic images

        # Apply transformations to the image
        img_resized = self.transform(image)
        img_np = (img_resized.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        # Generate pseudo-label (mask) using MedSAM
        predictor.set_image(img_np)
        h, w = img_np.shape[:2]
        
        # Create a full-image prompt by setting the box around the entire image
        box = np.array([0, 0, w, h])  # [x1, y1, x2, y2] coordinates for the box
        
        # Create point_coords and point_labels for guiding the segmentation
        point_coords = np.array([[w // 2, h // 2]])  # Use center point of the image
        point_labels = np.array([1])  # Label the point as part of the object
        
        # Predict the mask using the point_coords and point_labels
        masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels)
        
        mask = masks[0].float()

        # Resize the image and mask to match the target size
        resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        img_resized = resize(img_resized)
        mask_resized = resize(mask)

        # Normalize the image
        normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        img_normalized = normalize(img_resized)

        return img_normalized, mask_resized.unsqueeze(0)

# U-Net model initialization
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
).to(DEVICE)

# DataLoader initialization
train_dataset = BoneDatasetSAM(IMAGE_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, masks in train_loader:
        # Skip any problematic images
        if images is None or masks is None:
            continue
        
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

# Inference function to segment a single image
def infer_single_image(img_path, save_path='pred_mask.png'):
    model.eval()
    with torch.no_grad():
        image = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        output = model(img_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8) * 255
        cv2.imwrite(save_path, mask)
        print(f"Saved predicted mask to {save_path}")

# Example usage for inference
infer_single_image('/Users/akashsaha/Desktop/Ropar_Computer_Vision_Research/g-5-2SMI/JPEG/G-5-2_trab._Raw import  W0.00 L0.00_000.jpeg', save_path='./pred_mask.png')
