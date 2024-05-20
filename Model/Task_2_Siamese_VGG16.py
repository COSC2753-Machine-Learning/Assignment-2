import os
import torch
import torchvision.transforms as transforms
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import argparse

# Define the Siamese Network using PyTorch
class SiameseNetwork(nn.Module):
    def __init__(self, input_shape, embedding_dim=48):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.global_avg_pool(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x

# Euclidean distance layer
class EuclideanDistance(nn.Module):
    def forward(self, featsA, featsB):
        return F.pairwise_distance(featsA, featsB, keepdim=True)

# Define the complete Siamese Network model
class SiameseModel(nn.Module):
    def __init__(self, input_shape, embedding_dim=48):
        super(SiameseModel, self).__init__()
        self.feature_extractor = SiameseNetwork(input_shape, embedding_dim)
        self.euclidean_distance = EuclideanDistance()
        self.fc = nn.Linear(1, 1)

    def forward(self, inputA, inputB):
        featsA = self.feature_extractor(inputA)
        featsB = self.feature_extractor(inputB)
        distance = self.euclidean_distance(featsA, featsB)
        output = torch.sigmoid(self.fc(distance))
        return output

# Dataset class for inference
class InferenceDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Function to get all image paths from a folder
def get_image_paths(folder_path):
    img_paths = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                img_paths.append(os.path.join(root, file))
    return img_paths

# Image recommendation function
def recommend_images(target_image_path, dataset, model, top_k=10):
    target_image = Image.open(target_image_path)
    if transform:
        target_image = transform(target_image)
    target_image = target_image.unsqueeze(0).to(DEVICE)

    distances = []
    for img, path in dataset:
        img = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(target_image, img)
        distances.append((output.item(), path))

    # Sort by distance (ascending order)
    distances.sort(key=lambda x: x[0])
    return distances[:top_k]

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Image Recommendation using Siamese Network")
    parser.add_argument("--target_image", type=str, required=True, help="Path to the target image")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top recommendations to return")
    args = parser.parse_args()

    # Set device
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the model and transformation pipeline
    IMG_SHAPE = (120, 120, 3)  # Update this based on your image size
    model = SiameseModel(IMG_SHAPE).to(DEVICE)
    model.load_state_dict(torch.load('siamese_model.pth', map_location=DEVICE))
    model.eval()
    print("Model loaded for inference.")

    with open('transform.pkl', 'rb') as f:
        transform = pickle.load(f)
    print("Transformation pipeline loaded.")

    # Get all image paths from the folder
    img_paths = get_image_paths(args.image_folder)
    dataset = InferenceDataset(img_paths, transform)

    # Perform image recommendation
    recommendations = recommend_images(args.target_image, dataset, model, top_k=args.top_k)
    for dist, path in recommendations:
        print(f"Image: {path}, Distance: {dist}")
