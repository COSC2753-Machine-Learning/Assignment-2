import os
import torch
import torchvision.transforms as transforms
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import argparse

# Get the Output Dimension of VGG16    
def get_output_shape(model, image_dim, device):
    model = model.to(device)  # Ensure the model is on the same device as the dummy input
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, *image_dim).to(device)
        output = model(dummy_input)
    return output.shape[1]  # Return the number of output features

# Define the VGG16 used for feature extraction
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        
        self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_3 = nn.Sequential(        
                nn.Conv2d(in_channels=128,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),        
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_4 = nn.Sequential(   
                nn.Conv2d(in_channels=256,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),        
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),        
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),            
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_5 = nn.Sequential(
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),            
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),            
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),    
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))             
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()
                    
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # flatten
        return x
    
# Define the Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=48):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = VGG16FeatureExtractor()
        out_features = get_output_shape(self.feature_extractor, (120, 120), DEVICE)
        print(f"Output features: {out_features}")  # Debugging print statement
        self.fc = nn.Linear(out_features, embedding_dim)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Euclidean distance layer
class EuclideanDistance(nn.Module):
    def forward(self, featsA, featsB):
        return F.pairwise_distance(featsA, featsB, keepdim=True)

# Define the complete Siamese Network model
class SiameseModel(nn.Module):
    def __init__(self, embedding_dim=48):
        super(SiameseModel, self).__init__()
        self.feature_extractor = SiameseNetwork(embedding_dim)
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
def recommend_images(target_image_path, dataset, model, transform, device, top_k=10):
    target_image = Image.open(target_image_path)
    if transform:
        target_image = transform(target_image)
    target_image = target_image.unsqueeze(0).to(device)

    distances = []
    for img, path in dataset:
        img = img.unsqueeze(0).to(device)
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
    model = SiameseModel().to(DEVICE)
    model.load_state_dict(torch.load('Model/Task 2/Siamese_(VGG16).pth', map_location=DEVICE))
    model.eval()
    print("Model loaded for inference.")

    with open('Model/Task 2/Transform.pkl', 'rb') as f:
        transform = pickle.load(f)
    print("Transformation pipeline loaded.")

    # Get all image paths from the folder
    img_paths = get_image_paths(args.image_folder)
    dataset = InferenceDataset(img_paths, transform)

    # Perform image recommendation
    recommendations = recommend_images(args.target_image, dataset, model, transform, DEVICE, top_k=args.top_k)
    for dist, path in recommendations:
        print(f"Image: {path}, Distance: {dist}")
