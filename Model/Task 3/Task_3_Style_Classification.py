import os
import argparse
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from models import CombinedModel, SiameseModel
from torch.utils.data import Dataset

# Initialize the model
num_categories = 6  
num_styles = 17     
combined_model_path = 'Model/Task 3/ResNetStyle.pth'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the CombinedModel for both category and style classification
combined_model = CombinedModel(num_categories, num_styles)
combined_model.load_state_dict(torch.load(combined_model_path, map_location=DEVICE))
combined_model.eval()
combined_model.to(DEVICE)

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((120, 120)),        
    transforms.CenterCrop((110, 110)),            
    transforms.ToTensor(),                
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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
    distances.sort(key=lambda x: x[0], reverse=True)
    return distances[:top_k]

# Define a function to classify the input image
def classify_image(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    image = image.to(device)

    model.eval()
    with torch.no_grad():
        category_logits, style_logits = model(image)
        
        # Get the predicted category and style
        category_prob = F.softmax(category_logits, dim=1)
        style_prob = F.softmax(style_logits, dim=1)

        predicted_category = category_prob.argmax(dim=1).item()
        predicted_style = style_prob.argmax(dim=1).item()

    return predicted_category, predicted_style

def main():
    # Define category and style mappings
    category_mapping = {0: 'beds', 1: 'chairs', 2: 'dressers', 3: 'lamps', 4: 'sofas', 5: 'tables'}
    style_mapping = {
        0: 'Asian', 
        1: 'Beach', 
        2: 'Contemporary', 
        3: 'Craftsman', 
        4: 'Eclectic', 
        5: 'Farmhouse', 
        6: 'Industrial', 
        7: 'Mediterranean', 
        8: 'Midcentury', 
        9: 'Modern', 
        10: 'Rustic', 
        11: 'Scandinavian', 
        12: 'Southwestern', 
        13: 'Traditional', 
        14: 'Transitional', 
        15: 'Tropical', 
        16: 'Victorian'
    }

    parser = argparse.ArgumentParser(description='Classify an image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file')

    args = parser.parse_args()

    # Classify the image
    predicted_category_idx, predicted_style_idx = classify_image(args.image_path, combined_model, DEVICE)

    predicted_category = category_mapping[predicted_category_idx]
    predicted_style = style_mapping[predicted_style_idx]

    print(f'Predicted Category: {predicted_category}')
    print(f'Predicted Style: {predicted_style}')

    # Load the Siamese model and transformation pipeline for recommendation
    siamese_model_path = 'Model/Task 2/Siamese.pth'
    siamese_model = SiameseModel((120, 120, 3)).to(DEVICE)
    siamese_model.load_state_dict(torch.load(siamese_model_path, map_location=DEVICE))
    siamese_model.eval()
    print("Siamese model loaded for inference.")

    with open('Model/Task 2/Transform.pkl', 'rb') as f:
        siamese_transform = pickle.load(f)
    print("Transformation pipeline loaded.")

    # Construct the folder path
    folder_path = os.path.join('Data', 'Furniture_Data', predicted_category, predicted_style)
    print(f"Looking for images in: {folder_path}")

    # Check the parent directory
    parent_dir = os.path.dirname(folder_path)
    if not os.path.exists(parent_dir):
        print(f"Error: The parent directory {parent_dir} does not exist.")
        return

    if not os.path.exists(folder_path):
        print(f"Error: The path {folder_path} does not exist.")
        return

    img_paths = get_image_paths(folder_path)
    if not img_paths:
        print(f"No images found in the path {folder_path}.")
        return

    dataset = InferenceDataset(img_paths, siamese_transform)

    # Perform image recommendation
    recommendations = recommend_images(args.image_path, dataset, siamese_model, siamese_transform, DEVICE, top_k=10)
    for dist, path in recommendations:
        print(f"Image: {path}, Distance: {dist}")

if __name__ == '__main__':
    main()
