import os
import tkinter as tk
import torch

from pathlib import Path
from tkinter import filedialog, messagebox
from torchvision import transforms
from PIL import Image
from models import ResNet, BasicBlock

def main():
    # Get classes' name
    task_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(task_path, "../../Furniture_Data")
    classes = [Path(f.path).stem for f in os.scandir(data_path) if f.is_dir()]

    # Open dialog for user to select an image file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    # file_path is an empty string if user cancels
    if not file_path:
        return

    # Set up image transformation
    transform = transforms.Compose([
        transforms.Resize((120, 120)),        
        transforms.CenterCrop((110, 110)),            
        transforms.ToTensor(),                
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load and transform image
    image = Image.open(file_path)
    image_tensor = torch.unsqueeze(transform(image), 0)

    # Load model
    model_path = os.path.join(task_path, "resnet.pth")
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    # Make prediction
    with torch.no_grad():
        msg = f"The object in the image belongs to class '{classes[model(image_tensor).argmax()]}'"
    messagebox.showinfo("Prediction", msg)

if __name__ == '__main__':
    main()