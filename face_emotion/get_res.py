import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import cv2
from camera import take_picture

input_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transform = transforms.Compose(
    [
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 8)
model.load_state_dict(torch.load("ft_weights.pt", map_location=torch.device("cpu")))
optimizer = optim.Adam(model.parameters(), lr=1e-5, eps=1e-6)

criterion = nn.CrossEntropyLoss()

emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

if __name__ == "__main__":
    model.eval()
    img_path = "img_test.png"
    print(f"Image Path: {img_path}")
    img = Image.open(img_path).convert("RGB")
    img = data_transform(img)

    with torch.autograd.set_grad_enabled(False):
        output = model(img.unsqueeze(0))
        print(torch.softmax(output, dim=1).float())
        _, pred = torch.max(output, dim=1)
        print(emotions[int(pred)])
    print("done")
