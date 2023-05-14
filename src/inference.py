import argparse
import os

import cv2
import numpy as np
import torch
from torchvision import transforms

from model import CustomCNN


CLASS_NAMES = ('Boots', 'Heels', 'Sneakers')
IMAGE_SIZE = 256


def load_model(checkpoints_path: str, device: str) -> CustomCNN:
    model = CustomCNN(len(CLASS_NAMES))
    checkpoints = torch.load(checkpoints_path, map_location=device)

    model.load_state_dict(checkpoints['model_state_dict'])

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify an image using a custom CNN model.')
    parser.add_argument('--checkpoints_path', type=str, required=True, help='Path to the model checkpoints.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to classify.')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cpu or cuda).')
    args = parser.parse_args()
    
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = load_model(args.checkpoints_path, args.device)

    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(args.device)

    output = model(image)
    output = output.detach().numpy()
    pred_class_name = CLASS_NAMES[np.argmax(output[0])]

    print(pred_class_name)
