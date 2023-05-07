import os, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
from model import model_transference


# Define
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# Load Dataset
dataset = ImageFolder('dataset/val', T.ToTensor())
data_loader = DataLoader(dataset, 1, False, num_workers=os.cpu_count())
CLASSES = dataset.classes

# Model
model = model_transference(
    pre_trained=models.resnet18(weights='DEFAULT'),
    output_shape=2,
    load_model='Models/E10_L0.2659_A0.95.pth',
    eval=True,
    device=DEVICE
)

# Predict amount = rows * cols
rows, cols = 4, 4

plt.style.use('dark_background')
fig = plt.figure(figsize=(11, 9), num=f"{rows*cols} prediction")

for i in range(1, rows * cols + 1):
    # Select random image
    idx = torch.randint(0, len(data_loader), size=[1]).item()
    image, label = data_loader.dataset[idx]

    # Prediction
    inputs = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        outputs = model(inputs)
        prob = torch.sigmoid(outputs)
        _, pred = torch.max(outputs, 1)

    # Plot
    fig.add_subplot(rows, cols, i)
    plt.imshow(image.permute(1, 2, 0))
    prob_label = f'{CLASSES[pred]}: {prob[0, pred.item()]*100:.1f}%'
    if pred == label:
        plt.title(prob_label, fontsize=14, c='g')
    else:
        plt.title(prob_label, fontsize=14, c='r')
    plt.axis(False)
plt.show()
