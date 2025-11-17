import torch
import torchvision.models as models
from PIL import Image
from torchvision import datasets, transforms
import urllib.request


# Charger ResNet-50 pré-entraîné sur ImageNet
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Pour l'inférence
model.eval()


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Charger et prédire
img = Image.open("testManuel/verre.jpg")

img_tensor = preprocess(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    pred = output.argmax(1).item()

print(pred)

#Map l'index imageNet
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = [line.strip() for line in urllib.request.urlopen(url)]

#label la prediction avec son etiquette associée
predicted_label = imagenet_classes[pred]

print(f"Predicted class index: {pred}, label: {predicted_label}")

