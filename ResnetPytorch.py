import torch
import torchvision.models as models
from PIL import Image
from torchvision import datasets, transforms
import urllib.request
import cv2

# Charger ResNet-50 pré-entraîné sur ImageNet
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# Pour l'inférence
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cap = cv2.VideoCapture(1)

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir BGR->RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Prétraiter et prédire
        pil_image = Image.fromarray(rgb_frame)
        input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
        output = model(input_tensor)
        pred_class = output.argmax(1).item()
        confidence = torch.softmax(output, 1)[0][pred_class].item()

        #Map l'index imageNet
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        imagenet_classes = [line.strip() for line in urllib.request.urlopen(url)]

        predicted_label = imagenet_classes[pred_class]

        # Afficher sur la frame
        text = f"Class: {predicted_label}, Conf: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        cv2.imshow('ResNet-50 Live', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()