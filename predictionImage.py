import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

#def du modèle CNN (id a l'entrainement)
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #deux convolutions pour extraire les features du data-set
        self.conv1 = nn.Conv2d(1, 32, 3, 1)      # entrée 1 canal (image 28x28), pas de padding donc image de 26*26 en sortie 
        self.conv2 = nn.Conv2d(32, 64, 3, 1)    #idem image de 24*24 en sortie

        #1/4 des neuronnes sont aleatoirement deconnecter durant l'entrainement a chaque epoch afin d'eviter d'overfit le cnn
        self.dropout1 = nn.Dropout(0.25)
        
        #Flattent layer de 9216 neuronnes d'entrée (64*12*12) --- 12 car le maxpool 2*2 divise la taille de la feature maps par 4 24*24-->12*12
        self.fc1 = nn.Linear(9216, 128) #coucche intermédiare dans le NN pour smooth l'apprentissage
        self.fc2 = nn.Linear(128, 10) # creation de l'out layer de 0 a 9 soit 10 neuronnes 

    def forward(self, x):
        x = F.relu(self.conv1(x)) #relu classique comme fonction d'activation pour chaque convolution 
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2) # pooling 2*2 pour reduire la taille des features map
        x = self.dropout1(x) #voir init
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) #couche de sortie brut renvoier
        return x


#Charger le modèle
model = CNN()
model.load_state_dict(torch.load("./model/mnist_cnn1.pth"))
model.eval()

#mise en forme de l'image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # assure que l'image est en niveau de gris
    transforms.Resize((28, 28)),                   # redimensionne à 28x28
    transforms.ToTensor(),                         # convertit en tenseur
    transforms.Normalize((0.1307,), (0.3081,))     # normalisation MNIST
])

#Charge l'image
img = Image.open("./testManuel/1.png")
img = ImageOps.invert(img.convert('L')) #l'image est blanche sur fond noir alors que le set est noir sur fond blanc donc convertion

# Appliquer la transformation
img_t = transform(img).unsqueeze(0)

# Faire la prédiction
with torch.no_grad():
    output = model(img_t)
    pred = output.argmax(dim=1, keepdim=True)
    print("Prédiction :", pred.item())
