import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#Aquisition des données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # moyenne et écart-type du dataset MNIST
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True) # 60k images d'entrainement
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True) # 10k images de test

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #entrainement par batch de 64 images
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False) #test sur 1000 images

#Définition du modèle CNN
class CNN(nn.Module):
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

model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # dans le cadre ou quelqu'un a CUDA, chez moi c'est sur le CPU :(
model.to(device)

#Définition de l’optimiseur et de la fonction de perte ici tres classique adam + log loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#def de l'entraînement
def train(model, device, train_loader, optimizer, epoch):
    model.train() #on appelle notre CNN a faire l'entrainement
    for batch_idx, (data, target) in enumerate(train_loader): #sur chaqu'un des batchs
        data, target = data.to(device), target.to(device) #envoie l'image d'entrainement et son label sur le CPU/GPU
        optimizer.zero_grad() #init des gradients
        output = model(data) #calcul le resultats le pach par un forward pass
        loss = criterion(output, target) #calcul de la fonction de cout
        loss.backward() #realise un backwardpass
        optimizer.step() #met a jour les parametres
        
        if batch_idx % 100 == 0: #affiche l'avancée de l'entrainement tout les 100 batchs
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

# test du modele
def test(model, device, test_loader):
    model.eval() #mets le modele en mode tet
    test_loss = 0 #init des indicateurs
    correct = 0
    
    with torch.no_grad(): #sans recalculer les gradients pour ne pas perde du temps
        for data, target in test_loader: #pour chaque image et label de notre set de test
            data, target = data.to(device), target.to(device)
            output = model(data) #on evalue avec notre modele 
            test_loss += criterion(output, target).item() #somme les ecarts avec le label
            pred = output.argmax(dim=1) #fait une prediction en choisisant le neurone de sortie le plus grand
            correct += pred.eq(target).sum().item() #somme les predictions justes

    test_loss /= len(test_loader) #estime l'erreur moyenne
    acc = 100. * correct / len(test_loader.dataset) #donne la précision du modele
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n")

#Boucle principale, on train et test 5 fois 
for epoch in range(1, 6):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

#Sauvegarde du modèle
torch.save(model.state_dict(), "./model/mnist_cnn1.pth")
print("Modèle sauvegardé sous './model/mnist_cnn1.pth'")
