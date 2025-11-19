import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import PIL
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,IterableDataset
import matplotlib.pyplot as plt
from datasets import load_dataset
from io import BytesIO
import requests
from huggingface_hub import login
import os

#s'authertifier avec son token sur hf, necessite de creer une variable d'environnement HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)
#chargement du dataset (ImageNet)

ImageNetTrain = load_dataset(
    "ILSVRC/imagenet-1k",
    split="train",
    streaming=True
)

ImageNetTest = load_dataset(
    "ILSVRC/imagenet-1k",
    split="test",
    streaming=True
)

#creation d'un dataset iterable
class ImageNetIT(IterableDataset):
    def __init__(self, ds_iterable, transform=None, limit=None):
        self.ds_iterable = ds_iterable
        self.transform = transform
        self.limit = limit

    def __iter__(self):
        for i, row in enumerate(self.ds_iterable):
            if i >= self.limit:
                break
            
            img = row["image"].convert("RGB")
            
            img = self.transform(img)
            yield img, row["label"]    

#Mise en forme du subset
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])  # moyenne et écart-type du dataset ImageNet
])

#creation de ds iterable
ImageNetTrainIT = ImageNetIT(ImageNetTrain,transform=transform,limit=2e6)
ImageNetTestIT = ImageNetIT(ImageNetTest,transform=transform,limit=1e4)


#Creation des batchs
ImageNetTrainLoader = DataLoader(ImageNetTrainIT, batch_size=32) #entrainement par batch de 32 images
ImageNetTestLoader = DataLoader(ImageNetTestIT, batch_size=1000) #test sur 1000 images


#definition du modele CNN (Resnet)
class Resnet(nn.Module):
    def __init__(self , *args, **kwargs):
        super().__init__(*args, **kwargs)

        #1er  bloc
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.bn64 = nn.BatchNorm2d(num_features=64)

        #2eme bloc, bottleneck
        self.bn256 = nn.BatchNorm2d(num_features=256)
        self.conv2_1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,stride=1,padding=0) #suite du maxpool 
        self.conv2_1bis = nn.Conv2d(in_channels=256,out_channels=64,kernel_size=1,stride=1,padding=0) 
        self.conv2_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1) #réduction
        self.conv2_3 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1,stride=1,padding=0) #expansion
        self.conv2_skip = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1,stride=1,padding=0) #resize du skip

        #3eme bloc , bottleneck 2
        self.bn128 = nn.BatchNorm2d(num_features=128)
        self.bn512 = nn.BatchNorm2d(num_features=512)
        self.conv3_1 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1,stride=1,padding=0) #suite bloc 3 
        self.conv3_1bis = nn.Conv2d(in_channels=512,out_channels=128,kernel_size=1,stride=1,padding=0) 
        self.conv3_2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1) #reduction b3
        self.conv3_2bis = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1) #reduction b3
        self.conv3_3 = nn.Conv2d(in_channels=128,out_channels=512,kernel_size=1,stride=1,padding=0) #expansion b3
        self.conv3_skip = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1,stride=2,padding=0) 
        
        #4eme bloc
        self.bn1024 = nn.BatchNorm2d(num_features=1024)
        self.conv4_1 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1,padding=0) #suite bloc 4 
        self.conv4_1bis = nn.Conv2d(in_channels=1024,out_channels=256,kernel_size=1,stride=1,padding=0) 
        self.conv4_2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1) #reduction b4
        self.conv4_2bis = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1) #reduction b4
        self.conv4_3 = nn.Conv2d(in_channels=256,out_channels=1024,kernel_size=1,stride=1,padding=0) #expansion b4
        self.conv4_skip = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=1,stride=2,padding=0)   

        #5eme bloc
        self.bn2048 = nn.BatchNorm2d(num_features=2048)
        self.conv5_1 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1,stride=1,padding=0) #suite bloc 4 
        self.conv5_1bis = nn.Conv2d(in_channels=2048,out_channels=512,kernel_size=1,stride=1,padding=0) 
        self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=2,padding=1) #reduction b5
        self.conv5_2bis = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1) #reduction b5
        self.conv5_3 = nn.Conv2d(in_channels=512,out_channels=2048,kernel_size=1,stride=1,padding=0) #expansion b5
        self.conv5_skip = nn.Conv2d(in_channels=1024,out_channels=2048,kernel_size=1,stride=2,padding=0)

        #6eme bloc
        self.fl = nn.Linear(in_features=2048,out_features=1000)

    def forward(self,x):
        
        #1er bloc
        x = F.relu(self.bn64(self.conv1(x))) #input 3c 224*224, output 64c112x112
        x = F.max_pool2d(input=x, kernel_size=3,stride=2) #out 64c56*56

        #2eme bloc
        #premier bottleneck
        x_skip = x #64c56*56
        x = F.relu(self.bn64(self.conv2_1(x)))  #out 64c56*56
        x = F.relu(self.bn64(self.conv2_2(x))) #out 64c56*56
        x = F.relu(self.bn256(self.conv2_3(x))) #out 256c56*56
        x_skip = self.conv2_skip(x_skip) #out 256c56*56
        x = F.relu(x+x_skip) #out 256c56*56
        
        #2 et 3 eme bottleneck
        for i in range(2):
            x_skip = x #256c56*65
            x = F.relu(self.bn64(self.conv2_1bis(x)))  #out 64c56*56
            x = F.relu(self.bn64(self.conv2_2(x))) #out 64c56*56
            x = self.bn256(self.conv2_3(x)) #out 256c56*56
            x = F.relu(x+x_skip) #out 256c56*56
        




        #3eme bloc
        #premier bottleneck
        x_skip = x #256c56*56
        x = F.relu(self.bn128(self.conv3_1(x)))  #out 128c56*56
        x = F.relu(self.bn128(self.conv3_2(x))) #out 128c28*28
        x = self.bn512(self.conv3_3(x)) #out 512c28*28
        x_skip = self.conv3_skip(x_skip) #out 512c28*28
        x = F.relu(x+x_skip) #out 512c128*28
        
        #bottleneck 2,3,4
        for i in range(3):
            x_skip = x #512c28*28
            x = F.relu(self.bn128(self.conv3_1bis(x)))  #out 512c28*28
            x = F.relu(self.bn128(self.conv3_2bis(x))) #out 512c28*28
            x = self.bn512(self.conv3_3(x)) #out 512c28*28
            x = F.relu(x+x_skip) #out 512c28*28
        






        #4eme bloc
        #premier bottleneck
        x_skip = x #512c28*28
        x = F.relu(self.bn256(self.conv4_1(x)))  #out 256c28*28
        x = F.relu(self.bn256(self.conv4_2(x))) #out 256c14*14
        x = self.bn1024(self.conv4_3(x)) #out 1024c14*14
        x_skip = self.conv4_skip(x_skip)
        x = F.relu(x+x_skip) #out 1024c14*14
        
        #bottleneck 2,3,4,5,6
        for i in range(5):
            x_skip = x #1024c14*14
            x = F.relu(self.bn256(self.conv4_1bis(x)))  ##out 256c14*14
            x = F.relu(self.bn256(self.conv4_2bis(x))) #out 256c14*14
            x = self.bn1024(self.conv4_3(x)) #out 1024c14*14
            x = F.relu(x+x_skip)    #out 1024c14*14     


        #5eme bloc
        #premier bottleneck
        x_skip = x # 1024c14*14
        x = F.relu(self.bn512(self.conv5_1(x)))  #out 512c14*14
        x = F.relu(self.bn512(self.conv5_2(x))) #out 512c7*7
        x = self.bn2048(self.conv5_3(x)) #out 2048c7*7
        x_skip = self.conv5_skip(x_skip)
        x = F.relu(x+x_skip) #out 2048c7*7
        
        #bottleneck 2,3
        for i in range(2):
            x_skip = x #out 2048c7*7
            x = F.relu(self.bn512(self.conv5_1bis(x)))  #out 512c7*7
            x = F.relu(self.bn512(self.conv5_2bis(x))) #out 512c7*7
            x = self.bn2048(self.conv5_3(x)) #out 2048c7*7
            x = F.relu(x+x_skip) #out 2048c7*7
    

        #6eme bloc
        x = F.avg_pool2d(x,kernel_size=7,stride=1)
        x = torch.flatten(x, 1)
        x = self.fl(x)

        return x
    

#execution
model = Resnet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # dans le cadre ou quelqu'un a CUDA, chez moi c'est sur le CPU :(
model.to(device)

#Définition de l’optimiseur et de la fonction de perte ici tres classique adam + log loss
optimizer = optim.SGD(params=model.parameters(),
                      lr=0.1,
                      momentum=0.9,
                      weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                      step_size=30,   # Diminue tous les 30 epochs le learning rate 
                                      gamma=0.1)  

criterion = nn.CrossEntropyLoss()



#def de l'entraînement
def entrainement(model, device, train_loader, optimizer, epoch):
    model.train() #on appelle notre CNN a faire l'entrainement


    for batch_idx, (image, label) in enumerate(train_loader): #sur chaqu'un des batchs
        image, label = image.to(device), label.to(device) #envoie l'image d'entrainement et son label sur le CPU/GPU


        #forwardpass
        optimizer.zero_grad() #init des gradients
        output = model(image) #calcul le resultats le pach par un forward pass
        loss = criterion(output, label) #calcul de la fonction de cout
        
        #backward pass
        loss.backward() #realise un backwardpass
        optimizer.step() #met à jour les parametres
        
        if batch_idx % 100 == 0: #affiche l'avancée de l'entrainement tout les 100 batchs
            print(f"Train Epoch: {epoch} [{batch_idx}]")
            
#def test du modele
def test(model, device, test_loader):



    model.eval() #mets le modele en mode tet
    test_loss = 0 #init des indicateurs
    correct = 0
    
    with torch.no_grad(): #sans recalculer les gradients pour ne pas perde du temps
        for image, label in test_loader: #pour chaque image et label de notre set de test
            image, label = image.to(device), label.to(device)
            output = model(image) #on evalue avec notre modele 
            test_loss += criterion(output, label).item() #somme les ecarts avec le label
            pred = output.argmax(dim=1) #fait une prediction en choisisant le neurone de sortie le plus grand
            correct += pred.eq(label).sum().item() #somme les predictions justes




#Boucle principale, on train et test 100 fois
numEpoch = 100

for epoch in range(numEpoch):
    entrainement(model, device, ImageNetTrainLoader, optimizer, epoch)
    test(model, device, ImageNetTestLoader)

#Sauvegarde du modèle
torch.save(model.state_dict(), "./model/Resnet50v1.pth")
print("Modèle sauvegardé sous './model/Resnet50v1.pth'")

