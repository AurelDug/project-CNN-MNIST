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
login(token=HF_TOKEN)

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
            response = requests.get(row["image"], timeout=10)
            img = PIL.Image.open(BytesIO(response.content)).convert("RGB")
            img = self.transform(img)
            yield img, row["text"]    

#Mise en forme du subset
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])  # moyenne et écart-type du dataset ImageNet
])

#creation de ds iterable
ImageNetTestIT = ImageNetIT(ImageNetTest,transform=transform,limit=1e4)
ImageNetTrainIT = ImageNetIT(ImageNetTest,transform=transform,limit=2e6)

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
        self.conv2_1bis = nn.Conv2d(in_channels=256,out_channels=64,kernel_size=1,stride=1,padding=0) #réduction
        self.conv2_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1) 
        self.conv2_3 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1,stride=1,padding=0) #expansion
        self.conv2_skip = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1,stride=1,padding=0) #resize du skip

        #3eme bloc , bottleneck 2
        self.bn1024 = nn.BatchNorm2d(num_features=1024)
        self.conv3_1 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,stride=1,padding=0) #suite bloc 2 
        self.conv3_1bis = nn.Conv2d(in_channels=1024,out_channels=256,kernel_size=1,stride=1,padding=0) #reduction b3
        self.conv3_2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1) #conv b3 
        self.conv3_3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,stride=1,padding=0) #expansion b3
        self.conv3_skip = nn.Conv2d(in_channels=256,out_channels=1024,kernel_size=1,stride=1,padding=0) #suite du maxpool 
        
        #4eme bloc



        #5eme bloc
        #1/4 des neuronnes sont aleatoirement deconnecter durant l'entrainement a chaque epoch afin d'eviter d'overfit le cnn
        self.dropout1 = nn.Dropout(0.25)





    def forward(self,x):
        skip = x

        #1er bloc
        x = F.relu(self.bn64(self.conv1(x))) #input 3c 224*224, output 64c112x112
        x = F.max_pool2d(input=x, kernel_size=3,stride=2) #out 64c56*56

        #2eme bloc
        #premier bottleneck
        x_skip = x
        x = F.relu(self.bn64(self.conv2_1(x)))  #out 64c56*56
        x = F.relu(self.bn64(self.conv2_2(x))) #out 64c56*56
        x = F.relu(self.bn256(self.conv2_3(x))) #out 256c56*56
        x_skip = self.conv2_skip(x)
        x = F.relu(x+x_skip) 
        
        #2 et 3 eme bottleneck
        for i in range(2):
            x_skip = x #256c*56*56
            x = F.relu(self.bn64(self.conv2_1bis(x)))  #out 64c56*56
            x = F.relu(self.bn64(self.conv2_2(x))) #out 64c56*56
            x = F.relu(self.bn256(self.conv2_3(x))) #out 256c56*56
            x_skip = self.conv2_skip(x)
            x = F.relu(x+x_skip) 

        #3eme bloc
        #premier bottleneck
        x_skip = x
        x = F.relu(self.bn256(self.conv3_1(x)))  #out 64c56*56
        x = F.relu(self.bn256(self.conv3_2(x))) #out 64c56*56
        x = F.relu(self.bn1024(self.conv3_3(x))) #out 256c56*56
        x_skip = self.conv3_skip(x)
        x = F.relu(x+x_skip) 
        
        #bottleneck 2,3,4
        for i in range(3):
            x_skip = x #256c*56*56
            x = F.relu(self.bn64(self.conv2_1bis(x)))  #out 64c56*56
            x = F.relu(self.bn64(self.conv2_2(x))) #out 64c56*56
            x = F.relu(self.bn256(self.conv2_3(x))) #out 256c56*56
            x_skip = self.conv2_skip(x)
            x = F.relu(x+x_skip) 
        

        #4eme bloc
        
        #5eme bloc
    
    
        return x
    

#execution
model = Resnet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # dans le cadre ou quelqu'un a CUDA, chez moi c'est sur le CPU :(
model.to(device)

#Définition de l’optimiseur et de la fonction de perte ici tres classique adam + log loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()



