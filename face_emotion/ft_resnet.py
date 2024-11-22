import random
import copy
import time
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

root_dir = 'affectnet-training-data'
all_imgs = []
for dirname, _, filenames in os.walk(root_dir):
    for filename in filenames:
        all_imgs.append(os.path.join(dirname, filename))
all_imgs[:] = [x for x in all_imgs if "labels.csv" not in x]

random.shuffle(all_imgs)
train_imgs = all_imgs



#train/validation split
train_set, valid_set = train_test_split(train_imgs, test_size=0.2)

class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
            
        label = img_path.split('/')[-2]
        emotion_dict = {'anger':0, 'disgust':1, 'fear':2, 'happy':3, 'neutral':4, 'sad':5, 'surprise': 6, 'contempt':7}
        label = emotion_dict[label]
        return img, label      

input_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val':
    transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}


image_datasets = {
    'train': ImageDataset(train_set, transform=data_transforms['train']),
    'val': ImageDataset(valid_set, transform=data_transforms['val'])
}

batch_size = 32

dataloaders_dict = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False)
}


# Number of output classes in the dataset (emotions)
num_classes = 8

# Number of epochs to train for 
num_epochs = 30

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



'''
model_ft = models.vgg16(weights='DEFAULT')
model_ft.classifier[-1] = nn.Linear(model_ft.classifier[-1].in_features, 8)
'''
model_ft = models.resnet18(pretrained=True)
model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
                 
# Send the model to GPU
model_ft = model_ft.to(device)

optimizer = optim.Adam(model_ft.parameters(), lr = 2e-5, eps = 1e-6)

criterion = nn.CrossEntropyLoss()

def train_model(model, dataloaders, optimizer, criterion, num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-" * 10)
        
        #train phase
        running_loss = 0.0
        running_acc = 0.0
        model.train()

        for inputs, labels in tqdm(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.autograd.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0) 
            running_acc += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_acc / len(dataloaders['train'].dataset)
        print("{} Loss: {:.4f} Acc: {:.4f}".format('train', epoch_loss, epoch_acc))
        
        #val phasef
        running_loss = 0.0
        running_acc = 0.0
        model.eval()

        for inputs, labels in tqdm(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.autograd.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                #print(labels[0], preds[0])


            running_loss += loss.item() * inputs.size(0) 
            running_acc += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['val'].dataset)
        epoch_acc = running_acc / len(dataloaders['val'].dataset)
        print("{} Loss: {:.4f} Acc: {:.4f}".format('val', epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print("Best val Acc: {:4f}".format(best_acc))
    
    # load best model weights
    model_ft.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft, dataloaders_dict, optimizer, criterion, num_epochs=num_epochs)
torch.save(model_ft.state_dict(), "ft_weights.pt")