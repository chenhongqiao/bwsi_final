# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# %matplotlib inline
import torch
import torchvision
from sklearn.model_selection import train_test_split

from tqdm import tqdm

gpu_check = torch.cuda.is_available()
if gpu_check:
    print("Training on GPU")
else:
    print("Training on CPU")

# %%
df = pd.read_csv("./data/train.csv")

# %%
Emotion = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# %%
ax = sns.countplot(df["Emotion"])
ax.set_xticks(range(len(Emotion)))
ax.set_xticklabels(Emotion.values(), rotation=45, ha="right")
ax.set_title("Dataset Count")

plt.show()

# %%
df.head()

# %%
val_size = 0.2

X = df["Pixels"].str.split(" ", expand=True)  # Split the 'Pixels' column based on space
X = np.asarray(X).astype(float)  # convert dataset to float
X = X / 255.0  # normalize
X = X.reshape(-1, 1, 48, 48)  # reshape to 4D array
y = np.asarray(df["Emotion"]).reshape(-1, 1).astype(int)  # 1D array for target values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# show the number of training and validation data
print(
    f"The number of items in X_train is {len(X_train)}.\nThe number of items in y_train is {len(y_train)}.\n"
)
print(
    f"The number of items in X_train is {len(X_val)}.\nThe number of items in y_train is {len(y_val)}."
)


# %%
# function to display samle data
def display_tensor(x, y):
    plt.figure(figsize=(5, 5))
    plt.imshow(x.reshape(48, 48), cmap="gray")
    plt.title(y)
    plt.show()


# %%
import random

ran = random.randint(0, len(X_train))
display_tensor(X[ran], Emotion[int(y[ran])])

# %%
# Display sample faces per emotion
fig = plt.figure(figsize=(25, 4))
for i in range(len(Emotion)):
    ax = fig.add_subplot(1, 7, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(X_train[np.argmax(y_train == i)].reshape(48, 48)), cmap="gray")
    ax.set_title(Emotion[int(y_train[np.argmax(y_train == i)])])

# %%
# Create PyTorch dataset from the numpy dataset
import torch.utils.data as utils
import torchvision.transforms as transforms

# Create tensors of the training dataset for PyTorch
tensor_x = torch.stack([torch.Tensor(i) for i in X_train])
tensor_y = torch.stack([torch.Tensor(i) for i in y_train])
train_data = utils.TensorDataset(tensor_x, tensor_y)

# Create tensors of the validation dataset for PyTorch
tensor_x = torch.stack([torch.Tensor(i) for i in X_val])
tensor_y = torch.stack([torch.Tensor(i) for i in y_val])
valid_data = utils.TensorDataset(tensor_x, tensor_y)

# %%
train_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.RandomHorizontalFlip()]
)
valid_transform = transforms.Compose([transforms.ToTensor()])

# %%
num_workers = 0
batch_size = 20

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
valid_loader = torch.utils.data.DataLoader(
    valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
)

# %%
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # CNN layer sees 48x48x1 image tensor
        self.conv1 = nn.Conv2d(1, 30, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(30)
        # CNN layer sees 24x24x20 image tensor
        self.conv2 = nn.Conv2d(30, 30, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(30)
        # CNN layer sees 12x12x20 image tensor
        self.conv3 = nn.Conv2d(30, 30, 7, padding=3)
        self.bn3 = nn.BatchNorm2d(30)

        self.maxpool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(30 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)

        self.dropout = nn.Dropout(p=0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # forward pass
    def forward(self, x):
        x = self.bn1(self.maxpool(F.relu(self.conv1(x))))
        x = self.bn2(self.maxpool(F.relu(self.conv2(x))))
        x = self.bn3(self.maxpool(F.relu(self.conv3(x))))
        x = x.view(-1, 6 * 6 * 30)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


model = Net()

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# %%
# training epochs
n_epoch = 200

valid_loss_ret = np.Inf
valid_loss_min = np.Inf

if gpu_check:
    model.to("cuda")

for epoch in tqdm(range(n_epoch)):
    train_loss = 0
    valid_loss = 0

    model.train()

    for data, target in train_loader:
        if gpu_check:
            data, target = data.to("cuda"), target.to("cuda")
        target = target.squeeze()
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

model.eval()

for data, target in valid_loader:
    if gpu_check:
        data, target = data.to("cuda"), target.to("cuda")
    target = target.squeeze()
    output = model(data)
    loss = criterion(output, target.long())
    valid_loss += loss.item() * data.size(0)

    # avg training loss of training and validation
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)

    if valid_loss < valid_loss_min:
        model_save_path = os.path.join("Fe_Models", "ED_model.pth")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        valid_loss_min = valid_loss

# %%
# load current best model based on previous training
model.load_state_dict(torch.load("Fe_Models/ED_model.pth"))


# %%
# predict output and compare with ground truth
def test_model(img, label, model):
    X = torch.Tensor(img.reshape(1, 1, 48, 48))
    output = model(X)
    _, pred = torch.max(output, 1)
    Emotion[int(pred)], label, pred
    display_tensor(X, f"Prediction: {Emotion[int(pred)]}\nActual:{label}")


# %%
model.to("cpu")
y_pred = []
corr = 0
wrong = []
for i in tqdm(range(len(X_val))):
    output = model(torch.Tensor(X_val[i].reshape(1, 1, 48, 48)))
    _, pred = torch.max(output, 1)
    y_pred.append(pred)
    if int(pred) == int(y_val[i]):
        corr += 1
    else:
        wrong.append(i)

print("The model's accuracy is {:.2f}%".format(corr * 100 / len(X_val)))

# %%
y_pred = np.array([int(pred.item()) for pred in y_pred])
y_val = y_val.squeeze().numpy().astype(int)

# confusion matrix
from sklearn.metrics import confusion_matrix

sns.set(rc={"figure.figsize": (6, 6)})
ax = sns.heatmap(
    confusion_matrix(y_pred, y_val),
    cmap="gray_r",
    annot=True,
    cbar=False,
    xticklabels=list(Emotion.values()),
    yticklabels=list(Emotion.values()),
)
ax.set(xlabel="Predicted", ylabel="Actual")
plt.show()
# %%
# Sample of incorrect prediction
ran = wrong[8]
test_model(X_val[ran], Emotion[int(y_val[ran])], model)
