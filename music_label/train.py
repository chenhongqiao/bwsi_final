import torch
import random
import os
from model import FCN
from torch.nn import BCEWithLogitsLoss
import torch
from torch.optim import Adam
from torchmetrics.classification import MultilabelAUROC
import os
import json
import matplotlib.pyplot as plt


def load_dataset(device):
    x_train = torch.load("dataset/train_src.pt", map_location=device)
    y_train = torch.load("dataset/train_tgt.pt", map_location=device)
    x_test = torch.load("dataset/test_src.pt", map_location=device)
    y_test = torch.load("dataset/test_tgt.pt", map_location=device)
    return x_train, y_train, x_test, y_test


def train_music_label(total_epoch, lr, batch_size):
    x_train, y_train, x_test, y_test = load_dataset(device="cuda")
    model = FCN().to("cuda")
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = BCEWithLogitsLoss().to("cuda")
    ml_auroc = MultilabelAUROC(num_labels=8, average="macro")
    metrics = []
    for epoch in range(1, total_epoch + 1):
        model.train()
        idxs = torch.arange(len(x_train))  # -> array([0, 1, ..., num_train-1])
        random.shuffle(idxs)  # shuffles indices in-place
        for batch_cnt in range(len(x_train) // batch_size):
            batch_indices = idxs[batch_cnt * batch_size : (batch_cnt + 1) * batch_size]
            optimizer.zero_grad()
            batch = x_train[
                batch_indices
            ]  # <COGSTUB> get the random batch of our training data
            truth = y_train[
                batch_indices
            ]  # <COGSTUB> get the true labels for this batch of images

            batch = batch[:, None, :, :]
            # compute your model's predictions for this batch
            prediction = model(batch)  # <COGSTUB>

            # compute the loss that compares the model's predictions to the true values
            loss = loss_fn(prediction, truth)  # <COGSTUB>  use softmax_cross_entropy

            # Use mygrad compute the derivatives for your model's parameters, so
            # that we can perform gradient descent.
            loss.backward()  # <COGLINE>

            # execute one step of gradient descent by calling optim.step()
            optimizer.step()  # <COGLINE>
            # set the training loss and accuracy

        model.eval()
        epoch_predictions = []
        epoch_truth = []
        total_loss = 0
        for batch_cnt in range(0, len(x_test) // batch_size):
            idxs = torch.arange(len(x_test))
            batch_indices = idxs[
                batch_cnt * batch_size : (batch_cnt + 1) * batch_size
            ]  # <COGSTUB>  get the batch of our **test** data
            batch = x_test[
                batch_indices
            ]  # <COGSTUB>  get the batch of our **test** labels
            batch = batch[:, None, :, :]

            with torch.no_grad():
                # get your model's prediction on the test-batch
                prediction = model(batch)  # <COGSTUB>

                # get the truth values for that test-batch
                truth = y_test[batch_indices]  # <COGSTUB>

                loss = loss_fn(prediction, truth)
                total_loss += loss.item()

                epoch_predictions.append(torch.sigmoid(prediction))
                epoch_truth.append(truth)

        epoch_predictions = torch.vstack(epoch_predictions)
        epoch_truth = torch.vstack(epoch_truth).int()

        score = ml_auroc(epoch_predictions, epoch_truth)

        print(f"Epoch {epoch} AUROC {score} Loss {total_loss}")

        metrics.append([score.item(), total_loss])

        if epoch % 5 == 0:
            if not os.path.exists(f"checkpoints/{epoch}"):
                os.mkdir(f"checkpoints/{epoch}")
            torch.save(model.state_dict(), f"checkpoints/{epoch}/model.pt")
            torch.save(optimizer.state_dict(), f"checkpoints/{epoch}/optimizer.pt")

            with open(f"checkpoints/{epoch}/metrics.json", "w") as file:
                file.write(json.dumps(metrics))


def graphing(epoch):
    with open(f"checkpoints/{epoch}/metrics.json", "r") as file:
        metrics = json.loads(file.read())

    auroc, loss = list(zip(*metrics))
    x = list(range(1, epoch + 1))
    plt.plot(x, auroc)
    plt.title("AUROC Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("AUROC")
    plt.savefig("figures/plot.png")


if __name__ == "__main__":
    train_music_label(50, lr=1e-6, batch_size=64)
    graphing(50)
