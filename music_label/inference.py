from .model import FCN
import torch

import os

model = FCN().to("cpu")
dir_path = os.path.dirname(os.path.realpath(__file__))
model.load_state_dict(
    torch.load(dir_path + "/checkpoints/45/model.pt", map_location=torch.device("cpu"))
)


# predictions [N, 25]
# truth [N, 25]
def accuracy(prediction, truth):
    match = torch.sum(prediction == (truth > 0.5), dim=1)
    # match [N]
    acc = torch.mean((match == 25).float())
    return acc


def pred_label(x):
    batch_size = 1

    model.eval()
    predictions = []
    idxs = torch.arange(len(x))
    for batch_cnt in range(0, len(x) // batch_size):
        batch_indices = idxs[
            batch_cnt * batch_size : (batch_cnt + 1) * batch_size
        ]  # get the batch of our **test** data
        batch = x[batch_indices]  #  get the batch of our **test** labels
        batch = batch[:, None, :, :]

        with torch.no_grad():
            # get your model's prediction on the test-batch
            prediction = model(batch)

            # get the truth values for that test-batch
            predictions.append(prediction)

    return torch.vstack(predictions).detach()
