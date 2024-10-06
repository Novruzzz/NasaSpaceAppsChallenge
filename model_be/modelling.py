import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data    = torch.load("/data/NSAC/data.pt",    weights_only=False)
targets = torch.load("/data/NSAC/targets.pt", weights_only=False)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, targets) -> None:
        self.data    = data
        self.targets = targets

    def __len__(self) -> int:
        return targets.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

dataset = Dataset(data, targets)

train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size

train_data, val_data = torch.utils.data.random_split(dataset, (train_size, val_size))

batch_size = 1

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data  , batch_size=batch_size, shuffle=True)

from sklearn.metrics import recall_score, precision_score, f1_score

# ipython-input-69-010b094198ac.py
def compute_metrics(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Convert continuous predictions to binary using a threshold (e.g., 0.5)
    y_pred = (y_pred > 0.5).astype(int)

    r = recall_score(   y_true, y_pred, average='micro', pos_label=1)
    p = precision_score(y_true, y_pred, average='micro', pos_label=1)
    f = f1_score(       y_true, y_pred, average='micro', pos_label=1)

    return {"recall": r, "precision": p, "f1": f}

def run_epoch(model, optimizer, loss_func, data_loader, coll_dict, desc, run_type, device):

    start_time = time()

    a_loss = []
    y_true = []
    y_pred = []

    for inputs, labels in tqdm(data_loader, desc=desc, leave=True):


        inputs = inputs[0].to(device)
        labels = labels[0].to(device)

        if model.training:
            optimizer.zero_grad()
            y_hat = model(inputs)
            loss  = loss_func(y_hat.squeeze(1), labels)
            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                y_hat = model(inputs)
                loss  = loss_func(y_hat, labels)

        a_loss.append(loss.item())

        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend( y_hat.detach().cpu().numpy())

    epoch_loss = np.mean(a_loss)
    coll_dict[f"{run_type} loss"].append(epoch_loss)
    print(f"\n{run_type} loss: {epoch_loss}", end="\t")

    metrics = compute_metrics(y_pred, y_true)
    for items in list(metrics.items()):
        coll_dict[f"{run_type} {items[0]}"].append(items[1])
        print(f"{items[0]}: {items[1]}", end="\t")

    coll_dict[f"{run_type} time"].append(time() - start_time)

    if run_type == "val":
        return epoch_loss

class CNNQuakeDetector(nn.Module):
    def __init__(self, in_shape=(1, 1, 5, 129), out_channels=64, n_classes=1) -> None:
        super(CNNQuakeDetector, self).__init__()

        B, C, W, H = in_shape

        self.CNN = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.2)
            )

        self.Classifier = nn.Linear(out_channels * (W // 4) * (H // 4), n_classes)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, X):

        out = self.CNN(X)

        out = self.Classifier(out)
        out = self.Sigmoid(out)

        return out

results_dict = {
    "epoch"           : [],
    "epoch time"      : [],

    "train loss"      : [],
    "train recall"    : [],
    "train precision" : [],
    "train f1"        : [],
    "train time"      : [],

    "val loss"        : [],
    "val recall"      : [],
    "val precision"   : [],
    "val f1"          : [],
    "val time"        : [],
    }

model = CNNQuakeDetector(n_classes=1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

epochs = 10
eta_0  = 0.05

checkpoint_file = "/content/drive/MyDrive/NSAC/model.pt"
#loss_func      = torch.nn.MSELoss()
loss_func       = FocalLoss()
optimizer       = torch.optim.AdamW(model.parameters(), lr=eta_0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5)

for epoch in tqdm(range(epochs), total=epochs):

    epoch_start_time = time()

    # Training
    model = model.train().to(device)
    run_epoch(model, optimizer, loss_func, train_loader, results_dict, "Training", "train", device)

    # Validating
    val_loss = run_epoch(model, optimizer, loss_func, val_loader, results_dict, "Validating", "val", device)
    scheduler.step(val_loss)

    results_dict["epoch time"].append(time() - epoch_start_time)

torch.save(model.to(device).state_dict(), checkpoint_file)

