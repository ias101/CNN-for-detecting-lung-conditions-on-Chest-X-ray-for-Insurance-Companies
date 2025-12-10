import torchmetrics
from torch import Tensor
from tqdm import tqdm
import torch
import sys
import os
from batch_sampler import BatchSampler
from typing import Callable, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmap visualization
import torchmetrics

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from config import Net

class_names = ['Atelectasis', 'Effusion', 'Infiltration', 'Healthy', 'Nodule', 'Pneumothorax']


def train_model(
        model: Net,
        train_sampler: BatchSampler,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Lets keep track of all the losses:
    losses = []
    # Put the model in train mode:
    model.train()
    # Feed all the batches one by one:
    for batch in tqdm(train_sampler):
        # Get a batch:
        x, y = batch
        # Making sure our samples are stored on the same device as our model:
        x, y = x.to(device), y.to(device)
        # Get predictions:
        predictions = model.forward(x)
        loss = loss_function(predictions, y)
        losses.append(loss)
        # We first need to make sure we reset our optimizer at the start.
        # We want to learn from each batch separately,
        # not from the entire dataset at once.
        optimizer.zero_grad()
        # We now backpropagate our loss through our model:
        loss.backward()
        # We then make the optimizer take a step in the right direction.
        optimizer.step()
    return losses


def test_model(
        model: Net,
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    # Setting the model to evaluation mode:
    model.eval()
    losses = []

    # Initialize metrics
    accuracy = torchmetrics.Accuracy(num_classes=6, task="multiclass").to(device)
    precision = torchmetrics.Precision(num_classes=6, average="macro", task="multiclass").to(device)
    recall = torchmetrics.Recall(num_classes=6, average="macro", task="multiclass").to(device)
    f1 = torchmetrics.F1Score(num_classes=6, average="macro", task="multiclass").to(device)
    cohen_kappa = torchmetrics.CohenKappa(num_classes=6, task="multiclass").to(device)
    avg_pre = torchmetrics.AveragePrecision(num_classes=6, average="macro", task="multiclass").to(device)
    conf_matrix = torchmetrics.ConfusionMatrix(num_classes=6, task="multiclass").to(device)

    # We need to make sure we do not update our model based on the test data:
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            prediction = model.forward(x)
            loss = loss_function(prediction, y)
            losses.append(loss)

            accuracy.update(prediction, y)
            precision.update(prediction, y)
            recall.update(prediction, y)
            f1.update(prediction, y)
            cohen_kappa.update(prediction.argmax(dim=1), y)
            avg_pre.update(prediction, y)
            conf_matrix.update(prediction.argmax(dim=1), y)

        metrics = {
            "accuracy": accuracy.compute().item(),
            "precision": precision.compute().item(),
            "recall": recall.compute().item(),
            "f1_score": f1.compute().item(),
            "cohen_kappa": cohen_kappa.compute().item(),
            "auc_pr": avg_pre.compute().item(),
        }

        # Plot confusion matrix

        conf_matrix = conf_matrix.compute().cpu().numpy()
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix")
        plt.savefig(f"artifacts/confusion_matrix_{metrics['accuracy']}.png")

    return losses, metrics
