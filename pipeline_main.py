# Custom imports
from batch_sampler import BatchSampler
from experimental_nets import FocalLoss
from image_dataset import ImageDataset
from train_test import train_model, test_model

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import warnings  # type: ignore
import os
import sys
import argparse
import plotext  # type: ignore

# pipeline import
from argparse import Namespace
from datetime import datetime
from pathlib import Path
import optuna

# config imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from config import Net, hyptune, nb_epochs as n_ep, batch_size as bat_sz

# Disable FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Check if /artifacts/ subdir exists
if not Path("artifacts/").exists():
    os.mkdir(Path("artifacts/"))

class TrainingPipeline:
    """
    A class for training a neural network pipeline.

    Args:
        args (argparse.Namespace): command line arguments
        activeloop (bool, optional): whether to run the main training loop. Defaults to True.

    Attributes:
        args (argparse.Namespace): command line arguments
        activeloop (bool): whether to run the main training loop
        model (torch.nn.Module): the neural network model
        optimizer (torch.optim): the optimizer for the model
        loss_function (torch.nn.Module): the loss function for the model
        train_dataset (ImageDataset): the training dataset
        test_dataset (ImageDataset): the testing dataset
        train_sampler (BatchSampler): the training batch sampler
        test_sampler (BatchSampler): the testing batch sampler
        device (str): the device to run the model on (e.g. "cuda")
        mean_losses_train (list): a list of mean training losses for each epoch
        mean_losses_test (list): a list of mean testing losses for each epoch
    """

    def __init__(self, args: Namespace, activeloop: bool = True):
        self.criterion = None
        self.args = args
        self.activeloop = activeloop
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.loss_test_function = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_sampler = None
        self.test_sampler = None
        self.device = None
        self.mean_losses_train = []
        self.mean_losses_test = []

    def load_datasets(self):
        """Load the train and test data set."""
        # Load the train and test data set
        self.train_dataset = ImageDataset(Path("../data/X_train.npy"), Path("../data/Y_train.npy"))
        self.test_dataset = ImageDataset(Path("../data/X_test.npy"), Path("../data/Y_test.npy"))

    def optimize_hyperparameters(self, n_trials=100):
        def objective(trial):
            # Define the hyperparameters
            global accuracy
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

            # weighted loss function cross function
            loss_function_name = trial.suggest_categorical('loss_function',
                                                           ['cross_entropy', 'focal', 'weighted_cross_entropy'])
            optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", 'AdamW'])  # optimizer

            # convolution params
            num_layers = trial.suggest_int("number_layers", 1, 4)
            num_residual_blocks = trial.suggest_int("num_res_blocks", 1, 3)  # number of residual blocks 1-4
            drop_conv1 = trial.suggest_float("drop_conv1", 0.125, 0.5)
            drop_conv2 = trial.suggest_float("drop_conv2", 0.125, 0.5)  # dropout p1 p2 0.125 0.25 0.375 0.5

            pooling_kernel_size = [2] * (num_residual_blocks + 1)  # list of kernel sizes for pooling layers
            kern_size = [3] * (num_residual_blocks + 1)  # list of kernel sizes for conv layers
            pooling_kernel_size[0] = trial.suggest_int(f'pooling_kernel_size_0_0', 2, 4, 1)
            kern_size[0] = trial.suggest_int("kernel_size_0_0", 4, 11, 1)

            # param for each residual block
            # for i in range(num_residual_blocks):
            # pooling_kernel_size[1] = trial.suggest_int(f'resblock_pooling_kernel_size_{1}', 2, 4, 1)
            # kern_size[1] = (2 * trial.suggest_int(f'resblock_kernel_size_{1}', 0, 2, 1)) + 1  # kernel size
            # print(kern_size[1])
            # kern_size[2 * i + 2] = trial.suggest_int(f'resblock_pooling_kernel_size_{i}_1', 1, 4, 1)  # kernel size

            pooling_type = trial.suggest_categorical("pooling", ["avg", "max"])  # pooling type avg max
            base_channel = trial.suggest_categorical('base_channel', [16, 32, 64])  # base channels 32 64 128

            # Set up the model with the suggested hyperparameters
            print(pooling_kernel_size, kern_size)
            self.setup_model(learning_rate, loss_function_name, optimizer_name, num_residual_blocks,
                             drop_conv1, drop_conv2, pooling_kernel_size, pooling_type, base_channel, kern_size,
                             num_layers)
            self.setup_device()

            # Train and test the model
            train_losses = []
            test_losses = []
            for e in range(self.args.nb_epochs):
                train_loss = train_model(self.model, self.train_sampler, self.optimizer, self.loss_function,
                                         self.device)
                mean_loss = sum(train_loss) / len(train_loss)
                print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

                test_loss, metrics = test_model(self.model, self.test_sampler, self.loss_function, self.device)

                accuracy = metrics['accuracy']
                mean_test_loss = sum(test_loss) / len(test_loss)
                print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_test_loss} accuracy: {accuracy}\n")

                # TODO need to find wtf do we base our pruning on and what we try to maximize
                trial.report(accuracy, e)

                train_losses.append(train_loss)
                test_losses.append(test_loss)

                #  TODO if mean train_loss 10% higher than mean test_loss then prune trail
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Return the mean test loss
            print(accuracy)
            return accuracy

        # Run the optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        print('go there')

        # Print the best hyperparameters
        best_params = study.best_params
        print('Best hyperparameters: ', best_params)
        print('Best trial:', study.best_trial)

    def setup_model(self, learning_rate, loss_function_name, optimizer_name, num_residual_blocks,
                    drop_conv1, drop_conv2, pooling_kern, pooling_type, base_channel, kern_size, num_layers):
        """Load the Neural Net. NOTE: set the number of distinct labels here"""
        self.model = Net(
            n_classes=6,
            in_channels=1,
            base_channels=base_channel,
            residual_blocks=num_residual_blocks,
            pooling_type=pooling_type,
            pooling_kernel_size=pooling_kern,
            dropout_p1=drop_conv1,
            dropout_p2=drop_conv2,
            kernel_sizes=kern_size,
            num_layers=num_layers
        )

        # Initialize optimizer(s) and loss function(s)
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.1)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.1)
            raise Warning(f"Optimizer {optimizer_name} not supported, SGD used instead!")

        uniqueTr, countsTr = self.train_sampler.labels

        if loss_function_name == 'cross_entropy':
            self.loss_function = nn.CrossEntropyLoss()
        elif loss_function_name == 'focal':
            self.loss_function = FocalLoss()
        elif loss_function_name == 'weighted_cross_entropy':
            self.loss_function = nn.CrossEntropyLoss(weight=(weights(countsTr).to(torch.float32)))
        else:
            raise ValueError(f"Invalid loss function: {loss_function_name}")

    def setup_device(self):
        """Moving our model to the right device (CUDA will speed training up significantly!)"""
        if torch.cuda.is_available():
            print("@@@ CUDA device found, enabling CUDA training...")
            self.device = "cuda"
            self.model.to(self.device)
            # Creating a summary of our model and its layers:
            summary(self.model, (1, 128, 128), device=self.device)
        elif torch.backends.mps.is_available():
            print("@@@ Apple silicon device enabled, training with Metal backend...")
            self.device = "cpu"
            self.model.to(self.device)
        else:
            print("@@@ No GPU boosting device found, training on CPU...")
            self.device = "cpu"
            # Creating a summary of our model and its layers:
            summary(self.model, (1, 128, 128), device=self.device)

    def setup_samplers(self):
        """Let's now train and test our model for multiple epochs:"""
        self.train_sampler = BatchSampler(
            batch_size=self.args.batch_size, dataset=self.train_dataset, balanced=self.args.balanced_batches
        )
        self.test_sampler = BatchSampler(
            batch_size=100, dataset=self.test_dataset, balanced=self.args.balanced_batches
        )

    def train_and_test(self):
        """Train and test the model."""
        for e in range(self.args.nb_epochs):
            if self.activeloop:
                # Training:
                losses = train_model(self.model, self.train_sampler, self.optimizer, self.loss_function, self.device)
                # Calculating and printing statistics:
                mean_loss = sum(losses) / len(losses)
                self.mean_losses_train.append(mean_loss)
                print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

                # Testing:
                losses, metrics = test_model(self.model, self.test_sampler, self.loss_function, self.device)
                print(losses)
                
                # # Calculating and printing statistics:
                mean_loss = sum(losses) / len(losses)
                self.mean_losses_test.append(mean_loss)
                print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")

                # Plotting during training
                plotext.clf()
                plotext.scatter(self.mean_losses_train, label="train")
                plotext.scatter(self.mean_losses_test, label="test")
                plotext.title("Train and test loss")
                plotext.xticks([i for i in range(len(self.mean_losses_train) + 1)])

                plotext.show()

    def save_results(self):
        """Save the results of the training."""
        # retrieve the current time to label artifacts
        now = datetime.now()

        # check if model_weights/ subdir exists
        if not Path("model_weights/").exists():
            os.mkdir(Path("model_weights/"))

        # Saving the model
        torch.save(self.model.state_dict(),
                   f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")

        # Create a plot of losses
        figure(figsize=(9, 10), dpi=80)
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)

        ax1.plot(range(1, 1 + self.args.nb_epochs), [x.detach().cpu() for x in self.mean_losses_train], label="Train",
                 color="blue")
        ax2.plot(range(1, 1 + self.args.nb_epochs), [x.detach().cpu() for x in self.mean_losses_test], label="Test",
                 color="red")
        fig.legend()

        # Check if /artifacts/ subdir exists
        if not Path("artifacts/").exists():
            os.mkdir(Path("artifacts/"))

        # save the plot of losses
        fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

    def run_pipeline(self):
        """Run the entire training pipeline."""
        self.load_datasets()
        self.setup_samplers()
        self.setup_model(learning_rate=1.3364134397486015e-05,  # here we use the best parameters found by hyperparameter optimization
                         loss_function_name='cross_entropy', 
                         optimizer_name='AdamW', 
                         num_layers=4, 
                         num_residual_blocks=1, 
                         drop_conv1=0.1452268541179967, 
                         drop_conv2=0.4491861204418486, 
                         pooling_kern=[3,2], 
                         kern_size=[10,3], 
                         pooling_type='avg', 
                         base_channel=64)
        if hyptune:
            self.optimize_hyperparameters()
        self.setup_device()
        self.train_and_test()
        self.save_results()


def weights(counts):
    total = sum(counts)
    frequencies = [count / total for count in counts]
    w = [1 / freq for freq in frequencies]
    return torch.tensor(w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nb_epochs", help="number of training iterations", default=n_ep, type=int)
    parser.add_argument("--batch_size", help="batch_size", default=bat_sz, type=int)
    parser.add_argument("--balanced_batches", help="whether to balance batches for class labels", default=True,
                        type=bool)
    args = parser.parse_args()

    pipeline = TrainingPipeline(args)
    pipeline.run_pipeline()
