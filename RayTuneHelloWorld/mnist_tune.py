import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import WandbLoggerCallback

# Simple CNN model
class Net(nn.Module):
    def __init__(self, l1=32, l2=64):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, l1, 3, 1)
        self.conv2 = nn.Conv2d(l1, l2, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Training function that will be called by Ray Tune
def train_mnist(config):
    # Initialize W&B if API key is available
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb.init(project="ray-tune-mnist", config=config)
    
    # Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(
        train_data,
        batch_size=int(config["batch_size"]),
        shuffle=True)
    
    test_loader = DataLoader(
        test_data,
        batch_size=int(config["batch_size"]),
        shuffle=False)
    
    # Build model
    model = Net(
        l1=config["layer1_size"],
        l2=config["layer2_size"]
    ).to(device)
    
    # Training setup
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"])
    
    # Training loop
    for epoch in range(10):  # Just 10 epochs for hello world
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                # Calculate validation accuracy periodically
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        outputs = model(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                
                accuracy = correct / total
                # Report to Ray Tune
                tune.report(loss=loss.item(), accuracy=accuracy)
                
                # Also log to W&B if available
                if os.environ.get("WANDB_API_KEY"):
                    wandb.log({"loss": loss.item(), "accuracy": accuracy, "epoch": epoch})
                
                model.train()
        
    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    final_accuracy = correct / total
    tune.report(loss=loss.item(), accuracy=final_accuracy)
    
    if os.environ.get("WANDB_API_KEY"):
        wandb.log({"final_accuracy": final_accuracy})
        wandb.finish()

def main():
    # Connect to Ray cluster
    ray.init(address="auto")
    
    # Define the search space
    config = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "layer1_size": tune.choice([16, 32, 64]),
        "layer2_size": tune.choice([32, 64, 128]),
        "batch_size": tune.choice([32, 64, 128]),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
    }
    
    # Set up the ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="accuracy",
        mode="max",
        max_t=10,  # Maximum number of epochs
        grace_period=2,  # Try all configs for at least 2 epochs
        reduction_factor=2
    )
    
    # W&B callback
    wandb_callback = None
    if os.environ.get("WANDB_API_KEY"):
        wandb_callback = WandbLoggerCallback(
            project="ray-tune-mnist",
            log_config=True,
            api_key_file=None  # Will use env var WANDB_API_KEY
        )
        callbacks = [wandb_callback]
    else:
        callbacks = []
    
    # Set up the tuner
    tuner = tune.Tuner(
        train_mnist,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=8,  # Try 8 different hyperparameter combinations
            resources_per_trial={"gpu": 1, "cpu": 2}  # Each trial uses 1 GPU
        ),
        param_space=config,
        run_config=ray.air.RunConfig(
            name="mnist_tune_example",
            callbacks=callbacks
        )
    )
    
    # Run the hyperparameter search
    results = tuner.fit()
    
    # Get the best trial
    best_trial = results.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final accuracy: {}".format(best_trial.metrics["accuracy"]))

if __name__ == "__main__":
    main()