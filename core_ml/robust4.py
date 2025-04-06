import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import random
import matplotlib.pyplot as plt
import csv # Import csv module
import os  # Import os module for directory creation

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBrunch(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(ConvBrunch, self).__init__()
        padding = (kernel_size - 1) // 2
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU())

    def forward(self, x):
        return self.out_conv(x)

class Net(nn.Module):
    def __init__(self, type='CIFAR10'):
        super(Net, self).__init__()
        #self.type = type
        self.block1 = nn.Sequential(
                ConvBrunch(3, 64, 3),
                ConvBrunch(64, 64, 3),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.block2 = nn.Sequential(
            ConvBrunch(64, 128, 3),
            ConvBrunch(128, 128, 3),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.block3 = nn.Sequential(
            ConvBrunch(128, 196, 3),
            ConvBrunch(196, 196, 3),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(4*4*196, 256),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.fc2 = nn.Linear(256, 10)
        self.fc_size = 4*4*196
    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.global_avg_pool(x)
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, self.fc_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()

class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()

class NCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)


class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, scale=1.0, gamma=0, num_classes=10, alpha=None, size_average=True):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class CELoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, num_classes=10):
        super(CELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.ce(outputs, targets)


def add_label_noise(dataset, symmetric_noise_ratio=0.0, asymmetric_noise_ratio=0.0):
    targets = np.array(dataset.targets)
    num_classes = 10
    num_samples = len(targets)
    indices = np.arange(num_samples)

    if symmetric_noise_ratio > 0:
        num_noisy = int(symmetric_noise_ratio * num_samples)
        noisy_indices = np.random.choice(indices, num_noisy, replace=False)
        for i in noisy_indices:
            new_label = np.random.choice([x for x in range(num_classes) if x != targets[i]])
            targets[i] = new_label

    if asymmetric_noise_ratio > 0:
        for c in range(num_classes):
            class_indices = np.where(targets == c)[0]
            num_noisy = int(asymmetric_noise_ratio * len(class_indices))
            noisy_indices = np.random.choice(class_indices, num_noisy, replace=False)
            for i in noisy_indices:
                # Asymmetric noise: map class c to (c+1)%num_classes
                new_label = (targets[i] + 1) % num_classes
                targets[i] = new_label

    dataset.targets = targets.tolist()
    return dataset


# Define training and testing routines
# MODIFIED train_model to calculate, print, and return epoch-wise metrics
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=10):
    epoch_metrics = [] # Store metrics per epoch
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

        # Calculate metrics for the epoch
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        test_accuracy = test_model(model, test_loader) # Test after each epoch

        # Print metrics to terminal
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")

        # Store metrics
        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_accuracy,
            'test_acc': test_accuracy
        })

    return model, epoch_metrics # Return metrics along with the model

# test_model remains the same
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

# Helper function to save metrics to CSV
def save_metrics_to_csv(metrics, filename):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as csvfile:
        if not metrics: # Handle empty metrics list
            return
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Epoch-wise metrics saved to {filename}")

# Helper function to plot epoch-wise metrics
def plot_epoch_metrics(metrics, title, filename):
     # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not metrics: # Handle empty metrics list
        print(f"No metrics to plot for {title}")
        return
    epochs = [m['epoch'] for m in metrics]
    train_acc = [m['train_acc'] for m in metrics]
    test_acc = [m['test_acc'] for m in metrics]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, test_acc, label='Test Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename) # Save the plot
    plt.close() # Close the plot to avoid displaying it now
    print(f"Epoch-wise plot saved to {filename}")


# --- Main Script Execution ---

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # Use standard CIFAR10 normalization
])

# Load CIFAR10 data
# Use clean datasets for test_loader consistency
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False) # Define test_loader once

# Define a list of noise ratios
noise_levels = [0.1, 0.2, 0.3, 0.4]

# Dictionaries to store FINAL test accuracies for the summary plot
results = {
    'CrossEntropy': [],
    'FocalLoss': [],
    'Normalized CrossEntropy': [],
    'Normalized FocalLoss': [],
    'APL': []
}

# Define loss functions and their names
loss_functions = {
        'APL': NCEandRCE(alpha=1.0, beta=1.0, num_classes=10) # Adjusted alpha/beta based on common usage, can be tuned
}


# Training parameters
num_epochs = 15
batch_size = 256
learning_rate = 0.001

# --- Main Loop ---
# Create directories for outputs
output_dir = "training_results"
csv_dir = os.path.join(output_dir, "csv")
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


# Iterate over different noise levels
for noise in noise_levels:
    print(f"\n=============================================")
    print(f"Training with Asymmetric Noise Ratio: {noise}")
    print(f"=============================================")

    # Create a noisy copy of the training dataset for this noise level
    # Load clean train dataset first inside the loop to avoid cumulative noise
    train_dataset_clean = CIFAR10(root='./data', train=True, download=True, transform=transform)
    noisy_train_dataset = add_label_noise(train_dataset_clean, asymmetric_noise_ratio=noise)
    train_loader = DataLoader(noisy_train_dataset, batch_size=batch_size, shuffle=True)


    # Iterate through each loss function
    for loss_name, criterion in loss_functions.items():
        print(f"\n--- Training with {loss_name} (Noise: {noise}) ---")

        # Initialize a new model for each loss function and noise level
        model = Net().to(device)
        model._reset_prams() # Reset parameters if needed, or remove if default init is fine
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model and get epoch-wise metrics
        model, epoch_metrics = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=num_epochs)

        # Get final test accuracy from the last epoch
        final_test_acc = epoch_metrics[-1]['test_acc'] if epoch_metrics else 0.0
        results[loss_name].append(final_test_acc)
        print(f"--- Final Test Accuracy for {loss_name} (Noise: {noise}): {final_test_acc:.4f} ---")

        # Save epoch-wise metrics to CSV
        csv_filename = os.path.join(csv_dir, f"{loss_name}_noise_{noise:.1f}_metrics.csv")
        save_metrics_to_csv(epoch_metrics, csv_filename)

        # Generate and save epoch-wise plot
        plot_filename = os.path.join(plot_dir, f"{loss_name}_noise_{noise:.1f}_epoch_plot.png")
        plot_title = f"{loss_name} Training (Noise: {noise:.1f})"
        plot_epoch_metrics(epoch_metrics, plot_title, plot_filename)


# --- Final Plot Generation ---
print("\nGenerating final summary plot...")
plt.figure(figsize=(10, 6))
for loss_name, acc_list in results.items():
     # Ensure noise_levels and acc_list have the same length for plotting
    if len(noise_levels) == len(acc_list):
        plt.plot(noise_levels, acc_list, marker='o', label=loss_name)
    else:
        print(f"Warning: Mismatch in length for {loss_name}. Expected {len(noise_levels)}, got {len(acc_list)}. Skipping plot.")


plt.xlabel("Asymmetric Noise Ratio")
plt.ylabel("Final Test Accuracy")
plt.title("Model Performance vs. Asymmetric Noise Ratio")
plt.legend()
plt.grid(True)
final_plot_filename = os.path.join(output_dir, "noise_vs_accuracy_summary.png")
plt.savefig(final_plot_filename)
print(f"Final summary plot saved to {final_plot_filename}")
# plt.show() # Optionally display the plot interactively

print("\nScript finished.")