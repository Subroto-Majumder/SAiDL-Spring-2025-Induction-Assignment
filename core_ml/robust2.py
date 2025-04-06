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
import csv
import os

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
                new_label = (targets[i] + 1) % num_classes
                targets[i] = new_label

    dataset.targets = targets.tolist()
    return dataset


# Define training and testing routines
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=10, 
                loss_name="", noise_level=0.0, csv_writer=None, epoch_accuracies=None):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_accuracy = correct / total
        
        # Evaluation phase
        test_accuracy = test_model(model, test_loader)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
        
        # Record metrics to CSV
        if csv_writer:
            csv_writer.writerow([loss_name, noise_level, epoch+1, 
                                running_loss/len(train_loader), 
                                train_accuracy, test_accuracy])
        
        # Store accuracy for plotting
        if epoch_accuracies is not None:
            if epoch not in epoch_accuracies:
                epoch_accuracies[epoch] = {}
            if loss_name not in epoch_accuracies[epoch]:
                epoch_accuracies[epoch][loss_name] = {}
            epoch_accuracies[epoch][loss_name][noise_level] = test_accuracy
            
    return model

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

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR10 data
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define a list of noise ratios
noise_levels = [0.2, 0.4, 0.6, 0.8]

# Dictionaries to store test accuracies for each loss method
results = {
    'CrossEntropy': [],
    'FocalLoss': [],
    'Normalized CrossEntropy': [],
    'Normalized FocalLoss': [],
    'APL': []
}

# Dictionary to store epoch-wise accuracies
epoch_accuracies = {}

# Create directory for results if it doesn't exist
os.makedirs('results', exist_ok=True)

# Create CSV file for detailed metrics
csv_file = open('results/training_metrics.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Loss Function', 'Noise Level', 'Epoch', 'Loss', 'Train Accuracy', 'Test Accuracy'])

# Training parameters
num_epochs = 50  # Increase as needed for better performance
batch_size = 128
learning_rate = 0.001

# Iterate over different noise levels
for noise in noise_levels:
    print(f"\n===== Training with symmetric noise ratio: {noise} =====")
    
    # Create a noisy copy of the training dataset
    noisy_train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    noisy_train_dataset = add_label_noise(noisy_train_dataset, symmetric_noise_ratio=noise)
    
    train_loader = DataLoader(noisy_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # For each loss function, train a new model from scratch
    # 1. Vanilla CrossEntropy Loss
    print("\nTraining with CrossEntropy Loss:")
    model_ce = Net().to(device)
    criterion_ce = CELoss()  # using the provided CELoss wrapper
    optimizer_ce = optim.Adam(model_ce.parameters(), lr=learning_rate)
    train_model(model_ce, train_loader, test_loader, criterion_ce, optimizer_ce, 
                epochs=num_epochs, loss_name="CrossEntropy", noise_level=noise, 
                csv_writer=csv_writer, epoch_accuracies=epoch_accuracies)
    acc_ce = test_model(model_ce, test_loader)
    results['CrossEntropy'].append(acc_ce)
    print(f"Final CrossEntropy Accuracy: {acc_ce:.4f}")
    
    # 2. Vanilla Focal Loss
    print("\nTraining with Focal Loss:")
    model_focal = Net().to(device)
    criterion_focal = FocalLoss(gamma=2)  # gamma chosen as an example
    optimizer_focal = optim.Adam(model_focal.parameters(), lr=learning_rate)
    train_model(model_focal, train_loader, test_loader, criterion_focal, optimizer_focal, 
                epochs=num_epochs, loss_name="FocalLoss", noise_level=noise, 
                csv_writer=csv_writer, epoch_accuracies=epoch_accuracies)
    acc_focal = test_model(model_focal, test_loader)
    results['FocalLoss'].append(acc_focal)
    print(f"Final FocalLoss Accuracy: {acc_focal:.4f}")
    
    # 3. Normalized CrossEntropy Loss
    print("\nTraining with Normalized CrossEntropy Loss:")
    model_nce = Net().to(device)
    criterion_nce = NormalizedCrossEntropy(num_classes=10, scale=1.0)
    optimizer_nce = optim.Adam(model_nce.parameters(), lr=learning_rate)
    train_model(model_nce, train_loader, test_loader, criterion_nce, optimizer_nce, 
                epochs=num_epochs, loss_name="Normalized CrossEntropy", noise_level=noise, 
                csv_writer=csv_writer, epoch_accuracies=epoch_accuracies)
    acc_nce = test_model(model_nce, test_loader)
    results['Normalized CrossEntropy'].append(acc_nce)
    print(f"Final Normalized CrossEntropy Accuracy: {acc_nce:.4f}")
    
    # 4. Normalized Focal Loss
    print("\nTraining with Normalized Focal Loss:")
    model_nfl = Net().to(device)
    criterion_nfl = NormalizedFocalLoss(scale=1.0, gamma=2, num_classes=10)
    optimizer_nfl = optim.Adam(model_nfl.parameters(), lr=learning_rate)
    train_model(model_nfl, train_loader, test_loader, criterion_nfl, optimizer_nfl, 
                epochs=num_epochs, loss_name="Normalized FocalLoss", noise_level=noise, 
                csv_writer=csv_writer, epoch_accuracies=epoch_accuracies)
    acc_nfl = test_model(model_nfl, test_loader)
    results['Normalized FocalLoss'].append(acc_nfl)
    print(f"Final Normalized FocalLoss Accuracy: {acc_nfl:.4f}")
    
    # 5. APL Loss (Active-Passive Loss)
    print("\nTraining with APL Loss:")
    model_apl = Net().to(device)
    criterion_apl = NCEandRCE(alpha=0.1, beta=0.1, num_classes=10)
    optimizer_apl = optim.Adam(model_apl.parameters(), lr=learning_rate)
    train_model(model_apl, train_loader, test_loader, criterion_apl, optimizer_apl, 
                epochs=num_epochs, loss_name="APL", noise_level=noise, 
                csv_writer=csv_writer, epoch_accuracies=epoch_accuracies)
    acc_apl = test_model(model_apl, test_loader)
    results['APL'].append(acc_apl)
    print(f"Final APL Accuracy: {acc_apl:.4f}")

# Close the CSV file
csv_file.close()

# Plot the final results
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, results['CrossEntropy'], marker='o', label='CrossEntropy')
plt.plot(noise_levels, results['FocalLoss'], marker='o', label='FocalLoss')
plt.plot(noise_levels, results['Normalized CrossEntropy'], marker='o', label='Normalized CrossEntropy')
plt.plot(noise_levels, results['Normalized FocalLoss'], marker='o', label='Normalized FocalLoss')
plt.plot(noise_levels, results['APL'], marker='o', label='APL')
plt.xlabel("Symmetric Noise Ratio")
plt.ylabel("Test Accuracy")
plt.title("Model Performance under Different Noise Levels")
plt.legend()
plt.grid(True)
plt.savefig('results/noise_comparison.png')
plt.show()

# Plot epoch-wise accuracy for different noise levels for each loss function
for loss_name in results.keys():
    plt.figure(figsize=(12, 8))
    for noise in noise_levels:
        epochs = list(range(1, num_epochs + 1))
        accuracies = [epoch_accuracies[e-1][loss_name][noise] for e in epochs]
        plt.plot(epochs, accuracies, marker='.', label=f'Noise {noise}')
    
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title(f"Epoch-wise Test Accuracy for {loss_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/epoch_accuracy_{loss_name.replace(" ", "_")}.png')
    plt.show()

# Export epoch_accuracies to CSV for additional analysis
with open('results/epoch_wise_accuracies.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Loss Function', 'Noise Level', 'Test Accuracy'])
    
    for epoch in epoch_accuracies:
        for loss_name in epoch_accuracies[epoch]:
            for noise in epoch_accuracies[epoch][loss_name]:
                writer.writerow([
                    epoch + 1,
                    loss_name,
                    noise,
                    epoch_accuracies[epoch][loss_name][noise]
                ])

print("\nAll training completed. Results saved to the 'results' directory.")