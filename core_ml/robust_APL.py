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
# import matplotlib.pyplot as plt # Removed if plotting is not desired

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


class ActivePassiveLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, num_classes=10):
        super(ActivePassiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.epsilon = 1e-7 # Using a small epsilon for stability

    def forward(self, outputs, targets):
        # Active Loss: Normalized Cross Entropy (as in original)
        log_probs = F.log_softmax(outputs, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float().to(outputs.device)
        # Ensure denominator stability
        log_probs_sum_per_sample = -log_probs.sum(dim=1)
        nce = -1 * (torch.sum(log_probs * one_hot, dim=1)) / (log_probs_sum_per_sample + self.epsilon)
        nce = nce.mean()

        # Passive Loss: Normalized Reverse Cross Entropy (as in original)
        pred = F.softmax(outputs, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0) # Clamp predictions
        label_one_hot = F.one_hot(targets, self.num_classes).float().to(outputs.device)
        # Clamp targets slightly away from 0 to avoid log(0) issues if needed, though one_hot should be fine
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        # RCE calculation - Ensure correct log calculation for one-hot
        # Original RCE definition often involves log(pred) * label_one_hot
        # The provided one was: -torch.sum(pred * torch.log(label_one_hot + self.epsilon), dim=1).mean()
        # Let's stick to the provided one:
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot + self.epsilon), dim=1))
        rce = rce.mean()


        total_loss = self.alpha * nce + self.beta * rce
        return total_loss


def add_label_noise(dataset, symmetric_noise_ratio=0.0, asymmetric_noise_ratio=0.0):
    # --- Keeping your original noise function ---
    targets = np.array(dataset.targets)
    num_classes = 10 # Assuming CIFAR-10
    num_samples = len(targets)
    indices = np.arange(num_samples)
    original_targets = targets.copy() # Keep for checking changes

    num_changed_sym = 0
    if symmetric_noise_ratio > 0:
        num_noisy = int(symmetric_noise_ratio * num_samples)
        noisy_indices = np.random.choice(indices, num_noisy, replace=False)
        for i in noisy_indices:
            old_label = targets[i]
            new_label = np.random.choice([x for x in range(num_classes) if x != old_label])
            if targets[i] != new_label: # Only count if label actually changes
                 num_changed_sym += 1
            targets[i] = new_label
        print(f"Applied symmetric noise. Targeted: {num_noisy}, Actually changed: {num_changed_sym}")


    num_changed_asym = 0
    if asymmetric_noise_ratio > 0:
        # Example: Only apply to certain classes and map them specifically
        # This part needs a defined mapping or logic if asymmetric noise is used
        # For simplicity here, let's assume a simple pair flip (e.g., class 2 -> 3)
        target_class = 2
        noisy_class = 3
        class_indices = np.where(original_targets == target_class)[0] # Use original targets to decide *who* gets noise
        num_noisy = int(asymmetric_noise_ratio * len(class_indices))
        noisy_indices = np.random.choice(class_indices, num_noisy, replace=False)
        for i in noisy_indices:
             if targets[i] == target_class: # Check if not already changed by symmetric noise
                targets[i] = noisy_class
                num_changed_asym +=1
        print(f"Applied asymmetric noise (Example: {target_class}->{noisy_class}). Targeted: {num_noisy}, Actually changed: {num_changed_asym}")
        # A real implementation would handle all classes/mappings as needed

    dataset.targets = targets.tolist()
    num_total_changed = np.sum(original_targets != np.array(dataset.targets))
    print(f"Total labels changed: {num_total_changed}/{num_samples} ({num_total_changed/num_samples:.2f})")
    return dataset


# Define training and testing routines
# MODIFIED train_model to accept test_loader and test_model function
def train_model(model, train_loader, test_loader, loss_fn, optimizer, test_fn, epochs=10):
    model.to(device) # Ensure model is on device
    for epoch in range(epochs):
        model.train() # Set model to training mode THIS IS IMPORTANT
        epoch_loss = 0.0
        num_batches = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        # --- Added evaluation step after each epoch ---
        current_accuracy = test_fn(model, test_loader)
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] | Average Train Loss: {avg_epoch_loss:.4f} | Test Accuracy: {current_accuracy:.4f}")
        # --- End of added evaluation step ---

    return model # Return only the model as in the original

# Original test_model function (added model.eval and torch.no_grad for correctness)
def test_model(model, test_loader):
    model.eval() # Set model to evaluation mode - IMPORTANT
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculations - IMPORTANT
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1) # Use torch.max for clarity
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

# --- Main execution keeping original structure ---

# Data transforms (using your original simple transform)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR10 data
train_dataset_clean = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define a noise level (e.g., 0.4 symmetric)
noise_level_sym = 0.8
noise_level_asym = 0.0 # Set asymmetric noise level if desired

# Training parameters
num_epochs = 50 # Keep your original value or adjust
batch_size = 128
learning_rate = 0.001

print(f"\nTraining with symmetric noise ratio: {noise_level_sym}, asymmetric: {noise_level_asym}")

# Create a noisy copy of the training dataset
# Important: Create a distinct copy for adding noise
noisy_train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
noisy_train_dataset = add_label_noise(noisy_train_dataset,
                                      symmetric_noise_ratio=noise_level_sym,
                                      asymmetric_noise_ratio=noise_level_asym)

# Create DataLoaders
train_loader = DataLoader(noisy_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# APL Loss setup
model_apl = Net().to(device)
# Note: Beta=0.1 might be low for APL, often Beta >= 1.0 is used, but keeping your value.
criterion_apl = NCEandRCE(alpha=0.1, beta=0.1, num_classes=10)
optimizer_apl = optim.Adam(model_apl.parameters(), lr=learning_rate)

print("\n--- Starting APL Training ---")
# Train the model using the modified train function that prints accuracy
# Pass the test_loader and the test_model function to train_model
model_apl = train_model(model_apl, train_loader, test_loader, criterion_apl, optimizer_apl, test_fn=test_model, epochs=num_epochs)

# Final Test (optional, as the last epoch printout shows the final accuracy)
final_acc_apl = test_model(model_apl, test_loader)
print(f"\n--- APL Training Finished ---")
print(f"Final APL Accuracy after {num_epochs} epochs: {final_acc_apl:.4f}")