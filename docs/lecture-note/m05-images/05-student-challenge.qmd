---
title: "Part 5: Your Challenge"
jupyter: python3
---

::: {.callout-note title="What you'll learn in this challenge"}
This challenge tests your ability to choose appropriate architectures, apply transfer learning, and diagnose model behavior.

You'll make decisions that practitioners face daily. Which model architecture should you use? How much data augmentation? When should you stop training? By completing this project, you'll synthesize everything learned in this module into a working image classification system.
:::

## The Challenge: Image Classification Competition

You've learned what images are, explored the deep learning revolution, mastered practical CNN skills, and traced the innovation timeline. Now it's your turn to apply this knowledge to a real classification problem.

Your task: build the best possible image classifier for a custom dataset using transfer learning and the techniques we've covered.

## Dataset

For this challenge, you'll work with the **CIFAR-100** dataset. It contains 60,000 color images, each 32×32 pixels, across 100 classes (600 images per class). The classes span diverse categories ranging from animals and vehicles to household objects and natural scenes.

```{python}
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transforms.ToTensor()
)
test_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transforms.ToTensor()
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Image shape: {train_dataset[0][0].shape}")
```

Let's visualize some examples:

```{python}
# Display sample images
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    img = img.permute(1, 2, 0)  # CHW -> HWC for display
    ax.imshow(img)
    ax.set_title(train_dataset.classes[label], fontsize=8)
    ax.axis('off')

plt.suptitle("Sample Images from CIFAR-100", fontsize=16)
plt.tight_layout()
plt.show()
```

## Evaluation Metric

Your model will be evaluated on **top-1 accuracy**. This is the percentage of test images where the highest-confidence prediction matches the true label.

Let's set some targets. The baseline to beat (random guessing with 100 classes) is **1%**. A reasonable target is **70%+**, which is achievable with transfer learning and good data augmentation. An excellent result is **80%+**, which requires careful architecture selection and tuning.

## Starter Code

Here's a template to get you started. You'll need to fill in key components and make design decisions.

### Step 1: Data Loading and Augmentation

```{python}
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

# Detect device (CUDA, MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

# TODO: Design your data augmentation strategy
train_transform = transforms.Compose([
    # Add your augmentation techniques here
    # Suggestions: RandomCrop, RandomHorizontalFlip, ColorJitter, RandomRotation
    transforms.Resize(224),  # Resize to match ImageNet pre-training
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet statistics
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Create datasets
train_dataset_full = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=val_transform
)

# Split training into train/validation
train_size = int(0.9 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(
    train_dataset_full, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Update validation dataset transform
val_dataset.dataset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=val_transform
)

# Create data loaders
batch_size = 128  # Adjust based on your GPU memory
# Use num_workers=0 for MPS (macOS) or 2 for CUDA
num_workers = 0 if str(device) == 'mps' else 2

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
```

### Step 2: Model Selection

```{python}
# TODO: Choose your architecture
# Options: resnet18, resnet50, resnet101, efficientnet_b0, vgg16, mobilenet_v2, etc.

def create_model(arch='resnet50', num_classes=100, pretrained=True):
    """
    Create a model for CIFAR-100 classification.

    Args:
        arch: Architecture name (e.g., 'resnet50', 'efficientnet_b0')
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pre-trained weights

    Returns:
        model: PyTorch model ready for training
    """
    if arch == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    # Add more architectures as needed

    return model

# Create your model
model = create_model(arch='resnet18', num_classes=100, pretrained=True)
model = model.to(device)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

### Step 3: Training Loop

```{python}
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# TODO: Configure training hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# Training loop
num_epochs = 20
best_val_acc = 0.0
train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  Saved new best model with val_acc: {val_acc:.4f}")

    # Update learning rate
    scheduler.step(val_loss)

print(f"\nBest validation accuracy: {best_val_acc:.4f}")
```

### Step 4: Evaluation and Analysis

```{python}
# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate on test set
test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(train_accs, label='Train Accuracy')
ax2.plot(val_accs, label='Val Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Guided Questions

Let's talk about the key decisions you'll make. As you work on this challenge, consider these questions. Document your answers in your final report.

### Architecture Selection

**Q1: Which architecture did you choose and why?**

Consider several factors. What's the parameter count (memory constraints)? What's the computational cost (training time)? Are pre-trained weights available? What's the reported performance on similar tasks?

**Q2: Did you use feature extraction or fine-tuning? Why?**

Feature extraction freezes early layers and trains only the classifier. Fine-tuning updates the entire network. What guided your choice?

### Data Augmentation

**Q3: What data augmentation techniques did you apply?**

CIFAR-100 images are small at 32×32 pixels. Common augmentations include random crops, horizontal flips, color jittering, random rotation, Cutout or RandomErasing, and MixUp or CutMix. Explain your choices and their expected benefits.

**Q4: How did augmentation affect your results?**

Compare training with and without augmentation. What changed?

### Training Strategy

**Q5: What learning rate and optimizer did you use?**

Did you use a learning rate schedule? How did you choose the initial learning rate?

**Q6: How did you prevent overfitting?**

You have several techniques at your disposal. Data augmentation helps by creating variations of training examples. Dropout randomly disables neurons during training. Weight decay (L2 regularization) penalizes large weights. Early stopping based on validation loss prevents the model from memorizing the training set. Which worked best for your model?

### Performance Analysis

**Q7: Which classes does your model struggle with most?**

Analyze confusion patterns. Are certain classes frequently confused? Why might this happen?

**Q8: How does performance compare to baselines?**

Research state-of-the-art results on CIFAR-100. How does your model compare? What techniques do top-performing models use that you didn't?

## Optional Extensions

Want to push your understanding further? Try these advanced challenges.

### Extension 1: Model Ensemble

Train multiple models with different architectures or initializations. Combine their predictions through voting or averaging. Does the ensemble outperform individual models?

```{python}
# Ensemble prediction example
def ensemble_predict(models, dataloader, device):
    """
    Ensemble prediction from multiple models.

    Args:
        models: List of PyTorch models
        dataloader: Test data loader
        device: Device to run inference on

    Returns:
        accuracy: Ensemble accuracy
    """
    all_predictions = []

    for model in models:
        model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions.append(outputs.cpu())

        all_predictions.append(torch.cat(predictions))

    # Average predictions
    ensemble_output = torch.stack(all_predictions).mean(dim=0)
    ensemble_preds = ensemble_output.argmax(dim=1)

    # Calculate accuracy
    true_labels = torch.tensor([label for _, label in dataloader.dataset])
    accuracy = (ensemble_preds == true_labels).float().mean().item()

    return accuracy
```

### Extension 2: Visualize Learned Features

Extract and visualize feature maps from intermediate layers. What patterns does your model detect?

```{python}
def visualize_features(model, image, layer_name):
    """
    Visualize feature maps from a specific layer.

    Args:
        model: Trained model
        image: Input image (CHW format)
        layer_name: Name of layer to visualize (e.g., 'layer2.0.conv1')
    """
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hook
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(get_activation(layer_name))

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(image.unsqueeze(0).to(device))

    # Visualize activations
    act = activations[layer_name].squeeze(0)
    num_filters = min(32, act.shape[0])

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            ax.imshow(act[i].cpu(), cmap='viridis')
        ax.axis('off')

    plt.suptitle(f"Feature Maps from {layer_name}", fontsize=16)
    plt.tight_layout()
    plt.show()
```

### Extension 3: Analyze Failure Cases

Identify images your model misclassifies. What makes them difficult? Can you identify patterns in failures?

```{python}
def analyze_failures(model, dataloader, device, num_samples=16):
    """Display misclassified images."""
    model.eval()
    failures = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            incorrect = (predicted != labels).nonzero(as_tuple=True)[0]
            for idx in incorrect:
                failures.append({
                    'image': inputs[idx].cpu(),
                    'true': labels[idx].item(),
                    'pred': predicted[idx].item()
                })

            if len(failures) >= num_samples:
                break

    # Visualize failures
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(failures):
            img = failures[i]['image'].permute(1, 2, 0)
            # Denormalize
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img = torch.clamp(img, 0, 1)

            ax.imshow(img)
            ax.set_title(
                f"True: {test_dataset.classes[failures[i]['true']]}\n"
                f"Pred: {test_dataset.classes[failures[i]['pred']]}",
                fontsize=8
            )
        ax.axis('off')

    plt.suptitle("Misclassified Images", fontsize=16)
    plt.tight_layout()
    plt.show()
```

### Extension 4: CNN vs. Vision Transformer

Compare a CNN (like ResNet) with a Vision Transformer on the same dataset. What are the differences in accuracy, training time, and computational cost?

## Deliverables

Submit the following:

**Jupyter Notebook**: Your complete implementation with code, outputs, and markdown explanations.

**Best Model Checkpoint**: Saved weights of your best-performing model (`best_model.pth`).

**Short Report (1-2 pages)**: A document answering the guided questions above. Include your architecture choice and rationale, data augmentation strategy, training hyperparameters, results (training curves, test accuracy), analysis of successes and failures, and lessons learned.

**Test Set Predictions**: CSV file with predictions for the entire test set.

## Evaluation Rubric

Your submission will be evaluated on four criteria.

**Test Accuracy (40%)**: How well does your model perform?

**Technical Quality (30%)**: Is your code clean, well-documented, and correct?

**Experimental Rigor (20%)**: Did you try multiple approaches? Did you compare results systematically?

**Analysis Depth (10%)**: Do you understand why your model succeeds or fails?

## Tips for Success

Here are some practical strategies to help you succeed.

**Start simple**. Begin with a small model (ResNet-18) to iterate quickly. Scale up once your pipeline works.

**Monitor for overfitting**. If training accuracy is much higher than validation accuracy, increase regularization.

**Experiment systematically**. Change one thing at a time. Document what works and what doesn't.

**Use pre-trained weights**. Transfer learning from ImageNet gives you a huge head start.

**Augment aggressively**. CIFAR-100 is small. Data augmentation is crucial for good generalization.

**Be patient**. Training 20+ epochs might be necessary. Use learning rate schedules to help convergence.

**Compare to baselines**. Look up published results on CIFAR-100 to calibrate your expectations.

## Conclusion

This challenge puts everything you've learned into practice. You'll make real architectural decisions, debug training issues, and analyze model behavior. These skills transfer directly to real-world computer vision problems.

Good luck! Remember that the goal isn't just high accuracy, though that's certainly satisfying. The goal is understanding why your choices lead to the results you see. That understanding is what makes you an effective deep learning practitioner.
