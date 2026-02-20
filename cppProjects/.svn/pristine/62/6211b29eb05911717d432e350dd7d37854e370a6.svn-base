import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# ======================
# CONFIGURATION
# ======================
CHECKPOINT_DIR = "./checkpointsOP"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class VerboseReduceLROnPlateau:
    """Wrapper for ReduceLROnPlateau with verbose output"""
    def __init__(self, optimizer, mode='max', factor=0.5, patience=3, verbose=True):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
        self.verbose = verbose
        self.old_lr = optimizer.param_groups[0]['lr']
        self.patience = patience
        self.bad_epochs = 0
        
    def step(self, metrics):
        self.scheduler.step(metrics)
        new_lr = self.scheduler.optimizer.param_groups[0]['lr']
        
        if self.verbose and new_lr != self.old_lr:
            print(f"✓ Learning rate reduced from {self.old_lr:.6f} to {new_lr:.6f}")
            self.old_lr = new_lr

class CheckpointManager:
    """Manages saving and loading checkpoints"""
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self.best_accuracy = 0
        self.checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
        self.stats_path = os.path.join(checkpoint_dir, "training_stats.json")
        
    def save_checkpoint(self, model, optimizer, epoch, train_losses, val_accuracy, 
                        model_config=None, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_accuracy': val_accuracy,
            'model_config': model_config or {},
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Always save current checkpoint
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
        
        # Save as best if accuracy improved
        if is_best:
            torch.save(checkpoint, self.checkpoint_path)
            self.best_accuracy = val_accuracy
            print(f"✓ New best model saved with accuracy: {val_accuracy:.2f}%")
    
    def load_checkpoint(self, model, optimizer=None):
        """Load the best checkpoint"""
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
            print(f"  Previous accuracy: {checkpoint['val_accuracy']:.2f}%")
            print(f"  Saved on: {checkpoint['timestamp']}")
            
            return checkpoint
        return None
    
    def load_latest_checkpoint(self, model, optimizer=None):
        """Load the most recent checkpoint"""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, latest))
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"✓ Loaded latest checkpoint from epoch {checkpoint['epoch']}")
            return checkpoint
        return None

# ======================
# 1. LOAD AND PREPARE DATA
# ======================
print("Loading MNIST dataset...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Training data
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    transform=transform, 
    download=True
)

# Split training into train and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=64,
    shuffle=False
)

# Test data
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    transform=transform, 
    download=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ======================
# 2. DEFINE NEURAL NETWORK
# ======================
class SimpleNN(nn.Module):
    def __init__(self, layer_config=None):
        super(SimpleNN, self).__init__()
        # Default configuration
        default_config = {
            'layer_sizes': [784, 128, 128, 128, 10],
            'activations': ['relu', 'relu', 'relu', None]  # No activation after last layer
        }
        
        config = layer_config or default_config
        layer_sizes = config['layer_sizes']
        activations = config['activations']
        
        # Build layers dynamically
        layers = []
        layers.append(nn.Flatten())
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if activations[i]:  # Add activation if specified
                if activations[i].lower() == 'relu':
                    layers.append(nn.ReLU())
                elif activations[i].lower() == 'prelu':
                    layers.append(nn.PReLU())
                elif activations[i].lower() == 'relu6':
                    layers.append(nn.ReLU6())
                elif activations[i].lower() == 'leakyrelu':
                    layers.append(nn.LeakyReLU())
                elif activations[i].lower() == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activations[i].lower() == 'tanh':
                    layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
        self.config = config
    
    def forward(self, x):
        return self.model(x)

# ======================
# 3. INITIALIZE MODEL AND CHECKPOINT MANAGER
# ======================
model = SimpleNN()
checkpoint_manager = CheckpointManager()

# Try to load previous best model
previous_checkpoint = checkpoint_manager.load_checkpoint(model)
if previous_checkpoint:
    start_epoch = previous_checkpoint['epoch'] + 1
    train_losses = previous_checkpoint['train_losses']
    best_accuracy = previous_checkpoint['val_accuracy']
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0
    train_losses = []
    best_accuracy = 0
    print("No checkpoint found. Starting fresh training.")

# ======================
# 4. SETUP TRAINING
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load optimizer state if available
if previous_checkpoint and 'optimizer_state_dict' in previous_checkpoint:
    optimizer.load_state_dict(previous_checkpoint['optimizer_state_dict'])

# Add learning rate scheduler with verbose wrapper
scheduler = VerboseReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# ======================
# 5. TRAIN THE MODEL
# ======================
def validate_model(model, data_loader):
    """Calculate accuracy on validation set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

print("\nStarting training...")
num_epochs = 10
early_stop_patience = 5
patience_counter = 0

for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{start_epoch + num_epochs}], '
                  f'Batch [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    # Calculate average loss for this epoch
    avg_loss = total_loss / batch_count
    train_losses.append(avg_loss)
    
    # Validate model
    val_accuracy = validate_model(model, val_loader)
    
    # Update learning rate based on validation accuracy
    scheduler.step(val_accuracy)
    
    print(f'Epoch [{epoch+1}/{start_epoch + num_epochs}] completed. '
          f'Avg Loss: {avg_loss:.4f}, '
          f'Val Accuracy: {val_accuracy:.2f}%, '
          f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Check if this is the best model
    is_best = val_accuracy > best_accuracy
    
    if is_best:
        best_accuracy = val_accuracy
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1
    
    # Save checkpoint
    checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        train_losses=train_losses,
        val_accuracy=val_accuracy,
        model_config=model.config,
        is_best=is_best
    )
    
    # Early stopping
    if patience_counter >= early_stop_patience:
        print(f"\n⚠️  Early stopping triggered! No improvement for {early_stop_patience} epochs.")
        print(f"Best validation accuracy: {best_accuracy:.2f}%")
        break

# ======================
# 6. TEST THE MODEL
# ======================
print("\nTesting model...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# ======================
# 7. VISUALIZE RESULTS
# ======================
plt.figure(figsize=(15, 5))

# Plot training loss
plt.subplot(1, 3, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-o', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss History', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Show model architecture info
plt.subplot(1, 3, 2)
plt.text(0.1, 0.5, 
         f"Model Configuration:\n\n"
         f"Layers: {model.config['layer_sizes']}\n"
         f"Activations: {model.config['activations']}\n\n"
         f"Results:\n"
         f"Best Val Accuracy: {best_accuracy:.2f}%\n"
         f"Test Accuracy: {accuracy:.2f}%\n"
         f"Total Epochs: {len(train_losses)}",
         fontsize=12, 
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.axis('off')
plt.title('Model Info', fontsize=14)

# Show some predictions
plt.subplot(1, 3, 3)
model.eval()
with torch.no_grad():
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    outputs = model(images)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    _, predictions = torch.max(outputs, 1)
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        img = images[i].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        pred_prob = probabilities[i][predictions[i]].item() * 100
        color = 'green' if predictions[i] == labels[i] else 'red'
        plt.title(f'Pred: {predictions[i].item()} ({pred_prob:.1f}%)\nTrue: {labels[i].item()}', 
                 color=color, fontsize=9)
        plt.axis('off')

plt.suptitle(f'MNIST Classifier Results (Accuracy: {accuracy:.2f}%)', fontsize=16)
plt.tight_layout()
plt.show()

# ======================
# 8. MAKE PREDICTIONS
# ======================
print("\n" + "="*50)
print("MAKING PREDICTIONS")
print("="*50)

model.eval()
with torch.no_grad():
    # Test on 5 random samples
    for i in range(5):
        idx = torch.randint(0, len(test_dataset), (1,)).item()
        sample_image, sample_label = test_dataset[idx]
        sample_image_batch = sample_image.unsqueeze(0)
        
        output = model(sample_image_batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(output).item()
        confidence = probabilities[0][prediction].item() * 100
        
        status = "✓ CORRECT" if prediction == sample_label else "✗ WRONG"
        print(f"Sample {i+1}: {status}")
        print(f"  Predicted: {prediction} ({confidence:.1f}% confidence)")
        print(f"  Actual: {sample_label}")
        
        # Show top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        print(f"  Top 3 predictions:")
        for j in range(3):
            print(f"    {top3_indices[0][j].item()}: {top3_probs[0][j].item()*100:.1f}%")
        print()

# ======================
# 9. SAVE FINAL MODEL
# ======================
print("\nSaving final model...")
final_model_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'config': model.config,
    'accuracy': accuracy,
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}, final_model_path)
print(f"Final model saved to: {final_model_path}")

# List all checkpoints
print("\n" + "="*50)
print("AVAILABLE CHECKPOINTS")
print("="*50)
checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
for checkpoint in checkpoints:
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint)
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        print(f"{checkpoint}:")
        print(f"  Epoch: {checkpoint_data.get('epoch', 'N/A')}")
        print(f"  Accuracy: {checkpoint_data.get('val_accuracy', checkpoint_data.get('accuracy', 'N/A')):.2f}%")
        print(f"  Saved: {checkpoint_data.get('timestamp', 'N/A')}")
    except:
        print(f"{checkpoint}: Could not load")
    print()

print("="*50)
print("TRAINING COMPLETE!")
print("="*50)