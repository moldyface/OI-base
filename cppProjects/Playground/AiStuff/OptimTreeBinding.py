import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import json
import copy
import random
import numpy as np
from datetime import datetime

# ======================
# CONFIGURATION
# ======================
CHECKPOINT_DIR = "./checkpoints_Treebind3"
TREE_BINDING_DIR = "./tree_binding_data3"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TREE_BINDING_DIR, exist_ok=True)

# Tree Binding Configuration
TREE_CONFIG = {
    'num_models': 10,           # Start with 10 models
    'keep_top': 3,              # Keep best 3 after each epoch
    'mutation_rate': 0.25,
    'mutation_strength': 0.05,
}

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
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
            print(f"  Previous accuracy: {checkpoint['val_accuracy']:.2f}%")
            print(f"  Saved on: {checkpoint['timestamp']}")
            
            return checkpoint
        return None

# ======================
# TREE BINDING MANAGER (SPEED OPTIMIZED)
# ======================
class TreeBindingManager:
    """Manages tree binding evolution - Optimized for speed"""
    def __init__(self, base_model, config=TREE_CONFIG):
        self.config = config
        self.base_model = base_model
        self.population = []
        self.generation = 0
        self.best_accuracy = 0
        self.best_model = None
        
    def initialize_population(self):
        """Initialize with 10 models"""
        print(f"\nInitializing Tree Binding with {self.config['num_models']} models...")
        
        self.population = []
        
        # Speed optimization: Get base state once
        base_state = self.base_model.state_dict()
        
        for i in range(self.config['num_models']):
            model = copy.deepcopy(self.base_model)
            
            # Add small weight variations (except first model) - optimized
            if i > 0:
                model_state = model.state_dict()
                # Vectorized noise addition for speed
                for key in model_state:
                    if 'weight' in key or 'bias' in key:
                        model_state[key] += torch.randn_like(model_state[key]) * 0.01
                model.load_state_dict(model_state)
            
            # Different learning rates for diversity
            lr = 0.001 * (0.8 + 0.6 * random.random())  # 0.0008 to 0.0012
            
            self.population.append({
                'model': model,
                'optimizer': optim.Adam(model.parameters(), lr=lr),
                'accuracy': 0.0,
                'id': i,
                'lr': lr
            })
        
        print(f"✓ Created {len(self.population)} initial models")
    
    def train_population_fast(self, train_loader, criterion):
        """Train all models quickly - Optimized"""
        print(f"\n[Tree Binding] Training Generation {self.generation + 1}...")
        
        # Train on fewer batches for speed (200 instead of 700)
        batches_to_train = 200
        
        # Train each model
        for idx, model_info in enumerate(self.population):
            model = model_info['model']
            optimizer = model_info['optimizer']
            
            model.train()
            
            batch_iter = iter(train_loader)
            
            for batch_idx in range(batches_to_train):
                # Get next batch (recycle if needed)
                try:
                    images, labels = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(train_loader)
                    images, labels = next(batch_iter)
                
                optimizer.zero_grad(set_to_none=True)  # Faster zero_grad
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Minimal progress output (only first model, every 50 batches)
                if idx == 0 and batch_idx % 50 == 0:
                    print(f'  Model 1, Batch [{batch_idx}/{batches_to_train}], Loss: {loss.item():.4f}')
        
        print(f"  ✓ Trained all {len(self.population)} models on {batches_to_train} batches")
    
    def evaluate_population(self, val_loader, criterion):
        """Evaluate all models - Optimized"""
        print("\n[Tree Binding] Evaluating models...")
        
        for model_info in self.population:
            model = model_info['model']
            model.eval()
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                # Use first 50 batches for quick evaluation
                batch_iter = iter(val_loader)
                for i in range(50):
                    try:
                        images, labels = next(batch_iter)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    except StopIteration:
                        break
            
            accuracy = 100 * correct / total if total > 0 else 0
            model_info['accuracy'] = accuracy
        
        # Sort by accuracy
        self.population.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Update best
        current_best = self.population[0]['accuracy']
        if current_best > self.best_accuracy:
            self.best_accuracy = current_best
            self.best_model = copy.deepcopy(self.population[0]['model'])
            print(f"  🎯 New Tree Binding Best: {current_best:.2f}%")
        
        # Show results
        print(f"\n  Top {min(3, len(self.population))} models:")
        for i in range(min(3, len(self.population))):
            print(f"    #{i+1}: Model {self.population[i]['id']} = {self.population[i]['accuracy']:.2f}%")
        
        return current_best
    
    def evolve_population(self):
        """Evolve: keep top 3, generate 10 new"""
        print(f"\n[Tree Binding] Evolving to Generation {self.generation + 2}...")
        
        # Keep top 3
        top_models = self.population[:self.config['keep_top']]
        
        # New population
        new_population = []
        next_id = 0
        
        # Add top 3 (elite)
        for top in top_models:
            new_population.append({
                'model': copy.deepcopy(top['model']),
                'optimizer': optim.Adam(top['model'].parameters(), lr=top['lr']),
                'accuracy': top['accuracy'],
                'id': next_id,
                'lr': top['lr']
            })
            next_id += 1
        
        # Create 7 new from top 3 - optimized mutation
        while len(new_population) < self.config['num_models']:
            parent = random.choice(top_models)
            
            # Create child with mutation
            child = copy.deepcopy(parent['model'])
            
            # Apply mutation with vectorized operations
            with torch.no_grad():
                child_state = child.state_dict()
                for key in child_state:
                    if 'weight' in key or 'bias' in key:
                        # Generate noise only where mutation occurs
                        mask = torch.rand_like(child_state[key]) < self.config['mutation_rate']
                        if mask.any():
                            noise = torch.randn_like(child_state[key]) * self.config['mutation_strength']
                            # Apply noise only to masked positions
                            child_state[key] = child_state[key] + noise * mask.float()
                child.load_state_dict(child_state)
            
            # Slightly different learning rate
            child_lr = parent['lr'] * (0.9 + 0.2 * random.random())
            
            new_population.append({
                'model': child,
                'optimizer': optim.Adam(child.parameters(), lr=child_lr),
                'accuracy': 0.0,
                'id': next_id,
                'lr': child_lr
            })
            next_id += 1
        
        self.population = new_population
        self.generation += 1
        
        print(f"  ✓ Evolved: Kept top {len(top_models)}, created {len(new_population)-len(top_models)} new")
        print(f"  Total: {len(self.population)} models ready")
        
        return True
    
    def get_best_model(self):
        return self.population[0]['model'] if self.population else None

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

# Use 0 workers for Windows compatibility
num_workers = 0

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=128,  # Larger batch size for speed
    shuffle=True,
    num_workers=num_workers
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=num_workers
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
    batch_size=128,
    shuffle=False,
    num_workers=num_workers
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: 128")

# ======================
# 2. DEFINE NEURAL NETWORK (SAME AS BEFORE)
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
# MAIN EXECUTION
# ======================
def main():
    print("\n" + "="*60)
    print("TREE BINDING TRAINING (SPEED OPTIMIZED)")
    print("="*60)

    model = SimpleNN()
    checkpoint_manager = CheckpointManager()
    tree_binding = TreeBindingManager(model)

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

    # Initialize tree binding
    tree_binding.initialize_population()

    print(f"\nTree Binding Configuration:")
    print(f"  Population: {TREE_CONFIG['num_models']} models")
    print(f"  Keep top: {TREE_CONFIG['keep_top']} models each generation")
    print(f"  Mutation rate: {TREE_CONFIG['mutation_rate']}")
    print(f"  Process: Train 10 → Evaluate → Keep 3 → Generate 10")

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
    # 5. TRAIN THE MODEL WITH TREE BINDING
    # ======================
    def validate_model(model, data_loader, num_batches=100):
        """Calculate accuracy on validation set - optimized"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            batch_iter = iter(data_loader)
            for _ in range(num_batches):
                try:
                    images, labels = next(batch_iter)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                except StopIteration:
                    break
        
        accuracy = 100 * correct / total if total > 0 else 0
        return accuracy

    print("\nStarting training with Tree Binding...")
    num_epochs = 10
    early_stop_patience = 5
    patience_counter = 0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{start_epoch + num_epochs}")
        print(f"{'='*60}")
        
        # ====================================
        # PHASE 1: REGULAR TRAINING
        # ====================================
        print("\n[Phase 1] Training main model...")
        model.train()
        total_loss = 0
        batch_count = 0
        
        # Speed optimization: Train on fewer batches
        batches_to_train = 200
        data_iter = iter(train_loader)
        
        for batch_idx in range(batches_to_train):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad(set_to_none=True)  # Faster zero_grad
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f'  Batch [{batch_idx}/{batches_to_train}], Loss: {loss.item():.4f}')
        
        # Calculate average loss for this epoch
        avg_loss = total_loss / batch_count
        train_losses.append(avg_loss)
        
        # Validate model - faster validation
        val_accuracy = validate_model(model, val_loader, num_batches=100)
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_accuracy)
        
        print(f"\n  Main model results:")
        print(f"    Loss: {avg_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        print(f"    Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # ====================================
        # PHASE 2: TREE BINDING EVOLUTION
        # ====================================
        print(f"\n[Phase 2] Tree Binding (Generation {tree_binding.generation + 1})...")
        
        # Train tree binding population
        tree_binding.train_population_fast(train_loader, criterion)
        
        # Evaluate tree binding population
        tree_accuracy = tree_binding.evaluate_population(val_loader, criterion)
        
        print(f"\n  Tree binding results:")
        print(f"    Best accuracy: {tree_accuracy:.2f}%")
        print(f"    Population: {len(tree_binding.population)} models")
        
        # ====================================
        # PHASE 3: UPDATE MAIN MODEL IF TREE IS BETTER
        # ====================================
        print(f"\n[Phase 3] Model comparison...")
        
        if tree_accuracy > val_accuracy:
            print(f"  ✓ Tree binding is better! ({tree_accuracy:.2f}% vs {val_accuracy:.2f}%)")
            print(f"  ↳ Updating main model with tree binding best")
            model.load_state_dict(tree_binding.get_best_model().state_dict())
            val_accuracy = tree_accuracy  # Use tree accuracy
        else:
            print(f"  ✓ Main model is better ({val_accuracy:.2f}% vs {tree_accuracy:.2f}%)")
        
        # Check if this is the best model
        is_best = val_accuracy > best_accuracy
        
        if is_best:
            best_accuracy = val_accuracy
            patience_counter = 0  # Reset patience counter
            print(f"  🎯 NEW OVERALL BEST: {best_accuracy:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stop_patience})")
        
        # ====================================
        # PHASE 4: EVOLVE TREE BINDING POPULATION
        # ====================================
        print(f"\n[Phase 4] Evolving tree binding...")
        
        # Evolve to next generation
        tree_binding.evolve_population()
        
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
        
        # Show epoch timing
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        print(f"\n  Epoch {epoch + 1} completed in {epoch_time:.1f} seconds")
        
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
        # Test on fewer batches for speed
        batches_to_test = 200
        batch_iter = iter(test_loader)
        
        for _ in range(batches_to_test):
            try:
                images, labels = next(batch_iter)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            except StopIteration:
                break

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    # ======================
    # 7. TEST ENSEMBLE OF TOP 3 TREE MODELS
    # ======================
    print("\n" + "="*60)
    print("TESTING TREE BINDING ENSEMBLE")
    print("="*60)

    # Test ensemble of top 3 tree models
    top_models = tree_binding.population[:3]

    def ensemble_predict(models, images):
        """Combine predictions from multiple models"""
        predictions = []
        for model_info in models:
            model = model_info['model']
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                predictions.append(outputs)
        
        # Average predictions
        avg_output = torch.stack(predictions).mean(dim=0)
        return avg_output

    ensemble_correct = 0
    ensemble_total = 0

    with torch.no_grad():
        # Test on fewer batches for speed
        batches_to_test = 200
        batch_iter = iter(test_loader)
        
        for _ in range(batches_to_test):
            try:
                images, labels = next(batch_iter)
                outputs = ensemble_predict(top_models, images)
                _, predicted = torch.max(outputs.data, 1)
                ensemble_total += labels.size(0)
                ensemble_correct += (predicted == labels).sum().item()
            except StopIteration:
                break

    ensemble_accuracy = 100 * ensemble_correct / ensemble_total
    print(f"Ensemble of top 3 tree models: {ensemble_accuracy:.2f}%")

    # ======================
    # 8. VISUALIZE RESULTS
    # ======================
    plt.figure(figsize=(18, 6))

    # Plot 1: Training loss
    plt.subplot(2, 3, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-o', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss History', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Plot 2: Tree binding model accuracies
    plt.subplot(2, 3, 2)
    accuracies = [m['accuracy'] for m in tree_binding.population]
    colors = ['green'] * TREE_CONFIG['keep_top'] + ['blue'] * (len(accuracies) - TREE_CONFIG['keep_top'])
    plt.bar(range(len(accuracies)), accuracies, color=colors)
    plt.xlabel('Model Index', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Generation {tree_binding.generation + 1} Models', fontsize=14)
    plt.axhline(y=TREE_CONFIG['keep_top'], color='red', linestyle=':', alpha=0.3)
    plt.grid(True, alpha=0.3, axis='y')

    # Plot 3: Performance comparison
    plt.subplot(2, 3, 3)
    labels = ['Main Model', 'Tree Best', 'Ensemble (Top 3)']
    accuracies = [accuracy, tree_binding.best_accuracy, ensemble_accuracy]
    colors = ['blue', 'green', 'orange']

    bars = plt.bar(labels, accuracies, color=colors)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Comparison', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)

    # Plot 4: Model info
    plt.subplot(2, 3, 4)
    plt.text(0.1, 0.5, 
            f"Model Configuration:\n\n"
            f"Layers: {model.config['layer_sizes']}\n"
            f"Activations: {model.config['activations']}\n\n"
            f"Tree Binding:\n"
            f"  Generations: {tree_binding.generation + 1}\n"
            f"  Population: {len(tree_binding.population)}\n"
            f"  Keep Top: {TREE_CONFIG['keep_top']}\n"
            f"  Mutation Rate: {TREE_CONFIG['mutation_rate']}",
            fontsize=11, 
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')

    # Plot 5-6: Sample predictions
    plt.subplot(2, 3, 5)
    model.eval()
    with torch.no_grad():
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        outputs = model(images[:6])
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
        
        for i in range(3):
            plt.subplot(2, 3, i + 6)
            img = images[i].squeeze().numpy()
            plt.imshow(img, cmap='gray')
            pred_prob = probabilities[i][predictions[i]].item() * 100
            color = 'green' if predictions[i] == labels[i] else 'red'
            plt.title(f'Pred: {predictions[i].item()} ({pred_prob:.1f}%)\nTrue: {labels[i].item()}', 
                    color=color, fontsize=9)
            plt.axis('off')

    plt.suptitle(f'Tree Binding Training - Test: {accuracy:.2f}%', fontsize=16)
    plt.tight_layout()
    plt.show()

    # ======================
    # 9. MAKE PREDICTIONS
    # ======================
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)

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
    # 10. SAVE FINAL MODEL
    # ======================
    print("\nSaving final model...")
    final_model_path = os.path.join(CHECKPOINT_DIR, "final_tree_binding_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'accuracy': accuracy,
        'ensemble_accuracy': ensemble_accuracy,
        'tree_generations': tree_binding.generation + 1,
        'tree_best_accuracy': tree_binding.best_accuracy,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Save tree binding results
    tree_results_path = os.path.join(CHECKPOINT_DIR, "tree_binding_results.json")
    with open(tree_results_path, 'w') as f:
        json.dump({
            'test_accuracy': accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'tree_best_accuracy': tree_binding.best_accuracy,
            'tree_generations': tree_binding.generation + 1,
            'tree_config': TREE_CONFIG,
            'model_config': model.config,
            'timestamp': datetime.now().strftime("%Y-%m-d %H:%M:%S")
        }, f, indent=2)
    print(f"Tree binding results saved to: {tree_results_path}")

    print("="*60)
    print("TREE BINDING TRAINING COMPLETE!")
    print("="*60)
    print(f"Main Model Accuracy: {accuracy:.2f}%")
    print(f"Tree Binding Best: {tree_binding.best_accuracy:.2f}%")
    print(f"Ensemble (Top 3): {ensemble_accuracy:.2f}%")
    print(f"Tree Generations: {tree_binding.generation + 1}")
    print("="*60)

if __name__ == "__main__":
    main()