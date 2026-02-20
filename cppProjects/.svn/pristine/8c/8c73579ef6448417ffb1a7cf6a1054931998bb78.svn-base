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
CHECKPOINT_DIR = "./checkpoints_TreebindLRbetter"
TREE_BINDING_DIR = "./tree_binding_dataLRbetter"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TREE_BINDING_DIR, exist_ok=True)

# Tree Binding Configuration
TREE_CONFIG = {
    'num_models': 6,           # Start with 6 models
    'keep_top': 2,              # Keep best 2 after each epoch
    'mutation_rate': 0.3,
    'mutation_strength': 0.1,
}

class VerboseReduceLROnPlateau:
    """Wrapper for ReduceLROnPlateau with verbose output"""
    def __init__(self, optimizer, mode='max', factor=0.5, patience=3, verbose=True):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
        self.verbose = verbose
        self.old_lrs = [group['lr'] for group in optimizer.param_groups]
        self.patience = patience
        self.bad_epochs = 0
        
    def step(self, metrics):
        self.scheduler.step(metrics)
        new_lrs = [group['lr'] for group in self.scheduler.optimizer.param_groups]
        
        if self.verbose:
            for i, (old_lr, new_lr) in enumerate(zip(self.old_lrs, new_lrs)):
                if new_lr != old_lr:
                    print(f"✓ Learning rate for layer group {i+1} reduced from {old_lr:.6f} to {new_lr:.6f}")
            self.old_lrs = new_lrs

class MultiLROptimizer:
    """Optimizer with different learning rates for each layer"""
    def __init__(self, model, base_lr=0.001, lr_ratios=None):
        """
        Args:
            model: The neural network model
            base_lr: Base learning rate
            lr_ratios: List of ratios for each layer (e.g., [1.0, 0.5, 0.25] for 3 layers)
                      If None, uses default ratios
        """
        self.model = model
        
        # Default learning rate ratios if not provided
        if lr_ratios is None:
            # Count linear layers
            linear_layers = [module for module in model.modules() 
                           if isinstance(module, nn.Linear)]
            # Higher learning rate for earlier layers, lower for later layers
            lr_ratios = [1.0 / (i + 1) for i in range(len(linear_layers))]
            # Normalize so the first layer has base_lr
            lr_ratios = [ratio / lr_ratios[0] for ratio in lr_ratios]
        
        # Create parameter groups with different learning rates
        param_groups = []
        
        # Group parameters by layer
        for idx, (name, param) in enumerate(model.named_parameters()):
            if 'weight' in name or 'bias' in name:
                # Determine which layer this parameter belongs to
                layer_idx = 0
                if '1.' in name:
                    layer_idx = 1
                elif '2.' in name:
                    layer_idx = 2
                elif '3.' in name:
                    layer_idx = 3
                # Cap layer_idx to lr_ratios length
                layer_idx = min(layer_idx, len(lr_ratios) - 1)
                
                param_groups.append({
                    'params': param,
                    'lr': base_lr * lr_ratios[layer_idx],
                    'name': name,
                    'layer': layer_idx
                })
        
        # Create optimizer with these parameter groups
        self.optimizer = optim.Adam(param_groups)
        self.param_groups = param_groups
        self.base_lr = base_lr
        self.lr_ratios = lr_ratios
        
        # Print layer-wise learning rates
        print(f"\nLayer-wise learning rates:")
        for i, ratio in enumerate(lr_ratios):
            print(f"  Layer {i}: {base_lr * ratio:.6f} (ratio: {ratio:.2f})")
    
    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self):
        self.optimizer.step()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
    
    def get_lr_info(self):
        """Get current learning rates for each layer"""
        lr_info = {}
        for group in self.param_groups:
            lr_info[group['name']] = group['lr']
        return lr_info

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
        self.next_id = 0
        
    def initialize_population(self):
        """Initialize with 6 models"""
        print(f"\nInitializing Tree Binding with {self.config['num_models']} models...")
        
        self.population = []
        self.next_id = 0
        
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
            
            # Different learning rate configurations for diversity
            lr_configs = [
                [1.0, 0.8, 0.6, 0.4],    # Decreasing
                [0.4, 0.6, 0.8, 1.0],    # Increasing
                [1.0, 0.5, 0.25, 0.125], # Steep decreasing
                [0.125, 0.25, 0.5, 1.0], # Steep increasing
                [1.0, 1.0, 1.0, 1.0],    # Uniform (global)
                [0.5, 1.0, 0.5, 1.0],    # Alternating
            ]
            
            lr_ratios = random.choice(lr_configs)
            base_lr = 0.001 * (0.8 + 0.4 * random.random())  # 0.0008 to 0.0012
            
            optimizer = MultiLROptimizer(model, base_lr=base_lr, lr_ratios=lr_ratios)
            
            self.population.append({
                'model': model,
                'optimizer': optimizer,
                'accuracy': 0.0,
                'id': self.next_id,
                'base_lr': base_lr,
                'lr_ratios': lr_ratios,
                'origin': 'initial'
            })
            self.next_id += 1
        
        print(f"✓ Created {len(self.population)} initial models with diverse LR configurations")
    
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
        
        # Show results with learning rate info
        print(f"\n  Top {min(3, len(self.population))} models:")
        for i in range(min(3, len(self.population))):
            model_info = self.population[i]
            lr_str = f"LRs: {model_info['base_lr']:.5f} * {model_info['lr_ratios']}"
            origin = model_info.get('origin', 'unknown')
            print(f"    #{i+1}: Model {model_info['id']} ({origin}) = {model_info['accuracy']:.2f}% | {lr_str}")
        
        # Show worst model
        if len(self.population) > 0:
            worst_model = self.population[-1]
            print(f"  Worst: Model {worst_model['id']} = {worst_model['accuracy']:.2f}%")
        
        return current_best
    
    def add_main_model(self, main_model, main_accuracy, optimizer, base_lr=0.001, lr_ratios=None):
        """Add the main model to the population, replacing the worst performing model"""
        print(f"\n[Tree Binding] Adding main model to population...")
        
        if len(self.population) == 0:
            print("  Population is empty. Cannot add main model.")
            return False
        
        # Find worst performing model
        worst_idx = -1
        worst_accuracy = float('inf')
        
        for idx, model_info in enumerate(self.population):
            if model_info['accuracy'] < worst_accuracy:
                worst_accuracy = model_info['accuracy']
                worst_idx = idx
        
        if worst_idx == -1:
            print("  Could not find worst model.")
            return False
        
        # Create a copy of the main model
        main_model_copy = copy.deepcopy(main_model)
        
        # Use the optimizer's learning rate configuration
        if lr_ratios is None:
            lr_ratios = [1.0, 0.8, 0.6, 0.4]  # Default decreasing pattern
        
        # Create optimizer for the main model copy
        main_optimizer = MultiLROptimizer(main_model_copy, base_lr=base_lr, lr_ratios=lr_ratios)
        
        # Replace worst model with main model
        removed_model = self.population[worst_idx]
        print(f"  Replacing Model {removed_model['id']} ({removed_model['origin']}) = {removed_model['accuracy']:.2f}%")
        print(f"  with Main Model (accuracy: {main_accuracy:.2f}%)")
        
        self.population[worst_idx] = {
            'model': main_model_copy,
            'optimizer': main_optimizer,
            'accuracy': main_accuracy,
            'id': self.next_id,
            'base_lr': base_lr,
            'lr_ratios': lr_ratios,
            'origin': 'main_model_replacement'
        }
        self.next_id += 1
        
        # Re-sort population
        self.population.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"  ✓ Main model added to population at position {worst_idx + 1}")
        return True
    
    def evolve_population(self):
        """Evolve: keep top 2, generate new ones"""
        print(f"\n[Tree Binding] Evolving to Generation {self.generation + 2}...")
        
        # Keep top 2
        top_models = self.population[:self.config['keep_top']]
        
        # New population
        new_population = []
        
        # Add top 2 (elite)
        for top in top_models:
            new_population.append({
                'model': copy.deepcopy(top['model']),
                'optimizer': MultiLROptimizer(top['model'], base_lr=top['base_lr'], lr_ratios=top['lr_ratios']),
                'accuracy': top['accuracy'],
                'id': self.next_id,
                'base_lr': top['base_lr'],
                'lr_ratios': top['lr_ratios'],
                'origin': f'elite_gen_{self.generation + 1}'
            })
            self.next_id += 1
        
        # Create new models from top 2 - optimized mutation
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
            
            # Mutate learning rate configuration
            lr_ratios = parent['lr_ratios'].copy()
            
            # With 30% probability, change the LR pattern
            if random.random() < 0.3:
                # Choose a new LR pattern
                lr_configs = [
                    [1.0, 0.8, 0.6, 0.4],    # Decreasing
                    [0.4, 0.6, 0.8, 1.0],    # Increasing
                    [1.0, 0.5, 0.25, 0.125], # Steep decreasing
                    [0.125, 0.25, 0.5, 1.0], # Steep increasing
                    [1.0, 1.0, 1.0, 1.0],    # Uniform (global)
                    [0.5, 1.0, 0.5, 1.0],    # Alternating
                ]
                lr_ratios = random.choice(lr_configs)
            else:
                # Slightly mutate existing ratios
                for i in range(len(lr_ratios)):
                    if random.random() < 0.2:  # 20% chance to mutate each ratio
                        lr_ratios[i] *= (0.9 + 0.2 * random.random())
            
            # Slightly different base learning rate
            child_base_lr = parent['base_lr'] * (0.9 + 0.2 * random.random())
            
            new_population.append({
                'model': child,
                'optimizer': MultiLROptimizer(child, base_lr=child_base_lr, lr_ratios=lr_ratios),
                'accuracy': 0.0,
                'id': self.next_id,
                'base_lr': child_base_lr,
                'lr_ratios': lr_ratios,
                'origin': f'mutated_gen_{self.generation + 1}'
            })
            self.next_id += 1
        
        self.population = new_population
        self.generation += 1
        
        print(f"  ✓ Evolved: Kept top {len(top_models)}, created {len(new_population)-len(top_models)} new")
        print(f"  Total: {len(self.population)} models ready")
        
        return True
    
    def get_best_model(self):
        return self.population[0]['model'] if self.population else None
    
    def get_population_stats(self):
        """Get statistics about the population"""
        accuracies = [m['accuracy'] for m in self.population]
        origins = [m.get('origin', 'unknown') for m in self.population]
        
        return {
            'size': len(self.population),
            'avg_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
            'best_accuracy': max(accuracies) if accuracies else 0,
            'worst_accuracy': min(accuracies) if accuracies else 0,
            'origins': origins
        }

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
# 2. DEFINE NEURAL NETWORK - CHANGED TO 784-128-64-10
# ======================
class SimpleNN(nn.Module):
    def __init__(self, layer_config=None):
        super(SimpleNN, self).__init__()
        # Changed configuration: 784-128-128-128-10
        default_config = { 
            'layer_sizes': [784, 128, 128, 128, 10],
            'activations': ['relu', 'relu', 'relu', None]
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
    print("TREE BINDING TRAINING WITH MAIN MODEL INTEGRATION")
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
    print(f"  Process: Train {TREE_CONFIG['num_models']} → Evaluate → Keep {TREE_CONFIG['keep_top']} → Generate {TREE_CONFIG['num_models']}")
    print(f"  NEW: Main model replaces worst tree model when it performs better")
    
    print(f"\nNetwork Architecture: {model.config['layer_sizes']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ======================
    # 4. SETUP TRAINING WITH LAYER-WISE LEARNING RATES
    # ======================
    criterion = nn.CrossEntropyLoss()
    
    # Use MultiLROptimizer for main model with custom layer ratios
    # Example: [1.0, 0.8, 0.6, 0.4] - decreasing learning rate for deeper layers
    optimizer = MultiLROptimizer(model, base_lr=0.001, lr_ratios=[1.0, 0.8, 0.6, 0.4])

    # Load optimizer state if available
    if previous_checkpoint and 'optimizer_state_dict' in previous_checkpoint:
        optimizer.load_state_dict(previous_checkpoint['optimizer_state_dict'])

    # Add learning rate scheduler with verbose wrapper
    scheduler = VerboseReduceLROnPlateau(optimizer.optimizer, mode='max', factor=0.5, patience=3, verbose=True)

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
        print("\n[Phase 1] Training main model with layer-wise learning rates...")
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
        
        # Print current layer-wise learning rates
        lr_info = optimizer.get_lr_info()
        print(f"\n  Main model results:")
        print(f"    Loss: {avg_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        print(f"    Layer-wise Learning Rates:")
        for param_name, lr in lr_info.items():
            print(f"      {param_name}: {lr:.6f}")
        
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
        # PHASE 3: MODEL COMPARISON & INTEGRATION
        # ====================================
        print(f"\n[Phase 3] Model comparison & integration...")
        
        if tree_accuracy > val_accuracy:
            print(f"  ✓ Tree binding is better! ({tree_accuracy:.2f}% vs {val_accuracy:.2f}%)")
            print(f"  ↳ Updating main model with tree binding best")
            model.load_state_dict(tree_binding.get_best_model().state_dict())
            val_accuracy = tree_accuracy  # Use tree accuracy
        else:
            print(f"  ✓ Main model is better ({val_accuracy:.2f}% vs {tree_accuracy:.2f}%)")
            print(f"  ↳ Adding main model to tree binding population...")
            
            # Get current base learning rate from optimizer
            current_base_lr = optimizer.base_lr
            
            # Add main model to tree binding, replacing worst performing model
            success = tree_binding.add_main_model(
                main_model=model,
                main_accuracy=val_accuracy,
                optimizer=optimizer,
                base_lr=current_base_lr,
                lr_ratios=optimizer.lr_ratios
            )
            
            if success:
                print(f"  ✓ Main model integrated into tree binding population")
            else:
                print(f"  ✗ Failed to integrate main model into population")
        
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
            optimizer=optimizer.optimizer,
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
    
    # Color code by origin
    colors = []
    for model_info in tree_binding.population:
        origin = model_info.get('origin', 'unknown')
        if 'main_model' in origin:
            colors.append('red')  # Main model replacements
        elif 'elite' in origin:
            colors.append('green')  # Elite models
        elif 'mutated' in origin:
            colors.append('blue')  # Mutated models
        elif 'initial' in origin:
            colors.append('orange')  # Initial models
        else:
            colors.append('gray')  # Unknown origin
    
    plt.bar(range(len(accuracies)), accuracies, color=colors)
    plt.xlabel('Model Index', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Generation {tree_binding.generation + 1} Models', fontsize=14)
    plt.axhline(y=TREE_CONFIG['keep_top'], color='red', linestyle=':', alpha=0.3)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color='green', label='Elite'),
        mpatches.Patch(color='blue', label='Mutated'),
        mpatches.Patch(color='red', label='Main Model'),
        mpatches.Patch(color='orange', label='Initial'),
    ]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=8)

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
    stats = tree_binding.get_population_stats()
    plt.text(0.1, 0.5, 
            f"Model Configuration:\n\n"
            f"Layers: {model.config['layer_sizes']}\n"
            f"Activations: {model.config['activations']}\n\n"
            f"Tree Binding:\n"
            f"  Generations: {tree_binding.generation + 1}\n"
            f"  Population: {stats['size']}\n"
            f"  Avg Accuracy: {stats['avg_accuracy']:.2f}%\n"
            f"  Best Accuracy: {stats['best_accuracy']:.2f}%\n"
            f"  Keep Top: {TREE_CONFIG['keep_top']}\n"
            f"  Mutation Rate: {TREE_CONFIG['mutation_rate']}\n\n"
            f"Main Model Integration:\n"
            f"  When main model is better,\n"
            f"  it replaces worst tree model",
            fontsize=10, 
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
            plt.subplot(2, 3, i + 3)
            img = images[i].squeeze().numpy()
            plt.imshow(img, cmap='gray')
            pred_prob = probabilities[i][predictions[i]].item() * 100
            color = 'green' if predictions[i] == labels[i] else 'red'
            plt.title(f'Pred: {predictions[i].item()} ({pred_prob:.1f}%)\nTrue: {labels[i].item()}', 
                    color=color, fontsize=9)
            plt.axis('off')

    plt.suptitle(f'Tree Binding with Main Model Integration - Test: {accuracy:.2f}%', fontsize=16)
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
        'optimizer_lr_ratios': optimizer.lr_ratios,
        'population_stats': tree_binding.get_population_stats(),
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
            'main_model_lr_ratios': optimizer.lr_ratios,
            'population_stats': tree_binding.get_population_stats(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"Tree binding results saved to: {tree_results_path}")

    print("="*60)
    print("TREE BINDING TRAINING COMPLETE!")
    print("="*60)
    print(f"Main Model Accuracy: {accuracy:.2f}%")
    print(f"Tree Binding Best: {tree_binding.best_accuracy:.2f}%")
    print(f"Ensemble (Top 3): {ensemble_accuracy:.2f}%")
    print(f"Tree Generations: {tree_binding.generation + 1}")
    print(f"Main Model Layer-wise LR Ratios: {optimizer.lr_ratios}")
    
    # Show population statistics
    stats = tree_binding.get_population_stats()
    print(f"\nFinal Population Statistics:")
    print(f"  Size: {stats['size']} models")
    print(f"  Average Accuracy: {stats['avg_accuracy']:.2f}%")
    print(f"  Best Accuracy: {stats['best_accuracy']:.2f}%")
    print(f"  Worst Accuracy: {stats['worst_accuracy']:.2f}%")
    print(f"  Model Origins: {', '.join(stats['origins'])}")
    
    print("="*60)

if __name__ == "__main__":
    main()