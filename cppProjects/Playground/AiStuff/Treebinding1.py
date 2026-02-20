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
CHECKPOINT_DIR = "./checkpoints_Treebind_Augmented4"
TREE_BINDING_DIR = "./tree_binding_data_Augmented4"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TREE_BINDING_DIR, exist_ok=True)

# Tree Binding Configuration
TREE_CONFIG = {
    'num_models': 4,
    'keep_top': 2,
    'mutation_rate': 0.2,
    'mutation_strength': 0.5,
    
    # LR Configuration
    'base_lr_range': [0.0005, 0.003],
    'min_lr': 1e-6,
    'max_lr': 0.02,
    
    # Backtracking Configuration
    'backtracking_enabled': True,
    'backtrack_generations': 2,
    'backtracking_threshold': 0.05,
    'backtracking_lr_factor': 0.5,  # Lower LR when backtracking
    
    # ACCURACY-BASED LR ADJUSTMENT
    'high_accuracy_threshold': 98.0,  # When accuracy >= 98%, use small LR
    'low_accuracy_threshold': 95.0,   # When accuracy <= 95%, use large LR
    'high_acc_lr_factor': 0.25,       # 0.25x when accuracy is HIGH (fine-tune)
    'low_acc_lr_factor': 1.3,         # 1.4x when accuracy is LOW (explore)
    
    # Plateau Detection
    'plateau_patience': 3,
    
    # Mutation Strategies
    'normal_mutation_rate': 0.3,
    'aggressive_mutation_rate': 0.6,
    'normal_mutation_strength': 0.1,
    'aggressive_mutation_strength': 0.3,
}

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
            'train_losses': train_losses,
            'val_accuracy': val_accuracy,
            'model_config': model_config or {},
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if hasattr(optimizer, 'state_dict'):
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_path)
            self.best_accuracy = val_accuracy
            print(f"✓ New best model saved with accuracy: {val_accuracy:.2f}%")
    
    def load_checkpoint(self, model, optimizer=None, ignore_optimizer=False):
        """Load the best checkpoint"""
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint and not ignore_optimizer:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"✓ Loaded optimizer state from checkpoint")
                except Exception as e:
                    print(f"⚠️ Could not load optimizer state: {e}")
            
            print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
            print(f"  Previous accuracy: {checkpoint['val_accuracy']:.2f}%")
            print(f"  Saved on: {checkpoint['timestamp']}")
            
            return checkpoint
        return None

class AccuracyBasedLROptimizer:
    """LR Optimizer that adjusts based on accuracy: High acc → Low LR, Low acc → High LR"""
    def __init__(self, model, base_lr=None, lr_ratios=None, config=TREE_CONFIG):
        self.model = model
        self.config = config
        
        if base_lr is None:
            base_lr = random.uniform(config['base_lr_range'][0], config['base_lr_range'][1])
        
        if lr_ratios is None:
            linear_layers = [module for module in model.modules() 
                           if isinstance(module, nn.Linear)]
            lr_ratios = [1.0 / (i + 1) for i in range(len(linear_layers))]
            lr_ratios = [ratio / lr_ratios[0] for ratio in lr_ratios]
        
        self.param_groups_list = []
        for idx, (name, param) in enumerate(model.named_parameters()):
            if 'weight' in name or 'bias' in name:
                layer_idx = self._get_layer_index(name, len(lr_ratios))
                initial_lr = base_lr * lr_ratios[layer_idx]
                
                self.param_groups_list.append({
                    'params': param,
                    'lr': initial_lr,
                    'name': name,
                    'layer': layer_idx,
                    'base_ratio': lr_ratios[layer_idx],
                    'original_lr': initial_lr,
                })
        
        self.optimizer = optim.AdamW(self.param_groups_list, weight_decay=1e-4)
        self.base_lr = base_lr
        self.lr_ratios = lr_ratios
        self.accuracy_history = []
        self.best_accuracy = 0
        self.plateau_counter = 0
        self.current_accuracy = 0
        self.last_lr_adjustment = "initial"
        
        print(f"\nAccuracy-Based LR Optimizer:")
        print(f"  Base LR: {base_lr:.6f}")
        print(f"  LR Ratios: {lr_ratios}")
        print(f"  Strategy: High acc({config['high_accuracy_threshold']}%+) → LR×{config['high_acc_lr_factor']}")
        print(f"             Low acc({config['low_accuracy_threshold']}%-) → LR×{config['low_acc_lr_factor']}")
    
    def _get_layer_index(self, param_name, max_layers):
        for i in range(max_layers):
            if f'{i}.' in param_name:
                return i
        return 0
    
    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self):
        self.optimizer.step()
    
    def state_dict(self):
        return {
            'optimizer_state': self.optimizer.state_dict(),
            'base_lr': self.base_lr,
            'lr_ratios': self.lr_ratios,
            'accuracy_history': self.accuracy_history,
            'best_accuracy': self.best_accuracy,
        }
    
    def load_state_dict(self, state_dict):
        if 'optimizer_state' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer_state'])
        if 'base_lr' in state_dict:
            self.base_lr = state_dict['base_lr']
        if 'lr_ratios' in state_dict:
            self.lr_ratios = state_dict['lr_ratios']
        if 'accuracy_history' in state_dict:
            self.accuracy_history = state_dict['accuracy_history']
        if 'best_accuracy' in state_dict:
            self.best_accuracy = state_dict['best_accuracy']
    
    def get_lr_info(self):
        return {group['name']: group['lr'] for group in self.param_groups_list}
    
    def adjust_lr_based_on_accuracy(self, current_accuracy):
        """Adjust LR based on accuracy: High acc → Low LR, Low acc → High LR"""
        self.current_accuracy = current_accuracy
        self.accuracy_history.append(current_accuracy)
        if len(self.accuracy_history) > 20:
            self.accuracy_history.pop(0)
        
        # Update best accuracy
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
        
        # Determine LR adjustment factor based on accuracy
        old_base_lr = self.base_lr
        adjustment_factor = 1.0
        reason = "no_change"
        
        if current_accuracy >= self.config['high_accuracy_threshold']:
            # HIGH ACCURACY: Use small LR (0.25x) for fine-tuning near global minimum
            adjustment_factor = self.config['high_acc_lr_factor']
            reason = f"high_acc({current_accuracy:.1f}%≥{self.config['high_accuracy_threshold']}%)"
            
        elif current_accuracy <= self.config['low_accuracy_threshold']:
            # LOW ACCURACY: Use large LR (2.0x) to explore more aggressively
            adjustment_factor = self.config['low_acc_lr_factor']
            reason = f"low_acc({current_accuracy:.1f}%≤{self.config['low_accuracy_threshold']}%)"
            
        elif self.plateau_counter >= self.config['plateau_patience']:
            # PLATEAU: Slight increase to escape
            adjustment_factor = 1.5
            reason = f"plateau({self.plateau_counter}gens)"
            self.plateau_counter = 0
        
        # Apply LR adjustment
        if adjustment_factor != 1.0:
            self.last_lr_adjustment = reason
            
            # Calculate new base LR
            if adjustment_factor > 1.0:
                new_base_lr = min(self.base_lr * adjustment_factor, self.config['max_lr'])
            else:
                new_base_lr = max(self.base_lr * adjustment_factor, self.config['min_lr'])
            
            # Apply to all parameter groups
            for group in self.param_groups_list:
                if adjustment_factor > 1.0:
                    new_lr = min(group['lr'] * adjustment_factor, self.config['max_lr'])
                else:
                    new_lr = max(group['lr'] * adjustment_factor, self.config['min_lr'])
                group['lr'] = new_lr
            
            self.base_lr = new_base_lr
            
            # Print adjustment info
            direction = "↑" if adjustment_factor > 1.0 else "↓"
            print(f"  {direction} LR {reason}: {old_base_lr:.6f}→{self.base_lr:.6f} (×{adjustment_factor:.2f})")
    
    def apply_backtracking_lr(self):
        """Apply backtracking LR reduction (0.25x)"""
        print(f"  🔙 Applying backtracking LR reduction")
        
        old_base_lr = self.base_lr
        adjustment_factor = self.config['backtracking_lr_factor']
        
        # Reduce LR for careful recovery
        self.base_lr = max(self.base_lr * adjustment_factor, self.config['min_lr'])
        
        # Apply to all parameter groups
        for group in self.param_groups_list:
            new_lr = max(group['lr'] * adjustment_factor, self.config['min_lr'])
            group['lr'] = new_lr
        
        self.last_lr_adjustment = f"backtracking(×{adjustment_factor})"
        print(f"  ↳ LR reduced: {old_base_lr:.6f}→{self.base_lr:.6f} (×{adjustment_factor})")
    
    def reset_tracking(self):
        """Reset tracking"""
        self.plateau_counter = 0
        self.accuracy_history = []
        self.current_accuracy = 0

class TreeBindingManager:
    """Tree Binding Manager with accuracy-based LR adjustment and backtracking"""
    def __init__(self, base_model, config=TREE_CONFIG):
        self.config = config
        self.base_model = base_model
        self.population = []
        self.generation = 0
        self.best_accuracy = 0
        self.best_model = None
        self.next_id = 0
        
        # Backtracking history
        self.generation_history = []
        self.best_population_history = []
        self.backtrack_count = 0
        
    def initialize_population(self):
        """Initialize population"""
        print(f"\nInitializing Tree Binding with {self.config['num_models']} models...")
        
        self.population = []
        self.next_id = 0
        
        for i in range(self.config['num_models']):
            model = copy.deepcopy(self.base_model)
            
            if i > 0:
                model_state = model.state_dict()
                for key in model_state:
                    if 'weight' in key or 'bias' in key:
                        model_state[key] += torch.randn_like(model_state[key]) * 0.01
                model.load_state_dict(model_state)
            
            lr_configs = [
                [1.0, 0.8, 0.6, 0.4],
                [0.4, 0.6, 0.8, 1.0],
                [1.0, 0.5, 0.25, 0.125],
                [0.125, 0.25, 0.5, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.5, 1.0, 0.5, 1.0],
            ]
            
            lr_ratios = random.choice(lr_configs)
            base_lr = random.uniform(self.config['base_lr_range'][0], 
                                   self.config['base_lr_range'][1])
            
            optimizer = AccuracyBasedLROptimizer(model, base_lr=base_lr, 
                                                lr_ratios=lr_ratios, config=self.config)
            
            self.population.append({
                'model': model,
                'optimizer': optimizer,
                'accuracy': 0.0,
                'id': self.next_id,
                'base_lr': base_lr,
                'lr_ratios': lr_ratios,
                'origin': 'initial',
                'generation': 0,
            })
            self.next_id += 1
        
        print(f"✓ Created {len(self.population)} initial models")
    
    def train_population_fast(self, train_loader, criterion):
        """Train all models"""
        print(f"\n[Tree Binding] Training Generation {self.generation + 1}...")
        
        batches_to_train = 200
        
        for idx, model_info in enumerate(self.population):
            model = model_info['model']
            optimizer = model_info['optimizer']
            
            model.train()
            batch_iter = iter(train_loader)
            
            for batch_idx in range(batches_to_train):
                try:
                    images, labels = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(train_loader)
                    images, labels = next(batch_iter)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        print(f"  ✓ Trained all models")
    
    def evaluate_population(self, val_loader, criterion):
        """Evaluate all models and adjust LRs based on accuracy"""
        print("\n[Tree Binding] Evaluating models...")
        
        for model_info in self.population:
            model = model_info['model']
            optimizer = model_info['optimizer']
            
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
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
            
            # Adjust LR based on accuracy
            optimizer.adjust_lr_based_on_accuracy(accuracy)
        
        # Sort by accuracy
        self.population.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Update generation history
        current_best = self.population[0]['accuracy']
        self.generation_history.append({
            'generation': self.generation,
            'best_accuracy': current_best,
            'avg_accuracy': sum(m['accuracy'] for m in self.population) / len(self.population),
        })
        
        # Store top models for backtracking
        if len(self.best_population_history) < self.config['backtrack_generations']:
            self.best_population_history.append(copy.deepcopy(self.population[:self.config['keep_top']]))
        else:
            self.best_population_history.pop(0)
            self.best_population_history.append(copy.deepcopy(self.population[:self.config['keep_top']]))
        
        # Update best model
        if current_best > self.best_accuracy:
            self.best_accuracy = current_best
            self.best_model = copy.deepcopy(self.population[0]['model'])
            print(f"  🎯 New Tree Binding Best: {current_best:.2f}%")
        
        # Show results
        print(f"\n  Top {min(3, len(self.population))} models:")
        for i in range(min(3, len(self.population))):
            model_info = self.population[i]
            print(f"    #{i+1}: Model {model_info['id']} = {model_info['accuracy']:.2f}% | LR: {model_info['base_lr']:.6f}")
        
        return current_best
    
    def evolve_population(self):
        """Evolve population with backtracking"""
        print(f"\n[Tree Binding] Evolving to Generation {self.generation + 2}...")
        
        # Check if we should backtrack
        should_backtrack = False
        if (self.config['backtracking_enabled'] and 
            len(self.generation_history) >= 2 and
            len(self.best_population_history) >= 1):
            
            current_best = self.population[0]['accuracy']
            
            # Find best historical accuracy
            best_historical_accuracy = 0
            for gen_data in self.generation_history[-self.config['backtrack_generations']:]:
                if gen_data['best_accuracy'] > best_historical_accuracy:
                    best_historical_accuracy = gen_data['best_accuracy']
            
            # Check if current generation is worse
            if current_best < best_historical_accuracy - self.config['backtracking_threshold']:
                should_backtrack = True
                print(f"  ⚠️ Performance dropped: {current_best:.2f}% < {best_historical_accuracy:.2f}%")
        
        # Choose evolution strategy
        if should_backtrack:
            return self._backtrack_evolution()
        else:
            return self._normal_evolution()
    
    def _normal_evolution(self):
        """Normal evolution"""
        print(f"  Strategy: Normal evolution")
        
        top_models = self.population[:self.config['keep_top']]
        new_population = []
        
        # Keep top models
        for top in top_models:
            model_copy = copy.deepcopy(top['model'])
            optimizer = AccuracyBasedLROptimizer(
                model_copy, 
                base_lr=top['base_lr'], 
                lr_ratios=top['lr_ratios'], 
                config=self.config
            )
            
            new_population.append({
                'model': model_copy,
                'optimizer': optimizer,
                'accuracy': top['accuracy'],
                'id': self.next_id,
                'base_lr': top['base_lr'],
                'lr_ratios': top['lr_ratios'],
                'origin': f'elite_gen_{self.generation + 1}',
                'generation': self.generation + 1,
            })
            self.next_id += 1
        
        # Generate new models
        while len(new_population) < self.config['num_models']:
            parent = random.choice(top_models)
            child = self._create_mutated_child(parent)
            
            new_population.append({
                'model': child['model'],
                'optimizer': child['optimizer'],
                'accuracy': 0.0,
                'id': self.next_id,
                'base_lr': child['base_lr'],
                'lr_ratios': child['lr_ratios'],
                'origin': f'mutated_gen_{self.generation + 1}',
                'generation': self.generation + 1,
            })
            self.next_id += 1
        
        self.population = new_population
        self.generation += 1
        
        print(f"  ✓ Evolved: Kept top {len(top_models)}, created {len(new_population)-len(top_models)} new")
        return True
    
    def _backtrack_evolution(self):
        """Backtrack to previous best generation and apply low LR"""
        print(f"  Strategy: BACKTRACKING evolution")
        
        self.backtrack_count += 1
        
        if not self.best_population_history:
            print("  No history to backtrack to, using normal evolution")
            return self._normal_evolution()
        
        # Use models from previous best generation
        historical_top_models = self.best_population_history[-1]
        new_population = []
        
        # Restore models with backtracking LR reduction
        for historical_model in historical_top_models:
            model_copy = copy.deepcopy(historical_model['model'])
            
            # Apply backtracking LR reduction
            backtrack_base_lr = max(
                historical_model['base_lr'] * self.config['backtracking_lr_factor'],
                self.config['min_lr']
            )
            
            optimizer = AccuracyBasedLROptimizer(
                model_copy, 
                base_lr=backtrack_base_lr, 
                lr_ratios=historical_model['lr_ratios'], 
                config=self.config
            )
            
            # Apply backtracking LR adjustment
            optimizer.apply_backtracking_lr()
            
            new_population.append({
                'model': model_copy,
                'optimizer': optimizer,
                'accuracy': historical_model['accuracy'],
                'id': self.next_id,
                'base_lr': backtrack_base_lr,
                'lr_ratios': historical_model['lr_ratios'],
                'origin': f'backtrack_elite_gen_{self.generation + 1}',
                'generation': self.generation + 1,
            })
            self.next_id += 1
        
        # Create careful mutations
        while len(new_population) < self.config['num_models']:
            parent = random.choice(historical_top_models)
            child = self._create_mutated_child(parent, careful=True)
            
            # Apply even lower LR for careful mutations
            child_base_lr = max(
                child['base_lr'] * 0.5,
                self.config['min_lr']
            )
            child['optimizer'].base_lr = child_base_lr
            
            new_population.append({
                'model': child['model'],
                'optimizer': child['optimizer'],
                'accuracy': 0.0,
                'id': self.next_id,
                'base_lr': child_base_lr,
                'lr_ratios': child['lr_ratios'],
                'origin': f'backtrack_careful_gen_{self.generation + 1}',
                'generation': self.generation + 1,
            })
            self.next_id += 1
        
        self.population = new_population
        self.generation += 1
        
        print(f"  ✓ Backtracking: Restored from previous best generation")
        print(f"  ↳ Applied low LR ({self.config['backtracking_lr_factor']}x) for careful recovery")
        return True
    
    def _create_mutated_child(self, parent, careful=False):
        """Create a mutated child from parent"""
        child = copy.deepcopy(parent['model'])
        
        # Apply mutation
        mutation_rate = self.config['normal_mutation_rate'] if not careful else 0.15
        mutation_strength = self.config['normal_mutation_strength'] if not careful else 0.05
        
        with torch.no_grad():
            child_state = child.state_dict()
            for key in child_state:
                if 'weight' in key or 'bias' in key:
                    mask = torch.rand_like(child_state[key]) < mutation_rate
                    if mask.any():
                        noise = torch.randn_like(child_state[key]) * mutation_strength
                        child_state[key] = child_state[key] + noise * mask.float()
            child.load_state_dict(child_state)
        
        # Mutate LR configuration
        lr_ratios = parent['lr_ratios'].copy()
        if random.random() < 0.4:
            lr_patterns = [
                [1.0, 0.8, 0.6, 0.4],
                [0.4, 0.6, 0.8, 1.0],
                [1.0, 0.5, 0.25, 0.125],
                [0.125, 0.25, 0.5, 1.0],
            ]
            lr_ratios = random.choice(lr_patterns)
        
        # Mutate base LR
        base_lr = parent['base_lr'] * (0.7 + 0.6 * random.random())
        base_lr = max(self.config['min_lr'], min(self.config['max_lr'], base_lr))
        
        optimizer = AccuracyBasedLROptimizer(child, base_lr=base_lr, 
                                            lr_ratios=lr_ratios, config=self.config)
        
        return {
            'model': child,
            'optimizer': optimizer,
            'base_lr': base_lr,
            'lr_ratios': lr_ratios,
        }
    
    def get_best_model(self):
        return self.population[0]['model'] if self.population else None

# ======================
# DATA LOADING AND NETWORK
# ======================
print("Loading MNIST dataset...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=128, shuffle=True, num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=128, shuffle=False, num_workers=0
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=128, shuffle=False, num_workers=0
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

class SimpleNN(nn.Module):
    def __init__(self, layer_config=None):
        super(SimpleNN, self).__init__()
        default_config = { 
            'layer_sizes': [784, 128, 128, 128, 10],
            'activations': ['relu', 'relu', 'relu', None]
        }
        config = layer_config or default_config
        layer_sizes = config['layer_sizes']
        activations = config['activations']
        
        layers = [nn.Flatten()]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if activations[i]:
                if activations[i].lower() == 'relu':
                    layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)
        self.config = config
    
    def forward(self, x):
        return self.model(x)

# ======================
# MAIN TRAINING
# ======================
def main():
    print("\n" + "="*60)
    print("TREE BINDING WITH ACCURACY-BASED LR ADJUSTMENT")
    print("="*60)
    
    model = SimpleNN()
    checkpoint_manager = CheckpointManager()
    
    # Create optimizer for main model
    optimizer = AccuracyBasedLROptimizer(
        model, 
        base_lr=0.001,
        lr_ratios=[1.0, 0.8, 0.6, 0.4],
        config=TREE_CONFIG
    )
    
    tree_binding = TreeBindingManager(model)
    
    # Load checkpoint
    previous_checkpoint = checkpoint_manager.load_checkpoint(model, optimizer, ignore_optimizer=True)
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
    print(f"  Keep top: {TREE_CONFIG['keep_top']} models")
    print(f"  Backtracking: {'ENABLED' if TREE_CONFIG['backtracking_enabled'] else 'DISABLED'}")
    print(f"  LR Strategy: High acc→×{TREE_CONFIG['high_acc_lr_factor']}, Low acc→×{TREE_CONFIG['low_acc_lr_factor']}")
    
    print(f"\nNetwork Architecture: {model.config['layer_sizes']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    print("\nStarting training...")
    
    num_epochs = 200
    early_stop_patience = 5
    patience_counter = 0
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{start_epoch + num_epochs}")
        print(f"{'='*60}")
        
        # PHASE 1: Train main model
        print("\n[Phase 1] Training main model...")
        model.train()
        total_loss = 0
        batch_count = 0
        
        batches_to_train = 200
        data_iter = iter(train_loader)
        
        for batch_idx in range(batches_to_train):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        train_losses.append(avg_loss)
        
        # Validate main model
        def validate_model(model, data_loader, num_batches=100):
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
            return 100 * correct / total if total > 0 else 0
        
        val_accuracy = validate_model(model, val_loader, num_batches=100)
        optimizer.adjust_lr_based_on_accuracy(val_accuracy)
        
        print(f"\n  Main model results:")
        print(f"    Loss: {avg_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        print(f"    Base LR: {optimizer.base_lr:.6f}")
        
        # PHASE 2: Tree Binding
        print(f"\n[Phase 2] Tree Binding (Generation {tree_binding.generation + 1})...")
        tree_binding.train_population_fast(train_loader, criterion)
        tree_accuracy = tree_binding.evaluate_population(val_loader, criterion)
        
        print(f"\n  Tree binding results:")
        print(f"    Best accuracy: {tree_accuracy:.2f}%")
        
        # PHASE 3: Model comparison
        print(f"\n[Phase 3] Model comparison...")
        if tree_accuracy > val_accuracy:
            print(f"  ✓ Tree binding is better! ({tree_accuracy:.2f}% vs {val_accuracy:.2f}%)")
            print(f"  ↳ Updating main model with tree binding best")
            model.load_state_dict(tree_binding.get_best_model().state_dict())
            val_accuracy = tree_accuracy
            optimizer.reset_tracking()
        else:
            print(f"  ✓ Main model is better ({val_accuracy:.2f}% vs {tree_accuracy:.2f}%)")
        
        # Check for best model
        is_best = val_accuracy > best_accuracy
        if is_best:
            best_accuracy = val_accuracy
            patience_counter = 0
            print(f"  🎯 NEW OVERALL BEST: {best_accuracy:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stop_patience})")
        
        # PHASE 4: Evolve
        print(f"\n[Phase 4] Evolving tree binding...")
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
        
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        print(f"\n  Epoch {epoch + 1} completed in {epoch_time:.1f} seconds")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️  Early stopping triggered!")
            print(f"Best validation accuracy: {best_accuracy:.2f}%")
            break
    
    # Testing
    print("\nTesting model...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
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
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Main Model Accuracy: {accuracy:.2f}%")
    print(f"Tree Binding Best: {tree_binding.best_accuracy:.2f}%")
    print(f"Backtrack Events: {tree_binding.backtrack_count}")
    print("="*60)

if __name__ == "__main__":
    main()