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
CHECKPOINT_DIR = "./checkpoints_TreebindALR"
TREE_BINDING_DIR = "./tree_binding_dataALR"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TREE_BINDING_DIR, exist_ok=True)

# Tree Binding Configuration
TREE_CONFIG = {
    'num_models': 12,           # Start with 6 models
    'keep_top': 3,              # Keep best 2 after each epoch
    'mutation_rate': 0.3,
    'mutation_strength': 0.1,
    'lr_adjust_factor': 0.8,    # Factor to adjust LR based on performance
    'lr_adjust_threshold': 1.0, # Minimum accuracy improvement to avoid LR reduction
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

class AdaptiveLROptimizer:
    """Optimizer with different learning rates for each layer that adapts based on performance"""
    def __init__(self, model, base_lr=0.001, lr_ratios=None, config=TREE_CONFIG):
        """
        Args:
            model: The neural network model
            base_lr: Base learning rate
            lr_ratios: List of ratios for each layer
            config: Configuration dictionary with LR adjustment parameters
        """
        self.model = model
        self.config = config
        
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
        self.param_groups_list = []
        
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
                
                self.param_groups_list.append({
                    'params': param,
                    'lr': base_lr * lr_ratios[layer_idx],
                    'name': name,
                    'layer': layer_idx,
                    'base_ratio': lr_ratios[layer_idx],
                    'original_lr': base_lr * lr_ratios[layer_idx]
                })
        
        # Create optimizer with these parameter groups
        self.optimizer = optim.Adam(self.param_groups_list)
        self.base_lr = base_lr
        self.lr_ratios = lr_ratios
        
        # Performance tracking
        self.previous_accuracy = 0
        self.accuracy_trend = []  # Track last 3 accuracies
        self.consecutive_decreases = 0
        self.consecutive_increases = 0
        self.best_accuracy = 0
        
        # Print layer-wise learning rates
        print(f"\nLayer-wise learning rates:")
        for i, ratio in enumerate(lr_ratios):
            print(f"  Layer {i}: {base_lr * ratio:.6f} (ratio: {ratio:.2f})")
    
    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self):
        self.optimizer.step()
    
    def state_dict(self):
        return {
            'optimizer_state': self.optimizer.state_dict(),
            'base_lr': self.base_lr,
            'lr_ratios': self.lr_ratios,
            'previous_accuracy': self.previous_accuracy,
            'best_accuracy': self.best_accuracy,
            'accuracy_trend': self.accuracy_trend
        }
    
    def load_state_dict(self, state_dict):
        if 'optimizer_state' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer_state'])
        if 'base_lr' in state_dict:
            self.base_lr = state_dict['base_lr']
        if 'lr_ratios' in state_dict:
            self.lr_ratios = state_dict['lr_ratios']
        if 'previous_accuracy' in state_dict:
            self.previous_accuracy = state_dict['previous_accuracy']
        if 'best_accuracy' in state_dict:
            self.best_accuracy = state_dict['best_accuracy']
        if 'accuracy_trend' in state_dict:
            self.accuracy_trend = state_dict['accuracy_trend']
    
    def get_lr_info(self):
        """Get current learning rates for each layer"""
        lr_info = {}
        for group in self.param_groups_list:
            lr_info[group['name']] = group['lr']
        return lr_info
    
    def update_lr_based_on_performance(self, current_accuracy, verbose=True):
        """
        Adjust learning rates based on performance trend
        Returns: whether LR was adjusted and the adjustment factor
        """
        # Update tracking
        self.accuracy_trend.append(current_accuracy)
        if len(self.accuracy_trend) > 3:
            self.accuracy_trend.pop(0)
        
        # Check if accuracy improved
        accuracy_change = current_accuracy - self.previous_accuracy
        self.previous_accuracy = current_accuracy
        
        # Update best accuracy
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.consecutive_decreases = 0
            self.consecutive_increases += 1
        else:
            self.consecutive_increases = 0
            self.consecutive_decreases += 1
        
        # Determine LR adjustment
        lr_adjusted = False
        adjustment_factor = 1.0
        adjustment_type = "no_change"
        
        # Logic for LR adjustment based on performance
        if accuracy_change < -self.config['lr_adjust_threshold']:
            # Significant decrease in accuracy - reduce LR
            adjustment_factor = self.config['lr_adjust_factor']
            adjustment_type = "decrease"
            lr_adjusted = True
            
            if verbose:
                print(f"  ⚠️ Accuracy decreased by {-accuracy_change:.2f}%, reducing LR by factor {adjustment_factor}")
                
        elif self.consecutive_decreases >= 2:
            # Multiple consecutive decreases - reduce LR
            adjustment_factor = self.config['lr_adjust_factor']
            adjustment_type = "consecutive_decrease"
            lr_adjusted = True
            
            if verbose:
                print(f"  ⚠️ {self.consecutive_decreases} consecutive decreases, reducing LR by factor {adjustment_factor}")
                
        elif accuracy_change > self.config['lr_adjust_threshold'] and self.consecutive_increases >= 2:
            # Good improvement trend - slightly increase LR
            adjustment_factor = 1.0 / self.config['lr_adjust_factor']
            adjustment_type = "increase"
            lr_adjusted = True
            
            if verbose:
                print(f"  ↗️ {self.consecutive_increases} consecutive increases, increasing LR by factor {adjustment_factor:.2f}")
        
        # Apply LR adjustment if needed
        if lr_adjusted:
            old_lrs = []
            new_lrs = []
            
            for group in self.param_groups_list:
                old_lrs.append(group['lr'])
                group['lr'] = max(group['lr'] * adjustment_factor, 1e-7)  # Don't go below minimum
                new_lrs.append(group['lr'])
            
            # Update base LR for consistency
            self.base_lr = max(self.base_lr * adjustment_factor, 1e-7)
            
            if verbose:
                print(f"  ↳ LR adjustment applied: {adjustment_type}")
                print(f"    Old LRs: {[f'{lr:.6f}' for lr in old_lrs[:3]]}...")
                print(f"    New LRs: {[f'{lr:.6f}' for lr in new_lrs[:3]]}...")
        
        return lr_adjusted, adjustment_factor, adjustment_type
    
    def adaptive_step(self, loss):
        """Perform optimization step with adaptive gradient clipping based on LR"""
        # Clip gradients based on current learning rates
        max_grad_norm = 1.0 / (self.base_lr * 1000)  # Adaptive clipping
        
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            max_norm=max_grad_norm,
            norm_type=2
        )
        
        self.optimizer.step()
    
    def reset_tracking(self):
        """Reset performance tracking (useful after major changes)"""
        self.consecutive_decreases = 0
        self.consecutive_increases = 0
        self.accuracy_trend = []

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
        
        # Save optimizer state if it has state_dict method
        if hasattr(optimizer, 'state_dict'):
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Always save current checkpoint
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
        
        # Save as best if accuracy improved
        if is_best:
            torch.save(checkpoint, self.checkpoint_path)
            self.best_accuracy = val_accuracy
            print(f"✓ New best model saved with accuracy: {val_accuracy:.2f}%")
    
    def load_checkpoint(self, model, optimizer=None, ignore_optimizer=False):
        """Load the best checkpoint, handling different optimizer types"""
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint and not ignore_optimizer:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"✓ Loaded optimizer state from checkpoint")
                except Exception as e:
                    print(f"⚠️ Could not load optimizer state: {e}")
                    print(f"  Starting with fresh optimizer")
            
            print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
            print(f"  Previous accuracy: {checkpoint['val_accuracy']:.2f}%")
            print(f"  Saved on: {checkpoint['timestamp']}")
            
            return checkpoint
        return None

# ======================
# TREE BINDING MANAGER WITH ADAPTIVE LRS
# ======================
class TreeBindingManager:
    """Manages tree binding evolution with adaptive learning rates"""
    def __init__(self, base_model, config=TREE_CONFIG):
        self.config = config
        self.base_model = base_model
        self.population = []
        self.generation = 0
        self.best_accuracy = 0
        self.best_model = None
        self.next_id = 0
        self.lr_adjustment_history = []
        
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
            
            optimizer = AdaptiveLROptimizer(model, base_lr=base_lr, lr_ratios=lr_ratios, config=self.config)
            
            self.population.append({
                'model': model,
                'optimizer': optimizer,
                'accuracy': 0.0,
                'previous_accuracy': 0.0,
                'id': self.next_id,
                'base_lr': base_lr,
                'lr_ratios': lr_ratios,
                'origin': 'initial',
                'lr_adjustments': 0,
                'performance_trend': []
            })
            self.next_id += 1
        
        print(f"✓ Created {len(self.population)} initial models with adaptive LR configurations")
    
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
                optimizer.adaptive_step(loss)  # Use adaptive step
                
                # Minimal progress output (only first model, every 50 batches)
                if idx == 0 and batch_idx % 50 == 0:
                    current_lrs = list(optimizer.get_lr_info().values())
                    avg_lr = sum(current_lrs) / len(current_lrs)
                    print(f'  Model 1, Batch [{batch_idx}/{batches_to_train}], Loss: {loss.item():.4f}, Avg LR: {avg_lr:.6f}')
        
        print(f"  ✓ Trained all {len(self.population)} models on {batches_to_train} batches")
    
    def evaluate_population(self, val_loader, criterion):
        """Evaluate all models and adjust LRs based on performance"""
        print("\n[Tree Binding] Evaluating models and adjusting LRs...")
        
        performance_summary = []
        
        for model_info in self.population:
            model = model_info['model']
            optimizer = model_info['optimizer']
            
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
            
            # Adjust learning rate based on performance
            previous_accuracy = model_info['previous_accuracy']
            model_info['previous_accuracy'] = accuracy
            model_info['accuracy'] = accuracy
            
            # Update performance trend
            model_info['performance_trend'].append(accuracy)
            if len(model_info['performance_trend']) > 5:
                model_info['performance_trend'].pop(0)
            
            # Perform LR adjustment based on performance
            lr_adjusted, adjustment_factor, adjustment_type = optimizer.update_lr_based_on_performance(
                accuracy, 
                verbose=(model_info['id'] < 2)  # Verbose for first 2 models
            )
            
            if lr_adjusted:
                model_info['lr_adjustments'] += 1
                model_info['base_lr'] = optimizer.base_lr
                
                # Record adjustment
                self.lr_adjustment_history.append({
                    'generation': self.generation,
                    'model_id': model_info['id'],
                    'adjustment_type': adjustment_type,
                    'factor': adjustment_factor,
                    'accuracy': accuracy,
                    'previous_accuracy': previous_accuracy
                })
            
            # Store summary
            performance_summary.append({
                'id': model_info['id'],
                'accuracy': accuracy,
                'base_lr': optimizer.base_lr,
                'lr_adjustments': model_info['lr_adjustments'],
                'origin': model_info['origin']
            })
        
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
            lr_str = f"LR: {model_info['base_lr']:.5f} (adj: {model_info['lr_adjustments']})"
            print(f"    #{i+1}: Model {model_info['id']} ({model_info['origin']}) = {model_info['accuracy']:.2f}% | {lr_str}")
        
        # Show LR adjustment statistics
        recent_adjustments = [adj for adj in self.lr_adjustment_history 
                             if adj['generation'] == self.generation]
        if recent_adjustments:
            decrease_count = sum(1 for adj in recent_adjustments 
                               if adj['adjustment_type'] in ['decrease', 'consecutive_decrease'])
            increase_count = sum(1 for adj in recent_adjustments 
                               if adj['adjustment_type'] == 'increase')
            print(f"  LR adjustments this gen: {len(recent_adjustments)} total ({increase_count}↑, {decrease_count}↓)")
        
        # Show worst model
        if len(self.population) > 0:
            worst_model = self.population[-1]
            print(f"  Worst: Model {worst_model['id']} = {worst_model['accuracy']:.2f}% (LR adj: {worst_model['lr_adjustments']})")
        
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
        
        # Create adaptive optimizer for the main model copy
        main_optimizer = AdaptiveLROptimizer(main_model_copy, base_lr=base_lr, 
                                            lr_ratios=lr_ratios, config=self.config)
        
        # Replace worst model with main model
        removed_model = self.population[worst_idx]
        print(f"  Replacing Model {removed_model['id']} ({removed_model['origin']}) = {removed_model['accuracy']:.2f}%")
        print(f"  with Main Model (accuracy: {main_accuracy:.2f}%)")
        
        self.population[worst_idx] = {
            'model': main_model_copy,
            'optimizer': main_optimizer,
            'accuracy': main_accuracy,
            'previous_accuracy': main_accuracy,
            'id': self.next_id,
            'base_lr': base_lr,
            'lr_ratios': lr_ratios,
            'origin': 'main_model_replacement',
            'lr_adjustments': 0,
            'performance_trend': [main_accuracy]
        }
        self.next_id += 1
        
        # Re-sort population
        self.population.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"  ✓ Main model added to population at position {worst_idx + 1}")
        return True
    
    def evolve_population(self):
        """Evolve: keep top 2, generate new ones with inherited LR knowledge"""
        print(f"\n[Tree Binding] Evolving to Generation {self.generation + 2}...")
        
        # Keep top 2
        top_models = self.population[:self.config['keep_top']]
        
        # New population
        new_population = []
        
        # Add top 2 (elite) - they keep their adaptive LR optimizers
        for top in top_models:
            # Create a fresh adaptive optimizer with the same configuration
            model_copy = copy.deepcopy(top['model'])
            new_optimizer = AdaptiveLROptimizer(
                model_copy, 
                base_lr=top['base_lr'], 
                lr_ratios=top['lr_ratios'], 
                config=self.config
            )
            
            # Carry over some performance knowledge
            if top['performance_trend']:
                new_optimizer.previous_accuracy = top['performance_trend'][-1]
                new_optimizer.best_accuracy = max(top['performance_trend'])
            
            new_population.append({
                'model': model_copy,
                'optimizer': new_optimizer,
                'accuracy': top['accuracy'],
                'previous_accuracy': top['accuracy'],
                'id': self.next_id,
                'base_lr': top['base_lr'],
                'lr_ratios': top['lr_ratios'],
                'origin': f'elite_gen_{self.generation + 1}',
                'lr_adjustments': top['lr_adjustments'],
                'performance_trend': top['performance_trend'][-3:] if top['performance_trend'] else []
            })
            self.next_id += 1
        
        # Create new models from top 2 with adaptive LR inheritance
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
            
            # Inherit and mutate learning rate configuration
            lr_ratios = parent['lr_ratios'].copy()
            
            # Inherit LR adjustment behavior from parent
            base_lr = parent['base_lr']
            
            # With 40% probability, significantly change the LR pattern
            if random.random() < 0.4:
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
                
                # Also change base LR more significantly
                base_lr = parent['base_lr'] * (0.7 + 0.6 * random.random())
            else:
                # Slightly mutate existing ratios
                for i in range(len(lr_ratios)):
                    if random.random() < 0.3:  # 30% chance to mutate each ratio
                        lr_ratios[i] = max(0.1, min(2.0, lr_ratios[i] * (0.8 + 0.4 * random.random())))
                
                # Slightly different base learning rate
                base_lr = parent['base_lr'] * (0.9 + 0.2 * random.random())
            
            # Create adaptive optimizer for child
            child_optimizer = AdaptiveLROptimizer(child, base_lr=base_lr, 
                                                 lr_ratios=lr_ratios, config=self.config)
            
            # Inherit some performance knowledge if parent has it
            if parent['performance_trend']:
                child_optimizer.previous_accuracy = parent['performance_trend'][-1]
            
            new_population.append({
                'model': child,
                'optimizer': child_optimizer,
                'accuracy': 0.0,
                'previous_accuracy': 0.0,
                'id': self.next_id,
                'base_lr': base_lr,
                'lr_ratios': lr_ratios,
                'origin': f'mutated_gen_{self.generation + 1}',
                'lr_adjustments': 0,
                'performance_trend': []
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
        lr_adjustments = [m['lr_adjustments'] for m in self.population]
        base_lrs = [m['base_lr'] for m in self.population]
        origins = [m.get('origin', 'unknown') for m in self.population]
        
        return {
            'size': len(self.population),
            'avg_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
            'best_accuracy': max(accuracies) if accuracies else 0,
            'worst_accuracy': min(accuracies) if accuracies else 0,
            'avg_lr_adjustments': sum(lr_adjustments) / len(lr_adjustments) if lr_adjustments else 0,
            'avg_base_lr': sum(base_lrs) / len(base_lrs) if base_lrs else 0,
            'origins': origins,
            'total_lr_adjustments': len(self.lr_adjustment_history)
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
# 2. DEFINE NEURAL NETWORK
# ======================
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
    print("TREE BINDING TRAINING WITH ADAPTIVE LEARNING RATES")
    print("="*60)

    model = SimpleNN()
    checkpoint_manager = CheckpointManager()
    
    # Create adaptive optimizer for main model
    optimizer = AdaptiveLROptimizer(model, base_lr=0.001, 
                                   lr_ratios=[1.0, 0.8, 0.6, 0.4], 
                                   config=TREE_CONFIG)
    
    tree_binding = TreeBindingManager(model)

    # Try to load previous best model (ignore optimizer state for now)
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
    print(f"  Keep top: {TREE_CONFIG['keep_top']} models each generation")
    print(f"  Mutation rate: {TREE_CONFIG['mutation_rate']}")
    print(f"  LR adjust factor: {TREE_CONFIG['lr_adjust_factor']}")
    print(f"  LR adjust threshold: {TREE_CONFIG['lr_adjust_threshold']}%")
    print(f"  Process: Train {TREE_CONFIG['num_models']} → Evaluate → Keep {TREE_CONFIG['keep_top']} → Generate {TREE_CONFIG['num_models']}")
    print(f"  NEW: Adaptive LRs that adjust based on performance")
    
    print(f"\nNetwork Architecture: {model.config['layer_sizes']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ======================
    # 4. SETUP TRAINING
    # ======================
    criterion = nn.CrossEntropyLoss()
    
    # Add learning rate scheduler for main model (still works with AdaptiveLROptimizer)
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

    print("\nStarting training with Adaptive Tree Binding...")
    num_epochs = 10
    early_stop_patience = 5
    patience_counter = 0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{start_epoch + num_epochs}")
        print(f"{'='*60}")
        
        # ====================================
        # PHASE 1: REGULAR TRAINING WITH ADAPTIVE LR
        # ====================================
        print("\n[Phase 1] Training main model with adaptive learning rates...")
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
            
            # Backward pass and optimize with adaptive gradient clipping
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.adaptive_step(loss)
            
            total_loss += loss.item()
            batch_count += 1
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                current_lrs = list(optimizer.get_lr_info().values())
                avg_lr = sum(current_lrs) / len(current_lrs)
                print(f'  Batch [{batch_idx}/{batches_to_train}], Loss: {loss.item():.4f}, Avg LR: {avg_lr:.6f}')
        
        # Calculate average loss for this epoch
        avg_loss = total_loss / batch_count
        train_losses.append(avg_loss)
        
        # Validate model - faster validation
        val_accuracy = validate_model(model, val_loader, num_batches=100)
        
        # Update main model's learning rate based on performance
        lr_adjusted, adjustment_factor, adjustment_type = optimizer.update_lr_based_on_performance(
            val_accuracy, 
            verbose=True
        )
        
        # Also use scheduler for additional LR adjustment
        scheduler.step(val_accuracy)
        
        # Print current layer-wise learning rates
        lr_info = optimizer.get_lr_info()
        print(f"\n  Main model results:")
        print(f"    Loss: {avg_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        print(f"    Base LR: {optimizer.base_lr:.6f}, LR adjustments: {optimizer.consecutive_increases}↑/{optimizer.consecutive_decreases}↓")
        
        # ====================================
        # PHASE 2: TREE BINDING EVOLUTION WITH ADAPTIVE LRS
        # ====================================
        print(f"\n[Phase 2] Tree Binding (Generation {tree_binding.generation + 1})...")
        
        # Train tree binding population
        tree_binding.train_population_fast(train_loader, criterion)
        
        # Evaluate tree binding population (includes LR adjustment)
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
            
            # Reset main model's LR tracking since we got new weights
            optimizer.reset_tracking()
        else:
            print(f"  ✓ Main model is better ({val_accuracy:.2f}% vs {tree_accuracy:.2f}%)")
            print(f"  ↳ Adding main model to tree binding population...")
            
            # Add main model to tree binding, replacing worst performing model
            success = tree_binding.add_main_model(
                main_model=model,
                main_accuracy=val_accuracy,
                optimizer=optimizer,
                base_lr=optimizer.base_lr,
                lr_ratios=optimizer.lr_ratios
            )
            
            if success:
                print(f"  ✓ Main model integrated into tree binding population")
        
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
            f"Tree Binding Stats:\n"
            f"  Generations: {tree_binding.generation + 1}\n"
            f"  Population: {stats['size']}\n"
            f"  Avg Accuracy: {stats['avg_accuracy']:.2f}%\n"
            f"  Avg Base LR: {stats['avg_base_lr']:.6f}\n"
            f"  LR Adjustments: {stats['total_lr_adjustments']}\n\n"
            f"Adaptive LR System:\n"
            f"  Adjusts LRs based on\n"
            f"  performance trends",
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

    plt.suptitle(f'Tree Binding with Adaptive LRs - Test: {accuracy:.2f}%', fontsize=16)
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
        'optimizer_base_lr': optimizer.base_lr,
        'population_stats': tree_binding.get_population_stats(),
        'lr_adjustment_history': tree_binding.lr_adjustment_history,
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
            'main_model_base_lr': optimizer.base_lr,
            'population_stats': tree_binding.get_population_stats(),
            'lr_adjustment_history': tree_binding.lr_adjustment_history[:20],  # First 20 adjustments
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
    print(f"Main Model Base LR: {optimizer.base_lr:.6f}")
    print(f"Total LR Adjustments: {len(tree_binding.lr_adjustment_history)}")
    
    # Show population statistics
    stats = tree_binding.get_population_stats()
    print(f"\nFinal Population Statistics:")
    print(f"  Size: {stats['size']} models")
    print(f"  Average Accuracy: {stats['avg_accuracy']:.2f}%")
    print(f"  Best Accuracy: {stats['best_accuracy']:.2f}%")
    print(f"  Worst Accuracy: {stats['worst_accuracy']:.2f}%")
    print(f"  Average Base LR: {stats['avg_base_lr']:.6f}")
    print(f"  Average LR Adjustments per model: {stats['avg_lr_adjustments']:.1f}")
    
    print("="*60)

if __name__ == "__main__":
    main()