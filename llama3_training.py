"""
Llama 3 Pre-training Pipeline
AdamW, Cosine Schedule, Mixed Precision, Gradient Accumulation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
from pathlib import Path
import json
from tqdm import tqdm
from tokenizers import Tokenizer


class TextDataset(Dataset):
    """Pre-training text dataset"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read text
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize
        encoded = tokenizer.encode(text)
        self.input_ids = encoded.ids
        
        print(f"Dataset loaded: {len(self.input_ids):,} token")
    
    def __len__(self):
        return len(self.input_ids) // self.max_length
    
    def __getitem__(self, idx):
        start = idx * self.max_length
        end = start + self.max_length
        
        input_ids = self.input_ids[start:end]
        
        # Padding
        if len(input_ids) < self.max_length:
            pad_token = self.tokenizer.token_to_id("<pad>")
            input_ids = input_ids + [pad_token] * (self.max_length - len(input_ids))
        
        return torch.tensor(input_ids, dtype=torch.long)


class CosineScheduleWithWarmup:
    """Cosine learning rate schedule with warmup"""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        
        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class Trainer:
    """Llama 3 Pre-training Trainer"""
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        
        # Resolve tokenizer (handle Subset wrapping)
        ds = train_dataset
        while hasattr(ds, "dataset"):
            ds = ds.dataset
        self.tokenizer = ds.tokenizer
        
        # Optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            betas=(config['beta1'], config['beta2']),
            weight_decay=config['weight_decay'],
            eps=1e-8
        )
        
        # Learning rate scheduler
        total_steps = len(train_dataset) // config['batch_size'] * config['num_epochs']
        total_steps = total_steps // config['gradient_accumulation_steps']
        warmup_steps = int(total_steps * config['warmup_ratio'])
        
        self.scheduler = CosineScheduleWithWarmup(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=config['min_learning_rate']
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config['use_amp'] else None
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        print(f"\nTrainer ready:")
        print(f"  Device: {device}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Warmup steps: {warmup_steps:,}")
        print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
        print(f"  Mixed precision: {config['use_amp']}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, input_ids in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            
            # Forward pass
            if self.config['use_amp']:
                with torch.amp.autocast('cuda'):
                    logits = self.model(input_ids)
                    loss = self.compute_loss(logits, input_ids)
                    loss = loss / self.config['gradient_accumulation_steps']
            else:
                logits = self.model(input_ids)
                loss = self.compute_loss(logits, input_ids)
                loss = loss / self.config['gradient_accumulation_steps']
            
            # Backward pass
            if self.config['use_amp']:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                if self.config['use_amp']:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )
                
                # Optimizer step
                if self.config['use_amp']:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Scheduler step
                lr = self.scheduler.step()
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.config["gradient_accumulation_steps"]:.4f}',
                    'lr': f'{lr:.2e}'
                })
            
            total_loss += loss.item() * self.config['gradient_accumulation_steps']
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        
        for input_ids in tqdm(self.val_loader, desc="Validation"):
            input_ids = input_ids.to(self.device)
            
            logits = self.model(input_ids)
            loss = self.compute_loss(logits, input_ids)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = math.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def compute_loss(self, logits, input_ids):
        """Causal language modeling loss"""
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Cross entropy loss
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.token_to_id("<pad>")
        )
        
        return loss
    
    def save_checkpoint(self, path, epoch, val_loss):
        """Checkpoint save"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        print(f"\n[OK] Checkpoint saved: {path}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, perplexity = self.validate()
            
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    Path(self.config['output_dir']) / 'best_model.pt',
                    epoch,
                    val_loss
                )
            
            # Save regular checkpoint
            if epoch % self.config['save_every'] == 0:
                self.save_checkpoint(
                    Path(self.config['output_dir']) / f'checkpoint_epoch_{epoch}.pt',
                    epoch,
                    val_loss
                )
        
        print("\n" + "=" * 60)
        print("[OK] TRAINING COMPLETED!")
        print("=" * 60)


if __name__ == "__main__":
    from llama3_architecture import LlamaModel, ModelConfig
    
    # Training configuration
    train_config = {
        'learning_rate': 3e-4,
        'min_learning_rate': 3e-5,
        'beta1': 0.9,
        'beta2': 0.95,
        'weight_decay': 0.1,
        'warmup_ratio': 0.05,
        'batch_size': 4,
        'gradient_accumulation_steps': 4,  # Effective batch size: 16
        'num_epochs': 10,
        'max_grad_norm': 1.0,
        'use_amp': True,
        'save_every': 2,
        'output_dir': 'models/checkpoints'
    }
    
    # Create output directory
    Path(train_config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    print("After training configuration is set:")
    print("1. Load Tokenizer")
    print("2. Prepare Dataset")
    print("3. Create Model")
    print("4. Create Trainer and start training")