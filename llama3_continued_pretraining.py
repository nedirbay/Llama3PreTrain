"""
Llama 3 Continued Pre-training (CPT)
Pre-train edilen modeli täze data bilen dowam etdirmek
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
from tokenizers import Tokenizer
from typing import Optional, Dict, Any
import math

from llama3_architecture import LlamaModel, ModelConfig
from llama3_training import CosineScheduleWithWarmup, TextDataset
from llama3_monitoring import TrainingMonitor


class ContinuedPreTrainer:
    """Continued Pre-training Trainer"""
    
    def __init__(
        self,
        checkpoint_path: str,
        new_data_path: str,
        tokenizer_path: str,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        self.device = device
        self.config = config
        
        print("=" * 70)
        print("CONTINUED PRE-TRAINING BAŞLATILIYOR")
        print("=" * 70)
        
        # 1. Checkpoint yüklemek
        print("\n1. Checkpoint yüklenýär...")
        self.checkpoint = self.load_checkpoint(checkpoint_path)
        
        # 2. Model yaratmak
        print("\n2. Model ýüklenýär...")
        model_config = ModelConfig(**self.checkpoint['model_config'])
        self.model = LlamaModel(model_config).to(device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        
        print(f"   ✅ Model ýüklendi: {self.model.num_parameters():,} parameters")
        print(f"   📊 Training epoch: {self.checkpoint.get('epoch', 'Unknown')}")
        print(f"   📉 Best val loss: {self.checkpoint.get('val_loss', 'Unknown'):.4f}")
        
        # 3. Tokenizer yüklemek
        print("\n3. Tokenizer ýüklenýär...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        with open(Path(tokenizer_path).parent / "tokenizer_config.json") as f:
            self.tokenizer_config = json.load(f)
        print(f"   ✅ Vocab size: {self.tokenizer_config['vocab_size']:,}")
        
        # 4. Täze dataset yüklemek
        print("\n4. Täze dataset taýýarlanýar...")
        self.new_dataset = self.load_new_dataset(new_data_path, model_config)
        
        # 5. Training components
        self.setup_training(model_config)
        
        # 6. Monitor
        self.monitor = TrainingMonitor(
            project_name="llama3-continued-pretraining",
            use_wandb=config.get('use_wandb', False)
        )
        
        print("\n" + "=" * 70)
        print("✅ CONTINUED PRE-TRAINING TAÝÝAR")
        print("=" * 70)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Checkpoint yüklemek"""
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint tapylmady: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        required_keys = ['model_state_dict', 'model_config']
        for key in required_keys:
            if key not in checkpoint:
                # Eger model_config yoksa, config dict-den almaga çalış
                if key == 'model_config' and 'config' in checkpoint:
                    checkpoint['model_config'] = checkpoint['config']
                else:
                    raise KeyError(f"Checkpoint-de '{key}' tapylmady!")
        
        return checkpoint
    
    def load_new_dataset(self, data_path: str, model_config: ModelConfig):
        """Täze dataset yüklemek"""
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Dataset tapylmady: {data_path}")
        
        # Full dataset
        full_dataset = TextDataset(
            data_path,
            self.tokenizer,
            max_length=model_config.max_position_embeddings
        )
        
        # Train/Val split
        train_size = int(0.95 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        torch.manual_seed(42)  # Reproducibility
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size]
        )
        
        print(f"   📁 Train samples: {len(train_dataset):,}")
        print(f"   📁 Val samples: {len(val_dataset):,}")
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        return full_dataset
    
    def setup_training(self, model_config: ModelConfig):
        """Training components setup"""
        
        print("\n5. Training taýýarlanýar...")
        
        # Learning rate strategy için 3 option:
        lr_strategy = self.config.get('lr_strategy', 'lower')  # 'lower', 'same', 'warmup'
        base_lr = self.config.get('learning_rate', 1e-4)
        
        if lr_strategy == 'lower':
            # Option 1: Düşük LR (10x daha kiçi)
            learning_rate = base_lr / 10
            print(f"   📉 LR Strategy: LOWER (10x azaldylan)")
        elif lr_strategy == 'same':
            # Option 2: Öňki LR
            learning_rate = base_lr
            print(f"   📊 LR Strategy: SAME (öňki LR)")
        else:  # warmup
            # Option 3: Warmup bilen täzeden
            learning_rate = base_lr
            print(f"   🔥 LR Strategy: WARMUP (täze warmup)")
        
        print(f"   📈 Learning rate: {learning_rate:.2e}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.95)),
            weight_decay=self.config.get('weight_decay', 0.1),
            eps=1e-8
        )
        
        # Eğer checkpoint-de optimizer state varsa yükle (optional)
        if self.config.get('load_optimizer', False) and 'optimizer_state_dict' in self.checkpoint:
            try:
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                print(f"   ✅ Optimizer state yüklendi")
            except Exception as e:
                print(f"   ⚠️  Optimizer state yüklenemedi: {e}")
        
        # Scheduler
        total_steps = len(self.train_dataset) // self.config['batch_size'] * self.config['num_epochs']
        total_steps = total_steps // self.config.get('gradient_accumulation_steps', 1)
        warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.03))  # Daha kısa warmup
        
        self.scheduler = CosineScheduleWithWarmup(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=self.config.get('min_learning_rate', 1e-6)
        )
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.get('use_amp', True) else None
        
        # Training state
        self.global_step = self.checkpoint.get('global_step', 0)
        self.start_epoch = self.checkpoint.get('epoch', 0) + 1
        self.best_val_loss = self.checkpoint.get('val_loss', float('inf'))
        
        print(f"   🔢 Total steps: {total_steps:,}")
        print(f"   🔥 Warmup steps: {warmup_steps:,}")
        print(f"   📊 Starting epoch: {self.start_epoch}")
        print(f"   💾 Best val loss: {self.best_val_loss:.4f}")
    
    def train_epoch(self, epoch: int) -> float:
        """Bir epoch train etmek"""
        
        self.model.train()
        total_loss = 0
        grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, input_ids in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            
            # Forward pass
            if self.config.get('use_amp', True):
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids)
                    loss = self.compute_loss(logits, input_ids)
                    loss = loss / grad_accum_steps
            else:
                logits = self.model(input_ids)
                loss = self.compute_loss(logits, input_ids)
                loss = loss / grad_accum_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
                
                # Step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Scheduler
                lr = self.scheduler.step()
                self.global_step += 1
                
                # Progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item() * grad_accum_steps:.4f}',
                    'lr': f'{lr:.2e}'
                })
            
            total_loss += loss.item() * grad_accum_steps
            
            # GPU memory clear
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> tuple:
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
    
    def compute_loss(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Loss hesaplamak"""
        
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Cross entropy
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.token_to_id("<pad>")
        )
        
        return loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, filename: str):
        """Checkpoint kaydet"""
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'model_config': self.model.config.__dict__,
            'train_config': self.config,
            'continued_pretraining': True,  # Flag
            'original_checkpoint': str(self.config.get('checkpoint_path', 'unknown'))
        }
        
        save_path = Path(self.config['output_dir']) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, save_path)
        print(f"\n💾 Checkpoint kaydedildi: {save_path}")
    
    def train(self):
        """Ana training loop"""
        
        print("\n" + "=" * 70)
        print("🚀 CONTINUED PRE-TRAINING BAŞLADI")
        print("=" * 70)
        
        for epoch in range(self.start_epoch, self.start_epoch + self.config['num_epochs']):
            print(f"\n📊 Epoch {epoch}/{self.start_epoch + self.config['num_epochs'] - 1}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, perplexity = self.validate()
            
            # Log metrics
            if self.monitor:
                self.monitor.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'perplexity': perplexity,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                }, step=self.global_step)
            
            # Print results
            print(f"\n📈 Epoch {epoch} Sonuçlary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Perplexity: {perplexity:.2f}")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, 'best_cpt_model.pt')
                print(f"   ⭐ Täze iň gowy model!")
            
            # Regular checkpoint
            if epoch % self.config.get('save_every', 2) == 0:
                self.save_checkpoint(epoch, val_loss, f'cpt_checkpoint_epoch_{epoch}.pt')
        
        print("\n" + "=" * 70)
        print("✅ CONTINUED PRE-TRAINING TAMAMLANDI!")
        print("=" * 70)
        print(f"\n📊 Final Results:")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")
        print(f"   Total Epochs: {self.config['num_epochs']}")
        print(f"   Output Dir: {self.config['output_dir']}")
        
        # Generate samples
        print("\n" + "=" * 70)
        print("📝 SAMPLE GENERATION")
        print("=" * 70)
        self.generate_samples()
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int = 3):
        """Sample text generate etmek"""
        
        self.model.eval()
        
        prompts = [
            "Once upon a time",
            "The future of AI is",
            "In the year 2030,"
        ]
        
        for prompt in prompts[:num_samples]:
            print(f"\n💬 Prompt: {prompt}")
            print("   Generated: ", end="")
            
            # Encode
            encoded = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([encoded.ids]).to(self.device)
            
            # Generate
            max_length = 50
            for _ in range(max_length):
                logits = self.model(input_ids)
                next_token_logits = logits[:, -1, :]
                
                # Sample
                probs = torch.softmax(next_token_logits / 0.8, dim=-1)  # temperature
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Decode
                token_str = self.tokenizer.decode([next_token.item()])
                print(token_str, end="", flush=True)
                
                # Stop at EOS
                if next_token.item() == self.tokenizer.token_to_id("</s>"):
                    break
            
            print()


def main():
    """Main execution"""
    
    # Configuration
    config = {
        # Paths
        'checkpoint_path': 'models/checkpoints/best_model.pt',
        'new_data_path': 'data/raw/new_domain_data.txt',  # Täze dataset
        'tokenizer_path': 'data/tokenizer/tokenizer.json',
        'output_dir': 'models/continued_pretraining',
        
        # Training params
        'learning_rate': 1e-4,  # Base LR
        'min_learning_rate': 1e-6,
        'lr_strategy': 'lower',  # 'lower', 'same', 'warmup'
        'load_optimizer': False,  # Optimizer state yüklemek?
        
        'beta1': 0.9,
        'beta2': 0.95,
        'weight_decay': 0.1,
        'warmup_ratio': 0.03,  # Daha kısa warmup
        
        'batch_size': 8,
        'gradient_accumulation_steps': 2,
        'num_epochs': 3,  # Daha az epoch
        'max_grad_norm': 1.0,
        
        'use_amp': True,
        'use_wandb': False,
        'num_workers': 2,
        'save_every': 1,
    }
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create trainer
    try:
        trainer = ContinuedPreTrainer(
            checkpoint_path=config['checkpoint_path'],
            new_data_path=config['new_data_path'],
            tokenizer_path=config['tokenizer_path'],
            config=config,
            device=device
        )
        
        # Train
        trainer.train()
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n📋 Gerekli adımlar:")
        print("1. İlk önce base pre-training yapın: python llama3_main.py")
        print("2. Täze domain dataset hazırlayın: data/raw/new_domain_data.txt")
        print("3. Sonra CPT başlatın: python llama3_continued_pretraining.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()