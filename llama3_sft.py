"""
Llama 3 Supervised Fine-Tuning (SFT)
Pre-train edilen modeli instruction-following üçin fine-tune etmek
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
from tokenizers import Tokenizer
from typing import List, Dict, Optional
import math

from llama3_architecture import LlamaModel, ModelConfig
from llama3_training import CosineScheduleWithWarmup
from llama3_monitoring import TrainingMonitor


class InstructionDataset(Dataset):
    """
    SFT üçin Instruction Dataset
    
    Format requirements:
    1. JSON/JSONL format
    2. Her bir sample:
       {
         "instruction": "User instruction",
         "input": "Optional context/input", 
         "output": "Expected response"
       }
    
    Alternative format (conversation):
       {
         "conversations": [
           {"role": "user", "content": "..."},
           {"role": "assistant", "content": "..."}
         ]
       }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Tokenizer,
        max_length: int = 512,
        format_type: str = "alpaca",  # "alpaca" or "conversation"
        check_lengths: bool = True  # FIX 1: Dataset statistikalaryny barlamak
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_type = format_type
        
        # Load data
        self.data = self.load_data(data_path)
        
        # FIX 2: Boş ýa-da nädogry data aýyrmak
        self.data = self.validate_and_filter_data(self.data)
        
        # Special tokens
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        
        # FIX 3: Token ID-leri barlamak
        self.bos_id = tokenizer.token_to_id(self.bos_token)
        self.eos_id = tokenizer.token_to_id(self.eos_token)
        self.pad_id = tokenizer.token_to_id(self.pad_token)
        
        if self.bos_id is None or self.eos_id is None or self.pad_id is None:
            raise ValueError(
                f"Special tokens tapylmady! "
                f"BOS: {self.bos_id}, EOS: {self.eos_id}, PAD: {self.pad_id}"
            )
        
        print(f"[OK] SFT Dataset loaded: {len(self.data):,} samples")
        print(f"     Format: {format_type}")
        print(f"     Max length: {max_length}")
        
        # FIX 4: Dataset statistikalaryny görkez
        if check_lengths:
            self.print_dataset_statistics()
    
    def load_data(self, data_path: str) -> List[Dict]:
        """Load dataset from JSON/JSONL"""
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Dataset tapylmady: {data_path}")
        
        data = []
        
        # JSONL format
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # FIX 5: Boş setirler skip
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Setir {line_num} JSON parse edilemedi: {e}")
                        continue
        
        # JSON format
        elif data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                try:
                    loaded_data = json.load(f)
                    # FIX 6: List ýa-da dict bolup biler
                    if isinstance(loaded_data, list):
                        data = loaded_data
                    elif isinstance(loaded_data, dict):
                        # Käbir datasetler {"data": [...]} formatda
                        data = loaded_data.get('data', [loaded_data])
                    else:
                        raise ValueError(f"JSON format nädogry: {type(loaded_data)}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON parse error: {e}")
        
        else:
            raise ValueError("Dataset .json ýa-da .jsonl formatda bolmaly!")
        
        if not data:
            raise ValueError(f"Dataset boş! File: {data_path}")
        
        return data
    
    def validate_and_filter_data(self, data: List[Dict]) -> List[Dict]:
        """FIX 7: Data validation we filtering"""
        
        valid_data = []
        filtered_count = 0
        
        for idx, sample in enumerate(data):
            try:
                # Format-a görä barla
                if self.format_type == "alpaca":
                    instruction = sample.get('instruction', '').strip()
                    output = sample.get('output', '').strip()
                    
                    # Boş instruction ýa-da output bolmaly däl
                    if not instruction or not output:
                        filtered_count += 1
                        continue
                    
                    # Örän gysga response-lar skip (spam bolup biler)
                    if len(output.split()) < 3:
                        filtered_count += 1
                        continue
                
                elif self.format_type == "conversation":
                    conversations = sample.get('conversations', [])
                    
                    if not conversations or len(conversations) < 2:
                        filtered_count += 1
                        continue
                    
                    # Iň azyndan bir user we bir assistant bolmaly
                    has_user = any(msg.get('role') == 'user' for msg in conversations)
                    has_assistant = any(msg.get('role') == 'assistant' for msg in conversations)
                    
                    if not has_user or not has_assistant:
                        filtered_count += 1
                        continue
                
                valid_data.append(sample)
                
            except Exception as e:
                print(f"⚠️  Sample {idx} validate edilemedi: {e}")
                filtered_count += 1
                continue
        
        if filtered_count > 0:
            print(f"⚠️  {filtered_count} nädogry sample aýryldy")
        
        return valid_data
    
    def print_dataset_statistics(self):
        """FIX 8: Dataset statistikalaryny görkezmek"""
        
        print("\n" + "=" * 60)
        print("DATASET STATISTIKALAR")
        print("=" * 60)
        
        prompt_lengths = []
        response_lengths = []
        total_lengths = []
        
        for sample in self.data[:min(100, len(self.data))]:  # Ilk 100 sample
            try:
                prompt = self.format_prompt(sample)
                
                if self.format_type == "alpaca":
                    response = sample.get('output', '')
                else:
                    conversations = sample.get('conversations', [])
                    response = ""
                    for msg in reversed(conversations):
                        if msg.get('role') == 'assistant':
                            response = msg.get('content', '')
                            break
                
                # Tokenize
                prompt_tokens = len(self.tokenizer.encode(prompt).ids)
                response_tokens = len(self.tokenizer.encode(response).ids)
                total_tokens = prompt_tokens + response_tokens + 2  # BOS + EOS
                
                prompt_lengths.append(prompt_tokens)
                response_lengths.append(response_tokens)
                total_lengths.append(total_tokens)
                
            except Exception:
                continue
        
        if total_lengths:
            import statistics
            
            print(f"\n📊 Token Uzunlyklar (ilk {len(total_lengths)} sample):")
            print(f"   Prompt:")
            print(f"      Mean: {statistics.mean(prompt_lengths):.1f}")
            print(f"      Median: {statistics.median(prompt_lengths):.1f}")
            print(f"      Max: {max(prompt_lengths)}")
            
            print(f"   Response:")
            print(f"      Mean: {statistics.mean(response_lengths):.1f}")
            print(f"      Median: {statistics.median(response_lengths):.1f}")
            print(f"      Max: {max(response_lengths)}")
            
            print(f"   Total:")
            print(f"      Mean: {statistics.mean(total_lengths):.1f}")
            print(f"      Max: {max(total_lengths)}")
            print(f"      Max length setting: {self.max_length}")
            
            # Kesilen sample sany
            truncated = sum(1 for l in total_lengths if l > self.max_length)
            if truncated > 0:
                print(f"\n⚠️  {truncated}/{len(total_lengths)} sample kesiler (max_length={self.max_length})")
                print(f"   Göz öňünde tut: max_length artdyrmak maslahat berilýär!")
        
        print("=" * 60)
    
    def format_prompt(self, sample: Dict) -> str:
        """Format prompt based on data format"""
        
        if self.format_type == "alpaca":
            # Alpaca format
            instruction = sample.get('instruction', '')
            input_text = sample.get('input', '')
            
            if input_text:
                prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
            else:
                prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
            
            return prompt
        
        elif self.format_type == "conversation":
            # Conversation format
            conversations = sample.get('conversations', [])
            
            prompt = ""
            for msg in conversations:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'user':
                    prompt += f"User: {content}\n\n"
                elif role == 'assistant':
                    prompt += f"Assistant: {content}\n\n"
            
            return prompt.strip()
        
        else:
            raise ValueError(f"Format '{self.format_type}' goldanylmaýar!")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        try:
            # Format prompt
            prompt = self.format_prompt(sample)
            
            # Get response/output
            if self.format_type == "alpaca":
                response = sample.get('output', '')
            else:  # conversation
                # Last assistant message
                conversations = sample.get('conversations', [])
                response = ""
                for msg in reversed(conversations):
                    if msg.get('role') == 'assistant':
                        response = msg.get('content', '')
                        break
            
            # FIX 9: Boş response barlamak
            if not response.strip():
                print(f"⚠️  Sample {idx} boş response, default ulanylýar")
                response = "I understand."
            
            # Full text: prompt + response
            full_text = f"{self.bos_token}{prompt}{response}{self.eos_token}"
            
            # Tokenize
            encoded = self.tokenizer.encode(full_text)
            input_ids = encoded.ids
            
            # Calculate labels (only compute loss on response part)
            prompt_encoded = self.tokenizer.encode(f"{self.bos_token}{prompt}")
            prompt_length = len(prompt_encoded.ids)
            
            # FIX 10: Prompt uzynlyk barlamak
            if prompt_length >= self.max_length - 10:  # Iň azyndan 10 token response üçin
                print(f"⚠️  Sample {idx}: prompt örän uzyn ({prompt_length} tokens)")
                # Prompt-y gysgaltmak
                prompt_encoded_ids = prompt_encoded.ids[:self.max_length - 50]
                prompt_length = len(prompt_encoded_ids)
                input_ids = prompt_encoded_ids + encoded.ids[prompt_length:]
            
            # Create labels: -100 for prompt (ignore), actual tokens for response
            labels = [-100] * prompt_length + input_ids[prompt_length:]
            
            # Truncate or pad
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
            else:
                padding_length = self.max_length - len(input_ids)
                input_ids = input_ids + [self.pad_id] * padding_length
                labels = labels + [-100] * padding_length
            
            # FIX 11: Attention mask düzgün hasaplamak
            attention_mask = [1 if id != self.pad_id else 0 for id in input_ids]
            
            # FIX 12: Label barlamak - iň azyndan birnäçe token training üçin bolmaly
            non_ignored_labels = sum(1 for l in labels if l != -100)
            if non_ignored_labels < 5:  # Iň azyndan 5 token response
                print(f"⚠️  Sample {idx}: örän az training token ({non_ignored_labels})")
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }
            
        except Exception as e:
            print(f"❌ Sample {idx} process edilemedi: {e}")
            # FIX 13: Fallback - boş sample return etmek
            input_ids = [self.bos_id] + [self.pad_id] * (self.max_length - 1)
            labels = [-100] * self.max_length
            attention_mask = [1] + [0] * (self.max_length - 1)
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }


class SFTTrainer:
    """Supervised Fine-Tuning Trainer"""
    
    def __init__(
        self,
        checkpoint_path: str,
        train_dataset: InstructionDataset,
        val_dataset: Optional[InstructionDataset],
        tokenizer: Tokenizer,
        config: Dict,
        device: str = 'cuda'
    ):
        self.device = device
        self.config = config
        self.tokenizer = tokenizer
        
        print("=" * 70)
        print("SUPERVISED FINE-TUNING (SFT) BAŞLATILIYOR")
        print("=" * 70)
        
        # 1. Load checkpoint
        print("\n1. Pre-trained model ýüklenýär...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Model config
        if 'model_config' in checkpoint:
            model_config = ModelConfig(**checkpoint['model_config'])
        elif 'config' in checkpoint:
            model_config = ModelConfig(**checkpoint['config'])
        else:
            raise KeyError("Model config tapylmady!")
        
        # 2. Create model
        self.model = LlamaModel(model_config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"   ✅ Model ýüklendi: {self.model.num_parameters():,} parameters")
        
        # 3. Setup datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        print(f"\n2. Dataset:")
        print(f"   Train samples: {len(train_dataset):,}")
        if val_dataset:
            print(f"   Val samples: {len(val_dataset):,}")
        
        # 4. Setup training components
        self.setup_training()
        
        # 5. Monitor
        self.monitor = TrainingMonitor(
            project_name="llama3-sft",
            use_wandb=config.get('use_wandb', False)
        )
        
        print("\n" + "=" * 70)
        print("✅ SFT TRAINER TAÝÝAR")
        print("=" * 70)
    
    def setup_training(self):
        """Setup optimizer, scheduler, dataloaders"""
        
        print("\n3. Training components taýýarlanýar...")
        
        # Optimizer - SFT üçin kiçi learning rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 2e-5),  # Daha kiçi LR
            betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.999)),
            weight_decay=self.config.get('weight_decay', 0.01),  # Kiçi weight decay
            eps=1e-8
        )
        
        # Scheduler
        total_steps = (
            len(self.train_dataset) // 
            self.config['batch_size'] * 
            self.config['num_epochs']
        )
        total_steps = total_steps // self.config.get('gradient_accumulation_steps', 1)
        warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))
        
        self.scheduler = CosineScheduleWithWarmup(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=self.config.get('min_learning_rate', 0)
        )
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config.get('num_workers', 2),
                pin_memory=True
            )
        else:
            self.val_loader = None
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.get('use_amp', True) else None
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        print(f"   Learning rate: {self.config.get('learning_rate', 2e-5):.2e}")
        print(f"   Total steps: {total_steps:,}")
        print(f"   Warmup steps: {warmup_steps:,}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Gradient accumulation: {self.config.get('gradient_accumulation_steps', 1)}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch"""
        
        self.model.train()
        total_loss = 0
        total_samples = 0  # FIX 14: Hakyky sample sany üçin
        grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # FIX 15: Batch-da hakyky training tokenleriň sanyny barlat
            valid_tokens = (labels != -100).sum().item()
            if valid_tokens == 0:
                print(f"⚠️  Batch {batch_idx}: Hiç training token ýok, skip edilýär")
                continue
            
            # Forward pass
            if self.config.get('use_amp', True):
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids)
                    loss = self.compute_loss(logits, labels)
                    
                    # FIX 16: NaN/Inf barlamak
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️  Batch {batch_idx}: Loss NaN/Inf, skip edilýär")
                        continue
                    
                    loss = loss / grad_accum_steps
            else:
                logits = self.model(input_ids)
                loss = self.compute_loss(logits, labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  Batch {batch_idx}: Loss NaN/Inf, skip edilýär")
                    continue
                
                loss = loss / grad_accum_steps
            
            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % grad_accum_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                # FIX 17: Gradient norm-y hasapla we log et
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
                
                # FIX 18: Gradient explosion barlamak
                if grad_norm > 100:
                    print(f"⚠️  Step {self.global_step}: Uly gradient norm: {grad_norm:.2f}")
                
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                lr = self.scheduler.step()
                self.global_step += 1
                
                # FIX 19: Doly progress bar maglumatlary
                pbar.set_postfix({
                    'loss': f'{loss.item() * grad_accum_steps:.4f}',
                    'lr': f'{lr:.2e}',
                    'grad_norm': f'{grad_norm:.2f}',
                    'tokens': valid_tokens
                })
            
            total_loss += loss.item() * grad_accum_steps
            total_samples += 1
            
            # FIX 20: Memory management - her 10 batch-dan GPU memory temizlemek
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_loss / max(total_samples, 1)  # Division by zero önlemek
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """Validation"""
        
        if not self.val_loader:
            return None, None
        
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits = self.model(input_ids)
            loss = self.compute_loss(logits, labels)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        return avg_loss, perplexity
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss only on response tokens
        labels with -100 are ignored
        """
        
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # FIX 21: Hakyky training tokenleriň sanyny barlat
        valid_labels = (shift_labels != -100).sum()
        
        if valid_labels == 0:
            # Hiç training token ýok bolsa, kiçi dummy loss return et
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Cross entropy (ignore_index=-100)
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='sum'  # FIX 22: 'mean' däl 'sum' ulanmak
        )
        
        # FIX 23: Token sany boýunça normalize etmek (per-token loss)
        loss = loss / valid_labels
        
        return loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, filename: str):
        """Save checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'model_config': self.model.config.__dict__,
            'train_config': self.config,
            'training_type': 'SFT'
        }
        
        save_path = Path(self.config['output_dir']) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, save_path)
        print(f"\n💾 Checkpoint saved: {save_path}")
    
    def train(self):
        """Main training loop"""
        
        print("\n" + "=" * 70)
        print("🚀 SFT TRAINING BAŞLADI")
        print("=" * 70)
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\n📊 Epoch {epoch}/{self.config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, perplexity = self.validate()
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch': epoch
            }
            
            if val_loss is not None:
                metrics['val_loss'] = val_loss
                metrics['perplexity'] = perplexity
            
            if self.monitor:
                self.monitor.log_metrics(metrics, step=self.global_step)
            
            # Print results
            print(f"\n📈 Epoch {epoch} Netijeleri:")
            print(f"   Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"   Val Loss: {val_loss:.4f}")
                print(f"   Perplexity: {perplexity:.2f}")
            
            # Save best model
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                self.save_checkpoint(epoch, current_loss, 'best_sft_model.pt')
                print(f"   ⭐ Täze iň gowy model!")
            
            # Regular checkpoint
            if epoch % self.config.get('save_every', 1) == 0:
                self.save_checkpoint(epoch, current_loss, f'sft_checkpoint_epoch_{epoch}.pt')
        
        print("\n" + "=" * 70)
        print("✅ SFT TRAINING TAMAMLANDI!")
        print("=" * 70)
        print(f"\n📊 Final Results:")
        print(f"   Best Loss: {self.best_val_loss:.4f}")
        print(f"   Total Epochs: {self.config['num_epochs']}")
        print(f"   Output Dir: {self.config['output_dir']}")
        
        # Generate samples
        print("\n" + "=" * 70)
        print("🔍 SAMPLE GENERATION")
        print("=" * 70)
        self.generate_samples()
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int = 3):
        """Generate sample responses"""
        
        self.model.eval()
        
        test_prompts = [
            "Explain what is machine learning in simple terms.",
            "Write a short poem about the ocean.",
            "What are the benefits of reading books?"
        ]
        
        for prompt in test_prompts[:num_samples]:
            print(f"\n💬 Instruction: {prompt}")
            print("   Response: ", end="")
            
            # Format prompt
            formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
            
            # Encode
            encoded = self.tokenizer.encode(f"<s>{formatted_prompt}")
            input_ids = torch.tensor([encoded.ids]).to(self.device)
            
            # FIX 24: Generation config doly bilen
            max_new_tokens = 100
            temperature = 0.7
            top_k = 50
            top_p = 0.9  # FIX 25: Top-p (nucleus) sampling goşmak
            repetition_penalty = 1.1  # FIX 26: Repetition penalty
            
            generated = []
            prev_tokens = []  # Repetition track etmek üçin
            
            for step in range(max_new_tokens):
                # FIX 27: Max sequence length barlamak
                if input_ids.size(1) >= self.model.config.max_position_embeddings - 1:
                    print("\n⚠️  Max sequence length ýetdi, durýar")
                    break
                
                logits = self.model(input_ids)
                next_token_logits = logits[:, -1, :]
                
                # FIX 28: Repetition penalty aplikasiýasy
                if repetition_penalty != 1.0 and len(prev_tokens) > 0:
                    for token_id in set(prev_tokens[-50:]):  # Soňky 50 token
                        if next_token_logits[0, token_id] < 0:
                            next_token_logits[0, token_id] *= repetition_penalty
                        else:
                            next_token_logits[0, token_id] /= repetition_penalty
                
                # Temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # FIX 29: Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Decode
                token_id = next_token.item()
                prev_tokens.append(token_id)
                
                # Stop at EOS
                if token_id == self.tokenizer.token_to_id("</s>"):
                    break
                
                # FIX 30: PAD token skip etmek
                if token_id == self.tokenizer.token_to_id("<pad>"):
                    continue
                
                token_str = self.tokenizer.decode([token_id])
                generated.append(token_str)
                print(token_str, end="", flush=True)
            
            print("\n")


def create_sample_dataset():
    """
    SFT üçin sample dataset döretmek
    
    Bu funksiya size format näçeräk bolmalydygyny görkezer
    """
    
    # Alpaca format
    alpaca_samples = [
        {
            "instruction": "Explain what is artificial intelligence.",
            "input": "",
            "output": "Artificial intelligence (AI) is a branch of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. AI systems use algorithms and large amounts of data to recognize patterns and make decisions."
        },
        {
            "instruction": "Translate the following sentence to Turkish.",
            "input": "Hello, how are you?",
            "output": "Merhaba, nasılsın?"
        },
        {
            "instruction": "Write a haiku about spring.",
            "input": "",
            "output": "Cherry blossoms bloom,\nGentle breeze whispers softly,\nSpring awakens life."
        }
    ]
    
    # Conversation format
    conversation_samples = [
        {
            "conversations": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, automation, and artificial intelligence."}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "Can you help me with a math problem?"},
                {"role": "assistant", "content": "Of course! I'd be happy to help you with your math problem. Please share the problem and I'll do my best to explain and solve it."},
                {"role": "user", "content": "What is 15% of 200?"},
                {"role": "assistant", "content": "To find 15% of 200:\n\n15% = 15/100 = 0.15\n0.15 × 200 = 30\n\nSo 15% of 200 is 30."}
            ]
        }
    ]
    
    # Save samples
    output_dir = Path("data/sft")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Alpaca format
    with open(output_dir / "sample_alpaca.json", 'w', encoding='utf-8') as f:
        json.dump(alpaca_samples, f, indent=2, ensure_ascii=False)
    
    # Conversation format  
    with open(output_dir / "sample_conversation.jsonl", 'w', encoding='utf-8') as f:
        for sample in conversation_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print("=" * 70)
    print("SAMPLE DATASETS DÖREDILDI")
    print("=" * 70)
    print(f"\n✅ Files created:")
    print(f"   1. {output_dir / 'sample_alpaca.json'} (Alpaca format)")
    print(f"   2. {output_dir / 'sample_conversation.jsonl'} (Conversation format)")
    print("\n📋 Dataset Format Requirements:")
    print("\n1. Alpaca Format (JSON):")
    print("""   {
     "instruction": "User request/question",
     "input": "Optional context (can be empty)",
     "output": "Expected response"
   }""")
    print("\n2. Conversation Format (JSONL):")
    print("""   {
     "conversations": [
       {"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."}
     ]
   }""")


def main():
    """Main execution"""
    
    # First, create sample datasets
    print("1. Sample datasets döredilýär...\n")
    create_sample_dataset()
    
    print("\n" + "=" * 70)
    print("SFT TRAINING BAŞLATILIYOR")
    print("=" * 70)
    
    # Configuration
    config = {
        # Paths
        'checkpoint_path': 'models/checkpoints/best_model.pt',
        'train_data_path': 'data/sft/train.json',  # Your SFT data
        'val_data_path': 'data/sft/val.json',      # Optional
        'tokenizer_path': 'data/tokenizer/tokenizer.json',
        'output_dir': 'models/sft',
        
        # Data format
        'data_format': 'alpaca',  # or 'conversation'
        
        # Training hyperparameters (SFT üçin optimize)
        'learning_rate': 2e-5,      # Kiçi LR (pre-training: 3e-4)
        'min_learning_rate': 0,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.01,       # Kiçi weight decay
        'warmup_ratio': 0.1,        # Daha uzyn warmup
        
        'batch_size': 4,            # Kiçi batch (memory üçin)
        'gradient_accumulation_steps': 4,  # Effective batch: 16
        'num_epochs': 3,            # Az epoch (overfitting önlemek)
        'max_grad_norm': 1.0,
        
        'use_amp': True,
        'use_wandb': False,
        'num_workers': 2,
        'save_every': 1,
    }
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        # Load tokenizer
        print("\n2. Tokenizer ýüklenýär...")
        tokenizer = Tokenizer.from_file(config['tokenizer_path'])
        print(f"   ✅ Tokenizer loaded")
        
        # Load datasets
        print("\n3. Datasets ýüklenýär...")
        
        # Use sample data for demo
        train_data_path = 'data/sft/sample_alpaca.json'
        
        if Path(train_data_path).exists():
            train_dataset = InstructionDataset(
                train_data_path,
                tokenizer,
                max_length=512,
                format_type=config['data_format']
            )
            
            # For demo, use same data for validation
            val_dataset = InstructionDataset(
                train_data_path,
                tokenizer,
                max_length=512,
                format_type=config['data_format']
            )
        else:
            print(f"\n⚠️  Training data tapylmady: {train_data_path}")
            print("\nÖň taýýarlaň:")
            print("1. Sizin SFT dataset-yňyzy taýýarlaň")
            print("2. Ony şu formatlarda saklaň:")
            print("   - data/sft/train.json (Alpaca format)")
            print("   - data/sft/train.jsonl (Conversation format)")
            return
        
        # Create trainer
        print("\n4. SFT Trainer döredilýär...")
        trainer = SFTTrainer(
            checkpoint_path=config['checkpoint_path'],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            config=config,
            device=device
        )
        
        # Train
        print("\n5. Training başlaýar...")
        trainer.train()
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n📋 Gerekli adımlar:")
        print("1. Pre-training tamamlaň: python llama3_main.py")
        print("2. SFT dataset taýýarlaň (yukarıdaky formatda)")
        print("3. SFT başlatıň: python llama3_sft.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()