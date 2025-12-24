"""
Llama 3 Pre-training - Main Execution Script
Main script to combine the whole process
"""

import torch
from pathlib import Path
import json
from tokenizers import Tokenizer

# Import our modules
from llama3_architecture import LlamaModel, ModelConfig
from llama3_training import Trainer, TextDataset
from llama3_monitoring import TrainingMonitor

def setup_training():
    """Setup training environment"""
    
    print("=" * 70)
    print("LLAMA 3 PRE-TRAINING SETUP")
    print("=" * 70)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[OK] Device: {device}")
    
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load tokenizer
    print("\n1. Loading Tokenizer...")
    tokenizer_path = "data/tokenizer/tokenizer.json"
    
    if not Path(tokenizer_path).exists():
        print("[WARN] Tokenizer not found: {tokenizer_path}")
        print("First run 'python llama3_tokenizer.py'!")
        return None
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    with open("data/tokenizer/tokenizer_config.json") as f:
        tokenizer_config = json.load(f)
    
    vocab_size = tokenizer_config['vocab_size']
    print(f"  [OK] Vocab size: {vocab_size:,}")
    
    # Model configuration
    print("\n2. Model Configuration...")
    model_config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=768,              # Medium size
        num_layers=12,                # 12 layers
        num_attention_heads=12,       # 12 attention heads
        num_key_value_heads=4,        # 4 KV heads (GQA)
        intermediate_size=2048,       # FFN size
        max_position_embeddings=512,  # Context length
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_dropout=0.0,
        hidden_dropout=0.1
    )
    
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Layers: {model_config.num_layers}")
    print(f"  Attention heads: {model_config.num_attention_heads}")
    print(f"  KV heads (GQA): {model_config.num_key_value_heads}")
    print(f"  Context length: {model_config.max_position_embeddings}")
    
    # Create model
    print("\n3. Creating Model...")
    model = LlamaModel(model_config)
    num_params = model.num_parameters()
    print(f"  [OK] Total parameters: {num_params:,}")
    print(f"  Model size: ~{num_params * 4 / 1e6:.1f} MB (FP32)")
    
    # Load datasets
    print("\n4. Loading Datasets...")
    data_path = "data/raw/combined_dataset.txt"
    
    if not Path(data_path).exists():
        print(f"[WARN] Dataset not found: {data_path}")
        print("First prepare the dataset!")
        return None
    
    # Full dataset
    full_dataset = TextDataset(
        data_path,
        tokenizer,
        max_length=model_config.max_position_embeddings
    )
    
    # Split train/val (95/5)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )
    
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")
    
    # Training configuration
    print("\n5. Training Configuration...")
    train_config = {
        'learning_rate': 3e-4,
        'min_learning_rate': 3e-5,
        'beta1': 0.9,
        'beta2': 0.95,
        'weight_decay': 0.1,
        'warmup_ratio': 0.05,
        'batch_size': 16,
        'gradient_accumulation_steps': 2,  # Effective batch: 16
        'num_epochs': 5,
        'max_grad_norm': 1.0,
        'use_amp': torch.cuda.is_available(),  # Mixed precision if GPU
        'save_every': 2,
        'output_dir': 'models/checkpoints'
    }
    
    # Create output directory
    Path(train_config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    print(f"  Learning rate: {train_config['learning_rate']}")
    print(f"  Batch size: {train_config['batch_size']}")
    print(f"  Gradient accumulation: {train_config['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {train_config['batch_size'] * train_config['gradient_accumulation_steps']}")
    print(f"  Epochs: {train_config['num_epochs']}")
    print(f"  Mixed precision: {train_config['use_amp']}")
    
    # Save configs
    with open(Path(train_config['output_dir']) / 'model_config.json', 'w') as f:
        json.dump(model_config.__dict__, f, indent=2)
    
    with open(Path(train_config['output_dir']) / 'train_config.json', 'w') as f:
        json.dump(train_config, f, indent=2)
    
    print("\n[OK] Configuration saved")
    
    return {
        'model': model,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'train_config': train_config,
        'device': device,
        'tokenizer': tokenizer
    }


def main():
    """Main training execution"""
    
    # Setup
    setup_result = setup_training()
    
    if setup_result is None:
        print("\n[WARN] Setup failed. Check errors above.")
        return
    
    model = setup_result['model']
    train_dataset = setup_result['train_dataset']
    val_dataset = setup_result['val_dataset']
    train_config = setup_result['train_config']
    device = setup_result['device']
    
    monitor = TrainingMonitor(use_wandb=True)
    monitor.log_model_info(model) # Model barada maglumatlary hasaba alýar
    
    # Create trainer
    print("\n" + "=" * 70)
    print("INITIALIZING TRAINER")
    print("=" * 70)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        device=device,
        monitor=monitor
    )
    
    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print("\nPress Ctrl+C to stop training and save checkpoint\n")
    
    try:
        trainer.train()
        monitor.plot_training_curves()
        monitor.generate_report(model, setup_result['tokenizer'], device)
    except KeyboardInterrupt:
        print("\n\n[WARN] Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(
            Path(train_config['output_dir']) / 'interrupted_checkpoint.pt',
            epoch=-1,
            val_loss=trainer.best_val_loss
        )
    
    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")
    print(f"Model saved in: {train_config['output_dir']}")
    
    # Generate sample text
    print("\n" + "=" * 70)
    print("SAMPLE GENERATION")
    print("=" * 70)
    
    generate_sample(
        model,
        setup_result['tokenizer'],
        device,
        prompt="Once upon a time"
    )


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="Hello", max_length=50):
    """Generate sample text"""
    
    model.eval()
    
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids]).to(device)
    
    print(f"\nPrompt: {prompt}")
    print("Generated: ", end="")
    
    # Generate
    for _ in range(max_length):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        
        # Sample from distribution
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decode and print
        token_str = tokenizer.decode([next_token.item()])
        print(token_str, end="", flush=True)
        
        # Stop at EOS
        if next_token.item() == tokenizer.token_to_id("</s>"):
            break
    
    print("\n")


if __name__ == "__main__":
    main()