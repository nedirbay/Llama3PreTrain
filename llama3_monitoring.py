"""
Llama 3 Pre-training - Monitoring and Evaluation
WandB integration, metrics tracking, visualization
"""

import torch
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime


class TrainingMonitor:
    """Monitor and visualize training process"""
    
    def __init__(self, project_name="llama3-pretraining", use_wandb=False):
        self.use_wandb = use_wandb
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'perplexity': [],
            'learning_rate': [],
            'grad_norm': [],
            'epoch': []
        }
        
        if use_wandb:
            wandb.init(
                project=project_name,
                config={
                    "architecture": "Llama 3",
                    "dataset": "custom 10MB",
                }
            )
            print("[OK] WandB initialized")
    
    def log_metrics(self, metrics, step):
        """Log metrics"""
        
        # Save to history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Log to WandB
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_model_info(self, model):
        """Log model information"""
        
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            'total_parameters': num_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': num_params * 4 / 1e6  # FP32
        }
        
        if self.use_wandb:
            wandb.config.update(info)
        
        return info
    
    def log_gpu_stats(self):
        """Log GPU statistics"""
        
        if not torch.cuda.is_available():
            return {}
        
        stats = {
            'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1e6,
            'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1e6,
            'gpu_utilization': torch.cuda.utilization()
        }
        
        return stats
    
    def plot_training_curves(self, save_path="logs/training_curves.png"):
        """Plot training curves"""
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if self.metrics_history['train_loss']:
            axes[0, 0].plot(self.metrics_history['train_loss'], label='Train Loss')
            axes[0, 0].plot(self.metrics_history['val_loss'], label='Val Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training & Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Perplexity
        if self.metrics_history['perplexity']:
            axes[0, 1].plot(self.metrics_history['perplexity'])
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Perplexity')
            axes[0, 1].set_title('Validation Perplexity')
            axes[0, 1].grid(True)
        
        # Learning rate
        if self.metrics_history['learning_rate']:
            axes[1, 0].plot(self.metrics_history['learning_rate'])
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True)
        
        # Gradient norm
        if self.metrics_history['grad_norm']:
            axes[1, 1].plot(self.metrics_history['grad_norm'])
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Training curves saved: {save_path}")
        
        if self.use_wandb:
            wandb.log({"training_curves": wandb.Image(save_path)})
    
    def save_metrics(self, save_path="logs/metrics.json"):
        """Save metrics to JSON"""
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"[OK] Metrics saved: {save_path}")
    
    def generate_report(self, model, tokenizer, device):
        """Generate training report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': self.log_model_info(model),
            'training_summary': {
                'final_train_loss': self.metrics_history['train_loss'][-1] if self.metrics_history['train_loss'] else None,
                'final_val_loss': self.metrics_history['val_loss'][-1] if self.metrics_history['val_loss'] else None,
                'best_val_loss': min(self.metrics_history['val_loss']) if self.metrics_history['val_loss'] else None,
                'final_perplexity': self.metrics_history['perplexity'][-1] if self.metrics_history['perplexity'] else None,
                'total_epochs': len(self.metrics_history['epoch'])
            }
        }
        
        # Generate sample texts
        samples = self.generate_samples(model, tokenizer, device)
        report['sample_generations'] = samples
        
        # Save report
        report_path = "logs/training_report.json"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[OK] Training report saved: {report_path}")
        
        return report
    
    @torch.no_grad()
    def generate_samples(self, model, tokenizer, device, num_samples=5):
        """Generate sample texts"""
        
        model.eval()
        
        prompts = [
            "Once upon a time",
            "The meaning of life is",
            "In the future, technology will",
            "The most important thing in life is",
            "Artificial intelligence can"
        ]
        
        samples = []
        
        for prompt in prompts[:num_samples]:
            # Encode
            encoded = tokenizer.encode(prompt)
            input_ids = torch.tensor([encoded.ids]).to(device)
            
            # Generate
            generated = []
            for _ in range(30):
                logits = model(input_ids)
                next_token_logits = logits[:, -1, :]
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Decode
                token_str = tokenizer.decode([next_token.item()])
                generated.append(token_str)
                
                # Stop at EOS
                if next_token.item() == tokenizer.token_to_id("</s>"):
                    break
            
            samples.append({
                'prompt': prompt,
                'generated': ''.join(generated)
            })
        
        return samples
    
    def close(self):
        """Close monitor"""
        
        if self.use_wandb:
            wandb.finish()
            print("[OK] WandB finished")


class ModelEvaluator:
    """Model evaluation we testing"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def evaluate_perplexity(self, dataset):
        """Calculate perplexity"""
        
        self.model.eval()
        
        from torch.utils.data import DataLoader
        import torch.nn.functional as F
        
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        total_loss = 0
        total_tokens = 0
        
        for input_ids in loader:
            input_ids = input_ids.to(self.device)
            
            logits = self.model(input_ids)
            
            # Shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum',
                ignore_index=self.tokenizer.token_to_id("<pad>")
            )
            
            total_loss += loss.item()
            total_tokens += shift_labels.ne(self.tokenizer.token_to_id("<pad>")).sum().item()
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        
        return perplexity.item()
    
    @torch.no_grad()
    def generate_text(self, prompt, max_length=100, temperature=0.8, top_k=50):
        """Generate text (advanced)"""
        
        self.model.eval()
        
        # Encode
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids]).to(self.device)
        
        generated_tokens = []
        
        for _ in range(max_length):
            logits = self.model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token_idx)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_tokens.append(next_token.item())
            
            # Stop at EOS
            if next_token.item() == self.tokenizer.token_to_id("</s>"):
                break
        
        # Decode
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text


if __name__ == "__main__":
    print("=" * 60)
    print("MONITORING & EVALUATION UTILITIES")
    print("=" * 60)
    
    print("\nAbove are TrainingMonitor and ModelEvaluator classes:")
    print("\n1. TrainingMonitor:")
    print("   - Metrics tracking")
    print("   - WandB integration")
    print("   - Training curves")
    print("   - GPU monitoring")
    print("   - Report generation")
    
    print("\n2. ModelEvaluator:")
    print("   - Perplexity evaluation")
    print("   - Text generation")
    print("   - Top-k sampling")
    print("   - Temperature control")
    
    print("\nUsage example:")
    print("""
# Training loop-da:
monitor = TrainingMonitor(use_wandb=True)

for epoch in range(num_epochs):
    # ... training ...
    
    monitor.log_metrics({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'perplexity': perplexity,
        'learning_rate': lr
    }, step=global_step)

monitor.plot_training_curves()
monitor.generate_report(model, tokenizer, device)
    """)