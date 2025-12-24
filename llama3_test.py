import torch
import torch.nn.functional as F
from pathlib import Path
import json
from llama3_architecture import LlamaModel, ModelConfig
from llama3_tokenizer import load_tokenizer
import inspect

class LlamaInference:
    """Tälim berlen Llama 3 modelini barlamak we ulanmak üçin professional klass"""
    
    def __init__(self, checkpoint_path, tokenizer_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Tokenizer-y ýüklemek
        self.tokenizer, self.tokenizer_config = load_tokenizer(tokenizer_path)
        
        # 2. Checkpoint-y we konfigurasiýany ýüklemek
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_config_dict = checkpoint['config']
        
        # Professional süzgüçleme: Diňe ModelConfig-e degişli bolan açarlary saýlap alýarys
        model_config_keys = inspect.signature(ModelConfig).parameters.keys()
        model_config_dict = {k: v for k, v in model_config_dict.items() if k in model_config_keys}

        # ModelConfig obýektini döretmek
        config = ModelConfig(**model_config_dict)
        
        # 3. Modeli gurmak we agramlary ýüklemek
        self.model = LlamaModel(config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval() # Modeli test düzgünine geçirmek
        
        print(f"[OK] Model ýüklendi: {checkpoint_path}")
        print(f"     Device: {self.device}")

    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 100, 
        temperature: float = 0.8, 
        top_k: int = 50,
        eos_token: str = "</s>"
    ):
        """Tekst generasýasy (Inference)"""
        
        # Prompt-y tokenlere öwürmek
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(self.device)
        
        generated_ids = []
        eos_id = self.tokenizer.token_to_id(eos_token)
        
        for _ in range(max_new_tokens):
            # Model arkaly çaklama
            logits = self.model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature # Temperatura sazlamasy
            
            # Top-K Sampling
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            # Ähtimallyk boýunça saýlamak
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Täze tokeni giriş maglumatyna goşmak
            input_ids = torch.cat([input_ids, next_token], dim=1)
            token_id = next_token.item()
            
            if token_id == eos_id: # EOS (End of Sentence) gelende durmak
                break
                
            generated_ids.append(token_id)
            
        # Netijäni dekodlamak
        return self.tokenizer.decode(generated_ids)

def test_model():
    # Faýl ýollaryny kesitlemek
    CHECKPOINT = "models/checkpoints/best_model.pt"
    TOKENIZER = "data/tokenizer/tokenizer.json"
    
    if not Path(CHECKPOINT).exists():
        print(f"[ERROR] Checkpoint tapylmady: {CHECKPOINT}")
        return

    # Inferences-y başlatmak
    inference = LlamaInference(CHECKPOINT, TOKENIZER)
    
    # Test soraglary (Prompts)
    prompts = [
        "Ben loved to explore",
        "little boy named Tim went to the park",
        "Once upon a time there was a friendly"
    ]
    
    print("\n" + "="*50)
    print("TEST NETIJELERI")
    print("="*50)
    
    for p in prompts:
        print(f"\nPROMPT: {p}")
        output = inference.generate(p, max_new_tokens=100, temperature=0.7)
        print(f"GENERATED: {output}")
        print("-" * 30)

if __name__ == "__main__":
    test_model()