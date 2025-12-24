"""
Llama 3 Model Pre-training - BPE Tokenizer Training
Açyk çeşme tokenizers library bilen BPE tokenizer döretmek
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFKC, Sequence
from pathlib import Path
import json

class LlamaTokenizerTrainer:
    """Train BPE Tokenizer for Llama 3"""
    
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        
    def create_tokenizer(self):
        """BPE tokenizer döretmek"""
        
        # BPE model
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        
        # Normalization: NFKC unicode normalization
        self.tokenizer.normalizer = NFKC()
        
        # Pre-tokenization: Byte-level split
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
        
        # Decoder
        self.tokenizer.decoder = decoders.ByteLevel()
        
        print("[OK] Tokenizer architecture created")
        
    def train(self, data_path, output_dir="data/tokenizer"):
        """Train tokenizer"""
        
        # Special tokens
        special_tokens = [
            "<unk>",      # Unknown token
            "<s>",        # Begin of sequence
            "</s>",       # End of sequence  
            "<pad>",      # Padding
            "<|endoftext|>" # end of text
        ]
        
        # Trainer configuration
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=special_tokens,
            min_frequency=2,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        
        # Dataset okamak
        files = [str(data_path)]
        
        print(f"\nTokenizer training started...")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Dataset: {data_path}")
        print(f"  Special tokens: {len(special_tokens)}")
        
        # Train
        self.tokenizer.train(files=files, trainer=trainer)
        
        # Post-processor: Add special tokens
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[
                ("<s>", self.tokenizer.token_to_id("<s>")),
                ("</s>", self.tokenizer.token_to_id("</s>")),
            ],
        )
        
        # Save tokenizer
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer.save(str(output_path / "tokenizer.json"))
        
        # Metadata save
        metadata = {
            "vocab_size": self.vocab_size,
            "model_type": "BPE",
            "special_tokens": special_tokens,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
        
        with open(output_path / "tokenizer_config.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[OK] Tokenizer ready: {output_dir}")
        
        return self.tokenizer
    
    def test_tokenizer(self, texts):
        """Tokenizer test etmek"""
        
        print("\n" + "=" * 60)
        print("TOKENIZER TEST")
        print("=" * 60)
        
        for text in texts:
            encoded = self.tokenizer.encode(text)
            
            print(f"\nOriginal: {text}")
            print(f"Tokens: {encoded.tokens}")
            print(f"IDs: {encoded.ids}")
            print(f"Token count: {len(encoded.ids)}")
            
            # Decode
            decoded = self.tokenizer.decode(encoded.ids)
            print(f"Decoded: {decoded}")
            print(f"Match: {'[OK]' if decoded.strip() == text.strip() else '[FAIL]'}")


def load_tokenizer(tokenizer_path="data/tokenizer/tokenizer.json"):
    """Tokenizer ýüklemek"""
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Config ýüklemek
    config_path = Path(tokenizer_path).parent / "tokenizer_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"[OK] Tokenizer loaded: {tokenizer_path}")
    print(f"  Vocab size: {config['vocab_size']}")
    
    return tokenizer, config


if __name__ == "__main__":
    print("=" * 60)
    print("LLAMA 3 TOKENIZER TRAINING")
    print("=" * 60)
    
    # Dataset path
    data_path = "data/raw/TinyStoriesV2-GPT4-train.txt"
    
    # Check dataset exists
    if not Path(data_path).exists():
        print(f"\n[WARN] Dataset not found: {data_path}")
        print("First run prepare_dataset.py!")
        exit(1)
    
    # Tokenizer döretmek we tälim bermek
    trainer = LlamaTokenizerTrainer(vocab_size=32000)
    trainer.create_tokenizer()
    tokenizer = trainer.train(data_path)
    
    # Test samples
    test_texts = [
        "Hello, world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Llama 3 is a powerful language model architecture."
    ]
    
    trainer.test_tokenizer(test_texts)
    
    print("\n" + "=" * 60)
    print("[OK] Tokenizer training completed!")
    print("=" * 60)
    print("\nTokenizer files:")
    print("  - data/tokenizer/tokenizer.json")
    print("  - data/tokenizer/tokenizer_config.json")