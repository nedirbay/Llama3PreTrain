# Llama 3 Pre-training from Scratch

This project implements a complete pipeline for pre-training a Llama 3 language model from scratch. It includes custom model architecture, BPE tokenizer training, dataset preparation, and a training loop with monitoring.

## 🚀 Features

- **Llama 3 Architecture**: Full implementation including Rotary Embeddings (RoPE), Grouped Query Attention (GQA), SwiGLU activation, and RMSNorm.
- **Custom Tokenizer**: Training a BPE tokenizer using the `tokenizers` library.
- **Dataset Pipeline**: downloading and processing datasets (WikiText, FineWeb-Edu, TinyStories) to reach a target size (~10MB for demo).
- **Training Loop**:
  - AdamW Optimizer
  - Cosine Learning Rate Schedule with Warmup
  - Mixed Precision Training (AMP)
  - Gradient Accumulation
- **Monitoring**: Weights & Biases (WandB) integration, loss plotting, and sample generation.

## 🚀 Quick Start (Gollanma)

```bash
# 1. Dependencies gurmak
pip install -r requirements.txt

# 2. Dataset taýýarlamak
python llama3_dataset.py

# 3. Tokenizer tälim bermek
python llama3_tokenizer.py

# 4. Model training başlamak
python llama3_main.py
```

## 📂 Project Structure

- `llama3_architecture.py`: PyTorch implementation of the Llama 3 model.
- `llama3_dataset.py`: Script to download and merge datasets.
- `llama3_tokenizer.py`: Script to train the tokenizer.
- `llama3_training.py`: The `Trainer` class handling the training loop.
- `llama3_monitoring.py`: Utilities for logging metrics and generating reports.
- `llama3_main.py`: The entry point script that ties everything together.
- `requirements.txt`: List of Python dependencies.

## ⚙️ Configuration

You can adjust training parameters in `llama3_main.py`:

- `vocab_size`: 32,000
- `hidden_size`: 768
- `num_layers`: 12
- `batch_size`: 8 or 16
- `learning_rate`: 3e-4

## ⚠️ Windows Users Note

If you encounter encoding issues, ensure your terminal supports UTF-8, though the scripts have been updated to use ASCII for maximum compatibility.
