# Architecture

## Overview

The SLM project builds a decoder-only Transformer language model from scratch.
The final system is not just a model — it is a complete AI assistant pipeline.

## System Layers

```
User Input
    │
    ▼
Safety Pre-filter
    │
    ▼
Router (rule-based + classifier)
    │
    ├──► SLM (direct generation)
    ├──► Web Search → Retrieve → Filter → SLM
    ├──► RAG (document retrieval) → SLM
    └──► Tools (calculator, code executor) → SLM
    │
    ▼
Context Builder
    │
    ▼
SLM (final generation)
    │
    ▼
Post-filter / Verification
    │
    ▼
User Response
```

## Model Architecture: Decoder-Only Transformer

### Component Breakdown

| Component | Purpose |
|---|---|
| Token Embeddings | Map token IDs to dense vectors |
| RoPE (Rotary Position Encoding) | Encode position without learned embeddings |
| Causal Self-Attention | Each token attends only to past tokens |
| Feed-Forward Network (FFN) | Per-token transformation (4x hidden dim) |
| RMSNorm | Normalization (faster than LayerNorm, no bias) |
| Residual Connections | Enable gradient flow through depth |
| LM Head | Project hidden state → vocabulary logits |

### Why Decoder-Only?

- All target tasks are generative (predict next token)
- Simpler than encoder-decoder; easier to scale
- Matches GPT, Llama, Mistral architecture family
- Pretraining objective (next-token prediction) is natural

### Why RoPE?

- No upper limit on sequence length (unlike learned PE)
- Relative position encoded directly in attention
- State-of-the-art in modern SLMs/LLMs

### Why RMSNorm over LayerNorm?

- Faster (no mean subtraction, no bias)
- Pre-norm placement (before attention/FFN) prevents vanishing gradients

## Model Configurations

| Config | Params | Layers | d_model | Heads | FFN | Context | Min VRAM |
|---|---|---|---|---|---|---|---|
| tiny | ~5M | 4 | 256 | 4 | 1024 | 512 | <1 GB |
| 50m | ~50M | 8 | 512 | 8 | 2048 | 1024 | ~2 GB |

> Additional configs (100m, 150m, 300m) will be added after hardware is confirmed.

## Data Flow

```
raw/           → immutable source files
    │
    ▼ (validation)
interim/       → after format normalization
    │
    ▼ (cleaning, dedup, filtering)
processed/     → versioned, ready for tokenizer
    │
    ▼ (tokenization)
tokenized/     → token ID sequences (memory-mapped)
    │
    ▼ (train/val/test split)
dataset        → DataLoader → Trainer
```

## Key Design Decisions

### No Silent Failures
Every component validates its inputs before processing.
Missing files, bad configs, NaN loss — all raise named exceptions immediately.

### Immutable Raw Data
The `data/raw/` directory is never modified after download.
All transformations produce outputs in `data/interim/` or `data/processed/`.

### Versioned Artifacts
Tokenizer, dataset, and model all carry explicit version tags.
Incompatible version combinations are rejected at load time.

### Config-Driven, Not Code-Driven
Model size, learning rate, batch size — nothing is hardcoded.
All values live in `configs/*.yaml` and are validated at startup.
