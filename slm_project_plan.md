# SLM Project — Master Plan & Feasibility Assessment

> **Status:** Phase 0 — Requirements & Planning  
> **Date:** 2026-10-01  
> **Author:** Antigravity (Senior ML/SLM Engineer)

---

## Table of Contents

1. [Conceptual Definitions](#1-conceptual-definitions)
2. [Scope Definition](#2-scope-definition)
3. [Feasibility Assessment](#3-feasibility-assessment)
4. [Proposed Architecture](#4-proposed-architecture)
5. [Development Phases](#5-development-phases)
6. [Hardware Discovery Requirements](#6-hardware-discovery-requirements)
7. [Software Prerequisites](#7-software-prerequisites)
8. [Dataset Strategy](#8-dataset-strategy)
9. [Major Risks](#9-major-risks)
10. [Testing & Error-Handling Strategy](#10-testing--error-handling-strategy)
11. [Recommended First Model Size](#11-recommended-first-model-size)
12. [First Exact Setup Step](#12-first-exact-setup-step)

---

## 1. Conceptual Definitions

### What is an SLM?
A **Small Language Model** is a transformer-based neural network trained on large text corpora to predict the next token. "Small" is contextual but generally refers to models in the **1M–7B parameter** range. SLMs are designed to run on commodity hardware (consumer GPUs, even CPUs). They trade raw capability for efficiency, deployability, and cost. For this project, the target is a **50M–300M parameter** decoder-only transformer.

### What is an LLM?
A **Large Language Model** is the same class of model, but with **7B–1T+ parameters**, trained on vastly more data, requiring many high-end GPUs. LLMs (GPT-4, Llama-3, Claude) emerge capabilities that SLMs cannot match without augmentation (RAG, tools). We are explicitly **not** building an LLM.

### What is a Foundation Model?
A **foundation model** is a model trained broadly on diverse internet-scale data (general purpose), intended to be fine-tuned or adapted for downstream tasks. It is the "base" before instruction tuning or specialization. Our pretrained SLM will be a small foundation model.

### What is an AI Assistant?
An **AI assistant** is a *system*, not just a model. It combines:
- A language model (the "brain")
- A router (decides what to do)
- Tools (calculator, search, code runner)
- Memory or retrieval (RAG)
- An interface (CLI, API, UI)

The model alone is not an assistant. The assistant is what we are ultimately building around the SLM.

### What is RAG?
**Retrieval-Augmented Generation** is the technique of fetching relevant documents from an external knowledge base at inference time and injecting them into the model's context before generation. It allows the model to answer questions grounded in documents it was never trained on, without retraining. Architecture: `Query → Embed → Vector Search → Top-K Docs → Context → LM → Answer`.

### What is an AI Agent?
An **AI agent** is a system where the language model itself orchestrates multi-step workflows: it decides which tools to call, interprets results, and loops until a goal is satisfied. Agents are more complex than assistants. We will **not** build a full agent in early phases — we will build a controlled router+tool system instead.

### What is Tool Calling?
**Tool calling** (also called function calling) is the ability of an LM to emit structured requests (e.g., JSON) to invoke predefined external functions — a calculator, a search API, a code executor — and receive results back. The model does not execute code itself; it requests execution via a structured interface.

### What is Web Retrieval?
**Web retrieval** is the act of querying a live search engine (Bing, SerpAPI, DuckDuckGo, etc.) to fetch current, real-time information. Unlike RAG (which uses a static offline corpus), web retrieval is dynamic. It is used when the model's training data is stale or insufficient.

### How These Components Differ

| Component | Static/Dynamic | Training Needed? | External Data? | Orchestration |
|---|---|---|---|---|
| SLM | Static (weights) | YES | NO | NO |
| RAG | Static corpus | NO (embed only) | YES (offline) | Partial |
| Web Search | Dynamic | NO | YES (live) | Partial |
| Tool Calling | Dynamic | NO (format only) | YES (API) | YES |
| AI Agent | Dynamic | NO | YES | Full |
| AI Assistant | Mixed | NO | Mixed | Partial |

---

## 2. Scope Definition

### A. Capabilities Provided by the SLM (The Model Itself)
- Normal conversation / dialogue
- Basic question answering (from training knowledge)
- Grammar correction
- Spelling and punctuation correction
- Text rewriting / paraphrasing
- Summarization
- Simple reasoning (chain-of-thought style)
- Basic arithmetic (learned patterns, not reliable)
- Simple code generation (Python, SQL primarily)
- Code explanation
- Simple code debugging
- Instruction following

### B. Capabilities Provided by External Tools
- Reliable arithmetic → **Calculator tool**
- Current events / live information → **Web search tool**
- Code execution with output → **Sandboxed Python executor**
- File operations → **File retrieval tool** (Phase 23+)

### C. Capabilities Provided by RAG
- Source-grounded answers from a local document corpus
- Retrieval of relevant information from uploaded documents
- Answering questions about private/proprietary documents

### D. Capabilities Provided by the Application Layer
- Intent routing (grammar? search? code? RAG?)
- Session/conversation memory
- Safety filtering (pre/post generation)
- Source citation formatting
- Response verification
- API rate limiting
- User authentication (if deployed)

---

### MVP Specification

#### Must-Have (MVP — Phase 0–16)
- [ ] Working tokenizer (BPE, vocabulary ~16K–32K)
- [ ] Decoder-only transformer (50M parameters minimum)
- [ ] Pretraining on a small but clean corpus (~1–5GB text)
- [ ] Instruction tuning (SFT on curated instruction data)
- [ ] Grammar / spelling correction capability
- [ ] Basic Q&A capability
- [ ] Simple conversation capability
- [ ] CLI inference interface
- [ ] Checkpoint save/resume
- [ ] Hardware validation script
- [ ] Reproducible environment

#### Should-Have (Phase 17–24)
- [ ] Simple code generation (Python)
- [ ] Code explanation
- [ ] Calculator tool integration
- [ ] Web search integration (SerpAPI or DuckDuckGo)
- [ ] Basic RAG over a small document store
- [ ] REST API (FastAPI)
- [ ] Structured logging and experiment tracking

#### Future / Later Phases (Phase 25+)
- [ ] Web UI (Gradio or custom)
- [ ] Multi-turn memory
- [ ] Reranking in RAG
- [ ] Full agent loop
- [ ] C/C++/Java/JS code generation
- [ ] Safety classifier integration
- [ ] Production deployment (Docker, cloud)

#### Explicitly Out of Scope (This Project)
- Training a model larger than ~300M parameters locally (hardware-dependent)
- Competing with GPT-4 / Claude / Gemini in capability
- Multimodal (vision, audio)
- RLHF / PPO training (compute-intensive; future work)
- Real-time voice interface
- Training on proprietary or unlicensed data
- Any form of user data collection or telemetry

---

## 3. Feasibility Assessment

### Overall Verdict: FEASIBLE with hardware-dependent constraints

| Aspect | Assessment | Notes |
|---|---|---|
| Tokenizer training | ✅ Fully feasible | CPU-only, fast |
| Tiny model (5M–20M) | ✅ Fully feasible | GTX-class GPU adequate |
| 50M model pretraining | ✅ Feasible | ~1–3 days on GTX 1080/3060 |
| 100M model pretraining | ⚠️ Feasible with care | VRAM is the bottleneck |
| 300M model pretraining | ⚠️ May require cloud GPU | Depends on VRAM |
| Instruction tuning (LoRA) | ✅ Feasible | Very VRAM-efficient |
| RAG pipeline | ✅ Fully feasible | CPU-capable with FAISS |
| Web search integration | ✅ Fully feasible | API-dependent |
| REST API deployment | ✅ Fully feasible | FastAPI on local machine |

> **CRITICAL:** The exact feasibility of 100M+ pretraining depends on your GPU VRAM.  
> This will be determined in Phase 1 by `hardware_check.py`.  
> **We will not commit to a model size until the hardware is confirmed.**

---

## 4. Proposed Architecture

### Repository Layout

```
slm-project/
│
├── configs/                    # YAML configs (model, training, data)
│   ├── model_tiny.yaml
│   ├── model_50m.yaml
│   ├── model_100m.yaml
│   └── training.yaml
│
├── data/
│   ├── raw/                    # IMMUTABLE — never modified after download
│   ├── interim/                # After initial cleaning
│   ├── processed/              # Final, versioned datasets
│   └── validation/             # Validation reports per dataset version
│
├── tokenizer/
│   ├── train_tokenizer.py
│   ├── tokenizer_tests.py
│   └── artifacts/              # Trained tokenizer files
│
├── model/
│   ├── architecture.py         # Transformer definition
│   ├── config.py               # ModelConfig dataclass
│   └── utils.py
│
├── training/
│   ├── trainer.py              # Main training loop
│   ├── optimizer.py
│   ├── scheduler.py
│   └── monitor.py              # NaN/OOM/gradient detection
│
├── finetuning/
│   ├── sft_trainer.py          # Supervised fine-tuning
│   └── lora.py                 # LoRA adapter
│
├── evaluation/
│   ├── eval_lm.py              # Perplexity
│   ├── eval_grammar.py
│   ├── eval_qa.py
│   ├── eval_code.py
│   └── regression_suite.py     # Fixed regression test set
│
├── inference/
│   ├── generate.py             # Core generation
│   └── cli.py                  # CLI interface
│
├── tools/
│   ├── base_tool.py            # Abstract tool interface
│   ├── calculator.py
│   ├── web_search.py
│   └── code_executor.py        # Sandboxed executor only
│
├── rag/
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── retriever.py
│
├── router/
│   └── router.py               # Intent classification + dispatch
│
├── api/
│   ├── main.py                 # FastAPI app
│   ├── schemas.py
│   └── middleware.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
│
├── scripts/
│   ├── hardware_check.py       # Phase 1 — FIRST script to run
│   ├── setup_env.sh / .ps1
│   └── run_smoke_test.py
│
├── notebooks/                  # Exploration only — NOT production code
│
├── logs/                       # Structured JSON logs
├── checkpoints/                # Model checkpoints (git-ignored)
│
├── docs/
│   ├── architecture.md
│   ├── setup.md
│   ├── data.md
│   ├── training.md
│   ├── evaluation.md
│   ├── troubleshooting.md
│   ├── deployment.md
│   └── experiments.md
│
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── README.md
└── Dockerfile
```

### Final System Architecture (Target State)

```
                        USER
                          │
                          ▼
                   CLI / REST API / UI
                          │
                          ▼
                ┌─── SAFETY FILTER ───┐
                │         │           │
                │         ▼           │
                │      ROUTER         │
                │   (rule + classifier)│
                │         │           │
                │  ┌──────┼──────┐    │
                │  ▼      ▼      ▼    │
                │ SLM  WebSearch Tools │
                │  │      │      │    │
                │  │   Retrieval  │    │
                │  │      │      │    │
                │  └──────┴──────┘    │
                │         │           │
                │    Context Builder  │
                │         │           │
                │         ▼           │
                │       SLM           │
                │    (generate)       │
                │         │           │
                │   Post-Verification │
                └─────────┼───────────┘
                          ▼
                        USER
```

---

## 5. Development Phases

| Phase | Name | Gate Condition |
|---|---|---|
| **0** | Requirements & Planning | This document reviewed and accepted |
| **1** | Hardware & Environment Validation | `hardware_check.py` PASS, env tests PASS |
| **2** | Repository & Configuration Setup | Structure created, configs validated |
| **3** | Data Source Discovery | Sources identified, licenses verified |
| **4** | Data Ingestion | At least 1 source downloaded, checksummed |
| **5** | Data Validation | Validation report generated, thresholds met |
| **6** | Data Cleaning | Cleaning pipeline tested, no raw data mutated |
| **7** | Deduplication | Duplicate count reported, removed |
| **8** | Dataset Versioning | `dataset_v001` tagged, manifest created |
| **9** | Tokenizer | Train/test PASS, regression tests PASS |
| **10** | Tiny Transformer (5M–20M) | Architecture instantiated, param count verified |
| **11** | Tiny Model Training Test | 1 batch → forward → loss → backward → step PASS |
| **12** | Checkpoint & Recovery | Save → stop → reload → continue PASS |
| **13** | 100M-class SLM | Architecture verified, param count confirmed |
| **14** | Pretraining | Loss converges, no NaN, checkpoints valid |
| **15** | Evaluation Baseline | Perplexity measured, regression suite created |
| **16** | Instruction Tuning (SFT + LoRA) | Grammar/Q&A/code eval PASS vs baseline |
| **17** | Grammar Capability | Grammar eval before/after PASS |
| **18** | Code Capability | Code eval (Python) PASS |
| **19** | Safety Filtering | Basic safety eval PASS |
| **20** | Inference Optimization | Quantization / KV-cache benchmarked |
| **21** | Tool Calling | Calculator, code executor integrated & tested |
| **22** | Web Search | Search success/failure/fallback tested |
| **23** | RAG | Retrieval precision/recall measured |
| **24** | Router | Router accuracy measured on fixed test set |
| **25** | REST API | FastAPI integration test PASS |
| **26** | Web UI | End-to-end manual test PASS |
| **27** | End-to-End Testing | All regression tests PASS |
| **28** | Deployment | Docker build, startup, inference, shutdown PASS |

---

## 6. Hardware Discovery Requirements

Before any code is written, the following must be confirmed:

### What We Need to Know

| Item | Why It Matters |
|---|---|
| GPU model | Determines architecture-level CUDA support |
| GPU VRAM (GB) | Directly caps maximum model size in training |
| CPU cores & threads | Determines DataLoader worker count |
| RAM (GB) | Caps dataset in-memory processing |
| CUDA version | Must match PyTorch CUDA build |
| PyTorch version | Must match CUDA; determines feature availability |
| Disk space (free) | Datasets + checkpoints can be 10s of GB |
| OS + Python version | Environment reproducibility |
| NVMe vs HDD | Affects DataLoader I/O throughput |

### Decision Table (VRAM → Model Size)

| VRAM | Max Practical Model (bf16/fp16) | Training Strategy |
|---|---|---|
| < 4 GB | ~20M parameters | batch=1, grad accum |
| 4–6 GB | ~50M parameters | batch=2–4, grad accum |
| 6–8 GB | ~100M parameters | batch=4–8, mixed precision |
| 8–12 GB | ~150M–200M parameters | mixed precision |
| 12–16 GB | ~300M–500M parameters | mixed precision |
| 16–24 GB | ~1B+ parameters | mixed precision, LoRA for larger |

> **This table is a guide. The actual usable batch size depends on sequence length and architecture.**  
> **We will measure empirically after hardware check.**

---

## 7. Software Prerequisites

### Required (Must Have Before Phase 1)

| Software | Minimum Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| pip | 23+ | Package management |
| git | 2.x | Version control |
| CUDA Toolkit | 11.8+ (if GPU) | GPU computation |
| cuDNN | Matching CUDA | Deep learning ops |
| NVIDIA Driver | 520+ (if GPU) | GPU driver |

### Python Packages (to be pinned in requirements.txt)

| Package | Purpose |
|---|---|
| `torch` | Core deep learning |
| `transformers` | Tokenizer, model utilities |
| `datasets` | HuggingFace dataset loading |
| `tokenizers` | Fast BPE tokenizer training |
| `sentencepiece` | Alternative tokenizer |
| `numpy` | Array operations |
| `tqdm` | Progress bars |
| `pyyaml` | Config file parsing |
| `rich` | Beautiful terminal output |
| `loguru` | Structured logging |
| `pytest` | Testing framework |
| `psutil` | System resource monitoring |
| `gputil` | GPU monitoring |
| `faiss-cpu` or `faiss-gpu` | Vector similarity search (RAG) |
| `fastapi` + `uvicorn` | REST API |
| `httpx` | HTTP client for tools |
| `python-dotenv` | Environment variable management |

### Dev-Only Packages (requirements-dev.txt)

| Package | Purpose |
|---|---|
| `pytest-cov` | Coverage reporting |
| `black` | Code formatting |
| `flake8` | Linting |
| `mypy` | Type checking |
| `ipykernel` | Jupyter notebooks |
| `tensorboard` | Training visualization |

---

## 8. Dataset Strategy

### Recommended Sources for Each Capability

| Capability | Dataset | License | Size | Notes |
|---|---|---|---|---|
| General text | [The Pile](https://pile.eleuther.ai/) | Multiple (mixed) | 825 GB | Use subsets only — verify per subset |
| Books | [Project Gutenberg](https://www.gutenberg.org/) | Public Domain | ~60K books | Free, verifiable PD |
| Wikipedia | [Wikipedia dumps](https://dumps.wikimedia.org/) | CC-BY-SA 3.0 | ~20 GB text | Attribution required |
| Web text | [C4 (Colossal Clean Crawled Corpus)](https://huggingface.co/datasets/allenai/c4) | ODC-BY | ~300 GB | Use `en` subset |
| Code | [The Stack](https://huggingface.co/datasets/bigcode/the-stack) | Various OSS | ~3 TB | Use permissive licenses only |
| Code (small) | [CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net) | Multiple | ~20 GB | Python/Java/JS/Go/PHP/Ruby |
| Instruction | [FLAN collection](https://huggingface.co/datasets/Muennighoff/flan) | Apache 2.0 | Large | Instruction-following |
| Instruction (small) | [Alpaca cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | CC-BY-NC 4.0 | 52K samples | Non-commercial only |
| Grammar | [JFLEG](https://huggingface.co/datasets/jfleg) | CC-BY-NC | 1.5K sentences | Evaluation |
| Grammar training | [LANG-8](http://cl.naist.jp/nlp-software/lang-8-20111007-2.0.tgz) | Research only | 1M pairs | Non-commercial research |
| Math | [MATH dataset](https://huggingface.co/datasets/lighteval/MATH) | MIT | 12.5K problems | Evaluation |
| Conversation | [Daily Dialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog) | CC-BY-NC-SA 4.0 | 13K dialogues | Non-commercial |
| Q&A | [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) | CC-BY-SA 4.0 | 150K QA pairs | Attribution required |

### Dataset Strategy for MVP (Local Training on ~6–8 GB VRAM)

For a first training run, we will use a **dramatically smaller curated corpus**:

```
Target: ~1–5 GB of clean, deduplicated English text
Sources:
  - Wikipedia EN (subset: ~2 GB)
  - Project Gutenberg (subset: ~500 MB)
  - C4 EN (small subset: ~1 GB)
  - Alpaca cleaned (instruction tuning: small)

Total: ~3–4 GB text → ~800M–1B tokens at typical compression
```

> **Why small first?** Training a tiny model to convergence on a small corpus is faster, cheaper, and serves as a pipeline validation.  
> We scale data up only after the pipeline is verified end-to-end.

### License Philosophy
- **Never use data without verifying the license**
- **Never use NC (non-commercial) data for commercial deployments**
- **Always attribute CC-licensed data**
- **Keep a `data/licenses/` directory with license copies**

---

## 9. Major Risks

### Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| GPU VRAM too low for target model | HIGH | HIGH | Hardware check first; fall back to smaller model or cloud |
| CUDA/PyTorch incompatibility | MEDIUM | HIGH | Pin versions; test before training |
| Training NaN loss | MEDIUM | HIGH | Gradient clipping; learning rate warmup; early detection |
| Dataset licensing violation | LOW | VERY HIGH | License verification step mandatory |
| Disk space exhaustion during training | MEDIUM | MEDIUM | Pre-check; checkpoint rotation |
| Tokenizer incompatibility with model | LOW | HIGH | Tokenizer versioning; regression tests |
| Catastrophic forgetting after fine-tuning | MEDIUM | MEDIUM | Evaluate before/after; keep base checkpoint |
| Web search API rate limits / cost | MEDIUM | LOW | Use free APIs first (DuckDuckGo); implement backoff |
| Data contamination of test set | MEDIUM | HIGH | Deduplication before split; leakage checks |
| Training divergence after resuming checkpoint | LOW | HIGH | Verify checkpoint integrity; test resume explicitly |
| Out-of-memory during training | HIGH | MEDIUM | Gradient accumulation; smaller batch |
| Slow data loading bottleneck | MEDIUM | LOW | Pre-tokenize offline; memory-mapped datasets |

---

## 10. Testing & Error-Handling Strategy

### Test Taxonomy

```
Unit Tests       — test one function in isolation
Integration Tests — test two or more components together
Smoke Tests      — fast sanity check (1 batch, 1 request)
Regression Tests — fixed golden test set; run on every model change
```

### Phase Gates (Definition of Done)

Every phase must satisfy ALL of the following before moving on:

```
[ ] Implementation exists and is committed to git
[ ] Unit tests written and PASS
[ ] Integration test PASS (where applicable)
[ ] Smoke test PASS
[ ] Validation report generated
[ ] Errors are explicitly handled (no bare except)
[ ] Structured log output confirmed
[ ] Output artifact saved and versioned
[ ] Reproducibility confirmed (fixed seed)
[ ] Documentation updated
```

### Error Handling Contract

Every component must:
1. Raise a **specific, named exception** — never `Exception` or bare `except`
2. Include: **what failed**, **where**, **why**, **how to fix**
3. Log at appropriate level: `DEBUG / INFO / WARNING / ERROR / CRITICAL`
4. Never silently continue from a corrupted state
5. Fail fast on missing required inputs

### Status Labels Used Throughout

```
PASS         — verified to work
FAIL         — verified to be broken
WARNING      — works but with caveats
NOT TESTED   — not yet run
BLOCKED      — cannot proceed due to dependency
NOT APPLICABLE — irrelevant for this phase
```

---

## 11. Recommended First Model Size

### Recommendation: Start with a Tiny Model (5M–10M parameters)

**Why not start at 100M?**
- A tiny model trains in minutes, not days
- It validates the entire pipeline: data → tokenizer → model → training → checkpoint → inference
- If the pipeline is broken, you find out in 10 minutes, not 10 hours
- The architecture scales cleanly — the same code that runs 5M runs 100M

### Tiny Model Configuration (for pipeline validation)

```
layers:        4
hidden_dim:    256
attention_heads: 4
FFN multiplier: 4x (1024)
vocabulary:    16,384
context_length: 512
parameters:    ~5M
```

### Target SLM Configuration (after pipeline validation, hardware-dependent)

| Config | Params | Layers | Hidden | Heads | Context | VRAM (fp16) |
|---|---|---|---|---|---|---|
| Tiny | ~5M | 4 | 256 | 4 | 512 | < 1 GB |
| Small | ~50M | 8 | 512 | 8 | 1024 | ~2 GB |
| Medium | ~100M | 12 | 768 | 12 | 1024 | ~4 GB |
| Base | ~150M | 16 | 1024 | 16 | 2048 | ~6 GB |
| Target | ~300M | 24 | 1024 | 16 | 2048 | ~10–12 GB |

> **Final model size will be selected after `hardware_check.py` confirms your GPU VRAM.**

---

## 12. First Exact Setup Step

### Step 1 — Run `hardware_check.py`

This is the **very first thing** we will build and run.

**What it will check:**
- CPU model, cores, threads
- RAM (total and available)
- GPU model and VRAM (via `torch.cuda` and `gputil`)
- CUDA version
- PyTorch version and CUDA build match
- Python version
- Free disk space on target drive
- Operating system

**Expected output format:**
```
============================================================
SLM Hardware Diagnostic Report
============================================================
[PASS] CPU:        Intel Core i7-xxxx  (8 threads)
[PASS] RAM:        16.0 GB total | 12.3 GB available
[PASS] GPU:        NVIDIA GTX XXXX
[PASS] VRAM:       8.0 GB
[PASS] CUDA:       12.1
[PASS] PyTorch:    2.3.0+cu121
[PASS] Python:     3.11.4
[PASS] Disk (C:\): 150 GB free
[INFO] Recommended max model size: ~100M parameters (fp16)
============================================================
Overall Status: PASS — Safe to proceed to Phase 1
============================================================
```

**Before running it**, you will need Python 3.10+ and pip installed.

**Immediate next action:**
```
Tell me:
1. Your operating system (Windows 10 / 11?)
2. Your Python version (run: python --version)
3. Whether you have a Python virtual environment tool
   (venv / conda / pyenv?)
4. Whether PyTorch is already installed
5. Your approximate free disk space on the drive
   where this project will live
```

Once I have this information, I will generate `hardware_check.py`  
as the first and only deliverable of Phase 1.

---

## Phase Status Summary

| Phase | Name | Status |
|---|---|---|
| 0 | Requirements & Planning | 🔵 IN PROGRESS |
| 1 | Hardware & Environment | ⬜ NOT STARTED |
| 2–28 | All subsequent phases | ⬜ NOT STARTED |

---

*This document will be updated as each phase completes.*  
*Never mark a phase PASSED without evidence.*
