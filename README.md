# Zero-Shot Alignment via Retrieval

Align LLM outputs to user preferences by retrieving pre-trained style modules (LoRA adapters) at inference time, instead of fine-tuning per user. Given a natural language preference description (e.g., "be formal and academic"), the system retrieves the best-matching style adapter from a bank of pre-trained modules and composes it onto a base LLM to generate preference-aligned responses.

## Why not just fine-tune? And if it's "zero-shot," why is there a training step?

The term "zero-shot" here refers to what happens at inference time from the user's perspective. A new user shows up, describes what they want in plain English, and gets a styled response immediately. No fine-tuning, no data collection, no waiting. That part is zero-shot.

But the adapter bank itself has to exist first. We train a set of LoRA adapters offline, once, each one capturing a different writing style. Think of it like stocking a library: you write the books ahead of time so that readers can just walk in and pick one off the shelf. The training is a one-time cost to build the shelf. The retrieval is what makes it zero-shot for every user after that.

This matters because the alternative (fine-tuning a model per user) is expensive, slow, and doesn't scale. With retrieval, adding a new style just means training one more adapter and dropping it into the bank. Existing users aren't affected and no retraining of the full model is needed.

| Step | What happens | Runs when |
|---|---|---|
| Adapter training | Train 10 LoRA modules on curated style datasets | Once, offline |
| Index building | Embed style descriptions into FAISS index | Once, offline |
| Retrieval + generation | User query → nearest style → compose adapter → generate | Every request, zero-shot |

## Architecture

```
User Preference Query ──► Embedding Model ──► FAISS Index ──► Top-K Style Cards
        │                  (MiniLM-L6-v2)      (cosine sim)         │
        │                                                           │
        ▼                                                           ▼
   User Prompt ──────────────────────────────► Base LLM + LoRA ──► Styled Response
                                               (TinyLlama 1.1B)
```

### Components

| Component | Implementation | Purpose |
|---|---|---|
| Style Representation | Style Cards (JSONL) | Searchable descriptions with tags, instructions, and examples |
| Retrieval | Sentence embeddings + FAISS | Fast nearest-neighbor search over style descriptions |
| Adaptation | LoRA adapters (PEFT) | Lightweight style modules composed onto base model |
| Evaluation | Heuristic + LLM-as-judge scoring | Measures retrieval accuracy, style adherence, and win rates |

## Project Structure

```
├── run_pipeline.py              # Main entry point for all pipeline steps
├── requirements.txt             # Python dependencies
├── style_bank/
│   ├── style_cards.jsonl        # 10 style definitions with examples
│   └── adapters/                # Trained LoRA weights (created during training)
├── src/
│   ├── config.py                # Shared configuration (model, paths, hyperparameters)
│   ├── build_index.py           # Builds FAISS index from style cards
│   ├── retrieve.py              # Retrieves top-k styles via embedding similarity
│   ├── train_adapters.py        # Trains LoRA adapters for each style
│   ├── generate.py              # Generates responses with retrieved adapters
│   └── evaluate.py              # Evaluation: retrieval accuracy, style adherence, win rates
├── data/
│   ├── training/                # Curated JSONL datasets (20 examples per style)
│   └── ...                      # FAISS index and metadata (created during indexing)
└── results/                     # Evaluation results (created during evaluation)
```

## Setup

### Requirements

- Python 3.10+
- macOS with Apple Silicon (MPS) or a CUDA GPU
- ~8 GB RAM minimum

### Installation

```bash
git clone https://github.com/sumonesphantom/KRR-Zero-Shot-Alignment-via-Retrieval.git
cd KRR-Zero-Shot-Alignment-via-Retrieval
pip install -r requirements.txt
```

## Usage

### Run the Full Pipeline

```bash
python run_pipeline.py --step all
```

This runs all three steps sequentially: index building, adapter training, and evaluation.

### Run Individual Steps

```bash
# Step 1: Build the FAISS retrieval index from style cards
python run_pipeline.py --step index

# Step 2: Train LoRA adapters for all 10 styles
python run_pipeline.py --step train

# Step 3: Run the evaluation suite
python run_pipeline.py --step evaluate

# Optional: Add LLM-as-judge scoring (slower but more nuanced)
python run_pipeline.py --step evaluate --llm-judge

# Interactive demo: enter your own preferences and questions
python run_pipeline.py --step demo
```

### Interactive Demo

```bash
python run_pipeline.py --step demo
```

```
Your preference: explain things simply with fun analogies
Your question: How does Wi-Fi work?

Retrieving best style...
Top matches:
  #1 eli5_simple (score: 0.8234)
  #2 casual_friendly (score: 0.7102)
  #3 storytelling_narrative (score: 0.6543)

Generating with style: eli5_simple...

--- Response (eli5_simple) ---
Imagine your phone is sending invisible letters through the air...
```

## Implementation Details

### 1. Style Representation (Style Cards)

Each style is defined as a Style Card in `style_bank/style_cards.jsonl`:

```json
{
  "id": "formal_academic",
  "tags": ["formal", "academic", "detailed", "structured"],
  "instruction": "Answer in a formal academic tone. Use precise terminology...",
  "examples": [
    {
      "prompt": "Explain gradient descent.",
      "answer": "Gradient descent is a first-order iterative optimization..."
    }
  ],
  "adapter_path": "style_bank/adapters/formal_academic"
}
```

The 10 styles:

| Style | Tags | Description |
|---|---|---|
| `formal_academic` | formal, academic, detailed | Precise terminology, structured paragraphs |
| `casual_friendly` | casual, friendly, warm | Conversational, contractions, light humor |
| `concise_bullet` | concise, bullet points, minimal | Key facts only, no fluff |
| `eli5_simple` | simple, eli5, analogies | Explain like I'm 5, fun analogies |
| `technical_precise` | technical, precise, code-oriented | Specific details, formulas, numbers |
| `socratic_teaching` | socratic, teaching, questions | Guide understanding through questions |
| `storytelling_narrative` | storytelling, creative, engaging | Weave explanations into narratives |
| `professional_business` | professional, executive, actionable | ROI focus, strategic relevance |
| `empathetic_supportive` | empathetic, encouraging, patient | Warm, validating, gentle explanations |
| `debate_critical` | critical, analytical, balanced | Multiple perspectives, pros and cons |

### 2. Training Data

Each style has a curated JSONL dataset in `data/training/` with ~20 prompt-response pairs. The prompts include an explicit style tag (e.g., `Style: Formal Academic`) to anchor the LoRA to the right behavior and prevent style bleed. Topics are diverse — science, economics, technology, everyday tasks — so the adapter learns the style itself rather than memorizing subject-specific patterns.

The training code loads curated data first. If no curated file is found for a given style, it falls back to synthetic data generation using the base model.

### 3. Style Embedding and Retrieval

**Indexing** (`src/build_index.py`):
- Builds a text representation for each style card by combining its instruction, tags, and example Q&A pairs
- Encodes with `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings)
- Stores normalized embeddings in a FAISS `IndexFlatIP` index (inner product = cosine similarity on normalized vectors)

**Retrieval** (`src/retrieve.py`):
- Encodes the user's preference query with the same embedding model
- Performs FAISS nearest-neighbor search to find top-k matching styles
- Computes softmax weights over similarity scores (with temperature scaling) for potential weighted composition

```python
retriever = StyleRetriever()
results = retriever.retrieve("I want formal academic explanations", top_k=3)
# Returns: [(style_card, similarity_score, weight), ...]
```

### 4. LoRA Adapter Training

**Training** (`src/train_adapters.py`):
- Loads curated training data from `data/training/{style_id}.jsonl` (20 prompt-response pairs per style)
- Trains a LoRA adapter (rank=16, alpha=32) on the `q_proj`, `v_proj`, `k_proj`, `o_proj` attention layers
- Each adapter adds only ~4M trainable parameters vs 1.1B total — lightweight and composable

**LoRA Configuration:**
| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target Modules | q_proj, v_proj, k_proj, o_proj |
| Training Epochs | 3 |
| Learning Rate | 2e-4 |

### 5. Generation with Composed Adapters

**Generation** (`src/generate.py`):
- Loads the base model (TinyLlama 1.1B Chat) once
- For each request, loads the retrieved LoRA adapter on top
- Generates with the composed model (base + adapter)
- Supports comparison mode: generates base, retrieved-style, and random-style outputs for the same prompt

### 6. Evaluation

**Evaluation** (`src/evaluate.py`) measures three things:

#### a) Retrieval Accuracy
- Tests whether the retriever returns the correct style for 20 diverse preference queries
- Reports top-1 accuracy (exact match) and top-3 accuracy (correct style in top 3)

#### b) Style Adherence Scoring
Two scoring methods:

- Keyword heuristics: rule-based scoring per style (e.g., checking for bullet points in `concise_bullet`, question marks in `socratic_teaching`, formal vocabulary in `formal_academic`)
- LLM-as-judge (optional): uses the base model to rate style adherence on a 1-5 scale

#### c) Pairwise Win Rates
Compares outputs across three conditions:
- Retrieved adapter vs base model (no adapter)
- Retrieved adapter vs random adapter (wrong style)
- Random adapter vs base model

The retrieved adapter should win against both the base model and a random adapter.

#### Baselines

| Baseline | Description |
|---|---|
| Base model | TinyLlama 1.1B with no adapter applied |
| Random adapter | A randomly selected (wrong) style adapter |
| Retrieved adapter | Our method — style selected by retrieval |

## Configuration

All configurable parameters are in `src/config.py`:

```python
# Model
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4

# Retrieval
TOP_K = 5
TEMPERATURE = 0.1

# Generation
MAX_NEW_TOKENS = 256
```

To use a larger base model (e.g., Llama 3.1 8B, Mistral 7B), change `BASE_MODEL_NAME` and ensure you have sufficient VRAM/RAM.

## Expected Results

After running the full pipeline, results are saved to `results/evaluation_results.json` containing:

- Retrieval accuracy metrics
- Per-example style adherence scores for all three conditions
- Pairwise win rates
- Generated outputs for qualitative inspection

## Design Decisions

**Why retrieval instead of per-user fine-tuning?**
Fine-tuning a model for every new user is expensive and doesn't scale. With retrieval, you build the adapter bank once and every new user gets instant style matching. Adding a new style means training one small adapter, not touching the base model.

**Why LoRA?**
LoRA adapters are small (~16 MB each vs 4 GB for the full model), fast to train, and can be swapped at inference time without reloading the base model. This makes the retrieval-and-compose pattern practical.

**Why curated training data?**
We use small, hand-written datasets (20 examples per style) rather than large scraped corpora. This gives direct control over what each style looks like and avoids noise. The datasets are small enough that training is fast but targeted enough that the adapters learn clear style differences.

**Why FAISS?**
FAISS provides sub-millisecond nearest-neighbor search, making retrieval negligible compared to generation time.

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
