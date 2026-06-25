---
library_name: transformers
tags:
- norwegian
- summarization
- qlora
- peft
- causal-lm
language:
- 'no'
base_model: norallm/normistral-7b-warm
datasets:
- SamiaT/NorSumm
---

# normistral-7b-warm-norsumm

A QLoRA fine-tune of [norallm/normistral-7b-warm](https://huggingface.co/norallm/normistral-7b-warm)
for abstractive summarization of Norwegian text. Given structured sentiment snippets extracted from
book reviews, the model produces a fluent 3-5 sentence Norwegian prose summary of reader reception.

## Model Details

### Model Description

- **Developed by:** GloriaABK1
- **Model type:** Causal LM with LoRA adapters (QLoRA)
- **Language(s):** Norwegian (Bokmål + Nynorsk)
- **License:** Same as base model ([norallm/normistral-7b-warm](https://huggingface.co/norallm/normistral-7b-warm))
- **Finetuned from:** [norallm/normistral-7b-warm](https://huggingface.co/norallm/normistral-7b-warm)

### Model Sources

- **Repository:** https://github.com/GloriaABK/norwegian-book-review-summarization
- **Training notebook:** See repository

## Uses

### Direct Use

Generate abstractive summaries of Norwegian text. The model expects the prompt format below
and outputs a short prose paragraph in Norwegian.

### Downstream Use

Originally built as part of an end-to-end Norwegian book review summarization pipeline:

1. 218 k reviews from Bokelskere.no are preprocessed and filtered to the 40 most-reviewed books
2. [ltg/norbert3-large_TSA](https://huggingface.co/ltg/norbert3-large_TSA) tags positive and negative sentiment spans per review
3. Spans are aggregated into a structured input string per book
4. This model generates a final prose summary

The adapter can also be loaded on top of the base model for general Norwegian summarization tasks.

### Out-of-Scope Use

- Non-Norwegian text (model is not instruction-tuned for multilingual use)
- Long-document summarization (max input is 600 tokens)
- Factual question answering or retrieval

## How to Get Started with the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

repo = "GloriaABK1/normistral-7b-warm-norsumm"
model = AutoModelForCausalLM.from_pretrained(
    repo,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(repo)
model.eval()

def summarize(text: str) -> str:
    prompt = (
        "### Instruksjon:\n"
        "Du er en bokanmeldelsessammendragsgenerator. Gjor om folgende "
        "Leserne likte- og Leserne mislikte-tagger til et 3-5 setningers "
        "sammendrag pa norsk.\n\n"
        f"### Inndata:\n{text}\n\n### Sammendrag:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=600).to(model.device)
    inputs.pop("token_type_ids", None)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4,
            length_penalty=3.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).split("### Sammendrag:")[-1].strip()

text = "Leserne likte: historien, karakterene, språket. Leserne mislikte: slutten, tempoet."
print(summarize(text))
```

## Training Details

### Training Data

Fine-tuned on [SamiaT/NorSumm](https://huggingface.co/datasets/SamiaT/NorSumm), a Norwegian
summarization dataset with both Bokmål (`nb`) and Nynorsk (`nn`) subsets. Both subsets were
merged and shuffled before training. The validation split was further divided 80/20 to create
a dedicated validation set for early stopping.

| Split | Rows (approx.) |
|---|---|
| Train | ~48 |
| Validation | ~12 |
| Test | ~60 |

Each example consists of a Norwegian article and one or more reference summaries.
The `summaries` column was exploded so each (article, summary) pair forms its own training row.

### Training Procedure

**Method:** QLoRA - 4-bit quantized base model with LoRA adapters trained in bf16.

#### Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Training regime | bf16 mixed precision |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head |
| Quantization | 4-bit NF4 |
| Optimizer | paged_adamw_8bit |
| Learning rate | 1e-5 |
| LR scheduler | constant |
| Warmup ratio | 0.03 |
| Epochs | 4 (early stopping patience = 1 on eval ROUGE-L) |
| Per-device batch size | 4 |
| Gradient accumulation steps | 2 |
| Max gradient norm | 0.3 |
| Weight decay | 0.001 |
| Max input length | 512 tokens |
| Max target length | 120 tokens |

#### Speeds, Sizes, Times

- **Hardware:** Google Colab A100 (40 GB)
- **Adapter size:** ~849 MB
- **Training time:** approximately 1-2 hours on A100

## Evaluation

### Testing Data and Metrics

Evaluated on the NorSumm test split using ROUGE-L F1, computed with the `evaluate` library.
Early stopping during training used ROUGE-L on the validation set.

The model was also applied to the Bokelskere.no book review corpus and ROUGE-L was computed
against aggregated review text as a pseudo-reference, providing a domain-specific quality signal.

### Results

| Metric | Value |
|---|---|
| ROUGE-L F1 (NorSumm test) | see trainer.evaluate() output in training notebook |

## Bias, Risks, and Limitations

- The base model was pre-trained predominantly on English text; Norwegian cultural and linguistic
  nuances may be underrepresented compared to a natively trained Norwegian model.
- The NorSumm training set is small (~60 examples before splitting), which limits generalization
  and makes the model sensitive to prompt format.
- ROUGE-L against pseudo-references rewards extractive overlap; the model may copy source phrases
  rather than produce fully abstractive summaries.
- The model is not suitable for privacy-sensitive inference against real user data without
  on-premise deployment (see Downstream Use above).

### Recommendations

Use this model as part of a pipeline where inputs are pre-processed and validated. Do not use
for summarizing personally identifiable information without appropriate data governance controls.
Evaluate outputs with human review before any production deployment.

## Environmental Impact

- **Hardware type:** NVIDIA A100 40 GB
- **Cloud provider:** Google (Colab)
- **Compute region:** Unknown
- **Hours used:** approximately 1-2
- Carbon emissions can be estimated using the [ML Impact Calculator](https://mlco2.github.io/impact#compute).

## Technical Specifications

### Model Architecture and Objective

Causal language model (Mistral-7B architecture) with LoRA adapters on all attention projection
layers and MLP gate/up/down projections plus the LM head. Base weights are frozen and loaded
in 4-bit NF4 quantization via `bitsandbytes`. Only the LoRA adapter weights are trained.

**Objective:** Next-token prediction on (prompt + summary) sequences; the prompt portion is not
masked during loss computation (SFTTrainer default).

### Compute Infrastructure

#### Hardware
Google Colab A100 (40 GB HBM2)

#### Software
- `transformers` (HuggingFace)
- `peft` (LoRA implementation)
- `trl` (SFTTrainer)
- `bitsandbytes` (4-bit quantization)
- `evaluate` (ROUGE-L metrics)
