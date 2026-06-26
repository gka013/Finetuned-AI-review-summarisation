# Norwegian Sentiment-Aware Summarisation

A Norwegian NLP project for generating concise, sentiment-aware summaries of user-generated book reviews.

The project combines targeted sentiment analysis, parameter-efficient fine-tuning of Norwegian large language models, and quantitative evaluation. It was originally developed as a course project and has, out of curiosity, been extended with production-oriented components using Claude Code: a commercial API integration, a REST service layer, and an MCP server exposing the pipeline as an agent-callable tool.

While the current domain is Norwegian book reviews, the project demonstrates a transferable workflow for developing AI-powered summarisation components. With domain-specific data, evaluation, and appropriate safeguards, the same architecture could be adapted for document archives, email threads, support tickets, or case overviews.

> **Note:** The fine-tuned models were trained and evaluated for Norwegian summarisation tasks. They are not production-ready for administrative or sensitive documents without domain-specific evaluation, access control, privacy assessment, and human review.

## Overview

The system takes Norwegian book reviews for a given title and produces an abstractive summary of how readers received the book.

Rather than passing raw reviews directly to the summarisation model, the pipeline first identifies positive and negative sentiment targets using a dedicated Norwegian NLP model. These are used to construct a compact, sentiment-aware input for the language model.

```text
Norwegian book reviews (218 837 reviews, Bokelskere.no)
        ↓
Data cleaning and grouping by title (top 40 most-reviewed books)
        ↓
Targeted sentiment analysis (NorBERT3-TSA)
        ↓
Positive and negative sentiment snippets
        ↓
Prompt construction
        ↓
Fine-tuned Norwegian LLM  (Llama-2 or NorMistral, 4-bit QLoRA)
        ↓
Abstractive, sentiment-aware review summary
        ↓
ROUGE-L evaluation + human evaluation (SummEval dimensions)
```

## Why This Project?

Long, unstructured text can be difficult to navigate. In many workflows, users need a quick overview of:

* What a document or collection of texts is about
* The most important positive and negative points, where sentiment is relevant
* What information may be useful for follow-up, prioritisation, or decision-making

This project explores one reusable component in that broader workflow: turning multiple pieces of Norwegian text into a short, coherent, source-informed overview.

## Data

The project uses user-generated Norwegian book reviews from Bokelskere.no, obtained through the National Library of Norway.

The original corpus contains reviews with metadata including:

* Review text
* Book title and author
* User rating (1-6)
* Publication date
* User and post identifiers

For this project, the data was filtered to reviews associated with the 40 most-reviewed books (1 828 reviews after preprocessing).

### Preprocessing

The preprocessing pipeline:

* Removes incomplete or unsuitable records and replaces string `"NULL"` with missing values
* Drops records missing `main_title`
* Cleans URLs, digits, and non-standard symbols from review text
* Groups reviews by book title and aggregates into a book-level input
* Preserves relevant metadata such as author and mean rating

## Targeted Sentiment Analysis

The pipeline uses [`ltg/norbert3-large_TSA`](https://huggingface.co/ltg/norbert3-large_TSA) to identify positive and negative sentiment targets in review text. The extracted snippets are organised into a compact representation:

```text
Leserne likte: språk, karakterer, familiehistorie.
Leserne mislikte: forutsigbar handling og lav spenning.
```

This approach gives the summarisation model a focused representation of the most salient aspects of reader feedback, rather than requiring it to process every raw review directly.

## Fine-Tuning

Two Norwegian instruction-tuned 7B language models were fine-tuned for abstractive summarisation:

* [`RuterNorway/Llama-2-7b-chat-norwegian`](https://huggingface.co/RuterNorway/Llama-2-7b-chat-norwegian)
* [`norallm/normistral-7b-warm`](https://huggingface.co/norallm/normistral-7b-warm)

Fine-tuning used the [Norwegian Summarisation Benchmark Dataset (NorSumm)](https://huggingface.co/datasets/SamiaT/NorSumm), combining Bokmål and Nynorsk subsets, with supervised fine-tuning via QLoRA to reduce memory requirements.

| Setting | Value |
| --- | --- |
| Fine-tuning method | QLoRA + supervised fine-tuning |
| Quantisation | 4-bit NF4 |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, down projections + lm_head |
| Learning rate | 1e-5 |
| Epochs | 4 (early stopping on ROUGE-L) |
| Optimiser | `paged_adamw_8bit` |
| Hardware | Single NVIDIA A100 (Google Colab) |

The fine-tuned checkpoints are available on Hugging Face:

* [GloriaABK1/llama-2-7b-chat-norwegian-sum](https://huggingface.co/GloriaABK1/llama-2-7b-chat-norwegian-sum)
* [GloriaABK1/normistral-7b-warm-norsumm](https://huggingface.co/GloriaABK1/normistral-7b-warm-norsumm)

## Prompting and Generation

The models receive an instruction to transform the positive and negative sentiment snippets into a short, fluent Norwegian review summary. Generation uses:

* Beam search with `num_beams=4`
* Length penalty of `3.0`
* No-repeat n-gram size of `3`
* `max_new_tokens=120`

## Evaluation

### Automatic Evaluation (ROUGE-L)

Generated summaries were scored with **ROUGE-L F1** against the aggregated cleaned review text as a pseudo-reference, which is a reasonable proxy when no gold summaries exist for this domain.

| Model | ROUGE-L F1 |
| --- | --- |
| Llama-2-7b-chat-norwegian | *(see notebook)* |
| NorMistral-7b-warm | *(see notebook)* |

> ROUGE-L rewards longest-common-subsequence overlap and captures fluent paraphrasing beyond strict n-gram matching. Note that pseudo-reference scoring inflates scores for extractive outputs; human evaluation provides a complementary quality signal.

### Human Evaluation (SummEval)

Because the book-review dataset did not include gold-standard reference summaries, the generated summaries were also evaluated manually using four dimensions from the SummEval framework. Three annotators calibrated rating criteria and achieved acceptable inter-annotator agreement before evaluating all 40 generated summaries.

| Dimension | Description |
| --- | --- |
| Coherence | Whether the summary is well structured and logically organised |
| Consistency | Whether the summary is faithful to the source material |
| Fluency | Whether the language is grammatically correct and readable |
| Relevance | Whether the summary captures the most important content |

### Main Results

| Dimension | Fine-tuned Llama | Fine-tuned NorMistral |
| --- | ---: | ---: |
| Coherence | 2.18 | 2.05 |
| Consistency | 1.75 | 1.43 |
| Fluency | 2.10 | 2.25 |
| Relevance | 2.15 | 1.40 |

The fine-tuned Llama model outperformed NorMistral on coherence, consistency, and relevance. NorMistral performed slightly better on fluency.

## Extended Integrations

> These integrations were not part of the original course project. They were added afterwards out of curiosity — to explore what a more complete, production-oriented version of this pipeline could look like.

Beyond the core pipeline, the notebook demonstrates how this component would fit into a production AI system:

### Commercial API Comparison

The same Norwegian summarisation prompt is run through `claude-opus-4-8` (Anthropic Python SDK) alongside the fine-tuned open models, illustrating the quality/cost/data-residency trade-off relevant to any deployment decision:

| Axis | Fine-tuned open model | Claude API |
| --- | --- | --- |
| Cost per call | GPU amortised | Pay-per-token |
| Data residency | On-premise | Processed by Anthropic |
| Customisation | Domain-adapted (QLoRA) | Prompt-only |
| Norwegian quality | Fine-tuned on Bokmål + Nynorsk | General multilingual |

### REST Service Layer (FastAPI)

A production-ready FastAPI application wraps the NorMistral model as an HTTP endpoint with Pydantic request/response schemas, lifespan model loading (loaded once at startup), and a `/health` endpoint. On-premise deployment behind an API gateway satisfies GDPR data-transfer requirements for Norwegian user data.

### MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server exposes the summarisation function as a `summarize_book_reviews` tool, making it callable from any MCP-compatible LLM agent (Claude Desktop, Cursor, VS Code Copilot). This demonstrates the bridge between a standalone ML inference pipeline and agentic workflows.

### Local Deployment

The 4-bit quantised adapters can be served on-premise using **Ollama** or **vLLM**, eliminating data egress for privacy-sensitive deployments.

## Example

### Input

```text
Leserne likte: språket, karakterene, familiehistorien og den gode fortellingen.
Leserne mislikte: handlingen var tidvis lite spennende og forutsigbar.
```

### Generated Summary

```text
Boken ble generelt godt mottatt av leserne, som særlig fremhevet språket,
karakterene og den sterke familiehistorien. Flere opplevde fortellingen som
engasjerende og velskrevet. Enkelte mente imidlertid at handlingen kunne være
forutsigbar og manglet spenning i enkelte partier.
```

## Repository Structure

```text
.
├── norwegian_sentiment_summarization_pipeline.ipynb   # Full pipeline: fine-tuning, inference, evaluation, integrations
├── requirements.txt
└── README.md
```

The Bokelskere.no corpus (`2019_bokelskere.json`) is not included in this repository. It must be obtained separately through the National Library of Norway, and users are responsible for complying with the relevant terms of use.

The notebook was developed and run on Google Colab with an A100 GPU. The fine-tuned model checkpoints are hosted on Hugging Face and are loaded at inference time.

## Limitations and Responsible Use

This project is a research prototype with important limitations:

* The models were fine-tuned on Norwegian news summarisation data (NorSumm) and evaluated on book-review summarisation — not administrative or sensitive documents.
* Generated summaries may hallucinate, omit important details, repeat content, or produce incomplete output.
* User-generated text may contain inaccurate, biased, or harmful language that can be reflected in generated summaries.
* The current implementation does not include authentication, authorisation, audit logging, monitoring, or a human approval interface.

A real deployment for internal documents would require:

* Domain-specific training or evaluation data
* Privacy and data-protection assessment (GDPR)
* Access control aligned with the source system
* Logging and monitoring
* Output validation
* Clear disclosure of AI-generated content
* Human review before consequential use
